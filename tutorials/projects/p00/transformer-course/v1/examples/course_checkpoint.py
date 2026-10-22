"""Save immutable full training generations and restore compatible state."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
import uuid

import numpy as np
import torch

SCHEMA = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def runtime_identity(identity, model):
    """Include runtime semantics in the caller's model/tokenizer/data identity."""
    return {"contract": identity, "torch": str(torch.__version__),
            "numpy": str(np.__version__), "schema": SCHEMA,
            "model_class": type(model).__module__ + "." + type(model).__name__,
            "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
            "parameter_devices": sorted({str(p.device) for p in model.parameters()}),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "python": list(sys.version_info[:3]), "threads": torch.get_num_threads(),
            "matmul_precision": torch.get_float32_matmul_precision()}


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return deepcopy(value)


def _rng(generators, *, needs_cuda=False):
    numpy = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"name": numpy[0], "keys": numpy[1].tolist(),
                  "position": numpy[2], "has_gauss": numpy[3], "cached": numpy[4]},
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if needs_cuda else [],
        "generators": {name: generator.get_state() for name, generator in generators.items()},
    }


def _restore_rng(state, generators):
    if set(state["generators"]) != set(generators):
        raise ValueError("checkpoint generator names do not match the trainer")
    if state["cuda"] and (not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count()):
        raise ValueError("CUDA RNG mapping differs; exact continuation is unavailable")
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n["name"], np.array(n["keys"], dtype=np.uint32), n["position"], n["has_gauss"], n["cached"]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    for name, generator in generators.items():
        generator.set_state(state["generators"][name])


def _verified(folder):
    """Read only generations whose completion record and persisted hash agree."""
    try:
        complete = json.loads((folder / "complete.json").read_text())
        manifest_bytes = (folder / "manifest.json").read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != complete["manifest_sha256"]:
            return None
        manifest = json.loads(manifest_bytes)
        if manifest["schema"] != SCHEMA:
            return None
        blob = (folder / "state.pt").read_bytes()
        if len(blob) != manifest["bytes"] or hashlib.sha256(blob).hexdigest() != manifest["sha256"]:
            return None
        return manifest
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_generation(directory, *, model, optimizer, scheduler, step, identity,
                    data_state=None, generators=None, scaler=None, keep=2):
    """Save after a completed update; publish completion last and retain recovery."""
    if step < 0 or keep < 2:
        raise ValueError("nonnegative update count and at least two retained generations are required")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    generations = [folder for folder in directory.iterdir() if folder.is_dir()]
    ordinal = 1 + max((int(folder.name.split("-", 1)[0]) for folder in generations
                       if folder.name.split("-", 1)[0].isdigit()), default=0)
    folder = directory / f"{ordinal:08d}-{uuid.uuid4().hex[:8]}"
    folder.mkdir()
    generators = generators or {}
    identity = runtime_identity(identity, model)
    payload = {
        "schema": SCHEMA, "identity": identity, "step": step,
        "model": _cpu(model.state_dict()), "optimizer": _cpu(optimizer.state_dict()),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "data_state": _cpu(data_state or {}),
        "rng": _rng(generators, needs_cuda=any(p.device.type == "cuda" for p in model.parameters())
                    or any(generator.device.type == "cuda" for generator in generators.values())),
        "model_training": model.training,
    }
    # Each call creates a new generation; an interrupted write leaves no completion.
    torch.save(payload, folder / "state.pt")
    blob = (folder / "state.pt").read_bytes()
    # Check deserialization before marking these persisted bytes as recoverable.
    torch.load(folder / "state.pt", map_location="cpu", weights_only=True)
    manifest = {"schema": SCHEMA, "ordinal": ordinal, "step": step,
                "identity": identity, "bytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest()}
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    (folder / "manifest.json").write_bytes(manifest_bytes)
    (folder / "complete.json").write_text(json.dumps({"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}))
    if _verified(folder) is None:
        raise OSError("persisted checkpoint did not verify")
    valid = [(candidate, _verified(candidate)) for candidate in directory.iterdir() if candidate.is_dir()]
    valid = sorted(((candidate, info) for candidate, info in valid if info is not None),
                   key=lambda item: item[1]["ordinal"], reverse=True)
    # Rotate only compatible verified generations; preserve other run identities.
    compatible = [(candidate, info) for candidate, info in valid if info["identity"] == identity]
    for candidate, _ in compatible[keep:]:
        for filename in ("complete.json", "manifest.json", "state.pt"):
            (candidate / filename).unlink()
        candidate.rmdir()
    return folder


def load_latest(directory, *, model, optimizer, scheduler, identity,
                generators=None, scaler=None):
    """Restore the newest compatible verified generation and report skipped writes."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError("no checkpoint directory exists")
    identity = runtime_identity(identity, model)
    candidates = sorted((p for p in directory.iterdir() if p.is_dir()), reverse=True)
    skipped = []
    for folder in candidates:
        manifest = _verified(folder)
        if manifest is None:
            skipped.append({"generation": folder.name, "reason": "incomplete or corrupt"})
            continue
        if manifest["identity"] != identity:
            skipped.append({"generation": folder.name, "reason": "incompatible identity"})
            continue
        try:
            payload = torch.load(folder / "state.pt", map_location="cpu", weights_only=True)
        except (OSError, ValueError, RuntimeError):
            skipped.append({"generation": folder.name, "reason": "unloadable state"})
            continue
        if payload["schema"] != SCHEMA or payload["identity"] != identity:
            skipped.append({"generation": folder.name, "reason": "inconsistent payload"})
            continue
        expected = model.state_dict()
        if set(payload["model"]) != set(expected) or any(
            payload["model"][name].shape != tensor.shape for name, tensor in expected.items()
        ):
            raise ValueError("model state does not match the declared identity")
        if (scheduler is None) != (payload["scheduler"] is None) or (scaler is None) != (payload["scaler"] is None):
            raise ValueError("scheduler/scaler state does not match the trainer")
        if set(payload["rng"]["generators"]) != set(generators or {}):
            raise ValueError("checkpoint generator names do not match the trainer")
        model.load_state_dict(payload["model"], strict=True)
        model.train(payload["model_training"])
        optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None:
            scaler.load_state_dict(payload["scaler"])
        # Restore randomness after constructing/loading every component, before data.
        _restore_rng(payload["rng"], generators or {})
        return {"step": payload["step"], "data_state": payload["data_state"],
                "generation": folder.name, "skipped": skipped}
    raise ValueError("no compatible verified full-state checkpoint is available")
