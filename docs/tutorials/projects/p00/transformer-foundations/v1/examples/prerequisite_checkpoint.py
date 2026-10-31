"""Publish verified full-state generations and restore compatible CPU training."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import random
import shutil
import tempfile
import time
import uuid

import numpy as np
import torch

FORMAT = "prerequisite-mlp-full-state-v1"


def _copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy(item) for item in value)
    return deepcopy(value)


def random_state():
    numpy = np.random.get_state()
    return {"python": random.getstate(), "numpy": {"kind": numpy[0], "keys": numpy[1].tolist(), "position": numpy[2], "has_gauss": numpy[3], "cached": numpy[4]}, "torch_cpu": torch.get_rng_state().clone()}


def restore_random_state(saved):
    random.setstate(saved["python"])
    numpy = saved["numpy"]
    np.random.set_state((numpy["kind"], np.array(numpy["keys"], dtype=np.uint32), numpy["position"], numpy["has_gauss"], numpy["cached"]))
    torch.set_rng_state(saved["torch_cpu"])


def snapshot(state):
    if any(parameter.device.type != "cpu" or parameter.dtype != torch.float32 for parameter in state.model.parameters()):
        raise ValueError("The prerequisite checkpoint profile supports FP32 CPU parameters only")
    if any(parameter.grad is not None for parameter in state.model.parameters()):
        raise ValueError("Save only after a completed update with no pending gradients")
    return {"format": FORMAT, "identity": state.identity(), "model": _copy(state.model.state_dict()), "optimizer": _copy(state.optimizer.state_dict()), "scheduler": _copy(state.scheduler.state_dict()), "sampler": _copy(state.sampler.state_dict()), "step": state.step, "examples_seen": state.examples_seen, "metric_position": state.step, "best_validation_loss": state.best_validation_loss, "rng": random_state(), "model_training": state.model.training, "precision": {"dtype": "float32", "scaler": None}, "accumulation": {"pending": 0}}


def verified_generation(folder):
    try:
        folder = Path(folder)
        complete = json.loads((folder / "complete.json").read_text())
        manifest_bytes = (folder / "manifest.json").read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != complete["manifest_sha256"]:
            return None
        manifest = json.loads(manifest_bytes)
        blob = (folder / "state.pt").read_bytes()
        if manifest["format"] != FORMAT or len(blob) != manifest["bytes"] or hashlib.sha256(blob).hexdigest() != manifest["sha256"]:
            return None
        payload = torch.load(folder / "state.pt", map_location="cpu", weights_only=True)
        required = {"format", "identity", "model", "optimizer", "scheduler", "sampler", "step", "examples_seen", "metric_position", "best_validation_loss", "rng", "model_training", "precision", "accumulation"}
        if not isinstance(payload, dict) or set(payload) != required:
            return None
        if payload["format"] != FORMAT or payload["step"] != manifest["step"]:
            return None
        identity_bytes = json.dumps(payload["identity"], sort_keys=True).encode()
        if hashlib.sha256(identity_bytes).hexdigest() != manifest["identity_sha256"]:
            return None
        return manifest
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError):
        return None


def generations(directory):
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted([path for path in directory.iterdir() if path.is_dir() and path.name.split("-", 1)[0].isdigit()], key=lambda path: int(path.name.split("-", 1)[0]), reverse=True)


def storage_preflight(directory, blob_bytes, keep, identity_sha256, storage_budget_bytes, free_reserve_bytes):
    """Reserve one unpublished generation while protecting current compatible copies."""
    if storage_budget_bytes <= 0 or free_reserve_bytes < 0:
        raise ValueError("Storage budget must be positive and free-space reserve nonnegative")
    all_bytes = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    compatible_sizes = []
    for folder in generations(directory):
        manifest = verified_generation(folder)
        if manifest is not None and manifest["identity_sha256"] == identity_sha256:
            compatible_sizes.append(sum(path.stat().st_size for path in folder.iterdir() if path.is_file()))
    foreign_bytes = all_bytes - sum(compatible_sizes)
    generation_bytes = max([blob_bytes + 4096, *compatible_sizes])
    required_capacity = max(all_bytes + generation_bytes, foreign_bytes + (keep + 1) * generation_bytes)
    free_bytes = shutil.disk_usage(directory).free
    if required_capacity > storage_budget_bytes:
        raise OSError("Checkpoint budget cannot protect retained copies and a new generation; training must stop")
    if free_bytes < generation_bytes + free_reserve_bytes:
        raise OSError("Insufficient free storage for a new generation plus reserve; training must stop")
    return {"storage_budget_bytes": storage_budget_bytes, "free_reserve_bytes": free_reserve_bytes, "required_capacity_bytes": required_capacity, "free_before_bytes": free_bytes}


def save_generation(directory, state, keep=2, storage_budget_bytes=8 * 1024 * 1024, free_reserve_bytes=1024 * 1024):
    """Write locally, copy into a new persistent generation, then publish completion."""
    if keep < 2:
        raise ValueError("Retain at least two verified full-state generations")
    started = time.perf_counter()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    existing = generations(directory)
    ordinal = 1 + max((int(path.name.split("-", 1)[0]) for path in existing), default=0)
    destination = directory / f"{ordinal:08d}-{uuid.uuid4().hex[:8]}"
    payload = snapshot(state)
    identity_sha256 = hashlib.sha256(json.dumps(payload["identity"], sort_keys=True).encode()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="prerequisite-state-") as temporary:
        local = Path(temporary) / "state.pt"
        torch.save(payload, local)
        torch.load(local, map_location="cpu", weights_only=True)
        blob = local.read_bytes()
        storage = storage_preflight(directory, len(blob), keep, identity_sha256, storage_budget_bytes, free_reserve_bytes)
        destination.mkdir()
        (destination / "state.pt").write_bytes(blob)
    persisted = (destination / "state.pt").read_bytes()
    if persisted != blob:
        raise OSError("Persisted state differs; generation is incomplete and training must stop")
    torch.load(destination / "state.pt", map_location="cpu", weights_only=True)
    manifest = {"format": FORMAT, "step": state.step, "examples_seen": state.examples_seen, "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(), "identity_sha256": identity_sha256}
    manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode()
    (destination / "manifest.json").write_bytes(manifest_bytes)
    if (destination / "manifest.json").read_bytes() != manifest_bytes:
        raise OSError("Persisted manifest differs; training must stop")
    (destination / "complete.json").write_text(json.dumps({"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}))
    if verified_generation(destination) is None:
        raise OSError("Persistent generation failed verification; training must stop")
    # Rotation happens only after the new persistent generation has verified.
    valid = []
    for folder in generations(directory):
        verified = verified_generation(folder)
        if verified is not None and verified["identity_sha256"] == identity_sha256:
            valid.append(folder)
    for folder in valid[keep:]:
        for file in folder.iterdir():
            file.unlink()
        folder.rmdir()
    return {"generation": destination.name, "step": state.step, "bytes": len(blob), "sha256": manifest["sha256"], "persistence_seconds": time.perf_counter() - started, **storage}


def restore_payload(payload, state):
    if payload.get("format") != FORMAT or payload.get("identity") != state.identity():
        raise ValueError("Source, config, data or runtime identity differs; start a separate run")
    if payload.get("accumulation") != {"pending": 0} or payload.get("precision") != {"dtype": "float32", "scaler": None}:
        raise ValueError("Only completed FP32 updates can be restored")
    expected = state.model.state_dict()
    if payload["model"].keys() != expected.keys() or any(payload["model"][name].shape != tensor.shape or payload["model"][name].dtype != tensor.dtype for name, tensor in expected.items()):
        raise ValueError("Model tensor mapping differs")
    if not 0 <= payload["step"] <= state.config.total_steps or payload["examples_seen"] < 0 or payload["metric_position"] != payload["step"]:
        raise ValueError("Saved progress is invalid")
    order = payload["sampler"]["order"]
    if sorted(order.tolist()) != list(range(len(state.sampler.order))) or not 0 <= payload["sampler"]["cursor"] <= len(order):
        raise ValueError("Saved data cursor/order is invalid")
    state.model.load_state_dict(payload["model"], strict=True)
    state.optimizer.load_state_dict(payload["optimizer"])
    state.scheduler.load_state_dict(payload["scheduler"])
    state.sampler.load_state_dict(payload["sampler"])
    state.step = payload["step"]
    state.examples_seen = payload["examples_seen"]
    state.best_validation_loss = payload["best_validation_loss"]
    state.model.train(payload["model_training"])
    state.optimizer.zero_grad(set_to_none=True)
    # Setup and loading happen first; restoration is last before the next draw.
    restore_random_state(payload["rng"])


def load_latest(directory, state):
    expected = state.identity()
    skipped = []
    for folder in generations(directory):
        manifest = verified_generation(folder)
        if manifest is None:
            skipped.append({"generation": folder.name, "reason": "incomplete or corrupt"})
            continue
        payload = torch.load(folder / "state.pt", map_location="cpu", weights_only=True)
        if payload["identity"] != expected:
            skipped.append({"generation": folder.name, "reason": "incompatible identity"})
            continue
        restore_payload(payload, state)
        return {"generation": folder.name, "step": state.step, "examples_seen": state.examples_seen, "skipped": skipped}
    raise ValueError("No compatible verified generation exists; resume does not start a new run")
