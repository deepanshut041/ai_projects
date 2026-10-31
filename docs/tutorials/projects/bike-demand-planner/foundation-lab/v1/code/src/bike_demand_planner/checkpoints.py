"""Verified full-state save generations with an explicit previous-generation fallback."""

import math
import re
from pathlib import Path

import numpy as np

from .artifacts import digest, json_bytes, publish_directory, read_json
from .features import FEATURE_NAMES
from .optimization import BatchStream, TrialConfig, scheduled_rate
from .optimizers import OptimizerConfig, OptimizerState


def check_payload(payload: dict, context: dict) -> None:
    if payload.get("version") != "hourly-training-checkpoint-v1":
        raise ValueError("Unsupported training checkpoint version.")
    state = payload["trainer_state"]
    if (context.get("version") != "hourly-training-context-v1" or context["feature_names"] != list(FEATURE_NAMES)
            or state.get("version") != "linear-optimization-state-v1"):
        raise ValueError("Checkpoint schema or trainer version is unsupported.")
    config = TrialConfig(**{**state["configuration"], "optimizer": OptimizerConfig(**state["configuration"]["optimizer"])})
    if state["configuration"] != context["configuration"] or not state["resumable"]:
        raise ValueError("Checkpoint configuration or success state is incompatible.")
    size = len(context["feature_names"]) + 1
    theta = np.asarray(state["parameters"], dtype=np.float64)
    if theta.shape != (size,) or not np.isfinite(theta).all():
        raise ValueError("Checkpoint parameters have an invalid shape or nonfinite values.")
    stream = BatchStream.from_dict(state["batch_stream"])
    optimizer = OptimizerState.from_dict(state["optimizer"], size)
    expected_updates = stream.epoch*math.ceil(stream.count/stream.batch_size) + math.ceil(stream.cursor/stream.batch_size)
    if (stream.count != context["training_count"] or stream.batch_size != config.batch_size
            or optimizer.step != expected_updates or stream.examples_seen > config.max_examples
            or state["schedule"]["next_learning_rate"] != scheduled_rate(config, stream.examples_seen)
            or state["schedule"]["policy"] != ("inverse_time_by_examples" if config.decay_examples else "constant")
            or not math.isfinite(state["elapsed_seconds"]) or state["elapsed_seconds"] < 0):
        raise ValueError("Checkpoint optimizer, data cursor, schedule or budget disagree.")
    scaler = context["standardizer"]
    if (len(scaler["means"]) != size-1 or len(scaler["scales"]) != size-1
            or not np.isfinite(scaler["means"]).all() or not np.isfinite(scaler["scales"]).all()
            or min(scaler["scales"]) <= 0 or scaler["training_count"] != stream.count):
        raise ValueError("Saved training transform is invalid.")
    monitoring = payload["monitoring"]
    points = monitoring["history"]
    if (monitoring["early_stopping"] != "disabled; fixed example budget" or not points
            or points[-1]["updates"] != optimizer.step or points[-1]["examples_processed"] != stream.examples_seen
            or any(not math.isfinite(p["half_mse"]) or p["half_mse"] < 0 for p in points)
            or monitoring["best_training_objective"] != min(p["half_mse"] for p in points)):
        raise ValueError("Unsupported monitoring or early-stop policy.")


def save_generation(directory: Path, payload: dict, context: dict) -> Path:
    check_payload(payload, context)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".writer-lock"
    lock.mkdir()
    try:
        numbers = [int(p.name.split("-")[1]) for p in directory.iterdir() if re.fullmatch(r"generation-\d{6}", p.name)]
        destination = directory / f"generation-{max(numbers, default=0)+1:06d}"
        files = {"state.json": json_bytes(payload), "context.json": json_bytes(context)}
        receipt = {"version": "training-save-generation-v1", "artifacts_sha256": {n:digest(v) for n,v in files.items()},
                   "optimizer_step": payload["trainer_state"]["optimizer"]["step"]}
        publish_directory(destination, {**files, "receipt.json": json_bytes(receipt)})
        return destination
    finally:
        lock.rmdir()


def load_generation(directory: Path, expected_context: dict) -> tuple[dict, dict]:
    failures = []
    candidates = sorted((p for p in directory.iterdir() if re.fullmatch(r"generation-\d{6}", p.name)), reverse=True)
    for candidate in candidates:
        try:
            receipt = read_json(candidate / "receipt.json")
            if receipt.get("version") != "training-save-generation-v1" or set(receipt["artifacts_sha256"]) != {"state.json", "context.json"}:
                raise ValueError("Incomplete generation receipt.")
            for name, identity in receipt["artifacts_sha256"].items():
                if digest((candidate/name).read_bytes()) != identity:
                    raise ValueError(f"Generation {name} hash mismatch.")
            context = read_json(candidate / "context.json")
            if context != expected_context:
                raise ValueError("Saved source/data/schema/transform/config/runtime context differs.")
            payload = read_json(candidate / "state.json")
            check_payload(payload, context)
            if receipt["optimizer_step"] != payload["trainer_state"]["optimizer"]["step"]:
                raise ValueError("Generation step differs from its receipt.")
            return payload, {"loaded_generation": candidate.name, "rejected_generations": failures,
                             "ignored_incomplete": sorted(p.name for p in directory.iterdir() if p.name.endswith(".pending")),
                             "fallback_used": bool(failures)}
        except (ValueError, OSError, KeyError, TypeError) as error:
            failures.append({"generation": candidate.name, "reason": str(error)})
    raise ValueError(f"No compatible verified training generation remains: {failures}")
