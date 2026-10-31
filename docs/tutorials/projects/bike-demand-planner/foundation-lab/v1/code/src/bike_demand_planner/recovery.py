"""Fresh-process training continuation under one fixed local numerical contract."""

import argparse
import copy
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .artifacts import digest, implementation_identity, json_bytes, read_json
from .checkpoints import load_generation, save_generation
from .evaluation import load_labelled_partition
from .features import FEATURE_NAMES
from .optimization import BatchStream, TrialConfig, fit_trial, scheduled_rate
from .optimizers import OptimizerConfig, OptimizerState, optimizer_step
from .regression import FeatureStandardizer, fit_standardizer


def recovery_configuration(count: int) -> TrialConfig:
    return TrialConfig(optimizer=OptimizerConfig("adam"), learning_rate=.01, batch_size=32, seed=17,
                       max_examples=3*count, max_updates=10_000, max_seconds=300, decay_examples=count)


def payload_for(trial, previous: dict | None = None) -> dict:
    points = list(previous["monitoring"]["history"]) if previous else []
    for point in trial.history:
        value = {k:v for k,v in asdict(point).items() if k != "elapsed_seconds"}
        if points and points[-1]["updates"] == point.updates:
            points[-1] = value
        else:
            points.append(value)
    return {"version": "hourly-training-checkpoint-v1", "trainer_state": trial.state,
            "monitoring": {"history": points, "best_training_objective": min(p["half_mse"] for p in points),
                           "stop_reason": trial.stop_reason, "last_batch_size": trial.last_batch_size,
                           "early_stopping": "disabled; fixed example budget"}}


def _training(processed: Path):
    rows, manifest = load_labelled_partition(processed, "train")
    return np.asarray([r.features for r in rows]), np.asarray([r.target for r in rows]), manifest


def _context(project_root: Path, X, scaler, manifest, config):
    return {"version": "hourly-training-context-v1", "implementation": implementation_identity(project_root),
            "training_count": len(X), "feature_names": list(FEATURE_NAMES), "standardizer": asdict(scaler),
            "configuration": asdict(config), "dataset_manifest_sha256": manifest["manifest_sha256"],
            "training_sha256": manifest["artifacts_sha256"]["train.csv"], "dtype": "float64",
            "compute": "local CPU; one BLAS/OpenMP thread; fixed source/runtime; time cap is a guard"}


def _worker(mode: str, processed: Path, work: Path, project_root: Path, expected_context_sha256=None):
    X, y, manifest = _training(processed)
    config = recovery_configuration(len(X))
    if mode in ("reference", "pause"):
        scaler = fit_standardizer(X)
        context = _context(project_root, X, scaler, manifest, config)
        # JSON round-trip fixes tuple/list representation before comparing contexts.
        context = json.loads(json_bytes(context))
    else:
        if digest((work/"contract.json").read_bytes()) != expected_context_sha256:
            raise ValueError("Training context differs from its trusted identity.")
        context = read_json(work / "contract.json")
        scaler = FeatureStandardizer(**context["standardizer"])
        current = json.loads(json_bytes(_context(project_root, X, scaler, manifest, config)))
        if current != context:
            raise ValueError("Current source/runtime/data/schema/transform/config differs from the checkpoint context.")
    Z = scaler.transform(X)
    if mode == "reference":
        full = fit_trial(Z, y, config)
        (work/"contract.json").write_bytes(json_bytes(context))
        result = payload_for(full)
    elif mode == "pause":
        if context != read_json(work/"contract.json"):
            raise ValueError("Pause run differs from the reference contract.")
        first = fit_trial(Z, y, config, pause_after_updates=269)
        one = payload_for(first)
        save_generation(work/"generations", one, context)
        second = fit_trial(Z, y, config, continuation=one["trainer_state"], pause_after_updates=273)
        two = payload_for(second, one)
        save_generation(work/"generations", two, context)
        third = fit_trial(Z, y, config, continuation=two["trainer_state"], pause_after_updates=274)
        latest = save_generation(work/"generations", payload_for(third, two), context)
        # Deliberate fault injection retains the corrupt and incomplete artifacts.
        with (latest/"state.json").open("ab") as output:
            output.write(b"damaged-save-probe")
        incomplete = work/"generations/.generation-000004.pending"
        incomplete.mkdir()
        (incomplete/"state.json").write_text('{"interrupted":')
        result = {"paused_step": 273, "paused_examples": two["trainer_state"]["batch_stream"]["examples_seen"],
                  "next_rate": two["trainer_state"]["schedule"]["next_learning_rate"],
                  "next_batch": BatchStream.from_dict(two["trainer_state"]["batch_stream"]).next_indices().tolist(),
                  "injected_faults": ["newest state byte corruption", "unpublished incomplete directory"]}
    else:
        saved, fallback = load_generation(work/"generations", context)
        original = copy.deepcopy(saved)
        state = saved["trainer_state"]
        batch = BatchStream.from_dict(state["batch_stream"]).next_indices()
        theta = np.asarray(state["parameters"])
        error = Z[batch]@theta[:-1]+theta[-1]-y[batch]
        gradient = np.append(Z[batch].T@error/len(batch), error.mean())
        old_optimizer = OptimizerState.from_dict(state["optimizer"], len(theta))
        rate = scheduled_rate(config, state["batch_stream"]["examples_seen"])
        updated, updated_optimizer = optimizer_step(theta, gradient, old_optimizer, config.optimizer, rate)
        coordinate = 8
        trace = {"feature": FEATURE_NAMES[coordinate], "coordinate": coordinate, "batch_count": len(batch),
                 "before_step": old_optimizer.step, "after_step": updated_optimizer.step, "learning_rate": rate,
                 "old_weight": theta[coordinate], "gradient": gradient[coordinate],
                 "old_first": old_optimizer.first[coordinate], "old_second": old_optimizer.second[coordinate],
                 "new_first": updated_optimizer.first[coordinate], "new_second": updated_optimizer.second[coordinate],
                 "corrected_first": updated_optimizer.first[coordinate]/(1-config.optimizer.beta1**updated_optimizer.step),
                 "corrected_second": updated_optimizer.second[coordinate]/(1-config.optimizer.beta2**updated_optimizer.step),
                 "new_weight": updated[coordinate], "batch_positions": batch.tolist()}
        restored = fit_trial(Z, y, config, continuation=saved["trainer_state"])
        result = payload_for(restored, saved)
        result["recovery"] = fallback
        result["resumed_first_batch"] = list(restored.first_batches[0])
        result["next_update_trace"] = trace
        assert saved == original, "Continuation must not mutate the restored checkpoint"
        altered = copy.deepcopy(saved["trainer_state"])
        altered["optimizer"]["first"] = [0.0]*len(altered["parameters"])
        altered["optimizer"]["second"] = [0.0]*len(altered["parameters"])
        reset = fit_trial(Z, y, config, continuation=altered)
        result["zeroed_moment_control_max_parameter_difference"] = float(np.max(np.abs(reset.parameters-restored.parameters)))
    (work/(mode+".json")).write_bytes(json_bytes(result))


def run_recovery(processed: Path, destination: Path, project_root: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    protocol = {"version": "hourly-local-recovery-proof-v1", "seed": 17, "batch_size": 32,
                "optimizer": "adam", "rate": .01, "passes": 3, "decay": "inverse time by training examples",
                "pause_updates": [269, 273], "faults": ["corrupt generation 3", "incomplete pending generation"],
                "objective": "same numerical updates/order after restart; no model selection or new validation tuning",
                "max_seconds_per_training_path": 300, "test_evaluated": False, "hosted_recovery": False}
    (destination/"protocol.json").write_bytes(json_bytes(protocol))
    record = {"status": "running", "protocol": protocol}
    try:
        for mode in ("reference", "pause", "resume"):
            args = [sys.executable, "-m", "bike_demand_planner.recovery", "--mode", mode,
                    "--processed", str(processed), "--work", str(destination), "--project", str(project_root)]
            if mode == "resume":
                args += ["--context-sha256", digest((destination/"contract.json").read_bytes())]
            env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "GIT_OPTIONAL_LOCKS": "0"}
            with (destination/(mode+".stdout.txt")).open("w") as out, (destination/(mode+".stderr.txt")).open("w") as err:
                subprocess.run(args, cwd=destination, env=env, stdout=out, stderr=err, check=True, timeout=360)
        full, resumed, pause = (read_json(destination/(name+".json")) for name in ("reference", "resume", "pause"))
        state_keys = ("version", "configuration", "resumable", "parameters", "optimizer", "batch_stream", "schedule")
        if any(full["trainer_state"][key] != resumed["trainer_state"][key] for key in state_keys):
            raise ValueError("Fresh-process continuation differs from the uninterrupted numerical state.")
        if resumed["resumed_first_batch"] != pause["next_batch"]:
            raise ValueError("Resumed next batch differs from the paused stream.")
        common = {p["updates"]:p for p in resumed["monitoring"]["history"]}
        if any(common[p["updates"]] != p for p in full["monitoring"]["history"]):
            raise ValueError("Common monitoring observations differ after recovery.")
        record.update(status="completed", context_sha256=digest((destination/"contract.json").read_bytes()),
                      paused=pause, recovery=resumed["recovery"], final_updates=resumed["trainer_state"]["optimizer"]["step"],
                      final_examples=resumed["trainer_state"]["batch_stream"]["examples_seen"],
                      final_training_half_mse=resumed["monitoring"]["history"][-1]["half_mse"],
                      exact_numeric_state_parity=True, exact_next_batch_parity=True, common_monitoring_parity=True,
                      zeroed_moment_control_max_parameter_difference=resumed["zeroed_moment_control_max_parameter_difference"],
                      next_update_trace=resumed["next_update_trace"],
                      elapsed_seconds_scope="wall-clock fields may differ and are not part of bitwise numerical parity")
    except Exception as error:
        record.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        (destination/"recovery.json").write_bytes(json_bytes(record))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("reference", "pause", "resume"), required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--context-sha256")
    args = parser.parse_args()
    _worker(args.mode, args.processed, args.work, args.project, args.context_sha256)
