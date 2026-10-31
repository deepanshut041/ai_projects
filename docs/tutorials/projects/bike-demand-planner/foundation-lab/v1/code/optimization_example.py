"""Compare optimizer rules, batch policies and one rate schedule separately."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from bike_demand_planner.evaluation import load_labelled_partition
from bike_demand_planner.optimizers import OptimizerConfig, initial_state, optimizer_step
from bike_demand_planner.optimizer_evaluation import run_optimizer_study
from bike_demand_planner.regression import fit_standardizer, loss_and_gradients


def real_hour_probe(processed_directory: Path, learning_rate: float = .01) -> dict:
    """Repeat one real training observation to make optimizer history visible."""
    rows, _ = load_labelled_partition(processed_directory, "train")
    raw = np.asarray([row.features for row in rows], dtype=np.float64)
    targets = np.asarray([row.target for row in rows], dtype=np.float64)
    scaler = fit_standardizer(raw)
    index = next(index for index, row in enumerate(rows) if row.identity.record_id == 54)
    Z = scaler.transform(raw[index:index + 1])
    target = targets[index:index + 1]
    probe = {"scope": "Two updates repeating one real row; not a dataset training comparison",
             "record_id": 54, "target_time": rows[index].identity.target_time,
             "forecast_issue_time": rows[index].identity.forecast_issue_time, "target": float(target[0]),
             "learning_rate": learning_rate, "updates": {}}
    for kind in ("sgd", "momentum", "adam"):
        theta = np.append(np.zeros(raw.shape[1]), targets.mean())
        state, updates = initial_state(len(theta)), []
        for step in (1, 2):
            loss, dw, db = loss_and_gradients(Z, target, theta[:-1], theta[-1])
            prediction_before = float(Z[0] @ theta[:-1] + theta[-1])
            theta, state = optimizer_step(theta, np.append(dw, db), state, OptimizerConfig(kind), learning_rate)
            updates.append({"step": step, "prediction_before": prediction_before, "half_squared_error_before": loss,
                            "hr_08_gradient": float(dw[8]), "hr_08_after": float(theta[8]),
                            "first_moment_hr08": float(state.first[8]), "second_moment_hr08": float(state.second[8]),
                            "bias_after": float(theta[-1]), "prediction_after": float(Z[0] @ theta[:-1] + theta[-1])})
        probe["updates"][kind] = updates
    return probe


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    processed_directory = project_root / "data/processed"
    study_directory = project_root / "runs" / ("optimization-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    rates, seeds = (.001, .01, .1, 1.0), (17, 29, 43)
    passes, pilot_updates, max_updates, max_seconds = 10, 200, 100_000, 300
    batch_rate = .01
    try:
        probe = real_hour_probe(processed_directory)
        report = run_optimizer_study(processed_directory, study_directory, project_root, rates=rates, seeds=seeds,
                                     passes=passes, pilot_updates=pilot_updates, max_updates=max_updates,
                                     max_seconds=max_seconds, batch_rate=batch_rate)
        (study_directory / "updates.json").write_text(json.dumps(probe, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, FloatingPointError) as error:
        print(json.dumps({"status": "error", "message": str(error), "study_directory": str(study_directory)}, indent=2))
        raise SystemExit(1) from error
    print(json.dumps({"status": report["status"], "study_directory": str(study_directory),
                      "selected_rates": report["selected_rates"], "unique_main_trials": report["unique_main_trials"],
                      "summaries": report["summaries"], "test_evaluated": False}, indent=2))
