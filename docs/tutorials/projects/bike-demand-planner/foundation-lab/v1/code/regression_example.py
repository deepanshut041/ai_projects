"""Trace batch updates and compare calendar regression with both demand baselines."""

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from bike_demand_planner.evaluation import load_labelled_partition, run_hour_workingday_baseline, run_mean_baseline
from bike_demand_planner.features import FEATURE_NAMES
from bike_demand_planner.linear_evaluation import run_linear_regression
from bike_demand_planner.regression import batch_step, fit_standardizer, forward, loss_and_gradients


def update_trace(features, targets, weights, bias, learning_rate, steps, example_index):
    trace = []
    for step in range(steps + 1):
        loss, dw, db = loss_and_gradients(features, targets, weights, bias)
        trace.append({"step": step, "half_mse": loss, "prediction": float(forward(features, weights, bias)[example_index]),
                      "weights": np.asarray(weights).tolist(), "bias": bias, "weight_gradients": dw.tolist(), "bias_gradient": db})
        if step < steps:
            weights, bias = batch_step(features, targets, weights, bias, learning_rate)
    return trace


def compare_models(processed_directory, run_directory, project_root, learning_rate, max_steps, gradient_tolerance):
    records = {
        "global": run_mean_baseline(processed_directory, run_directory / "global", project_root),
        "hour-workingday": run_hour_workingday_baseline(processed_directory, run_directory / "hour-workingday", project_root),
        "linear": run_linear_regression(processed_directory, run_directory / "linear", project_root,
                                         learning_rate=learning_rate, max_steps=max_steps, gradient_tolerance=gradient_tolerance),
    }
    reference = records["global"]
    if any(record["dataset"] != reference["dataset"] or record["metric_definitions"] != reference["metric_definitions"]
           or record["prediction_policy"] != reference["prediction_policy"] for record in records.values()):
        raise ValueError("Model comparisons require identical data, metrics and prediction policy.")
    rows = {}
    for name in records:
        with (run_directory / name / "validation-predictions.csv").open() as source:
            rows[name] = list(csv.DictReader(source))
    identities = lambda values: [(row["record_id"], row["target_time"], row["forecast_issue_time"]) for row in values]
    if any(identities(values) != identities(rows["global"]) for values in rows.values()):
        raise ValueError("The saved validation forecast identities differ.")
    linear_rows, grouped_rows = rows["linear"], rows["hour-workingday"]
    improved = sum(float(a["absolute_error"]) < float(b["absolute_error"]) for a, b in zip(linear_rows, grouped_rows, strict=True))
    worse = sum(float(a["absolute_error"]) > float(b["absolute_error"]) for a, b in zip(linear_rows, grouped_rows, strict=True))
    example_hours = {"2012-01-09T02:00:00", "2012-01-09T08:00:00"}
    comparison = {
        "status": "completed", "run_directory": str(run_directory), "test_evaluated": False,
        "training": {name: record["metrics"]["train"] for name, record in records.items()},
        "validation": {name: record["metrics"]["validation"] for name, record in records.items()},
        "examples": {name: [row for row in values if row["target_time"] in example_hours] for name, values in rows.items()},
        "linear_optimization": records["linear"]["optimization"],
        "linear_reference": records["linear"]["reference"],
        "linear_prediction_ranges": records["linear"]["prediction_ranges"],
        "linear_vs_grouped_rows": {"improved": improved, "worse": worse, "equal": len(linear_rows) - improved - worse},
        "largest_linear_errors": sorted(linear_rows, key=lambda row: float(row["absolute_error"]), reverse=True)[:3],
    }
    (run_directory / "comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
    return comparison


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    processed_directory = project_root / "data/processed"
    run_directory = project_root / "runs" / ("linear-comparison-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    learning_rate, max_steps, gradient_tolerance = .1, 10_000, 1e-7
    small_features = [[1, 0], [0, 1], [1, 1], [2, -1]]
    small_targets = [5, 2, 4, 8]
    small_weights, small_bias = [1, 1], 0.0
    try:
        train, _ = load_labelled_partition(processed_directory, "train")
        features = np.asarray([row.features for row in train], dtype=np.float64)
        targets = np.asarray([row.target for row in train], dtype=np.float64)
        standardizer = fit_standardizer(features)
        example_index = next(index for index, row in enumerate(train) if row.identity.record_id == 54)
        traces = {
            "learning_rate": learning_rate, "feature_names": list(FEATURE_NAMES),
            "fixture": update_trace(small_features, small_targets, small_weights, small_bias, learning_rate, 2, 0),
            "training_example": {"record_id": 54, "target_time": train[example_index].identity.target_time,
                                 "forecast_issue_time": train[example_index].identity.forecast_issue_time,
                                 "target": train[example_index].target, "raw_features": features[example_index].tolist()},
            "real_updates": update_trace(standardizer.transform(features), targets, np.zeros(features.shape[1]),
                                         float(targets.mean()), learning_rate, 2, example_index),
        }
        comparison = compare_models(processed_directory, run_directory, project_root, learning_rate, max_steps, gradient_tolerance)
        (run_directory / "updates.json").write_text(json.dumps(traces, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"run_directory": str(run_directory), "training_example_updates": [row["prediction"] for row in traces["real_updates"]],
                          "validation": comparison["validation"], "optimization": comparison["linear_optimization"],
                          "prediction_ranges": comparison["linear_prediction_ranges"], "test_evaluated": False}, indent=2))
    except (OSError, ValueError, KeyError, FloatingPointError) as error:
        print(json.dumps({"status": "error", "message": str(error), "run_directory": str(run_directory)}, indent=2))
        raise SystemExit(1) from error
