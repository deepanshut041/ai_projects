"""Fit a calendar linear model and score the frozen train/validation membership."""

import csv
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from .data import sha256
from .evaluation import (
    METRIC_DEFINITIONS, Forecast, _source_identity, _write_json, _write_predictions,
    load_labelled_partition, score_forecasts,
)
from .features import FEATURE_NAMES
from .regression import fit_linear_regression, least_squares_reference


def run_linear_regression(
    processed_directory: Path, run_directory: Path, project_root: Path, *,
    learning_rate: float = .1, max_steps: int = 10_000, gradient_tolerance: float = 1e-7,
) -> dict:
    """Retain parameters, optimization history, aligned errors and run provenance."""
    run_directory.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    record = {
        "version": "hourly-regression-run-v1", "status": "running",
        "started_utc": datetime.now(UTC).isoformat(), "model": {"kind": "calendar_linear_regression"},
        "metric_definitions": METRIC_DEFINITIONS,
        "prediction_policy": "real-valued rentals/hour; no rounding or clipping",
        "test_evaluated": False, "timings_seconds": {},
        "configuration": {"learning_rate": learning_rate, "max_steps": max_steps,
                          "gradient_tolerance": gradient_tolerance, "dtype": "float64",
                          "batch": "all training rows per update", "random_seed": None,
                          "initial_weights": "zeros", "initial_bias": "training target mean",
                          "target_scaling": "none", "validation_used_for_fitting": False},
    }
    try:
        record["source"] = _source_identity(project_root)
        record["environment"] = {
            "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
            "dependencies_sha256": {name: sha256((project_root / name).read_bytes()) for name in ("pyproject.toml", "uv.lock")},
        }
        train, manifest = load_labelled_partition(processed_directory, "train")
        record["dataset"] = {key: manifest[key] for key in (
            "source_hour_sha256", "feature_schema_sha256", "membership_sha256", "manifest_sha256", "counts")}
        record["dataset"]["partition_sha256"] = {name: manifest["artifacts_sha256"][f"{name}.csv"] for name in ("train", "validation")}
        record["task"] = {
            "target": "observed system-wide rentals in one hour", "units": "rentals/hour",
            **{key: manifest[key] for key in ("forecast_lead_hours", "target_duration_hours", "availability_policy", "readiness_assumption", "timestamp_policy")},
            "feature_schema_columns": list(FEATURE_NAMES), "features_used": list(FEATURE_NAMES),
            "transform_fit": "training-only column means and population standard deviations; constants use scale one",
        }
        X = np.asarray([row.features for row in train], dtype=np.float64)
        y = np.asarray([row.target for row in train], dtype=np.float64)
        fit_started = perf_counter()
        fit = fit_linear_regression(X, y, learning_rate=learning_rate, max_steps=max_steps,
                                    gradient_tolerance=gradient_tolerance)
        record["timings_seconds"]["fit"] = perf_counter() - fit_started
        reference_started = perf_counter()
        reference, rank = least_squares_reference(X, y, fit.model.standardizer)
        record["timings_seconds"]["least_squares_reference"] = perf_counter() - reference_started
        raw_weights, raw_bias = fit.model.raw_parameters()
        record["model"].update(
            asdict(fit.model), feature_names=list(FEATURE_NAMES), fit_partition="train",
            fit_objective="sum((prediction - target) ** 2) / (2 * training_count)",
            raw_weights=list(raw_weights), raw_bias=raw_bias,
        )
        record["optimization"] = {
            "converged": fit.converged, "stop_reason": fit.stop_reason, "updates": fit.history[-1].step,
            "initial": asdict(fit.history[0]), "final": asdict(fit.history[-1]),
            "maximum_curvature": fit.maximum_curvature,
            "stability_limit_exclusive": 2 / fit.maximum_curvature,
        }
        record["reference"] = {
            "solver": "numpy.linalg.lstsq; rcond=None; training only",
            "design_shape": [len(train), len(FEATURE_NAMES) + 1], "rank": rank,
            "comparison": "Compare predictions/objectives; dependent columns do not identify unique coefficients.",
            "model": asdict(reference),
        }
        _write_json(run_directory / "model.json", record["model"])
        _write_json(run_directory / "reference.json", record["reference"])
        with (run_directory / "optimization.csv").open("w", newline="") as output:
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(["step", "half_mse", "gradient_norm"])
            writer.writerows((point.step, point.half_mse, point.gradient_norm) for point in fit.history)

        # Validation is first opened after both training-only solvers have finished.
        validation, validation_manifest = load_labelled_partition(processed_directory, "validation")
        if validation_manifest != manifest:
            raise ValueError("Dataset identity changed between loading training and validation.")
        record["metrics"], record["prediction_ranges"], record["reference"]["agreement"] = {}, {}, {}
        for name, rows in (("train", train), ("validation", validation)):
            evaluation_started = perf_counter()
            values = np.asarray([row.features for row in rows], dtype=np.float64)
            targets = np.asarray([row.target for row in rows], dtype=np.float64)
            predictions = fit.model.predict(values)
            forecasts = [Forecast(row.identity, float(value)) for row, value in zip(rows, predictions, strict=True)]
            record["metrics"][name] = score_forecasts(rows, forecasts)
            reference_predictions = reference.predict(values)
            record["reference"]["agreement"][name] = {
                "maximum_absolute_prediction_difference": float(np.max(np.abs(predictions - reference_predictions))),
                "iterative_half_mse": float(np.mean((predictions - targets) ** 2) / 2),
                "least_squares_half_mse": float(np.mean((reference_predictions - targets) ** 2) / 2),
            }
            record["prediction_ranges"][name] = {
                "minimum": float(predictions.min()), "maximum": float(predictions.max()),
                "negative_count": int(np.count_nonzero(predictions < 0)),
            }
            record["timings_seconds"][f"{name}_predict_and_score"] = perf_counter() - evaluation_started
            _write_predictions(run_directory / f"{name}-predictions.csv", rows, forecasts)
        _write_json(run_directory / "reference.json", record["reference"])
        record["artifacts_sha256"] = {path.name: sha256(path.read_bytes()) for path in sorted(run_directory.iterdir())}
        record["status"] = "completed"
    except Exception as error:
        record["status"] = "failed"
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        record["finished_utc"] = datetime.now(UTC).isoformat()
        record["timings_seconds"]["total"] = perf_counter() - started
        _write_json(run_directory / "run.json", record)
    return record
