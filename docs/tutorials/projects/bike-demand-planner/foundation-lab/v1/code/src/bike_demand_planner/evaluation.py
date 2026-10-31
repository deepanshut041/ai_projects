"""Frozen train/validation loading, aligned scoring and baseline run records."""

import csv
import io
import json
import math
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from .baselines import fit_hour_workingday_baseline, fit_mean_baseline
from .data import sha256
from .features import FEATURE_NAMES, calendar_schema, canonical_json
from .metrics import regression_metrics


IDENTITY_FIELDS = ("record_id", "target_time", "forecast_issue_time")
METRIC_DEFINITIONS = {
    "residual": "prediction - target",
    "mae": "sum(abs(residual)) / count; primary, rentals/hour",
    "rmse": "sqrt(sum(residual ** 2) / count); rentals/hour",
    "mean_signed_error": "sum(residual) / count; positive means overprediction",
    "mean_underprediction": "sum(max(-residual, 0)) / count; all rows",
    "underprediction_count": "number of rows with residual < 0",
    "mean_underprediction_when_under": "total deficit / underprediction_count; null when none",
}


@dataclass(frozen=True)
class HourIdentity:
    record_id: int
    target_time: str
    forecast_issue_time: str


@dataclass(frozen=True)
class LabelledHour:
    identity: HourIdentity
    features: tuple[float, ...]
    target: float


@dataclass(frozen=True)
class Forecast:
    identity: HourIdentity
    prediction: float


def hour_workingday_key(row: LabelledHour) -> tuple[int, int]:
    """Select only the requested hour and working-day flag from the fixed schema."""
    if len(row.features) != len(FEATURE_NAMES):
        raise ValueError("Expected the fixed calendar feature count.")
    hour_values = row.features[:24]
    workingday = row.features[FEATURE_NAMES.index("workingday")]
    if any(value not in (0.0, 1.0) for value in hour_values) or sum(hour_values) != 1 or workingday not in (0.0, 1.0):
        raise ValueError("Expected one active hour and a binary working-day flag.")
    return hour_values.index(1.0), int(workingday)


def _read_manifest(directory: Path) -> dict:
    manifest = json.loads((directory / "split-manifest.json").read_text())
    identity = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if sha256(canonical_json(identity)) != manifest["manifest_sha256"] or manifest["frozen"] is not True:
        raise ValueError("Frozen manifest identity is invalid.")
    if manifest["version"] != "chronological-calendar-v1" or manifest["feature_schema_sha256"] != calendar_schema()["schema_sha256"]:
        raise ValueError("Unsupported data or calendar schema version.")
    return manifest


def _verified_bytes(directory: Path, name: str, manifest: dict) -> bytes:
    content = (directory / name).read_bytes()
    if sha256(content) != manifest["artifacts_sha256"][name]:
        raise ValueError(f"Frozen artifact {name} differs from its manifest.")
    return content


def _identity(row: dict) -> HourIdentity:
    return HourIdentity(int(row["record_id"]), row["target_time"], row["forecast_issue_time"])


def load_labelled_partition(directory: Path, partition: str) -> tuple[list[LabelledHour], dict]:
    """Read only training or validation; reject test access before opening files."""
    if partition not in ("train", "validation"):
        raise ValueError("Only train and validation are available for baseline development.")
    manifest = _read_manifest(directory)
    schema = json.loads(_verified_bytes(directory, "calendar-schema.json", manifest))
    if schema != calendar_schema():
        raise ValueError("Calendar schema differs from the supported feature contract.")
    members = csv.DictReader(io.StringIO(_verified_bytes(directory, "membership.csv", manifest).decode()))
    expected = [_identity(row) for row in members if row["split"] == partition]
    reader = csv.DictReader(io.StringIO(_verified_bytes(directory, f"{partition}.csv", manifest).decode()))
    if reader.fieldnames != [*IDENTITY_FIELDS, *FEATURE_NAMES, "target"]:
        raise ValueError("Unexpected labelled partition columns or feature order.")
    rows = [LabelledHour(_identity(row), tuple(float(row[name]) for name in FEATURE_NAMES), float(row["target"])) for row in reader]
    identities = [row.identity for row in rows]
    if not rows or identities != expected or len(rows) != manifest["counts"][partition] or len(set(identities)) != len(rows):
        raise ValueError("Partition rows do not match frozen membership and order.")
    if any(not math.isfinite(row.target) or row.target < 0 or any(value not in (0.0, 1.0) for value in row.features) for row in rows):
        raise ValueError("Expected nonnegative finite targets and binary calendar values.")
    return rows, manifest


def score_forecasts(rows: list[LabelledHour], forecasts: list[Forecast]) -> dict:
    """Refuse missing, reordered or duplicated predictions before reducing errors."""
    identities = [row.identity for row in rows]
    if [forecast.identity for forecast in forecasts] != identities or len(set(identities)) != len(identities):
        raise ValueError("Forecast identities must match each target row in order.")
    return regression_metrics([row.target for row in rows], [forecast.prediction for forecast in forecasts])


def _source_identity(project_root: Path) -> dict:
    paths = sorted([*project_root.glob("*.py"), *project_root.glob("src/**/*.py")])
    files = {path.relative_to(project_root).as_posix(): sha256(path.read_bytes()) for path in paths}
    result = {"files_sha256": files, "content_sha256": sha256(canonical_json(files)), "git_head": None, "git_status": None}
    try:
        result["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, stderr=subprocess.DEVNULL, text=True).strip()
        result["git_status"] = subprocess.check_output(["git", "status", "--porcelain", "--", "."], cwd=project_root, stderr=subprocess.DEVNULL, text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        pass
    return result


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _write_predictions(path: Path, rows: list[LabelledHour], forecasts: list[Forecast]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow([*IDENTITY_FIELDS, "target", "prediction", "residual", "absolute_error", "squared_error", "underprediction"])
        for row, forecast in zip(rows, forecasts, strict=True):
            error = forecast.prediction - row.target
            writer.writerow([*asdict(row.identity).values(), row.target, forecast.prediction, error, abs(error), error * error, max(-error, 0.0)])


def _run_baseline(processed_directory: Path, run_directory: Path, project_root: Path, kind: str) -> dict:
    """Use the same loading, identity, scoring and receipt path for both baselines."""
    grouped = kind == "hour_workingday_mean"
    run_directory.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    record = {"version": "hourly-regression-run-v1", "status": "running", "started_utc": datetime.now(UTC).isoformat(),
              "model": {"kind": kind}, "metric_definitions": METRIC_DEFINITIONS,
              "prediction_policy": "real-valued rentals/hour; no rounding or clipping",
              "test_evaluated": False, "timings_seconds": {}}
    try:
        record["source"] = _source_identity(project_root)
        record["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
                                 "dependencies_sha256": {name: sha256((project_root / name).read_bytes()) for name in ("pyproject.toml", "uv.lock")}}
        train, manifest = load_labelled_partition(processed_directory, "train")
        record["dataset"] = {key: manifest[key] for key in ("source_hour_sha256", "feature_schema_sha256", "membership_sha256", "manifest_sha256", "counts")}
        record["dataset"]["partition_sha256"] = {name: manifest["artifacts_sha256"][f"{name}.csv"] for name in ("train", "validation")}
        record["task"] = {"target": "observed system-wide rentals in one hour", "units": "rentals/hour",
                          **{key: manifest[key] for key in ("forecast_lead_hours", "target_duration_hours", "availability_policy", "readiness_assumption", "timestamp_policy")},
                          "feature_schema_columns": list(FEATURE_NAMES),
                          "features_used": [*FEATURE_NAMES[:24], "workingday"] if grouped else [],
                          "transform_fit": "none"}
        fit_started = perf_counter()
        targets = [row.target for row in train]
        model = (fit_hour_workingday_baseline([hour_workingday_key(row) for row in train], targets)
                 if grouped else fit_mean_baseline(targets))
        record["timings_seconds"]["fit"] = perf_counter() - fit_started
        record["model"].update(asdict(model), fit_partition="train",
                               fit_objective="minimum training squared error within each observed group" if grouped else "minimum training squared error among constants",
                               random_seed=None)
        if grouped:
            record["model"].update(group_fields=["hr", "workingday"],
                                   observed_group_count=len(model.groups), possible_group_count=48,
                                   absent_group_policy="global training mean; never use validation outcomes")
        _write_json(run_directory / "model.json", record["model"])
        validation, validation_manifest = load_labelled_partition(processed_directory, "validation")
        if validation_manifest != manifest:
            raise ValueError("Dataset identity changed between loading training and validation.")
        record["metrics"] = {}
        record["fallback_counts"] = {}
        for name, rows in (("train", train), ("validation", validation)):
            evaluation_started = perf_counter()
            if grouped:
                keys = [hour_workingday_key(row) for row in rows]
                observed = {(group.hour, group.workingday) for group in model.groups}
                record["fallback_counts"][name] = sum(key not in observed for key in keys)
                predictions = model.predict(keys)
            else:
                predictions = model.predict(len(rows))
            forecasts = [Forecast(row.identity, prediction) for row, prediction in zip(rows, predictions, strict=True)]
            record["metrics"][name] = score_forecasts(rows, forecasts)
            record["timings_seconds"][f"{name}_predict_and_score"] = perf_counter() - evaluation_started
            _write_predictions(run_directory / f"{name}-predictions.csv", rows, forecasts)
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


def run_mean_baseline(processed_directory: Path, run_directory: Path, project_root: Path) -> dict:
    """Fit the global training mean and retain aligned train/validation results."""
    return _run_baseline(processed_directory, run_directory, project_root, "training_mean")


def run_hour_workingday_baseline(processed_directory: Path, run_directory: Path, project_root: Path) -> dict:
    """Fit calendar-group means with a training-only global-mean fallback."""
    return _run_baseline(processed_directory, run_directory, project_root, "hour_workingday_mean")
