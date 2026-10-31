"""Chronological membership and reproducible calendar dataset exports."""

import csv
import io
import json
from datetime import datetime, timedelta
from pathlib import Path

from .data import RAW_FIELDS, parse_hourly_row, sha256
from .features import FEATURE_NAMES, INPUT_FIELDS, calendar_schema, canonical_json, encode_calendar


DATA_START = datetime(2011, 1, 1)
VALIDATION_READY = datetime(2012, 1, 1)
TEST_READY = datetime(2012, 7, 1)
DATA_END = datetime(2013, 1, 1)
FORECAST_LEAD = timedelta(hours=1)
TARGET_DURATION = timedelta(hours=1)


def split_for_target(target_time: datetime) -> str:
    """Use target labels and issue-time readiness, never target values."""
    if target_time.tzinfo is not None or any((target_time.minute, target_time.second, target_time.microsecond)):
        raise ValueError("Expected a naive, hour-aligned source timestamp.")
    if not DATA_START <= target_time < DATA_END:
        raise ValueError("Supported target dates are in 2011–2012.")
    if target_time < VALIDATION_READY:
        return "train"
    if target_time - FORECAST_LEAD < VALIDATION_READY:
        return "embargo"
    if target_time < TEST_READY:
        return "validation"
    if target_time - FORECAST_LEAD < TEST_READY:
        return "embargo"
    return "test"


def _csv_bytes(header: list[str], rows: list[list]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode()


def prepare_calendar_dataset(csv_path: Path, output_directory: Path, expected_source_sha256: str) -> dict:
    """Freeze membership/features and omit reserved-test targets from exports."""
    source = csv_path.read_bytes()
    if sha256(source) != expected_source_sha256:
        raise ValueError("Hourly CSV differs from the acquired source identity.")
    reader = csv.DictReader(io.StringIO(source.decode("utf-8-sig")))
    if reader.fieldnames != list(RAW_FIELDS):
        raise ValueError("Unexpected hourly source schema.")
    records = [parse_hourly_row(row) for row in reader]
    if not records:
        raise ValueError("No hourly records to partition.")
    if len({row.record_id for row in records}) != len(records) or any(
        first.target_time >= second.target_time
        for first, second in zip(records, records[1:])
    ):
        raise ValueError("Expected unique IDs and strictly chronological date/hour identities.")

    partitions = {name: [] for name in ("train", "validation", "test")}
    members, excluded = [], []
    training_ones = [0] * len(FEATURE_NAMES)
    bounds = {}
    for record in records:
        split = split_for_target(record.target_time)
        issue_time = record.target_time - FORECAST_LEAD
        identity = [record.record_id, record.target_time.isoformat(), issue_time.isoformat()]
        members.append([*identity, split])
        if split == "embargo":
            excluded.append({"record_id": record.record_id, "target_time": record.target_time.isoformat(),
                             "reason": "Forecast issue precedes the preceding stage's label-availability boundary."})
            continue
        calendar = {name: record.calendar[name] for name in INPUT_FIELDS}
        features = encode_calendar(calendar)
        row = [*identity, *features]
        if split != "test":
            row.append(record.target)
        partitions[split].append(row)
        bounds.setdefault(split, {"first": record.target_time, "last": record.target_time})["last"] = record.target_time
        if split == "train":
            training_ones = [count + int(value) for count, value in zip(training_ones, features, strict=True)]
    if any(not rows for rows in partitions.values()):
        raise ValueError("Each training, validation and test window must contain observations.")
    if bounds["train"]["last"] + TARGET_DURATION > bounds["validation"]["first"] - FORECAST_LEAD:
        raise ValueError("A training label is unavailable at the first validation issue time.")
    if bounds["validation"]["last"] + TARGET_DURATION > bounds["test"]["first"] - FORECAST_LEAD:
        raise ValueError("A validation label is unavailable at the first test issue time.")

    identity_header = ["record_id", "target_time", "forecast_issue_time"]
    schema = calendar_schema()
    artifacts = {
        "calendar-schema.json": (json.dumps(schema, indent=2) + "\n").encode(),
        "membership.csv": _csv_bytes([*identity_header, "split"], members),
    }
    for name, rows in partitions.items():
        header = [*identity_header, *FEATURE_NAMES]
        if name != "test":
            header.append("target")
        artifacts[f"{name}.csv"] = _csv_bytes(header, rows)
    train_count = len(partitions["train"])
    manifest = {
        "version": "chronological-calendar-v1",
        "frozen": True,
        "source_hour_sha256": expected_source_sha256,
        "feature_schema_sha256": schema["schema_sha256"],
        "membership_sha256": sha256(artifacts["membership.csv"]),
        "feature_count": len(FEATURE_NAMES),
        "forecast_lead_hours": 1,
        "target_duration_hours": 1,
        "availability_policy": "Target hour is an interval start; its label is available at interval end, with zero reporting delay.",
        "readiness_assumption": "The preceding stage is treated as ready at its boundary; model-build delay is zero in this offline protocol.",
        "timestamp_policy": "Working protocol on naive source date/hour labels; source timezone/DST semantics are unspecified.",
        "windows": {
            "train": {"start_inclusive": DATA_START.isoformat(), "end_exclusive": VALIDATION_READY.isoformat()},
            "validation": {"start_inclusive": (VALIDATION_READY + FORECAST_LEAD).isoformat(), "end_exclusive": TEST_READY.isoformat()},
            "test": {"start_inclusive": (TEST_READY + FORECAST_LEAD).isoformat(), "end_exclusive": DATA_END.isoformat()},
        },
        "counts": {**{name: len(rows) for name, rows in partitions.items()}, "embargo": len(excluded)},
        "observed_bounds": {name: {key: value.isoformat() for key, value in extent.items()} for name, extent in bounds.items()},
        "excluded_boundary_rows": excluded,
        "training_feature_summary": {
            "rows": train_count,
            "ones_by_feature": dict(zip(FEATURE_NAMES, training_ones, strict=True)),
            "constant_features": [name for name, count in zip(FEATURE_NAMES, training_ones, strict=True) if count in (0, train_count)],
        },
        "transform_fit": "none; fixed category definitions and binary flags",
        "test_targets_exported": False,
        "test_policy": "Reserved for final evaluation; no target summaries or model selection.",
        "artifacts_sha256": {name: sha256(content) for name, content in artifacts.items()},
    }
    manifest["manifest_sha256"] = sha256(canonical_json(manifest))
    artifacts["split-manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    for name, content in artifacts.items():
        path = output_directory / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"Frozen artifact {name} differs; use an explicitly versioned dataset instead of overwriting it.")
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, content in artifacts.items():
        (output_directory / name).write_bytes(content)
    return manifest
