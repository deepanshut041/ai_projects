"""Acquire the UCI archive and audit hourly observations."""

import csv
import hashlib
import io
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile


RAW_FIELDS = (
    "instant", "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed", "casual",
    "registered", "cnt",
)
CALENDAR_FIELDS = ("season", "yr", "mnth", "hr", "holiday", "weekday", "workingday")
EXCLUDED_PREDICTORS = {
    "instant": "Record identity, not a predictor.",
    "dteday": "Source date retained as identity; calendar fields are selected explicitly.",
    "casual": "Part of the target-hour outcome; casual + registered equals cnt.",
    "registered": "Part of the target-hour outcome; casual + registered equals cnt.",
    "cnt": "Target, kept separate from prediction inputs.",
    "weathersit": "Observed target-hour weather is not an advance forecast.",
    "temp": "Observed target-hour weather is not an advance forecast.",
    "atemp": "Observed target-hour weather is not an advance forecast.",
    "hum": "Observed target-hour weather is not an advance forecast.",
    "windspeed": "Observed target-hour weather is not an advance forecast.",
}


@dataclass
class HourlyRecord:
    record_id: int
    target_time: datetime
    calendar: dict[str, int]
    target: int


def parse_hourly_row(row: dict[str, str]) -> HourlyRecord:
    """Validate one raw row and separate calendar inputs, identity and target."""
    if set(row) != set(RAW_FIELDS) or any(value in (None, "") for value in row.values()):
        raise ValueError("Expected all 17 hourly fields with no empty cells.")
    numbers = {name: int(row[name]) for name in RAW_FIELDS if name not in {
        "dteday", "temp", "atemp", "hum", "windspeed",
    }}
    day = date.fromisoformat(row["dteday"])
    domains = {
        "season": (1, 4), "yr": (0, 1), "mnth": (1, 12), "hr": (0, 23),
        "holiday": (0, 1), "weekday": (0, 6), "workingday": (0, 1),
        "weathersit": (1, 4),
    }
    for name, (low, high) in domains.items():
        if not low <= numbers[name] <= high:
            raise ValueError(f"{name} must be in [{low}, {high}].")
    if day.year not in (2011, 2012) or numbers["yr"] != day.year - 2011:
        raise ValueError("Date and year code disagree or fall outside 2011–2012.")
    if numbers["mnth"] != day.month or numbers["weekday"] != (day.weekday() + 1) % 7:
        raise ValueError("Date, month or Sunday-zero weekday code disagree.")
    expected_workingday = int(day.weekday() < 5 and numbers["holiday"] == 0)
    if numbers["workingday"] != expected_workingday:
        raise ValueError("Working-day flag disagrees with weekday/holiday flags.")
    for name in ("temp", "atemp", "hum", "windspeed"):
        value = float(row[name])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1].")
    if numbers["instant"] <= 0 or any(numbers[name] < 0 for name in ("casual", "registered", "cnt")):
        raise ValueError("Record IDs must be positive and rental counts nonnegative.")
    if numbers["casual"] + numbers["registered"] != numbers["cnt"]:
        raise ValueError("Target does not equal casual + registered.")
    return HourlyRecord(
        record_id=numbers["instant"],
        target_time=datetime.combine(day, time(hour=numbers["hr"])),
        calendar={name: numbers[name] for name in CALENDAR_FIELDS},
        target=numbers["cnt"],
    )


def audit_hourly_csv(path: Path) -> dict:
    """Check structural integrity without estimating model quality or filling gaps."""
    ids, timestamps = set(), set()
    months = Counter()
    gaps = []
    first = previous = None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(RAW_FIELDS):
            raise ValueError("Hourly CSV header or column order differs from the source schema.")
        for line_number, row in enumerate(reader, start=2):
            try:
                record = parse_hourly_row(row)
                if record.record_id in ids or record.target_time in timestamps:
                    raise ValueError("Duplicate record ID or date/hour identity.")
                if previous is not None:
                    hours = (record.target_time - previous).total_seconds() / 3600
                    if hours <= 0:
                        raise ValueError("Rows are not strictly chronological.")
                    if hours > 1:
                        gaps.append({
                            "after": previous.isoformat(),
                            "before": record.target_time.isoformat(),
                            "unrecorded_hours": int(hours) - 1,
                        })
            except (ValueError, TypeError) as error:
                raise ValueError(f"CSV line {line_number}: {error}") from error
            if first is None:
                first = record
            ids.add(record.record_id)
            timestamps.add(record.target_time)
            months[record.target_time.strftime("%Y-%m")] += 1
            previous = record.target_time
    if first is None:
        raise ValueError("Hourly CSV contains no observations.")
    expected_slots = int((previous - first.target_time).total_seconds() / 3600) + 1
    return {
        "status": "ok",
        "rows": len(ids),
        "first_target_time": first.target_time.isoformat(),
        "last_target_time": previous.isoformat(),
        "source_timestamp_policy": "Naive source date/hour labels; timezone and DST semantics are unspecified.",
        "expected_naive_hour_slots": expected_slots,
        "unrecorded_naive_hour_slots": expected_slots - len(ids),
        "gap_count": len(gaps),
        "first_gaps": gaps[:5],
        "rows_by_month": dict(sorted(months.items())),
        "empty_cells": 0,
        "duplicate_ids": 0,
        "duplicate_date_hours": 0,
        "target_component_mismatches": 0,
        "calendar_fields": list(CALENDAR_FIELDS),
        "excluded_predictors": EXCLUDED_PREDICTORS,
        "first_row": {
            "identity": {"record_id": first.record_id, "target_time": first.target_time.isoformat()},
            "calendar_inputs": first.calendar,
            "target": first.target,
        },
        "scope": "raw_structural_audit",
    }


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare_hourly_data(source_page: str, archive_url: str, data_directory: Path) -> dict:
    """Cache the official archive, verify retained bytes and save provenance/audit."""
    raw = data_directory / "raw"
    archive_path, provenance_path = raw / "bike-sharing.zip", raw / "source.json"
    if archive_path.exists():
        provenance = json.loads(provenance_path.read_text())
        archive_bytes = archive_path.read_bytes()
        if provenance["archive_url"] != archive_url or provenance["archive_sha256"] != sha256(archive_bytes):
            raise ValueError("Cached archive disagrees with its source identity.")
    else:
        request = Request(archive_url, headers={"User-Agent": "bike-demand-planner/0.1"})
        with urlopen(request, timeout=30) as response:
            archive_bytes = response.read(5_000_001)
        if len(archive_bytes) > 5_000_000:
            raise ValueError("Archive exceeds the expected download size limit.")
        provenance = {
            "source_page": source_page,
            "archive_url": archive_url,
            "acquired_at_utc": datetime.now(UTC).isoformat(),
            "attribution": "Fanaee-T, H. (2013). Bike Sharing. UCI Machine Learning Repository.",
            "doi": "https://doi.org/10.24432/C5W894",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "archive_sha256": sha256(archive_bytes),
        }
    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        members = {}
        for name in ("hour.csv", "Readme.txt"):
            if archive.getinfo(name).file_size > 2_000_000:
                raise ValueError(f"Archive member {name} exceeds the expected size limit.")
            members[name] = archive.read(name)
    member_hashes = {name: sha256(content) for name, content in members.items()}
    if "member_sha256" in provenance and provenance["member_sha256"] != member_hashes:
        raise ValueError("Archive members disagree with retained source hashes.")
    raw.mkdir(parents=True, exist_ok=True)
    for name, content in members.items():
        path = raw / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"Existing {name} differs from the retained archive.")
    archive_path.write_bytes(archive_bytes)
    for name, content in members.items():
        (raw / name).write_bytes(content)
    provenance["member_sha256"] = member_hashes
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    report = audit_hourly_csv(raw / "hour.csv")
    report["source"] = provenance
    (data_directory / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
