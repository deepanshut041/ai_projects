"""Inspect one training observation's calendar encoding and membership."""

import csv
import json
from datetime import datetime
from pathlib import Path

from bike_demand_planner.data import parse_hourly_row
from bike_demand_planner.features import FEATURE_NAMES, INPUT_FIELDS, encode_calendar
from bike_demand_planner.splits import FORECAST_LEAD, split_for_target


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    csv_path = project_root / "data/raw/hour.csv"
    target_time = datetime(2011, 1, 3, 8)
    if split_for_target(target_time) != "train":
        raise ValueError("This example inspects a training-period observation.")
    with csv_path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            record = parse_hourly_row(row)
            if record.target_time == target_time:
                calendar = {name: record.calendar[name] for name in INPUT_FIELDS}
                features = encode_calendar(calendar)
                print(json.dumps({
                    "record_id": record.record_id,
                    "target_time": target_time.isoformat(),
                    "forecast_issue_time": (target_time - FORECAST_LEAD).isoformat(),
                    "calendar_inputs": calendar,
                    "feature_count": len(features),
                    "active_features": {name: value for name, value in zip(FEATURE_NAMES, features, strict=True) if value},
                    "split": split_for_target(target_time),
                    "target_separate_from_inputs": record.target,
                }, indent=2))
                break
        else:
            raise ValueError("The requested source hour is not recorded.")
