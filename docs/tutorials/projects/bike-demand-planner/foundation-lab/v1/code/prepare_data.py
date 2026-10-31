"""Acquire, audit and partition the official hourly calendar dataset."""

import json
from pathlib import Path
from zipfile import BadZipFile

from bike_demand_planner.data import prepare_hourly_data
from bike_demand_planner.splits import prepare_calendar_dataset


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    source_page = "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"
    archive_url = "https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip"
    data_directory = project_root / "data"
    output_directory = data_directory / "processed"
    try:
        report = prepare_hourly_data(source_page, archive_url, data_directory)
        report["prepared_dataset"] = prepare_calendar_dataset(
            data_directory / "raw/hour.csv", output_directory,
            report["source"]["member_sha256"]["hour.csv"],
        )
    except (OSError, ValueError, KeyError, BadZipFile) as error:
        print(json.dumps({"status": "error", "message": str(error)}, indent=2))
        raise SystemExit(1) from error
    print(json.dumps(report, indent=2))
