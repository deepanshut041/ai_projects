"""Fit a constant predictor and inspect aligned validation errors."""

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from bike_demand_planner.evaluation import run_mean_baseline


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    processed_directory = project_root / "data/processed"
    run_directory = project_root / "runs" / ("training-mean-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    example_hours = {"2012-01-09T02:00:00", "2012-01-09T08:00:00"}
    try:
        record = run_mean_baseline(processed_directory, run_directory, project_root)
        with (run_directory / "validation-predictions.csv").open() as source:
            examples = [row for row in csv.DictReader(source) if row["target_time"] in example_hours]
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "error", "message": str(error), "run_directory": str(run_directory)}, indent=2))
        raise SystemExit(1) from error
    print(json.dumps({"status": record["status"], "model": record["model"], "metrics": record["metrics"],
                      "examples": examples, "run_directory": str(run_directory), "test_evaluated": False}, indent=2))
