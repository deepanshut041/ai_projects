"""Compare global and hour/working-day means on the frozen validation hours."""

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from bike_demand_planner.evaluation import run_hour_workingday_baseline, run_mean_baseline


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    processed_directory = project_root / "data/processed"
    run_directory = project_root / "runs" / ("calendar-comparison-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    example_hours = {"2012-01-09T02:00:00", "2012-01-09T08:00:00"}
    try:
        constant = run_mean_baseline(processed_directory, run_directory / "global", project_root)
        grouped = run_hour_workingday_baseline(processed_directory, run_directory / "hour-workingday", project_root)
        if constant["dataset"] != grouped["dataset"] or constant["metric_definitions"] != grouped["metric_definitions"]:
            raise ValueError("The comparison requires identical frozen data and metric definitions.")
        examples = {}
        for name in ("global", "hour-workingday"):
            with (run_directory / name / "validation-predictions.csv").open() as source:
                examples[name] = [row for row in csv.DictReader(source) if row["target_time"] in example_hours]
        global_mae = constant["metrics"]["validation"]["mae"]
        group_mae = grouped["metrics"]["validation"]["mae"]
        comparison = {
            "status": "completed", "run_directory": str(run_directory), "test_evaluated": False,
            "validation": {"global": constant["metrics"]["validation"], "hour-workingday": grouped["metrics"]["validation"]},
            "training": {"global": constant["metrics"]["train"], "hour-workingday": grouped["metrics"]["train"]},
            "mae_reduction": global_mae - group_mae,
            "mae_reduction_percent": 100 * (global_mae - group_mae) / global_mae,
            "group_coverage": {"observed": grouped["model"]["observed_group_count"], "possible": 48,
                               "fallback_counts": grouped["fallback_counts"]},
            "examples": examples,
        }
        (run_directory / "comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "error", "message": str(error), "run_directory": str(run_directory)}, indent=2))
        raise SystemExit(1) from error
    print(json.dumps(comparison, indent=2))
