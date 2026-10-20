"""Validate and summarize the latest compatible practical-model study."""

import json
from pathlib import Path

from bike_demand_planner.comparison import summarize_tree_study


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    run_directories = sorted((project_root / "runs").glob("trees-*/study.json"))
    report_directory = project_root / "artifacts" / "comparison"
    if not run_directories:
        print(json.dumps({"status": "needs_run", "message": "Run tree_example.py before comparing its saved results."}))
        raise SystemExit(2)
    run_directory = run_directories[-1].parent
    summary = summarize_tree_study(run_directory)
    report_directory.mkdir(parents=True, exist_ok=True)
    destination = report_directory/(run_directory.name+".json")
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, indent=2))
