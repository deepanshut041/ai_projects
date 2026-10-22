"""Review aligned hourly errors and retain the candidate configuration."""

import json
from datetime import UTC, datetime
from pathlib import Path

from bike_demand_planner.error_review import run_error_review


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    processed_directory = project_root/"data"/"processed"
    source_records = sorted((project_root/"runs").glob("trees-*/study.json"))
    if not source_records:
        print(json.dumps({"status": "needs_run", "message": "Run tree_example.py before reviewing its saved predictions."}))
        raise SystemExit(2)
    source_run = source_records[-1].parent
    run_directory = project_root/"runs"/("errors-"+datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    report = run_error_review(processed_directory,source_run,run_directory,project_root)
    selected = report["results"]["groups"]["validation"]["summary"]
    print(json.dumps({"run_directory": str(run_directory), "status": report["status"],
                      "selected": report["selected_candidate"], "validation": selected,
                      "examples": report["examples"], "contract_rejections": len(report["contract_probes"]["rejected"]),
                      "candidate_sha256": report["candidate_sha256"]},indent=2))
