"""Prepare the saved predictor, replay assets and local restart evidence."""

import json
from datetime import UTC, datetime
from pathlib import Path

from bike_demand_planner.application_run import latest_review, run_application_preparation
from bike_demand_planner.artifacts import json_bytes
from bike_demand_planner.saved_predictor import SavedPredictor


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    processed_directory = project_root/"data/processed"
    model_directory = project_root/"models/selected"
    run_directory = project_root/"runs"/("application-"+datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    try:
        review_directory = latest_review(project_root)
    except FileNotFoundError as error:
        print(json.dumps({"status": "needs_run", "message": str(error)}))
        raise SystemExit(2)
    report = run_application_preparation(processed_directory, review_directory, model_directory, run_directory, project_root)
    predictor = SavedPredictor(model_directory)
    request_path = project_root/"data/request.json"
    if not request_path.exists():
        request_path.write_bytes(json_bytes(predictor.request_for_target("2012-01-09T08:00:00")))
    print(json.dumps({"status": report["status"], "run_directory": str(run_directory),
                      "validation_predictions_verified": report["validation_predictions_verified"],
                      "fresh_process_exact_parity": report["fresh_process_exact_parity"],
                      "recovery": report["recovery_summary"], "example": report["examples"]["2012-01-09T08:00:00"]}, indent=2))
