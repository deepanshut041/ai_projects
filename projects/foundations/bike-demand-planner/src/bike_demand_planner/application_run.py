"""Prepare checked serving assets and separate fresh-process recovery evidence."""

import os
import subprocess
import sys
from pathlib import Path

from .application_data import export_predictor, prepare_application_data
from .artifacts import digest, implementation_identity, json_bytes, read_json
from .evaluation import load_labelled_partition
from .recovery import run_recovery


def latest_review(project_root: Path) -> Path:
    candidates = sorted((project_root/"runs").glob("errors-*/review.json"))
    if not candidates:
        raise FileNotFoundError("Run error_review_example.py before exporting a predictor.")
    return candidates[-1].parent


def run_application_preparation(processed: Path, review: Path, bundle: Path, destination: Path, project_root: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    record = {"version": "hourly-application-preparation-v1", "status": "running", "test_evaluated": False,
              "protocol": "unchanged selected group model; all validation-hour fresh-process parity; separate training-only recovery proof",
              "implementation": implementation_identity(project_root)}
    try:
        predictor = export_predictor(processed, review, bundle, project_root)
        source_review = read_json(review / "review.json")
        study = project_root/"runs"/source_review["protocol"]["source_run"]
        app_data = prepare_application_data(processed, review, study, destination/"application-data", predictor)
        validation, _ = load_labelled_partition(processed, "validation")
        requests = [predictor.request_for_target(row.identity.target_time) for row in validation]
        request_file = destination/"prediction-requests.json"
        request_file.write_bytes(json_bytes(requests))
        in_process = [predictor.predict(request) for request in requests]
        output = destination/"fresh-predictions.json"
        args = [sys.executable, "-m", "bike_demand_planner.prediction_worker", "--bundle", str(bundle),
                "--expected-sha256", predictor.identity, "--requests", str(request_file), "--output", str(output)]
        subprocess.run(args, cwd=destination, check=True, timeout=60,
                       env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}, capture_output=True)
        fresh = read_json(output)
        if fresh != in_process:
            raise ValueError("Fresh-process forecasts differ from the original saved predictor.")
        recovery = run_recovery(processed, destination/"recovery", project_root)
        examples = {}
        for time in ("2012-01-09T02:00:00", "2012-01-09T08:00:00", "2012-06-04T08:00:00"):
            examples[time] = predictor.predict(predictor.request_for_target(time))
        record.update(status="completed", bundle_directory=str(bundle.relative_to(project_root)), bundle_sha256=predictor.identity,
                      application_data=app_data, validation_predictions_verified=len(fresh), fresh_process_exact_parity=True,
                      examples=examples, recovery_summary=recovery,
                      artifacts_sha256={"prediction-requests.json": digest(request_file.read_bytes()),
                                        "fresh-predictions.json": digest(output.read_bytes())})
    except Exception as error:
        record.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        (destination/"application.json").write_bytes(json_bytes(record))
    return record
