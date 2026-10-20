"""Freeze the retained comparison and assess the reserved later hours once."""

import json
from datetime import UTC, datetime
from pathlib import Path

from bike_demand_planner.application_data import verified_review
from bike_demand_planner.application_run import latest_review
from bike_demand_planner.artifacts import digest, read_json
from bike_demand_planner.final_evaluation import freeze_assessment, run_final_assessment


def assessment_inputs(project: Path) -> tuple[Path, Path]:
    """Follow the reviewed study identity rather than a machine's old run name."""
    review_directory = latest_review(project)
    review = verified_review(review_directory)
    name = review["protocol"]["source_run"]
    if not isinstance(name, str) or name in (".", "..") or Path(name).name != name:
        raise ValueError("Expected a plain reviewed study directory name.")
    study_directory = project/"runs"/name
    if not (study_directory/"study.json").is_file():
        raise FileNotFoundError("The reviewed practical-model study is missing; inspect its saved review before continuing.")
    return review_directory, study_directory


def saved_assessment(project: Path, receipt: Path) -> dict:
    """Check retained evidence without reopening source targets or making a new choice."""
    result = read_json(receipt)
    if (result.get("version") != "hourly-final-assessment-v1" or result["status"] != "completed"
            or not result["test_evaluated"] or result["selection_changed"]):
        raise ValueError("The latest assessment is incomplete or failed; inspect its receipt before continuing.")
    if result["protocol_sha256"] != digest((project/"models/final-assessment/protocol.json").read_bytes()):
        raise ValueError("The saved assessment protocol changed.")
    for name, identity in result["artifacts_sha256"].items():
        if Path(name).name != name or digest((receipt.parent/name).read_bytes()) != identity:
            raise ValueError("Saved assessment evidence changed; inspect it before continuing.")
    return result


def main(project: Path) -> int:
    frozen = project/"models/final-assessment"
    existing = sorted((project/"runs").glob("final-*/assessment.json"))
    try:
        if existing:
            result = saved_assessment(project, existing[-1])
            print(json.dumps({"status":"existing_assessment", "receipt":str(existing[-1]),
                              "assessment_status":result["status"], "saved_evidence_verified":True,
                              "message":"The saved assessment is unchanged; this command does not reopen targets or refreeze choices."},indent=2))
            return 0
        review, study = assessment_inputs(project)
        if not (project/"models/selected/manifest.json").is_file():
            raise FileNotFoundError("Run application_example.py before the final assessment.")
        if not frozen.exists():
            freeze_assessment(project,review,study,frozen)
        protocol = frozen/"protocol.json"
        run = project/"runs"/("final-"+datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
        result = run_final_assessment(project,protocol,digest(protocol.read_bytes()),run)
        print(json.dumps({"status":result["status"],"run_directory":str(run),"selected":result["selected"],
                          "test":result["results"][result["selected"]]["test"]["metrics"],"examples":result["examples"]},indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status":"needs_run", "message":str(error)},indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main(Path(__file__).resolve().parent))
