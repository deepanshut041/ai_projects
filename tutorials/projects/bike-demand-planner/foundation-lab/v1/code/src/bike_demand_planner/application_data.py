"""Build immutable replay assets from the reviewed development evidence."""

from pathlib import Path

from .artifacts import digest, implementation_identity, json_bytes, publish_directory, read_json
from .evaluation import load_labelled_partition
from .features import FEATURE_NAMES
from .saved_predictor import SavedPredictor


def verified_review(review_directory: Path) -> dict:
    review = read_json(review_directory / "review.json")
    if (review.get("version") != "hourly-error-review-v1" or review.get("status") != "completed"
            or review["test_evaluated"] or review["models_refitted"] or review["selected_candidate"] != "groups/raw"):
        raise ValueError("Expected a completed, development-only group-candidate review.")
    for name in ("candidate.json", "candidate-model.json"):
        if digest((review_directory/name).read_bytes()) != review["artifacts_sha256"][name]:
            raise ValueError(f"Reviewed artifact {name} changed.")
    if digest((review_directory/"candidate.json").read_bytes()) != review["candidate_sha256"]:
        raise ValueError("Candidate identity changed.")
    return review


def export_predictor(processed: Path, review_directory: Path, destination: Path, project_root: Path) -> SavedPredictor:
    review = verified_review(review_directory)
    candidate = read_json(review_directory / "candidate.json")
    validation, manifest = load_labelled_partition(processed, "validation")
    if manifest["manifest_sha256"] != candidate["dataset"]["manifest_sha256"]:
        raise ValueError("Replay calendar and candidate have different frozen datasets.")
    dates = {}
    for row in validation:
        date = row.identity.target_time[:10]
        holiday = int(row.features[FEATURE_NAMES.index("holiday")])
        if date in dates and dates[date] != holiday:
            raise ValueError("Recorded holiday metadata disagrees within a date.")
        dates[date] = holiday
    calendar = {"version": "recorded-holiday-calendar-v1", "holidays_by_date": dates,
                "validation_partition_sha256": candidate["dataset"]["partition_sha256"]["validation"],
                "authority": "frozen source calendar flags; no observed rental outcomes"}
    files = {"candidate.json": (review_directory/"candidate.json").read_bytes(),
             "model.json": (review_directory/"candidate-model.json").read_bytes(), "calendar.json": json_bytes(calendar)}
    if destination.exists():
        existing = SavedPredictor(destination)
        if any(digest(content) != existing.manifest["artifacts_sha256"][name] for name, content in files.items()):
            raise ValueError("Existing export differs; choose a new directory instead of overwriting it.")
        return existing
    metadata = {"version": "hourly-predictor-bundle-v1", "artifacts_sha256": {n: digest(v) for n,v in files.items()},
                "review_sha256": digest((review_directory/"review.json").read_bytes()),
                "export_implementation": implementation_identity(project_root),
                "adapter_policy": "recorded calendar authority; strict historical replay; raw outputs; no fitting or targets at prediction",
                "test_evaluated": False}
    publish_directory(destination, {**files, "manifest.json": json_bytes(metadata)})
    return SavedPredictor(destination)


def prepare_application_data(processed: Path, review_directory: Path, study_directory: Path,
                             destination: Path, predictor: SavedPredictor) -> dict:
    review = verified_review(review_directory)
    if digest((study_directory/"study.json").read_bytes()) != review["input_study_sha256"]:
        raise ValueError("Benchmark study differs from the reviewed input.")
    study = read_json(study_directory / "study.json")
    validation, manifest = load_labelled_partition(processed, "validation")
    if manifest["manifest_sha256"] != predictor.candidate["dataset"]["manifest_sha256"]:
        raise ValueError("Observation membership differs from the predictor's dataset.")
    observations = {row.identity.target_time: {"record_id": row.identity.record_id, "target": row.target,
                    "forecast_issue_time": row.identity.forecast_issue_time} for row in validation}
    rows = []
    for name, result in review["results"].items():
        trial_directory = study_directory / study["trial_directories"][result["trial"]]
        trial_bytes = (trial_directory/"run.json").read_bytes()
        if digest(trial_bytes) != review["input_run_records_sha256"][result["trial"]]:
            raise ValueError("A benchmark trial record changed.")
        trial = read_json(trial_directory / "run.json")
        rows.append({"name": name, "policy": result["policy"], "selected": name == "groups",
                     "validation": result["validation"]["summary"]["metrics"],
                     "training": result["train"]["summary"]["metrics"], "fit": result["cost_record"],
                     "validation_snapshot_predict_seconds": trial["timings_seconds"]["validation_predict"],
                     "model_json_bytes": (trial_directory/"model.json").stat().st_size,
                     "model_sha256": result["source_model_sha256"], "trial": result["trial"],
                     "workingday_rush": result["validation"]["slices"]["workingday_rush"]["yes"]["metrics"],
                     "june": result["validation"]["slices"]["month"]["6"]["metrics"]})
    benchmark = {"version": "hourly-replay-benchmarks-v1", "dataset": predictor.candidate["dataset"],
                 "bundle_sha256": predictor.identity, "rows": rows, "failed_trials": study["failed_trials"],
                 "scope": "seven retained representative artifacts; prior development validation, not a new test",
                 "cost_scope": "local fit and transparent saved-state batch replay; JSON bytes are storage, not peak RAM",
                 "source_review_sha256": digest((review_directory/"review.json").read_bytes())}
    replay = {"version": "hourly-validation-observations-v1", "dataset": predictor.candidate["dataset"],
              "bundle_sha256": predictor.identity, "observations": observations, "test_evaluated": False}
    files = {"replay.json": json_bytes(replay), "benchmarks.json": json_bytes(benchmark)}
    identity = {"version": "hourly-application-data-v1", "bundle_sha256": predictor.identity,
                "artifacts_sha256": {n: digest(v) for n,v in files.items()}}
    publish_directory(destination, {**files, "manifest.json": json_bytes(identity)})
    return identity
