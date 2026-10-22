"""Frozen-model assessment with predictions persisted before held-out label access."""

import csv
import io
import platform
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter

import numpy as np

from .application_data import verified_review
from .artifacts import digest, durable_write, implementation_identity, json_bytes, publish_directory, read_json
from .baselines import CalendarGroupMean, HourWorkingDayBaseline, MeanBaseline
from .data import parse_hourly_row
from .error_analysis import error_summary, largest_misses, read_aligned_predictions, slice_diagnostics, temporal_diagnostics
from .evaluation import (IDENTITY_FIELDS, METRIC_DEFINITIONS, Forecast, HourIdentity, LabelledHour,
                         _read_manifest, _verified_bytes, _write_json, _write_predictions, load_labelled_partition)
from .features import FEATURE_NAMES, encode_calendar
from .objectives import ObjectiveModel, apply_prediction_policy
from .regression import FeatureStandardizer
from .trees import predict_snapshot


EXAMPLE_TIMES = ("2012-07-02T08:00:00", "2012-07-04T08:00:00", "2012-12-03T08:00:00")


@dataclass(frozen=True)
class CalendarHour:
    identity: HourIdentity
    features: tuple[float, ...]


def load_test_features(processed: Path) -> tuple[list[CalendarHour], dict]:
    manifest = _read_manifest(processed)
    members = csv.DictReader(io.StringIO(_verified_bytes(processed, "membership.csv", manifest).decode()))
    expected = [HourIdentity(int(r["record_id"]), r["target_time"], r["forecast_issue_time"])
                for r in members if r["split"] == "test"]
    reader = csv.DictReader(io.StringIO(_verified_bytes(processed, "test.csv", manifest).decode()))
    if reader.fieldnames != [*IDENTITY_FIELDS, *FEATURE_NAMES]:
        raise ValueError("Expected target-free test features in the frozen order.")
    rows = []
    for record in reader:
        identity = HourIdentity(int(record["record_id"]), record["target_time"], record["forecast_issue_time"])
        target_time = datetime.fromisoformat(identity.target_time)
        if (target_time.tzinfo is not None or target_time.minute or target_time.second or target_time.microsecond
                or (target_time-timedelta(hours=1)).isoformat() != identity.forecast_issue_time):
            raise ValueError("Test identity violates the naive hourly, one-hour lead contract.")
        values = tuple(float(record[n]) for n in FEATURE_NAMES)
        calendar = {"hr": target_time.hour, "weekday": (target_time.weekday()+1)%7,
                    "mnth": target_time.month, "holiday": int(values[43]), "workingday": int(values[44])}
        if tuple(encode_calendar(calendar)) != values:
            raise ValueError("Test features disagree with their calendar identity.")
        rows.append(CalendarHour(identity, values))
    if ([r.identity for r in rows] != expected or len(rows) != manifest["counts"]["test"]
            or len(set(expected)) != len(expected) or not rows):
        raise ValueError("Test identities differ from frozen membership.")
    return rows, manifest


def predict_frozen(name: str, snapshot: dict, features, policy: str):
    """Replay saved state only; targets are deliberately absent from this interface."""
    X = np.asarray(features, dtype=np.float64)
    if name == "global":
        values = MeanBaseline(**snapshot).predict(len(X))
    elif name == "groups":
        model = HourWorkingDayBaseline(tuple(CalendarGroupMean(**g) for g in snapshot["groups"]),
                                      MeanBaseline(**snapshot["fallback"]))
        values = model.predict([(int(row[:24].argmax()), int(row[44])) for row in X])
    elif name in ("linear-additive", "linear-interactions"):
        model = ObjectiveModel(tuple(snapshot["weights"]), snapshot["bias"],
                               FeatureStandardizer(**snapshot["standardizer"]), snapshot["representation"],
                               tuple(snapshot["feature_names"]))
        values = model.predict(X)
    elif name in ("tree", "forest", "boosting"):
        values = predict_snapshot(snapshot, X)
    else:
        raise ValueError("Unknown frozen model.")
    return apply_prediction_policy(values, policy)


def freeze_assessment(project: Path, review_directory: Path, study_directory: Path, destination: Path) -> str:
    """Bind development-selected artifacts before any final labels are requested."""
    if destination.exists():
        raise FileExistsError("Assessment already frozen; reuse its exact protocol instead of replacing it.")
    review = verified_review(review_directory)
    if digest((study_directory/"study.json").read_bytes()) != review["input_study_sha256"]:
        raise ValueError("Reviewed study identity changed.")
    study = read_json(study_directory/"study.json")
    validation, manifest = load_labelled_partition(project/"data/processed", "validation")
    if review["dataset"]["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Reviewed dataset differs from the frozen manifest.")
    choices = {}
    for name, result in review["results"].items():
        directory = study_directory/study["trial_directories"][result["trial"]]
        trial = read_json(directory/"run.json")
        if (digest((directory/"run.json").read_bytes()) != review["input_run_records_sha256"][result["trial"]]
                or digest((directory/"model.json").read_bytes()) != result["source_model_sha256"]):
            raise ValueError("A retained trial or model changed before freezing.")
        predicted = predict_frozen(name, read_json(directory/"model.json"), [r.features for r in validation], result["policy"])
        retained = read_aligned_predictions(directory/f'validation-{result["policy"]}-predictions.csv', validation)
        if not np.allclose(predicted, retained, rtol=1e-12, atol=1e-9):
            raise ValueError("Saved-state replay differs from the retained development predictions.")
        choices[name] = {"trial": result["trial"], "policy": result["policy"],
                         "model_path": str((directory/"model.json").relative_to(project)),
                         "model_sha256": result["source_model_sha256"],
                         "trial_sha256": digest((directory/"run.json").read_bytes()),
                         "fit": result["cost_record"], "configuration": trial["configuration"],
                         "model_json_bytes": (directory/"model.json").stat().st_size,
                         "validation": result["validation"]["summary"]["metrics"]}
    if set(choices) != {"global", "groups", "linear-additive", "linear-interactions", "tree", "forest", "boosting"}:
        raise ValueError("Expected the seven reviewed representative artifacts.")
    if choices["groups"]["model_sha256"] != read_json(project/"models/selected/candidate.json")["model_sha256"]:
        raise ValueError("Saved application and selected assessment model differ.")
    protocol = {"version": "hourly-final-assessment-protocol-v1", "frozen_utc": datetime.now(UTC).isoformat(),
                "selected": "groups", "selected_policy": "raw", "choices": choices,
                "selection_policy": "retain validation-selected groups/raw; report all seven fixed representatives; no test-driven reselection",
                "refitting": "none; retain original 2011 fits and training-only transforms", "new_seed_search": False,
                "seed_scope": "first predeclared seed 17 for tree families; deterministic references; not seed-averaged final scores",
                "metric_definitions": METRIC_DEFINITIONS, "primary_metric": "MAE in rentals/hour",
                "slices": {"hour": list(range(24)), "workingday": [0,1], "rush_hours": [7,8,9,16,17,18],
                           "month": list(range(1,13)), "season": {"winter":[12,1,2],"spring":[3,4,5],"summer":[6,7,8],"autumn":[9,10,11]},
                           "observed_demand": [50,200], "empty": "count zero; metrics null; no imputed hours"},
                "temporal": "daily and Monday-start weekly counts; adjacent-hour residual correlation excludes gaps; no IID confidence intervals",
                "largest_misses": 10, "example_times": list(EXAMPLE_TIMES),
                "budget": "seven saved-state prediction passes, no fits; post-run timing only; no adaptive stopping",
                "cost_scope": "retained historical fit cost; one local saved-snapshot batch prediction timing; JSON storage bytes; peak RAM unmeasured",
                "quality_criterion": "descriptive assessment; no operational acceptance threshold was specified",
                "post_test_policy": "changes become test-informed development and need new independent holdout evidence",
                "inference_domain": "batch assessment of frozen test features; original interactive predictor remains January-June validation replay",
                "dataset": manifest, "source_provenance": read_json(project/"data/raw/source.json"),
                "review_path": str(review_directory.relative_to(project)), "review_sha256": digest((review_directory/"review.json").read_bytes()),
                "bundle_sha256": digest((project/"models/selected/manifest.json").read_bytes()),
                "implementation": implementation_identity(project), "platform": platform.platform(),
                "validation_replay_verified": len(validation), "test_labels_opened": False}
    publish_directory(destination, {"protocol.json": json_bytes(protocol)})
    return digest((destination/"protocol.json").read_bytes())


def verify_protocol(project: Path, protocol_path: Path, expected_sha256: str) -> dict:
    if digest(protocol_path.read_bytes()) != expected_sha256:
        raise ValueError("Frozen protocol hash differs from the requested assessment.")
    protocol = read_json(protocol_path)
    if protocol.get("version") != "hourly-final-assessment-protocol-v1" or protocol["test_labels_opened"]:
        raise ValueError("Expected a pre-access frozen protocol.")
    if implementation_identity(project) != protocol["implementation"]:
        raise ValueError("Implementation/runtime differs from the frozen protocol; inspect before test access.")
    if _read_manifest(project/"data/processed") != protocol["dataset"]:
        raise ValueError("Frozen dataset changed.")
    if digest((project/"models/selected/manifest.json").read_bytes()) != protocol["bundle_sha256"]:
        raise ValueError("Serving bundle changed after freezing.")
    for choice in protocol["choices"].values():
        path = project/choice["model_path"]
        if (digest(path.read_bytes()) != choice["model_sha256"]
                or digest((path.parent/"run.json").read_bytes()) != choice["trial_sha256"]):
            raise ValueError("A frozen model or fitting receipt changed.")
    return protocol


def join_final_targets(raw_csv: Path, rows: list[CalendarHour], access_path: Path,
                       protocol_sha256: str, raw_sha256: str) -> list[LabelledHour]:
    """Require a durable pre-label forecast receipt before opening raw outcomes."""
    access = read_json(access_path)
    if (access.get("version") != "hourly-final-label-access-v1" or access["protocol_sha256"] != protocol_sha256
            or access["source_hour_sha256"] != raw_sha256 or access["count"] != len(rows)
            or not access["forecasts_saved_before_labels"]):
        raise ValueError("Held-out target access needs the matching pre-label receipt.")
    predictions = access_path.parent/"target-free-predictions.json"
    if digest(predictions.read_bytes()) != access["predictions_sha256"]:
        raise ValueError("Persisted forecasts changed before label access.")
    saved = read_json(predictions)
    identities = [[r.identity.record_id, r.identity.target_time, r.identity.forecast_issue_time] for r in rows]
    if saved["identities"] != identities or saved["protocol_sha256"] != protocol_sha256:
        raise ValueError("Pre-label prediction membership differs.")
    content = raw_csv.read_bytes()
    if digest(content) != raw_sha256:
        raise ValueError("Raw outcome source differs from the frozen source hash.")
    selected = {r.identity.record_id: r for r in rows}; found = {}; all_ids = set()
    for item in csv.DictReader(io.StringIO(content.decode("utf-8-sig"))):
        identity = int(item["instant"])
        if identity in all_ids:
            raise ValueError("Duplicate raw record identity.")
        all_ids.add(identity)
        if identity not in selected:
            continue
        raw = parse_hourly_row(item); expected = selected[identity]
        calendar = {n:raw.calendar[n] for n in ("hr","weekday","mnth","holiday","workingday")}
        if raw.target_time.isoformat() != expected.identity.target_time or tuple(encode_calendar(calendar)) != expected.features:
            raise ValueError("Raw target identity/calendar differs from the frozen test feature row.")
        found[identity] = LabelledHour(expected.identity, expected.features, float(raw.target))
    if len(found) != len(rows):
        raise ValueError("Raw source does not cover every frozen test row.")
    return [found[r.identity.record_id] for r in rows]


def run_final_assessment(project: Path, protocol_path: Path, expected_sha256: str, run_directory: Path) -> dict:
    """Execute the fixed report; failed attempts keep their access/forecast receipts."""
    protocol = verify_protocol(project, protocol_path, expected_sha256)
    run_directory.mkdir(parents=True, exist_ok=False)
    report = {"version": "hourly-final-assessment-v1", "status": "running", "started_utc": datetime.now(UTC).isoformat(),
              "protocol_sha256": expected_sha256, "selected": protocol["selected"], "new_fits": 0,
              "bundle_sha256": protocol["bundle_sha256"], "test_evaluated": False, "results": {}}
    started = perf_counter()
    try:
        rows, manifest = load_test_features(project/"data/processed")
        if manifest != protocol["dataset"]:
            raise ValueError("Test feature manifest changed.")
        forecasts = {}; timings = {}
        for name, choice in protocol["choices"].items():
            then = perf_counter()
            forecasts[name] = predict_frozen(name, read_json(project/choice["model_path"]), [r.features for r in rows], choice["policy"])
            timings[name] = perf_counter()-then
        saved = {"protocol_sha256": expected_sha256,
                 "identities": [[r.identity.record_id,r.identity.target_time,r.identity.forecast_issue_time] for r in rows],
                 "predictions": {n:v.tolist() for n,v in forecasts.items()}}
        durable_write(run_directory/"target-free-predictions.json", json_bytes(saved))
        access = {"version":"hourly-final-label-access-v1", "opened_utc":datetime.now(UTC).isoformat(),
                  "protocol_sha256":expected_sha256, "source_hour_sha256":manifest["source_hour_sha256"],
                  "count":len(rows), "forecasts_saved_before_labels":True,
                  "predictions_sha256":digest((run_directory/"target-free-predictions.json").read_bytes())}
        durable_write(run_directory/"label-access.json",json_bytes(access))
        labelled = join_final_targets(project/"data/raw/hour.csv",rows,run_directory/"label-access.json",
                                      expected_sha256,manifest["source_hour_sha256"])
        report["test_evaluated"] = True
        for name, values in forecasts.items():
            choice = protocol["choices"][name]
            report["results"][name] = {**choice, "test":error_summary(labelled,values),
                "slices":slice_diagnostics(labelled,values), "temporal":temporal_diagnostics(labelled,values),
                "largest_misses":largest_misses(labelled,values,protocol["largest_misses"]),
                "predict_batch_seconds":timings[name], "status":"completed"}
            _write_predictions(run_directory/f"{name}-test-predictions.csv",labelled,
                               [Forecast(r.identity,float(v)) for r,v in zip(labelled,values,strict=True)])
        selected = forecasts[protocol["selected"]]
        report["examples"] = [{"record_id":r.identity.record_id,"target_time":r.identity.target_time,
                               "forecast_issue_time":r.identity.forecast_issue_time,"target":r.target,
                               "prediction":float(selected[i]),"residual":float(selected[i])-r.target,
                               "hour":r.features[:24].index(1.0),"holiday":int(r.features[43]),"workingday":int(r.features[44])}
                              for i,r in enumerate(labelled) if r.identity.target_time in protocol["example_times"]]
        first=datetime.fromisoformat(rows[0].identity.target_time);last=datetime.fromisoformat(rows[-1].identity.target_time)
        report["coverage"]={"count":len(rows),"first":first.isoformat(),"last":last.isoformat(),
                            "unrecorded_hour_slots":int((last-first).total_seconds()/3600)+1-len(rows),"missing_targets_imputed":False}
        report.update(status="completed",selection_changed=False,source_provenance=protocol["source_provenance"],
                      metric_definitions=protocol["metric_definitions"],cost_scope=protocol["cost_scope"],
                      interpretation="fixed candidate assessment; no operational acceptance threshold or independent learner/hosted proof")
    except Exception as error:
        report.update(status="failed",failure={"type":type(error).__name__,"message":str(error)})
        raise
    finally:
        report.update(finished_utc=datetime.now(UTC).isoformat(),total_seconds=perf_counter()-started)
        report["artifacts_sha256"]={f.name:digest(f.read_bytes()) for f in sorted(run_directory.iterdir()) if f.is_file() and f.name!="assessment.json"}
        _write_json(run_directory/"assessment.json",report)
    return report
