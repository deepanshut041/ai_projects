"""Review saved forecast failures and freeze a versioned development candidate."""

import copy
import json
import math
import platform
import shutil
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .comparison import summarize_tree_study
from .data import sha256
from .error_analysis import (RUSH_HOURS, SEASONS, calendar_level_comparison, error_summary,
                             largest_misses, read_aligned_predictions, slice_diagnostics, temporal_diagnostics)
from .evaluation import _source_identity, _write_json, load_labelled_partition
from .features import FEATURE_NAMES, calendar_schema, canonical_json
from .review_contract import REPLAY_END, REPLAY_START, probe_group_response, review_request_features


def _request(time, *, holiday=0):
    target = datetime.fromisoformat(time); weekday = (target.weekday()+1)%7
    return {"schema_version": calendar_schema()["version"], "target_time": time,
            "forecast_issue_time": (target-timedelta(hours=1)).isoformat(),
            "calendar": {"hr": target.hour, "weekday": weekday, "mnth": target.month,
                         "holiday": holiday, "workingday": int(weekday in range(1,6) and not holiday)}}


def contract_probes(model):
    valid = {"quiet_monday": _request("2012-01-09T02:00:00"), "busy_monday": _request("2012-01-09T08:00:00"),
             "sunday_same_hour": _request("2012-01-08T08:00:00"), "june_monday_same_hour": _request("2012-06-04T08:00:00")}
    accepted = {name: {"request": request, "response": probe_group_response(model,request)} for name,request in valid.items()}
    base = valid["busy_monday"]; invalid = {}
    def changed(name, mutate):
        request = copy.deepcopy(base); mutate(request); invalid[name] = request
    changed("missing_hour", lambda r:r["calendar"].pop("hr"))
    changed("nonfinite_hour", lambda r:r["calendar"].update(hr=float("nan")))
    changed("unknown_hour_category", lambda r:r["calendar"].update(hr=24))
    changed("unknown_calendar_field", lambda r:r["calendar"].update(weather=1))
    changed("unknown_schema", lambda r:r.update(schema_version="calendar-v999"))
    changed("outcome_in_request", lambda r:r.update(target=420))
    changed("inconsistent_weekday", lambda r:r["calendar"].update(weekday=2))
    changed("inconsistent_workingday", lambda r:r["calendar"].update(workingday=0))
    changed("wrong_lead", lambda r:r.update(forecast_issue_time=r["target_time"]))
    changed("timezone_aware", lambda r:r.update(target_time=r["target_time"]+"+00:00"))
    changed("non_hourly_time", lambda r:r.update(target_time="2012-01-09T08:30:00"))
    invalid["reserved_test_date"] = _request("2012-07-01T01:00:00")
    invalid["future_year"] = _request("2013-01-07T08:00:00")
    rejected = {}
    for name,request in invalid.items():
        try:
            review_request_features(request)
        except ValueError as error:
            rejected[name] = {"status": "rejected", "reason": str(error)}
        else:
            raise ValueError("An unsupported request was accepted: "+name)
    names = list(FEATURE_NAMES); names[0],names[1] = names[1],names[0]
    try:
        review_request_features(base,feature_names=names)
    except ValueError as error:
        rejected["reordered_features"] = {"status": "rejected", "reason": str(error)}
    else:
        raise ValueError("Reordered model features were accepted.")
    return {"accepted": accepted, "rejected": rejected,
            "scope": "requests exercise structural/domain checks; invalid probes are excluded from benchmark metrics; holiday authority and observed-hour coverage are separate"}


def run_error_review(processed_directory: Path, source_run: Path, run_directory: Path, project_root: Path, *, make_plots=True):
    run_directory.mkdir(parents=True, exist_ok=False)
    report = {"version": "hourly-error-review-v1", "status": "running", "started_utc": datetime.now(UTC).isoformat(),
              "test_evaluated": False, "models_refitted": False}
    try:
        prior = summarize_tree_study(source_run)
        study = json.loads((source_run/"study.json").read_text())
        if prior["selection"]["configuration"] != "groups" or prior["selection"]["policy"] != "raw":
            raise ValueError("This candidate contract expects the prior groups/raw selection.")
        choices = {}
        for configuration in ("global", "groups", "linear-additive", "linear-interactions"):
            row = next(r for r in prior["selection"]["ranking"] if r["configuration"]==configuration)
            choices[configuration] = {"trial": row["representative_trial"], "policy": row["policy"]}
        for family,row in prior["selection"]["family_winners"].items():
            choices[family] = {"trial": row["representative_trial"], "policy": row["policy"]}
        protocol = {"version": "hourly-error-review-protocol-v1", "source_run": source_run.name,
                    "source_study_sha256": sha256((source_run/"study.json").read_bytes()), "choices": choices,
                    "new_model_fits": 0, "largest_miss_count": 10, "rush_hours": list(RUSH_HOURS),
                    "seasons_by_month": {k:list(v) for k,v in SEASONS.items()}, "month_bins": list(range(1,13)),
                    "observed_demand_bands": ["[0,50)", "[50,200)", "[200,infinity)"],
                    "denominators": "slice metrics divide by actual observed slice count; conditional underprediction divides by underpredicted rows",
                    "time_summaries": "actual-count daily and Monday-start weekly groups; consecutive-hour correlation excludes gaps; no IID intervals",
                    "level_diagnostic": "shared (month,hour,workingday) cells, Jan-Jun; training means reweighted by validation membership; not a candidate or causal effect",
                    "selection": "retain integrity-checked prior groups/raw; no new tuning from diagnostic slices",
                    "validation_already_used_in_prior_development": True, "test_evaluated": False}
        _write_json(run_directory/"protocol.json",protocol)
        report.update(protocol_sha256=sha256(canonical_json(protocol)), protocol=protocol,
                      analysis_source=_source_identity(project_root),
                      environment={"python": platform.python_version(), "platform": platform.platform(),
                                   "dependencies_sha256": {n:sha256((project_root/n).read_bytes()) for n in ("pyproject.toml","uv.lock")}},
                      input_study_sha256=protocol["source_study_sha256"], input_run_records_sha256={})
        training,manifest = load_labelled_partition(processed_directory,"train")
        validation,second = load_labelled_partition(processed_directory,"validation")
        if manifest != second or any(study["dataset"][k] != manifest[k] for k in ("source_hour_sha256","feature_schema_sha256","membership_sha256","manifest_sha256","counts")):
            raise ValueError("Saved study and frozen partitions disagree.")
        report["dataset"] = study["dataset"]; report["task"] = study["task"]
        results = {}; retained_predictions = {}
        for label,choice in choices.items():
            directory = source_run/study["trial_directories"][choice["trial"]]
            record = json.loads((directory/"run.json").read_text())
            if record["task"] != study["task"] or record["protocol_sha256"] != study["protocol_sha256"]:
                raise ValueError("A source trial has an incompatible task or protocol.")
            report["input_run_records_sha256"][choice["trial"]] = sha256((directory/"run.json").read_bytes())
            results[label] = {"trial": choice["trial"], "policy": choice["policy"], "source_model_sha256": record["artifacts_sha256"]["model.json"],
                              "cost_record": record["fit"], "score_scope": "one retained representative model; not the configuration mean across seeds"}
            for partition,rows in (("train",training),("validation",validation)):
                path = directory/f'{partition}-{choice["policy"]}-predictions.csv'
                predictions = read_aligned_predictions(path,rows)
                summary = error_summary(rows,predictions)
                for key,value in record["metrics"][partition][choice["policy"]].items():
                    actual = summary["metrics"][key]
                    if value is None:
                        if actual is not None: raise ValueError("Saved undefined metric changed.")
                    elif not math.isclose(value,actual,rel_tol=1e-12,abs_tol=1e-10):
                        raise ValueError("Saved metrics differ from aligned prediction reductions.")
                results[label][partition] = {"summary": summary, "slices": slice_diagnostics(rows,predictions),
                                            "temporal": temporal_diagnostics(rows,predictions),
                                            "largest_misses": largest_misses(rows,predictions)}
                if partition=="validation": retained_predictions[label] = predictions
        report["results"] = results
        level = calendar_level_comparison(training,validation)
        report["calendar_level_comparison"] = level
        groups_dir = source_run/study["trial_directories"][choices["groups"]["trial"]]
        model = json.loads((groups_dir/"model.json").read_text())
        report["contract_probes"] = contract_probes(model)
        report["examples"] = {}
        for i,row in enumerate(validation):
            if row.identity.target_time in ("2012-01-09T02:00:00","2012-01-09T08:00:00"):
                report["examples"][row.identity.target_time] = {"identity": asdict(row.identity), "target": row.target,
                   "predictions": {label:float(values[i]) for label,values in retained_predictions.items()}}
        shutil.copyfile(groups_dir/"model.json",run_directory/"candidate-model.json")
        candidate = {"version": "hourly-demand-candidate-v1", "model_kind": "hour_workingday_mean", "model_artifact": "candidate-model.json",
                     "model_sha256": sha256((run_directory/"candidate-model.json").read_bytes()),
                     "fit_source": study["source"], "source_study": source_run.name, "source_study_sha256": report["input_study_sha256"],
                     "source_protocol_sha256": study["protocol_sha256"], "review_protocol_sha256": report["protocol_sha256"],
                     "dataset": study["dataset"], "schema": calendar_schema(), "feature_names": list(FEATURE_NAMES),
                     "used_fields": ["hr","workingday"], "target": "system-wide observed rentals in one hour", "units": "rentals/hour",
                     "output_policy": "raw floating point; no clipping or rounding", "fallback_policy": "training-only global mean for a valid absent (hour,workingday) group",
                     "fit_partition": "2011 training only; no review refit", "validation_metrics": results["groups"]["validation"]["summary"]["metrics"],
                     "input_policy": {"request_fields": ["schema_version","target_time","forecast_issue_time","calendar"],
                                      "target_time_start": REPLAY_START, "target_time_end": REPLAY_END,
                                      "timestamp": "canonical naive hourly interval start; timezone/DST unspecified in source",
                                      "forecast_lead_hours": 1, "holiday_metadata": "trusted external/recorded flag; structural consistency checked, not holiday authority",
                                      "missing_observations": "valid estimates do not create a target or zero for unrecorded hours",
                                      "rejection": "missing/extra/nonfinite/inconsistent fields, reordered features, unknown schema, unsupported dates"},
                     "selection_basis": "prior validation-led model/policy choice retained after diagnostics; no retuning",
                     "known_limits": ["same hour/day-type key ignores month and demand-level change", "large rush-hour underprediction remains",
                                      "observed rentals do not measure unmet demand or station inventory", "no causal claim or calibrated predictive uncertainty"],
                     "test_evaluated": False, "serving_implemented": False, "recovery_verified": False}
        _write_json(run_directory/"candidate.json",candidate)
        report["candidate_sha256"] = sha256((run_directory/"candidate.json").read_bytes())
        if make_plots:
            from .error_plots import make_review_plots
            report["figures"] = make_review_plots(run_directory/"figures",validation,retained_predictions["groups"],results["groups"]["validation"],level)
        report["artifacts_sha256"] = {str(f.relative_to(run_directory)):sha256(f.read_bytes()) for f in sorted(run_directory.rglob("*"))
                                      if f.is_file() and ".matplotlib-cache" not in f.parts}
        report.update(status="completed",selected_candidate="groups/raw", validation_used_for_diagnostics=True)
    except Exception as error:
        report.update(status="failed",failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        report["finished_utc"] = datetime.now(UTC).isoformat()
        _write_json(run_directory/"review.json",report)
    return report
