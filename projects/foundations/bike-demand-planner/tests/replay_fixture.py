"""Small independent saved-state fixtures; no external dataset is required."""

from datetime import date, timedelta
from pathlib import Path

from bike_demand_planner.artifacts import digest, json_bytes
from bike_demand_planner.features import FEATURE_NAMES, calendar_schema
from bike_demand_planner.review_contract import REPLAY_END, REPLAY_START


def write_bundle(directory: Path):
    directory.mkdir()
    model = {"groups": [{"hour":8,"workingday":0,"value":10.,"training_count":1},
                        {"hour":8,"workingday":1,"value":30.,"training_count":1}],
             "fallback":{"value":20.,"training_count":2}}
    candidate = {"version":"hourly-demand-candidate-v1", "model_kind":"hour_workingday_mean",
                 "schema":calendar_schema(), "feature_names":list(FEATURE_NAMES),
                 "model_sha256":digest(json_bytes(model)), "used_fields":["hr","workingday"],
                 "units":"rentals/hour", "output_policy":"raw floating point; no clipping or rounding",
                 "input_policy":{"target_time_start":REPLAY_START,"target_time_end":REPLAY_END,"forecast_lead_hours":1},
                 "known_limits":["calendar pooling"], "dataset":{"partition_sha256":{"validation":"fixture"}}}
    start,end=date(2012,1,1),date(2012,6,30)
    calendar = {"version":"recorded-holiday-calendar-v1", "validation_partition_sha256":"fixture",
                "holidays_by_date":{(start+timedelta(days=i)).isoformat():0 for i in range((end-start).days+1)}}
    for name,value in [("model.json",model),("candidate.json",candidate),("calendar.json",calendar)]:
        (directory/name).write_bytes(json_bytes(value))
    rehash(directory)
    return candidate


def rehash(directory: Path):
    (directory/"manifest.json").write_bytes(json_bytes({"version":"hourly-predictor-bundle-v1",
        "artifacts_sha256":{n:digest((directory/n).read_bytes()) for n in ["model.json","candidate.json","calendar.json"]}}))


def write_app_data(directory: Path, predictor):
    directory.mkdir()
    observation = {"version":"hourly-validation-observations-v1", "bundle_sha256":predictor.identity,
                   "dataset":predictor.candidate["dataset"], "test_evaluated":False,
                   "observations":{"2012-01-09T08:00:00":{"target":35.,"record_id":1,"forecast_issue_time":"2012-01-09T07:00:00"}}}
    benchmark = {"version":"hourly-replay-benchmarks-v1", "dataset":predictor.candidate["dataset"],
                 "bundle_sha256":predictor.identity,"rows":[],"failed_trials":[]}
    for name,value in [("replay.json",observation),("benchmarks.json",benchmark)]:
        (directory/name).write_bytes(json_bytes(value))
    (directory/"manifest.json").write_bytes(json_bytes({"version":"hourly-application-data-v1","bundle_sha256":predictor.identity,
        "artifacts_sha256":{n:digest((directory/n).read_bytes()) for n in ["replay.json","benchmarks.json"]}}))
