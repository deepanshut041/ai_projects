"""Historical calendar-request checks and bounded sensitivity probes."""

import math
from datetime import datetime, timedelta

from .features import FEATURE_NAMES, calendar_schema, encode_calendar


REPLAY_START = "2012-01-01T01:00:00"
REPLAY_END = "2012-06-30T23:00:00"


def _hour(value):
    if not isinstance(value, str):
        raise ValueError("Expected a canonical naive hourly timestamp.")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("Invalid hourly timestamp.") from error
    if result.tzinfo is not None or result.minute or result.second or result.microsecond or result.isoformat() != value:
        raise ValueError("Use a naive timestamp at an exact hour, with seconds included.")
    return result


def review_request_features(request, *, feature_names=FEATURE_NAMES):
    """Validate a review request without fitting, coercing or inventing calendar fields."""
    if not isinstance(feature_names, (tuple,list)) or tuple(feature_names) != FEATURE_NAMES:
        raise ValueError("Feature names and order differ from the supported schema.")
    fields = {"schema_version", "target_time", "forecast_issue_time", "calendar"}
    if not isinstance(request, dict) or set(request) != fields:
        raise ValueError("Expected exactly the declared request fields; outcomes are not inputs.")
    if request["schema_version"] != calendar_schema()["version"]:
        raise ValueError("Unknown calendar schema version.")
    target = _hour(request["target_time"]); issue = _hour(request["forecast_issue_time"])
    if not datetime.fromisoformat(REPLAY_START) <= target <= datetime.fromisoformat(REPLAY_END):
        raise ValueError("Target time is outside the supported historical validation replay window.")
    if issue != target - timedelta(hours=1):
        raise ValueError("Issue time must be exactly one hour before the target interval.")
    calendar = request["calendar"]
    if not isinstance(calendar, dict):
        raise ValueError("Calendar inputs must be named fields.")
    features = encode_calendar(calendar)
    if (calendar["hr"], calendar["mnth"], calendar["weekday"]) != (target.hour, target.month, (target.weekday()+1) % 7):
        raise ValueError("Calendar hour, month or weekday disagrees with the target timestamp.")
    return tuple(features)


def probe_group_response(model, request, *, feature_names=FEATURE_NAMES):
    """Inspect the selected group's response; this is not a serving/export entry point."""
    features = review_request_features(request, feature_names=feature_names)
    key = (features[:24].index(1.0), int(features[44]))
    groups = {(group["hour"], group["workingday"]): group for group in model["groups"]}
    if len(groups) != len(model["groups"]):
        raise ValueError("Duplicate fitted calendar groups.")
    group = groups.get(key)
    value = model["fallback"]["value"] if group is None else group["value"]
    if not math.isfinite(value) or value < 0:
        raise ValueError("Fitted rental values must be finite and nonnegative.")
    return {"prediction": value, "units": "rentals/hour", "group": list(key),
            "fallback_used": group is None, "group_training_count": 0 if group is None else group["training_count"],
            "target_available": False, "scope": "calendar sensitivity; observation coverage is a separate question"}
