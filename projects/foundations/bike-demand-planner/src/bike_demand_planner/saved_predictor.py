"""Target-free prediction from a verified, calendar-only JSON bundle."""

import math
from datetime import timedelta
from pathlib import Path

from .artifacts import digest, read_json
from .features import FEATURE_NAMES, calendar_schema
from .review_contract import REPLAY_END, REPLAY_START, _hour, probe_group_response


class SavedPredictor:
    def __init__(self, directory: Path, *, expected_manifest_sha256: str | None = None):
        directory = Path(directory)
        manifest_bytes = (directory / "manifest.json").read_bytes()
        self.identity = digest(manifest_bytes)
        if expected_manifest_sha256 is not None and self.identity != expected_manifest_sha256:
            raise ValueError("Bundle manifest differs from the trusted identity.")
        self.manifest = read_json(directory / "manifest.json")
        if self.manifest.get("version") != "hourly-predictor-bundle-v1":
            raise ValueError("Unsupported predictor bundle version.")
        members = self.manifest["artifacts_sha256"]
        if set(members) != {"candidate.json", "model.json", "calendar.json"}:
            raise ValueError("Unexpected predictor bundle members.")
        for name, identity in members.items():
            if digest((directory/name).read_bytes()) != identity:
                raise ValueError(f"Bundle member {name} differs from its saved identity.")
        self.candidate = read_json(directory / "candidate.json")
        self.model = read_json(directory / "model.json")
        self.calendar = read_json(directory / "calendar.json")
        c = self.candidate
        if (c.get("version") != "hourly-demand-candidate-v1" or c.get("model_kind") != "hour_workingday_mean"
                or c["schema"] != calendar_schema() or c["feature_names"] != list(FEATURE_NAMES)
                or c["model_sha256"] != members["model.json"] or c["used_fields"] != ["hr", "workingday"]
                or c["units"] != "rentals/hour" or c["output_policy"] != "raw floating point; no clipping or rounding"
                or c["input_policy"]["target_time_start"] != REPLAY_START
                or c["input_policy"]["target_time_end"] != REPLAY_END
                or c["input_policy"]["forecast_lead_hours"] != 1):
            raise ValueError("Candidate differs from the supported prediction contract.")
        if (self.calendar.get("version") != "recorded-holiday-calendar-v1"
                or self.calendar["validation_partition_sha256"] != c["dataset"]["partition_sha256"]["validation"]):
            raise ValueError("Unknown calendar authority version.")
        self._check_model()

    def _check_model(self):
        if set(self.model) != {"groups", "fallback"}:
            raise ValueError("Unexpected fitted group model fields.")
        seen = set()
        for g in self.model["groups"]:
            key = g["hour"], g["workingday"]
            if (set(g) != {"hour", "workingday", "value", "training_count"}
                    or type(key[0]) is not int or key[0] not in range(24)
                    or type(key[1]) is not int or key[1] not in (0, 1) or key in seen):
                raise ValueError("Invalid or duplicate fitted calendar group.")
            seen.add(key)
            self._check_level(g)
        self._check_level(self.model["fallback"])
        if sum(g["training_count"] for g in self.model["groups"]) != self.model["fallback"]["training_count"]:
            raise ValueError("Fitted group counts disagree with the training population.")
        dates = self.calendar["holidays_by_date"]
        start, end = _hour(REPLAY_START).date(), _hour(REPLAY_END).date()
        expected = {(start+timedelta(days=i)).isoformat() for i in range((end-start).days+1)}
        if set(dates) != expected or any(type(v) is not int or v not in (0, 1) for v in dates.values()):
            raise ValueError("Calendar authority must cover every supported date with binary holiday flags.")

    @staticmethod
    def _check_level(level):
        value, count = level["value"], level["training_count"]
        if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
                or type(count) is not int or count < 1):
            raise ValueError("Fitted levels require finite nonnegative values and positive counts.")

    def request_for_target(self, target_time: str) -> dict:
        target = _hour(target_time)
        if not _hour(REPLAY_START) <= target <= _hour(REPLAY_END):
            raise ValueError("Target time is outside the supported historical validation replay window.")
        weekday = (target.weekday()+1) % 7
        holiday = self.calendar["holidays_by_date"][target.date().isoformat()]
        return {"schema_version": calendar_schema()["version"], "target_time": target_time,
                "forecast_issue_time": (target-timedelta(hours=1)).isoformat(),
                "calendar": {"hr": target.hour, "weekday": weekday, "mnth": target.month,
                             "holiday": holiday, "workingday": int(weekday in range(1, 6) and not holiday)}}

    def predict(self, request: dict) -> dict:
        result = probe_group_response(self.model, request, feature_names=self.candidate["feature_names"])
        expected = self.request_for_target(request["target_time"])
        if request["calendar"] != expected["calendar"]:
            raise ValueError("Calendar fields disagree with the saved recorded calendar authority.")
        return {**result, "target_time": request["target_time"], "forecast_issue_time": request["forecast_issue_time"],
                "baseline": self.model["fallback"]["value"], "baseline_kind": "training-only global mean",
                "model_kind": self.candidate["model_kind"], "model_sha256": self.candidate["model_sha256"],
                "bundle_sha256": self.identity, "schema_version": request["schema_version"],
                "task": "historical validation replay; calendar-only observed system-wide rentals",
                "scope": "saved calendar-only prediction; observation coverage is separate",
                "known_limits": self.candidate["known_limits"]}
