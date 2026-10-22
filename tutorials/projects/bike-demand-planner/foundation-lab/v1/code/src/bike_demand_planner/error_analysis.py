"""Aligned residual diagnostics with explicit slice and time denominators."""

import csv
import math
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

from .evaluation import IDENTITY_FIELDS, Forecast, score_forecasts
from .features import FEATURE_NAMES


RUSH_HOURS = (7, 8, 9, 16, 17, 18)
SEASONS = {"winter": (12, 1, 2), "spring": (3, 4, 5), "summer": (6, 7, 8), "autumn": (9, 10, 11)}
PREDICTION_COLUMNS = [*IDENTITY_FIELDS, "target", "prediction", "residual", "absolute_error", "squared_error", "underprediction"]


def read_aligned_predictions(path, rows):
    """Require frozen order and targets, including the exported residual arithmetic."""
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != PREDICTION_COLUMNS:
            raise ValueError("Unexpected prediction columns.")
        saved = list(reader)
    if len(saved) != len(rows):
        raise ValueError("Prediction count differs from the frozen partition.")
    values = []
    for row, record in zip(rows, saved, strict=True):
        expected = (str(row.identity.record_id), row.identity.target_time, row.identity.forecast_issue_time)
        if tuple(record[field] for field in IDENTITY_FIELDS) != expected or float(record["target"]) != row.target:
            raise ValueError("Prediction identities, order or targets differ from the frozen rows.")
        prediction = float(record["prediction"])
        if not math.isfinite(prediction):
            raise ValueError("Prediction must be finite.")
        error = prediction-row.target
        for field, value in (("residual", error), ("absolute_error", abs(error)),
                             ("squared_error", error*error), ("underprediction", max(-error, 0.0))):
            if not math.isclose(float(record[field]), value, rel_tol=1e-12, abs_tol=1e-10):
                raise ValueError("Exported error arithmetic differs from prediction minus target.")
        values.append(prediction)
    score_forecasts(rows, [Forecast(row.identity, p) for row,p in zip(rows,values,strict=True)])
    return np.asarray(values, dtype=np.float64)


def error_summary(rows, predictions):
    values = np.asarray(predictions, dtype=np.float64)
    if values.shape != (len(rows),) or not np.isfinite(values).all():
        raise ValueError("Expected aligned finite predictions.")
    if not rows:
        return {"count": 0, "metrics": None, "mean_target": None, "mean_prediction": None,
                "absolute_error_sum": 0.0, "squared_error_sum": 0.0, "residual_sum": 0.0,
                "deficit_sum": 0.0, "excess_sum": 0.0, "overprediction_count": 0,
                "mean_overprediction": None, "mean_overprediction_when_over": None}
    metrics = score_forecasts(rows, [Forecast(row.identity,float(v)) for row,v in zip(rows,values,strict=True)])
    errors = [float(v)-row.target for row,v in zip(rows,values,strict=True)]
    over = sum(e > 0 for e in errors); excess = math.fsum(max(e,0) for e in errors)
    return {"count": len(rows), "metrics": metrics,
            "mean_target": math.fsum(row.target for row in rows)/len(rows),
            "mean_prediction": math.fsum(values.tolist())/len(rows),
            "absolute_error_sum": math.fsum(abs(e) for e in errors),
            "squared_error_sum": math.fsum(e*e for e in errors), "residual_sum": math.fsum(errors),
            "deficit_sum": math.fsum(max(-e,0) for e in errors), "excess_sum": excess,
            "overprediction_count": over, "mean_overprediction": excess/len(rows),
            "mean_overprediction_when_over": excess/over if over else None}


def slice_diagnostics(rows, predictions):
    """Each named dimension partitions rows; dimensions overlap with each other."""
    values = np.asarray(predictions, dtype=np.float64)
    error_summary(rows, values)
    labels = {"hour": [str(h) for h in range(24)], "workingday": ["0", "1"],
              "hour_window": ["rush", "other"], "workingday_rush": ["yes", "no"],
              "month": [str(m) for m in range(1,13)], "season": list(SEASONS),
              "observed_demand": ["below_50", "50_to_below_200", "at_least_200"]}
    groups = {dimension: {label: [] for label in names} for dimension,names in labels.items()}
    for index,row in enumerate(rows):
        if len(row.features) != len(FEATURE_NAMES):
            raise ValueError("Expected the frozen calendar schema.")
        hour = row.features[:24].index(1.0); month = row.features[31:43].index(1.0)+1
        workingday = int(row.features[44]); rush = hour in RUSH_HOURS
        keys = {"hour": str(hour), "workingday": str(workingday), "hour_window": "rush" if rush else "other",
                "workingday_rush": "yes" if workingday and rush else "no", "month": str(month),
                "season": next(name for name,months in SEASONS.items() if month in months),
                "observed_demand": "below_50" if row.target < 50 else "50_to_below_200" if row.target < 200 else "at_least_200"}
        for dimension,label in keys.items(): groups[dimension][label].append(index)
    result = {}
    overall = error_summary(rows, values)
    for dimension,parts in groups.items():
        result[dimension] = {}
        for label,indices in parts.items():
            subset = [rows[i] for i in indices]
            summary = error_summary(subset, values[indices])
            summary["observed_months"] = sorted({datetime.fromisoformat(r.identity.target_time).month for r in subset})
            result[dimension][label] = summary
        assert sum(s["count"] for s in result[dimension].values()) == len(rows)
        for key in ("absolute_error_sum", "squared_error_sum", "residual_sum", "deficit_sum", "excess_sum"):
            if not math.isclose(math.fsum(s[key] for s in result[dimension].values()), overall[key], rel_tol=1e-12, abs_tol=1e-8):
                raise ValueError("Slice error mass does not reconstruct the full population.")
    return result


def largest_misses(rows, predictions, limit=10):
    if type(limit) is not int or limit < 1:
        raise ValueError("Use a positive largest-miss count.")
    values = np.asarray(predictions, dtype=np.float64); error_summary(rows, values)
    indices = sorted(range(len(rows)), key=lambda i: (-abs(float(values[i])-rows[i].target),
                     rows[i].identity.target_time, rows[i].identity.record_id))[:limit]
    return [{"record_id": rows[i].identity.record_id, "target_time": rows[i].identity.target_time,
             "forecast_issue_time": rows[i].identity.forecast_issue_time, "target": rows[i].target,
             "prediction": float(values[i]), "residual": float(values[i])-rows[i].target,
             "absolute_error": abs(float(values[i])-rows[i].target)} for i in indices]


def temporal_diagnostics(rows, predictions):
    values = np.asarray(predictions, dtype=np.float64); error_summary(rows, values)
    times = [datetime.fromisoformat(r.identity.target_time) for r in rows]
    if any(b <= a for a,b in zip(times,times[1:])):
        raise ValueError("Temporal diagnostics require strict chronological order.")
    errors = values-np.asarray([r.target for r in rows]); grouped = {"day": defaultdict(list), "week": defaultdict(list)}
    for i,time in enumerate(times):
        grouped["day"][time.date().isoformat()].append(i)
        grouped["week"][(time.date()-timedelta(days=time.weekday())).isoformat()].append(i)
    result = {}
    for dimension,groups in grouped.items():
        result[dimension] = [{"start_date": label, **error_summary([rows[i] for i in ix],values[ix]),
                              "observed_dates": len({times[i].date() for i in ix})} for label,ix in sorted(groups.items())]
    pairs = [(i,i+1) for i in range(len(times)-1) if times[i+1]-times[i] == timedelta(hours=1)]
    correlation = None
    if len(pairs) >= 2:
        x = np.array([errors[i] for i,_ in pairs]); y = np.array([errors[j] for _,j in pairs])
        if np.std(x) > 0 and np.std(y) > 0: correlation = float(np.corrcoef(x,y)[0,1])
    result["consecutive_hour_pairs"] = len(pairs)
    result["consecutive_hour_residual_correlation"] = correlation
    result["uncertainty_scope"] = "descriptive time structure; no IID confidence interval and no temporal uncertainty inferred from training seeds"
    result["gaps"] = [{"after": a.isoformat(), "before": b.isoformat(),
                       "unrecorded_slots_between": int((b-a).total_seconds()/3600)-1}
                      for a,b in zip(times,times[1:]) if b-a > timedelta(hours=1)]
    return result


def calendar_level_comparison(training_rows, validation_rows):
    """Diagnostic only: reweight historical calendar-cell means to later membership."""
    def key(row):
        return (row.features[31:43].index(1.0)+1, row.features[:24].index(1.0), int(row.features[44]))
    cells = defaultdict(list)
    for row in training_rows:
        if key(row)[0] <= 6: cells[key(row)].append(row.target)
    means = {k: math.fsum(targets)/len(targets) for k,targets in cells.items()}
    groups = {"all_shared": [], **{str(m): [] for m in range(1,7)}}; unmatched = 0
    for row in validation_rows:
        k = key(row)
        if k not in means: unmatched += 1; continue
        pair = (row.target, means[k]); groups["all_shared"].append(pair); groups[str(k[0])].append(pair)
    result = {}
    for label,pairs in groups.items():
        actual = math.fsum(p[0] for p in pairs)/len(pairs) if pairs else None
        historical = math.fsum(p[1] for p in pairs)/len(pairs) if pairs else None
        result[label] = {"count": len(pairs), "validation_target_mean": actual,
                         "historical_cell_mean_reweighted": historical,
                         "observed_level_difference": actual-historical if pairs else None}
    return {"version": "calendar-level-diagnostic-v1", "cell_fields": ["month", "hour", "workingday"],
            "shared_validation_count": len(groups["all_shared"]), "unmatched_validation_count": unmatched,
            "groups": result, "use": "retrospective diagnostic; not a fitted or ranked candidate and not a causal effect"}
