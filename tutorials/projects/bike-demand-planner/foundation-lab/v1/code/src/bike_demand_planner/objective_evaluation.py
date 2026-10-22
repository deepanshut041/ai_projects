"""Separate objective, penalty, conditioning and feature-capacity comparisons."""

import json
import os
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .baselines import fit_hour_workingday_baseline, fit_mean_baseline
from .data import sha256
from .evaluation import (METRIC_DEFINITIONS, Forecast, _source_identity, _write_json,
                         _write_predictions, load_labelled_partition, score_forecasts)
from .objectives import LossConfig, fit_objective_model


def _identity(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def run_objective_study(processed_directory: Path, run_directory: Path, project_root: Path, *,
                        rates=(.01, .1, 1.0), pilot_steps=200, main_steps=2000,
                        scaling_steps=200, max_seconds=60.0):
    """Freeze training choices before opening validation, then freeze a selection receipt."""
    if not rates or any(isinstance(r, bool) or not np.isfinite(r) or r <= 0 for r in rates):
        raise ValueError("Use finite positive pilot rates.")
    if any(type(n) is not int or n < 1 for n in (pilot_steps, main_steps, scaling_steps)):
        raise ValueError("Use positive integer update budgets.")
    if isinstance(max_seconds, bool) or not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("Use a finite positive time budget.")
    run_directory.mkdir(parents=True, exist_ok=False)
    report = {"version": "objective-structure-study-v1", "status": "running",
              "started_utc": datetime.now(UTC).isoformat(), "test_evaluated": False,
              "trial_directories": {}, "pilots": [], "selected_rates": {}, "families": {},
              "metric_definitions": METRIC_DEFINITIONS}
    try:
        train, manifest = load_labelled_partition(processed_directory, "train")
        X = np.asarray([row.features for row in train]); y = np.asarray([row.target for row in train])
        source = _source_identity(project_root)
        environment = {"python": platform.python_version(), "numpy": np.__version__,
                       "platform": platform.platform(), "processor": platform.processor(),
                       "machine": platform.machine(), "logical_cpu_count": os.cpu_count(),
                       "blas_threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                       "dependencies_sha256": {n: sha256((project_root/n).read_bytes()) for n in ("pyproject.toml", "uv.lock")}}
        report.update(source=source, environment=environment,
                      dataset={k: manifest[k] for k in ("source_hour_sha256", "feature_schema_sha256", "membership_sha256", "manifest_sha256", "counts")})
        report["dataset"]["partition_sha256"] = {name: manifest["artifacts_sha256"][f"{name}.csv"]
                                                   for name in ("train", "validation")}
        task = {k: manifest[k] for k in ("forecast_lead_hours", "target_duration_hours", "availability_policy", "readiness_assumption", "timestamp_policy")}
        task.update(target="observed system-wide rentals", units="rentals/hour", target_transform="identity")
        losses = {"half_mse": LossConfig(), "mae": LossConfig("mae"),
                  "huber10": LossConfig("huber", 10), "huber50": LossConfig("huber", 50)}
        for label, loss in losses.items():
            candidates = []
            for rate in rates:
                model, receipt = fit_objective_model(X, y, loss=loss, rate=rate,
                                                     max_steps=pilot_steps, max_seconds=max_seconds)
                directory = run_directory / "pilot" / f"{label}-{rate:g}"
                directory.mkdir(parents=True)
                _write_json(directory/"model.json", asdict(model)); _write_json(directory/"fit.json", receipt)
                entry = {"loss": label, "rate": rate, "status": receipt["status"],
                         "stop_reason": receipt["stop_reason"], "updates": receipt["updates"],
                         "training_objective": receipt["history"][-1]["objective"],
                         "artifacts_sha256": {p.name: sha256(p.read_bytes()) for p in directory.iterdir()},
                         "directory": str(directory.relative_to(run_directory))}
                report["pilots"].append(entry)
                if receipt["status"] == "completed" and receipt["updates"] == pilot_steps:
                    candidates.append(entry)
            if not candidates:
                raise ValueError(f"No completed pilot for {label}.")
            report["selected_rates"][label] = min(candidates, key=lambda c: (c["training_objective"], c["rate"]))["rate"]
        protocol = {"version": "objective-structure-protocol-v1", "selected_rates": report["selected_rates"],
                    "pilot_rates": list(rates), "pilot_steps": pilot_steps, "main_steps": main_steps,
                    "scaling_steps": scaling_steps, "max_seconds": max_seconds,
                    "l2_strengths": [0, .1, 1, 10], "losses": {k: asdict(v) for k,v in losses.items()},
                    "structure_representations": ["onehot45", "cyclic8", "hour_workingday69"],
                    "structure_solver": "training-only least squares; no L2",
                    "scaling_comparison": {"solver": "sgd", "rate": .1, "steps": scaling_steps, "l2": 0},
                    "full_batch_deterministic": True, "random_seed": None,
                    "validation_used_for_rate_selection": False,
                    "prediction_policies": ["raw", "nonnegative"],
                    "selection_rule": "minimum validation MAE; exact ties prefer raw policy, then model name",
                    "families": "objectives at fixed representation/Adam; exact L2 solves; fixed-work SGD scaling; exact-solve representation changes",
                    "l2_normalization": "mean data loss + lambda/2 * squared weight norm; intercept excluded",
                    "optional_target_transforms": "deferred; all current targets remain unscaled"}
        _write_json(run_directory/"protocol.json", protocol)
        report["protocol_sha256"] = _identity(protocol)
        fitted = {}

        def fit(name, **settings):
            loss = settings.pop("loss", LossConfig())
            config = dict(loss=asdict(loss), **settings)
            model, receipt = fit_objective_model(X, y, loss=loss, max_seconds=max_seconds, **settings)
            directory = run_directory/"trials"/name; directory.mkdir(parents=True)
            record = {"version": "hourly-regression-run-v1", "status": receipt["status"],
                      "configuration": config, "optimization": receipt, "source": source,
                      "environment": environment, "dataset": report["dataset"], "task": task,
                      "protocol_sha256": report["protocol_sha256"], "test_evaluated": False}
            _write_json(directory/"model.json", asdict(model)); _write_json(directory/"fit.json", receipt)
            _write_json(directory/"run.json", record)
            report["trial_directories"][name] = str(directory.relative_to(run_directory))
            if receipt["status"] == "completed": fitted[name] = (model, record, directory)
            return name

        report["families"]["loss"] = {label: fit("loss-"+label, loss=loss,
                    rate=report["selected_rates"][label], max_steps=main_steps) for label,loss in losses.items()}
        report["families"]["l2"] = {str(strength): fit(f"l2-{strength:g}", loss=LossConfig(l2=strength), solver="least_squares")
                                        for strength in protocol["l2_strengths"]}
        report["families"]["scaling"] = {scaling: fit("scaling-"+scaling, scaling=scaling,
                            solver="sgd", rate=.1, max_steps=scaling_steps) for scaling in ("raw", "standardized")}
        report["families"]["structure"] = {"onehot45": "l2-0"}
        for representation in ("cyclic8", "hour_workingday69"):
            report["families"]["structure"][representation] = fit("structure-"+representation,
                    representation=representation, solver="least_squares")
        # The same training-only least-squares function expressed in raw coordinates.
        raw_reference, raw_reference_fit = fit_objective_model(X, y, scaling="raw", solver="least_squares")
        standard_reference = fitted["l2-0"][0]
        report["scaling_reference"] = {"raw_model": asdict(raw_reference), "fit": raw_reference_fit,
                                      "purpose": "same unpenalized representable function; compare predictions, not coefficients"}
        global_model = fit_mean_baseline(y.tolist())
        group_model = fit_hour_workingday_baseline([(int(x[:24].argmax()), int(x[44])) for x in X], y.tolist())
        _write_json(run_directory/"baselines.json", {"global": asdict(global_model), "groups": asdict(group_model)})
        # Every rate choice, main model and baseline is fitted before the first validation load.
        validation, second_manifest = load_labelled_partition(processed_directory, "validation")
        if second_manifest != manifest: raise ValueError("Frozen data changed during the study.")
        all_results = {}
        for partition, rows in (("train", train), ("validation", validation)):
            values = np.asarray([r.features for r in rows])
            entries = {name: model.predict(values) for name,(model,_,_) in fitted.items()}
            entries["global"] = np.asarray(global_model.predict(len(rows)))
            entries["groups"] = np.asarray(group_model.predict([(int(x[:24].argmax()), int(x[44])) for x in values]))
            for name, predictions in entries.items():
                directory = fitted[name][2] if name in fitted else run_directory/"trials"/name
                directory.mkdir(parents=True, exist_ok=True)
                result = all_results.setdefault(name, {"metrics": {}, "ranges": {}, "examples": {}})
                result["metrics"][partition] = {}
                result["ranges"][partition] = {"negative_raw_count": int(np.count_nonzero(predictions < 0)),
                                                "raw_minimum": float(predictions.min()), "raw_maximum": float(predictions.max())}
                for policy in ("raw", "nonnegative"):
                    outputs = predictions if policy == "raw" else np.maximum(predictions, 0)
                    forecasts = [Forecast(row.identity, float(p)) for row,p in zip(rows,outputs,strict=True)]
                    result["metrics"][partition][policy] = score_forecasts(rows, forecasts)
                    _write_predictions(directory/f"{partition}-{policy}-predictions.csv", rows, forecasts)
                if partition == "validation":
                    for row,prediction in zip(rows,predictions,strict=True):
                        if str(row.identity.target_time)[:13] in ("2012-01-09T02", "2012-01-09T08"):
                            result["examples"][str(row.identity.target_time)] = {"target": row.target, "raw": float(prediction), "nonnegative": max(0,float(prediction))}
            report["scaling_reference"][partition+"_maximum_prediction_difference"] = float(np.max(np.abs(raw_reference.predict(values)-standard_reference.predict(values))))
        for name,result in all_results.items():
            directory = run_directory/"trials"/name
            if name in fitted:
                record = fitted[name][1]
            else:
                record = {"version": "hourly-regression-run-v1", "status": "completed", "source": source,
                          "environment": environment, "dataset": report["dataset"], "task": task,
                          "configuration": {"model": name}, "protocol_sha256": report["protocol_sha256"], "test_evaluated": False}
                _write_json(directory/"model.json", asdict(group_model if name=="groups" else global_model))
                report["trial_directories"][name] = str(directory.relative_to(run_directory))
            record.update(result)
            record["artifacts_sha256"] = {p.name: sha256(p.read_bytes()) for p in directory.iterdir() if p.name!="run.json"}
            _write_json(directory/"run.json", record)
        ranking = sorted((r["metrics"]["validation"][policy]["mae"], policy!="raw", name, policy)
                         for name,r in all_results.items() for policy in ("raw", "nonnegative"))
        best = ranking[0]
        selection = {"version": "simple-model-selection-v1", "model": best[2], "policy": best[3],
                     "validation_mae": best[0], "criterion": protocol["selection_rule"],
                     "protocol_sha256": report["protocol_sha256"], "validation_used_for_selection": True,
                     "test_evaluated": False, "serving_implemented": False,
                     "model_sha256": sha256((run_directory/"trials"/best[2]/"model.json").read_bytes()),
                     "ranking": [{"model": n,"policy": p,"validation_mae": m} for m,_,n,p in ranking]}
        _write_json(run_directory/"selection.json", selection)
        report.update(status="completed", results=all_results, selection=selection,
                      selected_configuration_frozen=True, validation_used_for_fitting=False)
    except Exception as error:
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        report["finished_utc"] = datetime.now(UTC).isoformat()
        _write_json(run_directory/"study.json", report)
    return report
