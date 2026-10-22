"""Bounded calendar-only tree comparisons on frozen chronological partitions."""

import json
import os
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import sklearn

from .baselines import fit_hour_workingday_baseline, fit_mean_baseline
from .data import sha256
from .evaluation import (METRIC_DEFINITIONS, Forecast, _source_identity, _write_json,
                         _write_predictions, hour_workingday_key,
                         load_labelled_partition, score_forecasts)
from .features import canonical_json
from .objectives import fit_objective_model
from .trees import TreeSpec, default_tree_candidates, fit_tree_model, predict_snapshot, trace_snapshot


def _digest(value):
    return sha256(canonical_json(value))


def run_tree_study(processed_directory: Path, run_directory: Path, project_root: Path, *,
                   candidates=None, seeds=(17, 29, 43), pilot_rows=1024,
                   pilot_seconds=5.0, fit_seconds_limit=60.0):
    """Freeze the sweep before validation; choose configurations across all declared seeds."""
    candidates = default_tree_candidates() if candidates is None else candidates
    if set(candidates) != {"tree", "forest", "boosting"} or any(not specs for specs in candidates.values()):
        raise ValueError("Provide nonempty candidate lists for all three families.")
    if any(not isinstance(spec, TreeSpec) or spec.family != family for family, specs in candidates.items() for spec in specs):
        raise ValueError("Candidate specification and family disagree.")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Declare distinct integer seeds.")
    if type(pilot_rows) is not int or pilot_rows < 1:
        raise ValueError("Use a positive pilot row count.")
    if any(isinstance(v, bool) or not np.isfinite(v) or v <= 0 for v in (pilot_seconds, fit_seconds_limit)):
        raise ValueError("Use finite positive timing guards.")
    run_directory.mkdir(parents=True, exist_ok=False)
    report = {"version": "calendar-tree-study-v1", "status": "running",
              "started_utc": datetime.now(UTC).isoformat(), "test_evaluated": False,
              "pilots": [], "trial_directories": {}, "failed_trials": [],
              "metric_definitions": METRIC_DEFINITIONS}
    try:
        train, manifest = load_labelled_partition(processed_directory, "train")
        X = np.asarray([r.features for r in train]); y = np.asarray([r.target for r in train])
        source = _source_identity(project_root)
        environment = {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                       "platform": platform.platform(), "processor": platform.processor(), "machine": platform.machine(),
                       "logical_cpu_count": os.cpu_count(), "forest_jobs": 1,
                       "thread_settings": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                       "dependencies_sha256": {n: sha256((project_root/n).read_bytes()) for n in ("pyproject.toml", "uv.lock")}}
        dataset = {k: manifest[k] for k in ("source_hour_sha256", "feature_schema_sha256", "membership_sha256", "manifest_sha256", "counts")}
        dataset["partition_sha256"] = {p: manifest["artifacts_sha256"][p+".csv"] for p in ("train", "validation")}
        task = {k: manifest[k] for k in ("forecast_lead_hours", "target_duration_hours", "availability_policy", "readiness_assumption", "timestamp_policy")}
        task.update(target="observed system-wide rentals", units="rentals/hour", target_transform="identity", inputs="fixed calendar-only 45 columns")
        protocol = {"version": "calendar-tree-protocol-v1", "candidates": {f: [asdict(s) for s in specs] for f, specs in candidates.items()},
                    "seeds": list(seeds), "pilot_rows": min(pilot_rows, len(train)), "pilot_seconds_guard": pilot_seconds,
                    "pilot_purpose": "training-only feasibility; never quality selection", "fit_seconds_guard": fit_seconds_limit,
                    "time_guard_semantics": "post-fit checks; not an interruptible hard timeout",
                    "structural_caps": {"leaves_per_tree": 1024, "forest_trees": 64, "boosting_stages": 300, "boosting_depth": 3},
                    "main_fit_count_by_family": {f: len(specs)*len(seeds) for f, specs in candidates.items()},
                    "preprocessing": "raw fixed binary calendar features; no learned tree preprocessing",
                    "references": ["global", "groups", "linear-additive", "linear-interactions"],
                    "prediction_policies": ["raw", "nonnegative"], "primary_metric": "validation MAE",
                    "selection_rule": "minimum configuration mean validation MAE over all declared seeds; exact ties prefer raw, then declared configuration order",
                    "representative_seed": seeds[0], "representative_policy": "first declared seed; never chosen by validation quality",
                    "replicate_interpretation": "descriptive algorithm-seed variation on the same held-out hours; not independent data replicates",
                    "validation_used_for_fitting": False, "test_evaluated": False}
        _write_json(run_directory/"protocol.json", protocol)
        report.update(protocol_sha256=_digest(protocol), source=source, environment=environment, dataset=dataset, task=task)
        fitted = {}; configurations = {}; model_kinds = {}

        def save_fit(name, model, receipt, configuration):
            directory = run_directory/"trials"/name; directory.mkdir(parents=True)
            record = {"version": "calendar-tree-trial-v1", "status": "completed", "configuration": configuration,
                      "fit": receipt, "source": source, "environment": environment, "dataset": dataset,
                      "task": task, "protocol_sha256": report["protocol_sha256"], "test_evaluated": False}
            _write_json(directory/"model.json", model); _write_json(directory/"run.json", record)
            fitted[name] = (model, record, directory)
            report["trial_directories"][name] = str(directory.relative_to(run_directory))

        for family, specs in candidates.items():
            directory = run_directory/"pilot"/family; directory.mkdir(parents=True)
            entry = {"family": family, "specification": asdict(specs[0]), "seed": seeds[0], "training_rows": min(pilot_rows, len(train)), "status": "running"}
            try:
                model, receipt = fit_tree_model(X[:pilot_rows], y[:pilot_rows], specs[0], seed=seeds[0])
                _write_json(directory/"model.json", model)
                entry.update(receipt)
                if receipt["fit_seconds"] > pilot_seconds:
                    raise RuntimeError("Training-only timing pilot exceeded its post-fit guard.")
                entry["artifacts_sha256"] = {"model.json": sha256((directory/"model.json").read_bytes())}
            except Exception as error:
                entry.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
                raise
            finally:
                _write_json(directory/"pilot.json", entry); report["pilots"].append(entry)

        for family, specs in candidates.items():
            for index, spec in enumerate(specs):
                config_name = f"{family}-c{index}"; configurations[config_name] = []
                for seed in seeds:
                    name = f"{config_name}-seed{seed}"; directory = run_directory/"trials"/name
                    try:
                        model, receipt = fit_tree_model(X, y, spec, seed=seed)
                        if receipt["fit_seconds"] > fit_seconds_limit:
                            directory.mkdir(parents=True)
                            _write_json(directory/"model.json", model)
                            _write_json(directory/"run.json", {"status": "over_budget", "fit": receipt, "test_evaluated": False})
                            raise RuntimeError("Main fit exceeded its post-fit time guard; later fits were not started.")
                        save_fit(name, model, receipt, {"specification": asdict(spec), "seed": seed, "configuration": config_name})
                        configurations[config_name].append(name); model_kinds[name] = "tree"
                    except RuntimeError:
                        raise
                    except Exception as error:
                        directory.mkdir(parents=True, exist_ok=True)
                        failure = {"name": name, "configuration": config_name, "status": "failed",
                                   "failure": {"type": type(error).__name__, "message": str(error)}, "test_evaluated": False}
                        _write_json(directory/"run.json", failure); report["failed_trials"].append(failure)

        # Deterministic references get one fit each and no newly tuned parameters.
        references = {}
        for name in ("global", "groups"):
            start = perf_counter()
            if name == "global": model = fit_mean_baseline(y.tolist())
            else: model = fit_hour_workingday_baseline([hour_workingday_key(r) for r in train], y.tolist())
            save_fit(name, asdict(model), {"fit_seconds": perf_counter()-start, "status": "completed"}, {"kind": name, "random_seed": None})
            references[name] = model; model_kinds[name] = name; configurations[name] = [name]
        for name, representation in (("linear-additive", "onehot45"), ("linear-interactions", "hour_workingday69")):
            start = perf_counter(); model, receipt = fit_objective_model(X, y, representation=representation, solver="least_squares")
            save_fit(name, asdict(model), {**receipt, "fit_seconds": perf_counter()-start}, {"representation": representation, "solver": "least_squares", "l2": 0})
            references[name] = model; model_kinds[name] = "linear"; configurations[name] = [name]
        _write_json(run_directory/"training-complete.json", {"validation_loaded": False, "protocol_sha256": report["protocol_sha256"],
                    "model_sha256": {n: sha256((d/"model.json").read_bytes()) for n,(_,_,d) in fitted.items()}})
        validation, second_manifest = load_labelled_partition(processed_directory, "validation")
        if second_manifest != manifest:
            raise ValueError("Frozen data changed during fitting.")
        results = {}
        for partition, rows in (("train", train), ("validation", validation)):
            values = np.asarray([r.features for r in rows]); keys = [hour_workingday_key(r) for r in rows]
            for name, (snapshot, record, directory) in fitted.items():
                start = perf_counter()
                kind = model_kinds[name]
                if kind == "tree": predictions = predict_snapshot(snapshot, values)
                elif kind == "global": predictions = np.asarray(references[name].predict(len(rows)))
                elif kind == "groups": predictions = np.asarray(references[name].predict(keys))
                else: predictions = references[name].predict(values)
                elapsed = perf_counter()-start
                result = results.setdefault(name, {"metrics": {}, "examples": {}, "timings_seconds": {}, "negative_raw_counts": {}})
                result["timings_seconds"][partition+"_predict"] = elapsed
                result["timings_seconds"]["predict_implementation"] = "transparent numerical snapshot replay for trees; native numerical reference otherwise"
                result["negative_raw_counts"][partition] = int(np.count_nonzero(predictions < 0))
                result["metrics"][partition] = {}
                for policy in ("raw", "nonnegative"):
                    outputs = predictions if policy == "raw" else np.maximum(predictions, 0)
                    forecasts = [Forecast(r.identity, float(p)) for r,p in zip(rows, outputs, strict=True)]
                    result["metrics"][partition][policy] = score_forecasts(rows, forecasts)
                    _write_predictions(directory/f"{partition}-{policy}-predictions.csv", rows, forecasts)
                if partition == "validation":
                    for i,row in enumerate(rows):
                        if row.identity.target_time[:13] in ("2012-01-09T02", "2012-01-09T08"):
                            example = {"target": row.target, "raw": float(predictions[i]), "nonnegative": max(0,float(predictions[i]))}
                            if kind == "tree": example["first_component_path"] = trace_snapshot(snapshot, row.features)
                            result["examples"][row.identity.target_time] = example
        for name,result in results.items():
            snapshot, record, directory = fitted[name]; record.update(result)
            record["artifacts_sha256"] = {p.name: sha256(p.read_bytes()) for p in directory.iterdir() if p.name != "run.json"}
            _write_json(directory/"run.json", record)

        # Seed-averaged scores select a configuration, never an unusually lucky seed.
        order = ["global", "groups", "linear-additive", "linear-interactions"] + [n for n in configurations if n not in references]
        ranking = []
        for name in order:
            trials = configurations[name]
            if name not in references and len(trials) != len(seeds):
                continue
            for policy in ("raw", "nonnegative"):
                scores = [results[n]["metrics"]["validation"][policy]["mae"] for n in trials]
                representative = trials[0]
                ranking.append({"configuration": name, "policy": policy, "mean_validation_mae": float(np.mean(scores)),
                                "seed_mae_population_sd": float(np.std(scores)), "minimum_seed_mae": min(scores), "maximum_seed_mae": max(scores),
                                "trials": trials, "representative_trial": representative,
                                "representative_validation_mae": results[representative]["metrics"]["validation"][policy]["mae"]})
        ranking.sort(key=lambda row: (row["mean_validation_mae"], row["policy"] != "raw", order.index(row["configuration"])))
        selected = ranking[0]
        selection = {"version": "calendar-model-selection-v1", **selected,
                     "criterion": protocol["selection_rule"], "protocol_sha256": report["protocol_sha256"],
                     "model_sha256": sha256((fitted[selected["representative_trial"]][2]/"model.json").read_bytes()),
                     "validation_used_for_selection": True, "test_evaluated": False, "serving_implemented": False,
                     "family_winners": {f: next(row for row in ranking if row["configuration"].startswith(f+"-c"))
                                        for f in candidates if any(row["configuration"].startswith(f+"-c") for row in ranking)},
                     "ranking": ranking}
        _write_json(run_directory/"selection.json", selection)
        report.update(status="completed", results=results, selection=selection, validation_used_for_fitting=False,
                      configurations=configurations, all_models_fitted_before_validation=True)
    except Exception as error:
        report.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        report["finished_utc"] = datetime.now(UTC).isoformat()
        _write_json(run_directory/"study.json", report)
    return report
