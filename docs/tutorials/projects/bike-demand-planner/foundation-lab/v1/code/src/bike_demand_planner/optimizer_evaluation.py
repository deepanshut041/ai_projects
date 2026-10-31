"""A training-only pilot and separate optimizer, batch and schedule comparisons."""

import csv
import os
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from .data import sha256
from .evaluation import METRIC_DEFINITIONS, Forecast, _source_identity, _write_json, _write_predictions, load_labelled_partition, score_forecasts
from .features import FEATURE_NAMES, canonical_json
from .optimization import TrialConfig, fit_trial
from .optimizers import OptimizerConfig, optimizer_definition
from .regression import LinearRegressionModel, fit_standardizer


def _hardware() -> dict:
    model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        model = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines() if line.startswith("model name")), model)
    return {"platform": platform.platform(), "machine": platform.machine(), "processor": model,
            "logical_cpus": os.cpu_count(), "python": platform.python_version(), "numpy": np.__version__,
            "thread_environment": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}


def _save_trial(directory, trial, config, scaler, common, stage, protocol_sha256):
    directory.mkdir(parents=True, exist_ok=False)
    model = LinearRegressionModel(tuple(trial.parameters[:-1].tolist()), float(trial.parameters[-1]), scaler)
    raw_weights, raw_bias = model.raw_parameters()
    model_record = {"kind": "calendar_linear_regression", **asdict(model), "raw_weights": raw_weights, "raw_bias": raw_bias,
                    "feature_names": FEATURE_NAMES, "fit_partition": "train", "fit_objective": "mean half squared error"}
    _write_json(directory / "model.json", model_record)
    _write_json(directory / "state.json", {**trial.state, "source_sha256": common["source"]["content_sha256"],
                                          "dataset": common["dataset"], "standardizer": asdict(scaler), "feature_names": FEATURE_NAMES})
    with (directory / "optimization.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(asdict(trial.history[0])), lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(point) for point in trial.history)
    record = {
        **common, "status": "failed" if trial.stop_reason in ("nonfinite", "diverged") else "completed",
        "stage": stage, "model": model_record, "configuration": asdict(config), "optimizer_definition": optimizer_definition(config.optimizer),
        "protocol_sha256": protocol_sha256, "metrics": {}, "timings_seconds": {"fit": trial.elapsed_seconds},
        "optimization": {"stop_reason": trial.stop_reason, "updates": trial.state["optimizer"]["step"],
                         "examples_processed": trial.state["batch_stream"]["examples_seen"],
                         "last_batch_size": trial.last_batch_size, "first_batches": trial.first_batches,
                         "initial": asdict(trial.history[0]), "last_observed": asdict(trial.history[-1]),
                         "convergence_criterion": "none; compare at the declared work budget"},
    }
    _write_json(directory / "run.json", record)
    return record, model


def _summarize(entries, records):
    groups = {}
    for entry in entries:
        groups.setdefault(entry["label"], []).append(records[entry["trial"]])
    summaries = {}
    for label, runs in groups.items():
        complete = [run for run in runs if run["status"] == "completed" and "validation" in run["metrics"]]
        def summary(values):
            return {"mean": mean(values), "sample_std": stdev(values) if len(values) > 1 else None,
                    "minimum": min(values), "maximum": max(values)}
        summaries[label] = {"trials": len(runs), "scored_trials": len(complete),
                            "seeds": [run["configuration"]["seed"] for run in runs],
                            "stop_reasons": [run["optimization"]["stop_reason"] for run in runs]}
        if complete:
            summaries[label].update({key: summary([run["metrics"]["validation"][key] for run in complete]) for key in ("mae", "rmse", "mean_signed_error")})
            summaries[label].update({key: summary([run["optimization"][key] for run in complete]) for key in ("updates", "examples_processed")})
            summaries[label]["fit_seconds"] = summary([run["timings_seconds"]["fit"] for run in complete])
            summaries[label]["training_half_mse"] = summary([run["optimization"]["last_observed"]["half_mse"] for run in complete])
    return summaries


def run_optimizer_study(processed_directory: Path, study_directory: Path, project_root: Path, *,
                        rates=(.001, .01, .1, 1.0), seeds=(17, 29, 43), passes=10, pilot_updates=200,
                        max_updates=100_000, max_seconds=300, batch_rate=.01) -> dict:
    """Freeze a training-only pilot protocol before fitting the paired main trials."""
    if not rates or not seeds or type(passes) is not int or passes < 1 or type(pilot_updates) is not int or pilot_updates < 1:
        raise ValueError("Use nonempty rate/seed sets and positive pass/pilot budgets.")
    for rate in (*rates, batch_rate):
        TrialConfig(learning_rate=rate, max_updates=max_updates, max_seconds=max_seconds)
    for seed in seeds:
        TrialConfig(seed=seed)
    study_directory.mkdir(parents=True, exist_ok=False)
    study = {"version": "linear-optimizer-study-v1", "status": "running", "started_utc": datetime.now(UTC).isoformat(),
             "test_evaluated": False}
    try:
        train, manifest = load_labelled_partition(processed_directory, "train")
        X = np.asarray([row.features for row in train], dtype=np.float64)
        y = np.asarray([row.target for row in train], dtype=np.float64)
        scaler = fit_standardizer(X)
        Z = scaler.transform(X)
        common = {
            "version": "hourly-regression-run-v1", "metric_definitions": METRIC_DEFINITIONS,
            "prediction_policy": "real-valued rentals/hour; no rounding or clipping", "test_evaluated": False,
            "source": _source_identity(project_root), "environment": _hardware(),
            "dataset": {key: manifest[key] for key in ("source_hour_sha256", "feature_schema_sha256", "membership_sha256", "manifest_sha256", "counts")},
            "task": {"target": "observed system-wide rentals in one hour", "units": "rentals/hour", "feature_names": FEATURE_NAMES,
                     **{key: manifest[key] for key in ("forecast_lead_hours", "target_duration_hours", "availability_policy", "readiness_assumption", "timestamp_policy")},
                     "transform_fit": "training-only population feature statistics; constant scale one", "target_scaling": "none"},
        }
        common["dataset"]["partition_sha256"] = {name: manifest["artifacts_sha256"][f"{name}.csv"] for name in ("train", "validation")}
        common["environment"]["dependencies_sha256"] = {name: sha256((project_root / name).read_bytes()) for name in ("pyproject.toml", "uv.lock")}
        pilot = []
        selected = {}
        for kind in ("sgd", "momentum", "adam"):
            candidates = []
            for index, rate in enumerate(rates):
                config = TrialConfig(OptimizerConfig(kind), rate, 32, seeds[0], max_examples=pilot_updates * 32,
                                     max_updates=pilot_updates, max_seconds=max_seconds)
                result = fit_trial(Z, y, config)
                directory = study_directory / "pilot" / f"{kind}-{index:02}"
                receipt, _ = _save_trial(directory, result, config, scaler, common, "training_only_pilot", None)
                row = {"optimizer": kind, "rate": rate, "status": receipt["status"],
                       "stop_reason": result.stop_reason, "updates": result.state["optimizer"]["step"],
                       "examples_processed": result.state["batch_stream"]["examples_seen"],
                       "training_half_mse": result.history[-1].half_mse, "fit_seconds": result.elapsed_seconds,
                       "directory": str(directory.relative_to(study_directory))}
                pilot.append(row)
                if receipt["status"] == "completed" and result.state["optimizer"]["step"] == pilot_updates:
                    candidates.append(row)
            if not candidates:
                raise ValueError(f"No {kind} pilot completed its declared update budget.")
            selected[kind] = min(candidates, key=lambda row: (row["training_half_mse"], row["rate"]))["rate"]
        protocol = {
            "version": "linear-optimizer-protocol-v1", "frozen_utc": datetime.now(UTC).isoformat(),
            "pilot_seed": seeds[0], "pilot_updates_per_candidate": pilot_updates, "rate_candidates_per_optimizer": list(rates),
            "rate_selection": "lowest final full-training half-MSE among candidates completing the same pilot updates; tie uses smaller rate",
            "validation_used_for_selection": False, "selected_rates": selected, "seeds": list(seeds),
            "training_count": len(train), "main_examples_per_trial": passes * len(train), "main_passes": passes,
            "max_updates": max_updates, "max_seconds_per_trial": max_seconds,
            "optimizer_family": "SGD/momentum/Adam at batch 32 with equal four-rate pilot allowances",
            "batch_family": {"optimizer": "sgd", "rate": batch_rate, "sizes": [1, 32, len(train)]},
            "schedule_family": {"optimizer": "sgd", "batch_size": 32, "rate": selected["sgd"],
                                "decay_examples": 5 * len(train), "formula": "base_rate / (1 + examples_seen / decay_examples); evaluated before the update"},
            "initialization": "all weights zero; intercept training target mean; optimizer buffers zero per trial",
            "batch_order": "NumPy PCG64, fresh without-replacement permutation per epoch; tail retained and averaged by its actual size",
            "convergence": "no gradient stop; retain example/update/time budget status and failures",
            "proposal_revision": "A 10000-update cap would truncate the size-one ten-pass trial. Use a 100000-update safety cap and equal examples instead. The four-rate grid is 0.001/0.01/0.1/1 to include useful Adam movement and an observable excessive-rate case; no cross-product sweep.",
        }
        protocol_sha = sha256(canonical_json(protocol))
        _write_json(study_directory / "pilot.json", {"trials": pilot})
        _write_json(study_directory / "protocol.json", {**protocol, "protocol_sha256": protocol_sha})

        families = {"optimizer": [], "batch": [], "schedule": []}
        trials, models, configurations = {}, {}, {}

        def add(family, label, config):
            identity = sha256(canonical_json(asdict(config)))[:16]
            families[family].append({"label": label, "seed": config.seed, "trial": identity})
            if identity not in trials:
                result = fit_trial(Z, y, config)
                trials[identity], models[identity] = _save_trial(study_directory / "trials" / identity, result, config, scaler,
                                                               common, "main_comparison", protocol_sha)
                configurations[identity] = asdict(config)

        for seed in seeds:
            def settings(kind, rate, size, decay=None):
                return TrialConfig(OptimizerConfig(kind), rate, size, seed, passes * len(train), max_updates, max_seconds, decay)
            for kind in ("sgd", "momentum", "adam"):
                add("optimizer", kind, settings(kind, selected[kind], 32))
            for size in (1, 32, len(train)):
                add("batch", "full" if size == len(train) else str(size), settings("sgd", batch_rate, size))
            add("schedule", "constant", settings("sgd", selected["sgd"], 32))
            add("schedule", "inverse_time", settings("sgd", selected["sgd"], 32, 5 * len(train)))

        # Neither rate selection nor fitting above can inspect validation outcomes.
        validation, validation_manifest = load_labelled_partition(processed_directory, "validation")
        if validation_manifest != manifest:
            raise ValueError("The frozen dataset changed during the study.")
        example_times = {"2012-01-09T02:00:00", "2012-01-09T08:00:00"}
        for identity, receipt in trials.items():
            directory = study_directory / "trials" / identity
            if receipt["status"] == "completed":
                receipt["prediction_ranges"], receipt["examples"] = {}, []
                for name, rows in (("train", train), ("validation", validation)):
                    values = models[identity].predict([row.features for row in rows])
                    forecasts = [Forecast(row.identity, float(value)) for row, value in zip(rows, values, strict=True)]
                    receipt["metrics"][name] = score_forecasts(rows, forecasts)
                    receipt["prediction_ranges"][name] = {"minimum": float(values.min()), "maximum": float(values.max()),
                                                          "negative_count": int(np.count_nonzero(values < 0))}
                    _write_predictions(directory / f"{name}-predictions.csv", rows, forecasts)
                    if name == "validation":
                        receipt["examples"] = [{**asdict(row.identity), "target": row.target, "prediction": forecast.prediction}
                                               for row, forecast in zip(rows, forecasts, strict=True) if row.identity.target_time in example_times]
            receipt["artifacts_sha256"] = {path.name: sha256(path.read_bytes()) for path in sorted(directory.iterdir()) if path.name != "run.json"}
            _write_json(directory / "run.json", receipt)
        study.update(status="completed", protocol_sha256=protocol_sha, selected_rates=selected,
                     pilot_trials=pilot, unique_main_trials=len(trials), families=families,
                     trial_directories={identity: "trials/" + identity for identity in trials},
                     summaries={family: _summarize(entries, trials) for family, entries in families.items()})
    except Exception as error:
        study.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        study["finished_utc"] = datetime.now(UTC).isoformat()
        _write_json(study_directory / "study.json", study)
    return study
