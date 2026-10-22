# Run the project and find the source

The course explains the completed Bike Demand Planner implementation at source commit `0b49c0aafe154888929d8b22dbe6637ba2e56661`. The source project lives under `ai_projects/projects/foundations/bike-demand-planner/` in the Projects repository. It is an ordinary project directory, without a nested repository.

## Start after the first calculation

Lesson 1 deliberately starts with the forecast rather than environment setup. After following the 40-rental example, run that same calculation from the project directory:

```bash
cd ai_projects/projects/foundations/bike-demand-planner
uv sync --frozen
uv run --frozen python check_setup.py
uv run --frozen python linear_example.py
```

Run the first `cd` from workspace home. The sync command installs the pinned environment; the setup check verifies it. The numerical example creates no bike-data result. It prints the two opposite-target updates described in the first lesson. Python is pinned to 3.12.13. The checked dependencies are NumPy 2.5.4, scikit-learn 1.9.1 and Matplotlib 3.11.2. The real checkout’s `uv.lock` owns dependency installation.

## Reproduce the data and saved results

The [source README snapshot](evidence/project-readme.txt) contains the complete preparation order and individual command contracts. The finite preparation commands download the official source, freeze membership, fit the specified models, retain comparisons and prepare the application. Generated data, weights and runs remain ignored. Do not launch a server in the middle of that finite sequence. The final assessment is a reproduction of the disclosed historical experiment; it is not another independent test.

For a prepared checkout, start the final application from that same project directory:

```bash
uv run --frozen python serve_final.py
```

Open [the final application](http://127.0.0.1:8504/). It stays running until Ctrl+C, checks saved assets at startup and does not fit models. The validation-only page uses `serve_app.py` at port 8503. Its interactive replay remains January–June 2012 even though the final batch covers July–December.

Run the source checks separately:

```bash
uv run --frozen python -m unittest discover -s tests -v
```

## Downloadable source snapshots

These files retain their package layout under `code/`; they are source references from the completed project, not a second training implementation. The actual checkout and its lock file are the runnable project.

- [application_example.py](code/application_example.py)
- [baseline_example.py](code/baseline_example.py)
- [calendar_example.py](code/calendar_example.py)
- [check_setup.py](code/check_setup.py)
- [compare.py](code/compare.py)
- [demo.py](code/demo.py)
- [error_review_example.py](code/error_review_example.py)
- [final_evaluation_example.py](code/final_evaluation_example.py)
- [group_baseline_example.py](code/group_baseline_example.py)
- [inference.py](code/inference.py)
- [linear_example.py](code/linear_example.py)
- [objective_example.py](code/objective_example.py)
- [optimization_example.py](code/optimization_example.py)
- [prepare_data.py](code/prepare_data.py)
- [regression_example.py](code/regression_example.py)
- [serve_app.py](code/serve_app.py)
- [serve_final.py](code/serve_final.py)
- [__init__.py](code/src/bike_demand_planner/__init__.py)
- [__main__.py](code/src/bike_demand_planner/__main__.py)
- [application.py](code/src/bike_demand_planner/application.py)
- [application_data.py](code/src/bike_demand_planner/application_data.py)
- [application_run.py](code/src/bike_demand_planner/application_run.py)
- [artifacts.py](code/src/bike_demand_planner/artifacts.py)
- [baselines.py](code/src/bike_demand_planner/baselines.py)
- [checkpoints.py](code/src/bike_demand_planner/checkpoints.py)
- [comparison.py](code/src/bike_demand_planner/comparison.py)
- [data.py](code/src/bike_demand_planner/data.py)
- [error_analysis.py](code/src/bike_demand_planner/error_analysis.py)
- [error_plots.py](code/src/bike_demand_planner/error_plots.py)
- [error_review.py](code/src/bike_demand_planner/error_review.py)
- [evaluation.py](code/src/bike_demand_planner/evaluation.py)
- [features.py](code/src/bike_demand_planner/features.py)
- [final_application.py](code/src/bike_demand_planner/final_application.py)
- [final_evaluation.py](code/src/bike_demand_planner/final_evaluation.py)
- [linear.py](code/src/bike_demand_planner/linear.py)
- [linear_evaluation.py](code/src/bike_demand_planner/linear_evaluation.py)
- [metrics.py](code/src/bike_demand_planner/metrics.py)
- [objective_evaluation.py](code/src/bike_demand_planner/objective_evaluation.py)
- [objectives.py](code/src/bike_demand_planner/objectives.py)
- [optimization.py](code/src/bike_demand_planner/optimization.py)
- [optimizer_evaluation.py](code/src/bike_demand_planner/optimizer_evaluation.py)
- [optimizers.py](code/src/bike_demand_planner/optimizers.py)
- [prediction_worker.py](code/src/bike_demand_planner/prediction_worker.py)
- [project.py](code/src/bike_demand_planner/project.py)
- [recovery.py](code/src/bike_demand_planner/recovery.py)
- [regression.py](code/src/bike_demand_planner/regression.py)
- [review_contract.py](code/src/bike_demand_planner/review_contract.py)
- [saved_predictor.py](code/src/bike_demand_planner/saved_predictor.py)
- [splits.py](code/src/bike_demand_planner/splits.py)
- [tree_evaluation.py](code/src/bike_demand_planner/tree_evaluation.py)
- [trees.py](code/src/bike_demand_planner/trees.py)
- [train.py](code/train.py)
- [tree_example.py](code/tree_example.py)

## Why the original scaffold was small


Updated: 2026-10-15 (America/Toronto). Setup is implemented and checked; the first applied learning phase has not started.

### Why the skeleton stops here

The project is prepared to estimate hourly rentals later. Today it contains a package, a locked Python environment and small standalone entry points. This creates a reproducible place to apply prior tutorial/PDF knowledge without silently implementing all future phases.

`demo.py` calls project metadata. It reports that preparation, training, comparison and inference are not implemented. The four unfinished operation files declare their inputs in their main blocks and report `not_implemented` with exit 2; they create no data or result files. A running command therefore proves infrastructure behavior, not that an estimator exists.

### What the checks actually establish

Frozen setup with Python 3.12.13 imports the package. Direct and module runs from an unrelated directory show paths are resolved from the source rather than a developer's current directory. A fresh extracted source and an isolated wheel work without personal docs. Source and wheel inspection establish that private material is excluded. None of those observations measures forecast quality, training or learner understanding.

The environment currently has no runtime numerical/model dependencies. Those will be pinned when a concrete phase needs them. The detailed model/data/numerical requirements remain in the [design reference](evidence/project-readme.txt) and the [phase route](project/course-outline.md), rather than being treated as implemented code.


