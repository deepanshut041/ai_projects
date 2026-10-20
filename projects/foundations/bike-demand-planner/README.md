# Bike Demand Planner

Bike Demand Planner is a Python project for hourly bike rental demand estimation and model comparison. It focuses on system-wide demand and prediction errors across hours, working days and seasons.

Planned functionality includes historical replay, comparisons of calendar baselines, linear regression and tree-based models, and a local application backed by a saved predictor.

**Current status:** the package includes official dataset acquisition, hourly checks, frozen chronological membership, calendar baselines, full-batch linear regression, controlled optimizer/objective/calendar-structure studies and bounded decision-tree, random-forest and gradient-boosting comparisons. Fits use training only, with aligned train/validation results and retained numerical model state. Saved-model inference and the application remain pending.

## Setup

From the workspace root:

```bash
cd ai_projects/projects/foundations/bike-demand-planner
uv sync --frozen
uv run --frozen python check_setup.py
uv run --frozen python demo.py
uv run --frozen python linear_example.py
uv run --frozen python prepare_data.py
uv run --frozen python calendar_example.py
uv run --frozen python baseline_example.py
uv run --frozen python group_baseline_example.py
uv run --frozen python regression_example.py
uv run --frozen python train.py
uv run --frozen python optimization_example.py
uv run --frozen python objective_example.py
uv run --frozen python tree_example.py
uv run --frozen python compare.py
```

Python is pinned to 3.12.13, NumPy to 2.5.4 and scikit-learn to 1.9.1. Dependencies are recorded in `uv.lock`.

The setup check reports `status: ok`. The demo prints project information and the availability of each operation. The package can also be run with:

```bash
uv run --frozen python -m bike_demand_planner
```

The numerical example uses illustrative values and compares targets of 45 and 35 from the same starting parameters. The initial prediction is 40.00 in both cases; one update moves it to 40.30 for target 45 and to 39.70 for target 35. Half squared error decreases from 12.5 to approximately 11.045 in each case. The example does not load bike-sharing data or measure forecast quality.

Run the checks with:

```bash
uv run --frozen python -m unittest discover -s tests -v
```

The data command downloads and caches the official archive, retains SHA-256 identities and source attribution, and writes `data/audit.json`. It checks schema, categories/ranges, calendar/count alignment, uniqueness, ordering and unrecorded hourly slots. It then writes the calendar schema, membership manifest and partition CSVs under `data/processed/`. Repeated runs verify the retained bytes; changed frozen artifacts are rejected. All downloaded files and generated reports remain local.

The calendar example traces record 54 at `2011-01-03 08:00` into its 45 ordered binary features and training membership, keeping its observed target of 154 separate.

The baseline command fits the mean of training targets and repeats that value without using future outcomes. It saves `model.json`, row-aligned training/validation predictions and `run.json` in a new directory under `runs/`. The run record retains source/dependency/data identities, the fit policy, metric definitions, timings and completion/failure status. It does not read the test partition. Predictions remain real-valued; they are not rounded or clipped.

The group-baseline command compares that constant with a separate training mean for each `(hour, workingday)` pair. A valid pair absent from training falls back to the global training mean. Both predictors use the same frozen validation hours and metric definitions, with group counts and fallback usage saved alongside the run records. On the current 4,357-hour validation window, MAE is 154.2452 for the constant and 91.5729 for the group predictor; this is a development comparison rather than final-test performance.

The regression example traces a small matrix calculation and two updates for a real training hour, then compares all three models on the same frozen membership. Linear regression fits the 45 calendar features with an intercept, using float64 full-batch gradients on mean half squared error. Column means and population standard deviations are fitted on training only; constant columns use scale one. Targets remain in rentals/hour. Weights start at zero and the intercept starts at the training target mean. The fixed learning rate is 0.1, with a 10,000-update limit and a training-gradient norm tolerance of 1e-7. Validation does not choose preprocessing or the stopping point.

The current linear fit converges in 200 updates and agrees with the stable least-squares reference within 1e-6 rentals/hour on the observed train/validation inputs. Validation MAE is 116.7312 and RMSE is 157.9791, worse than the hour/working-day baseline. The additive calendar model does not learn separate working-day and nonworking-day hourly curves. Its 724 negative validation forecasts are retained without clipping, so the reported metrics describe the actual unconstrained model. The run saves standardized and equivalent raw parameters, feature order, convergence history, prediction ranges and reference agreement alongside the common evaluation artifacts. `train.py` fits this same bounded linear model without rerunning the baselines.

The optimization example first repeats two updates on one real training hour, then runs separate controlled comparisons. A training-only pilot gives each optimizer the same four rates and 200-update allowance at batch 32. The selected rates are 0.01 for SGD, 0.001 for classical momentum and 1.0 for Adam. Main runs use paired seeds 17/29/43 and ten training passes (86,450 processed examples), with safety limits of 100,000 updates or five minutes. Batch-size comparisons hold SGD and rate 0.01 fixed; the schedule comparison changes only constant versus inverse-time decay by processed examples. Exact matching configurations reuse their trial rather than fitting twice.

Across the three seeds at batch 32, validation MAE averages 116.1661 for SGD, 116.6027 for momentum and 114.3608 for Adam. Adam has greater seed variation and remains worse than the group baseline. Decay brings training squared error closer to the least-squares optimum without improving mean validation MAE in this comparison. Failed excessive-rate pilots remain recorded. Each main run retains aligned predictions, training history, optimizer moments/index, rate policy, epoch permutation/cursor, RNG state and source/config/data identity. Local continuation checks reproduce a paused numerical run exactly; application recovery in a fresh process remains separate work.

The objective example compares mean half squared error, absolute error and Huber losses with training-only rate pilots, then runs separate L2, scaling and calendar-structure studies. L2 excludes the intercept and uses the explicit objective `mean(data_loss) + lambda * sum(weights**2) / 2`. Penalty and representation comparisons use stable training-only least-squares solves; the scaling comparison fixes SGD rate and work. Calendar variants retain all 45 original columns, replace cyclic categories with eight sine/cosine/flag columns, or append 24 hour/working-day products.

Raw predictions and a declared nonnegative policy are scored separately. On the frozen validation window, the exact additive model has MAE 116.7312 raw and 109.1436 clipped; the interaction model improves those to 108.6937 and 100.5164. The group baseline remains the selected simple reference at 91.5729, with raw output retained on an exact policy tie. Each study saves its protocol, candidate results and validation-led selection identity; the final test and serving API remain separate. These deterministic full-batch studies add no dependencies.

## Entry points

| File | Purpose | Current behavior |
| --- | --- | --- |
| `check_setup.py` | Check the source checkout and installed package | Runs the setup check |
| `demo.py` | Show project information | Reports the current status |
| `linear_example.py` | Compare prediction, gradient and update directions | Runs two illustrative targets from the same starting parameters |
| `prepare_data.py` | Acquire, validate and partition hourly records | Writes the audit, calendar schema and frozen chronological exports |
| `calendar_example.py` | Inspect one real calendar encoding | Reports active features and membership for a training observation |
| `baseline_example.py` | Fit and evaluate a constant demand predictor | Saves the training mean, aligned predictions and error summaries |
| `group_baseline_example.py` | Compare global and calendar-group predictors | Saves matched train/validation results, group counts and fallback usage |
| `regression_example.py` | Trace full-batch updates and compare three predictors | Saves checked updates, linear fitting history and matched results |
| `train.py` | Fit calendar linear regression | Saves fitted parameters, history, aligned predictions and run state |
| `optimization_example.py` | Compare batching, optimizer rules and one rate schedule | Saves a training-only pilot, frozen comparison protocol, paired trials and continuation state |
| `objective_example.py` | Compare loss, L2, scaling, calendar structure and output policy | Saves training-only pilots, separate fitted comparisons and a validation-led reference selection |
| `tree_example.py` | Trace a small tree and compare practical calendar estimators | Saves bounded candidates, numerical snapshots, seed-aggregated validation results and selection identity |
| `compare.py` | Summarize a saved practical-model study | Verifies protocol/artifact identities and writes the latest compatible comparison summary |
| `inference.py` | Load a saved estimator and predict | Returns `not_implemented` with exit code 2 |

Inputs are declared in each file's main block. The comparison command requires a completed `tree_example.py` run; it does not combine incompatible older study formats. Saved-model inference remains unavailable.

The tree comparison retains the same 45 raw calendar columns and identity rental target. Each of three model families receives three configurations and seeds 17, 29 and 43. Configuration selection uses mean validation MAE across the declared seeds, with the first seed retained as the representative artifact. It never selects the luckiest seed. Simple deterministic references receive one fit each. Raw and nonnegative output policies are reported separately; final-test evaluation remains reserved.

Each tree is bounded to 1,024 leaves. Forests contain 64 bootstrap trees with 70% of features considered at a split and one fitting job. Boosting uses depth-three squared-error residual trees, learning rate 0.1 and 100/200/300 stages, without validation-based early stopping. Training-only timing pilots check feasibility. Post-fit time guards stop later work when exceeded; they are not hard interrupts. Fitted numerical snapshots preserve tree tests, leaf values, support and aggregation rules as JSON. Reported prediction timings measure this transparent replay, not an optimized serving deployment.

## Data and prediction contract

The source is the [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), using `hour.csv` for hourly Capital Bikeshare records from 2011–2012. The dataset is attributed to Hadi Fanaee-T under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with [DOI 10.24432/C5W894](https://doi.org/10.24432/C5W894). The acquired hourly file contains 17,379 rows, spanning 2011-01-01 through 2012-12-31. It has 165 unrecorded slots against a naive hourly grid; no zero counts are inferred for those slots. Source timestamps remain date/hour labels because the archive does not specify timezone or daylight-saving semantics.

The target is total rentals in an hour (`cnt`). The initial predictor will use calendar information available in advance. `casual` and `registered` are excluded because together they reveal the target; row identifiers are kept separately from predictive features. Observed weather will require a separately labelled historical comparison.

The first schema has 45 binary columns: one-hot hour, weekday and month, then holiday and working-day flags. Year and season are omitted. Domains are fixed calendar definitions, with no fitted scaler in the data exports. Linear regression applies a separate training-fitted scaler without changing those exports. Feature order is saved; invalid, missing and extra input fields are rejected. Full one-hot groups and flags create dependent columns with an intercept, so individual coefficients need not be unique.

Training uses 2011 (8,645 rows), validation uses 2012-01-01 01:00 through June (4,357 rows), and the reserved test uses 2012-07-01 01:00 through December (4,375 rows). The two midnight boundary rows are excluded because their one-hour-ahead issue times precede preceding-stage label availability. The protocol assumes interval-start labels, outcomes available at interval end and zero reporting/model-build delay. Source timezone/DST semantics remain unspecified. Reserved-test targets are omitted from the exported test CSV.

Evaluation uses MAE in rentals per hour as its primary summary, alongside RMSE and mean signed error (`prediction - target`). Mean underprediction is the sum of hourly deficits divided by all evaluated rows; the underprediction count and its conditional mean are reported separately. An identity/order check precedes scoring. Any future learned preprocessing must fit only the relevant training partition. The tool estimates observed system-wide rentals; station inventory and unmet demand require other data.

## Layout

```text
bike-demand-planner/
├── pyproject.toml
├── uv.lock
├── check_setup.py
├── demo.py
├── linear_example.py
├── calendar_example.py
├── baseline_example.py
├── group_baseline_example.py
├── regression_example.py
├── optimization_example.py
├── objective_example.py
├── tree_example.py
├── prepare_data.py
├── train.py
├── compare.py
├── inference.py
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_splits.py
│   ├── test_baselines.py
│   ├── test_evaluation.py
│   ├── test_regression.py
│   ├── test_linear_evaluation.py
│   ├── test_optimization.py
│   ├── test_optimizer_evaluation.py
│   ├── test_objectives.py
│   ├── test_objective_evaluation.py
│   ├── test_trees.py
│   ├── test_tree_evaluation.py
│   └── test_linear.py
└── src/bike_demand_planner/
    ├── __init__.py
    ├── __main__.py
    ├── data.py
    ├── baselines.py
    ├── evaluation.py
    ├── features.py
    ├── linear.py
    ├── regression.py
    ├── linear_evaluation.py
    ├── optimizers.py
    ├── optimization.py
    ├── optimizer_evaluation.py
    ├── objectives.py
    ├── objective_evaluation.py
    ├── trees.py
    ├── tree_evaluation.py
    ├── comparison.py
    ├── metrics.py
    ├── splits.py
    └── project.py
```

Local data, fitted models, checkpoints and generated reports stay out of Git. This project is an ordinary directory in the AI Projects repository.
