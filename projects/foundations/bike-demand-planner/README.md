# Bike Demand Planner

Bike Demand Planner is a Python project for hourly bike rental demand estimation and model comparison. It focuses on system-wide demand and prediction errors across hours, working days and seasons.

Planned functionality includes historical replay, comparisons of calendar baselines, linear regression and tree-based models, and a local application backed by a saved predictor.

**Current status:** the package includes official dataset acquisition, hourly structural checks, fixed calendar encoding and frozen chronological membership. Linear prediction, half squared error, analytic gradients and a single-observation SGD example are also available. Fitted demand estimators, batch training, evaluation and the application remain pending.

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
```

Python is pinned to 3.12.13. The package currently uses the standard library; numerical and modelling dependencies will be added when needed.

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

## Entry points

| File | Purpose | Current behavior |
| --- | --- | --- |
| `check_setup.py` | Check the source checkout and installed package | Runs the setup check |
| `demo.py` | Show project information | Reports the current status |
| `linear_example.py` | Compare prediction, gradient and update directions | Runs two illustrative targets from the same starting parameters |
| `prepare_data.py` | Acquire, validate and partition hourly records | Writes the audit, calendar schema and frozen chronological exports |
| `calendar_example.py` | Inspect one real calendar encoding | Reports active features and membership for a training observation |
| `train.py` | Fit an estimator and save run state | Returns `not_implemented` with exit code 2 |
| `compare.py` | Compare evaluation results | Returns `not_implemented` with exit code 2 |
| `inference.py` | Load a saved estimator and predict | Returns `not_implemented` with exit code 2 |

Inputs are declared in each file's main block. Training, comparison and inference entry points remain unavailable.

## Data and prediction contract

The source is the [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), using `hour.csv` for hourly Capital Bikeshare records from 2011–2012. The dataset is attributed to Hadi Fanaee-T under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with [DOI 10.24432/C5W894](https://doi.org/10.24432/C5W894). The acquired hourly file contains 17,379 rows, spanning 2011-01-01 through 2012-12-31. It has 165 unrecorded slots against a naive hourly grid; no zero counts are inferred for those slots. Source timestamps remain date/hour labels because the archive does not specify timezone or daylight-saving semantics.

The target is total rentals in an hour (`cnt`). The initial predictor will use calendar information available in advance. `casual` and `registered` are excluded because together they reveal the target; row identifiers are kept separately from predictive features. Observed weather will require a separately labelled historical comparison.

The first schema has 45 binary columns: one-hot hour, weekday and month, then holiday and working-day flags. Year and season are omitted. Domains are fixed calendar definitions, with no fitted scaler. Feature order is saved; invalid, missing and extra input fields are rejected.

Training uses 2011 (8,645 rows), validation uses 2012-01-01 01:00 through June (4,357 rows), and the reserved test uses 2012-07-01 01:00 through December (4,375 rows). The two midnight boundary rows are excluded because their one-hour-ahead issue times precede preceding-stage label availability. The protocol assumes interval-start labels, outcomes available at interval end and zero reporting/model-build delay. Source timezone/DST semantics remain unspecified. Reserved-test targets are omitted from the exported test CSV.

Evaluation will use MAE in rentals per hour, supported by RMSE and error slices. Model settings remain pending. Any future learned preprocessing must fit only the relevant training partition. The tool estimates observed system-wide rentals; station inventory and unmet demand require other data.

## Layout

```text
bike-demand-planner/
├── pyproject.toml
├── uv.lock
├── check_setup.py
├── demo.py
├── linear_example.py
├── calendar_example.py
├── prepare_data.py
├── train.py
├── compare.py
├── inference.py
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_splits.py
│   └── test_linear.py
└── src/bike_demand_planner/
    ├── __init__.py
    ├── __main__.py
    ├── data.py
    ├── features.py
    ├── linear.py
    ├── splits.py
    └── project.py
```

Local data, fitted models, checkpoints and generated reports stay out of Git. This project is an ordinary directory in the AI Projects repository.
