# Bike Demand Planner

Bike Demand Planner is a Python project for hourly bike rental demand estimation and model comparison. It focuses on system-wide demand and prediction errors across hours, working days and seasons.

Planned functionality includes historical replay, comparisons of calendar baselines, linear regression and tree-based models, and a local application backed by a saved predictor.

**Current status:** the Python package and standalone entry points are set up. Data preparation, estimators, training, evaluation and the application are not implemented yet.

## Setup

From the workspace root:

```bash
cd ai_projects/projects/foundations/bike-demand-planner
uv sync --frozen
uv run --frozen python check_setup.py
uv run --frozen python demo.py
```

Python is pinned to 3.12.13. The package currently uses the standard library; numerical and modelling dependencies will be added when needed.

The setup check reports `status: ok`. The demo prints project information and the availability of each operation. The package can also be run with:

```bash
uv run --frozen python -m bike_demand_planner
```

## Entry points

| File | Purpose | Current behavior |
| --- | --- | --- |
| `check_setup.py` | Check the source checkout and installed package | Runs the setup check |
| `demo.py` | Show project information | Reports the current status |
| `prepare_data.py` | Acquire and validate hourly records | Returns `not_implemented` with exit code 2 |
| `train.py` | Fit an estimator and save run state | Returns `not_implemented` with exit code 2 |
| `compare.py` | Compare evaluation results | Returns `not_implemented` with exit code 2 |
| `inference.py` | Load a saved estimator and predict | Returns `not_implemented` with exit code 2 |

Inputs are declared in each file's main block. The unfinished entry points do not create data, predictions or result files.

## Data and prediction contract

The intended source is the [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), using `hour.csv` for hourly Capital Bikeshare records from 2011–2012. The dataset is attributed to Hadi Fanaee-T under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with [DOI 10.24432/C5W894](https://doi.org/10.24432/C5W894). Data has not been downloaded.

The target is total rentals in an hour (`cnt`). The initial predictor will use calendar information available in advance. `casual` and `registered` are excluded because together they reveal the target; row identifiers are kept separately from predictive features. Observed weather will require a separately labelled historical comparison.

Evaluation will use chronological splits, training-only preprocessing and MAE in rentals per hour, supported by RMSE and error slices. Final split boundaries and model settings have not been fixed. The tool will estimate observed system-wide rentals; station inventory and unmet demand require other data.

## Layout

```text
bike-demand-planner/
├── pyproject.toml
├── uv.lock
├── check_setup.py
├── demo.py
├── prepare_data.py
├── train.py
├── compare.py
├── inference.py
└── src/bike_demand_planner/
    ├── __init__.py
    ├── __main__.py
    └── project.py
```

Local data, fitted models, checkpoints and generated reports stay out of Git. This project is an ordinary directory in the AI Projects repository.
