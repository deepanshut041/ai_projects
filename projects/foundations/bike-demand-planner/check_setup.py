"""Check the local source checkout and installed package."""

import json
import platform
from importlib.metadata import version
from pathlib import Path


def main(project_root: Path) -> int:
    required = (
        "README.md", "pyproject.toml", "uv.lock", ".python-version", ".gitignore",
        "demo.py", "linear_example.py", "calendar_example.py", "baseline_example.py", "group_baseline_example.py", "regression_example.py",
        "optimization_example.py", "objective_example.py", "prepare_data.py", "train.py", "compare.py", "inference.py",
        "src/bike_demand_planner/__init__.py",
        "src/bike_demand_planner/__main__.py",
        "src/bike_demand_planner/project.py",
        "src/bike_demand_planner/linear.py",
        "src/bike_demand_planner/data.py",
        "src/bike_demand_planner/features.py",
        "src/bike_demand_planner/splits.py",
        "src/bike_demand_planner/baselines.py",
        "src/bike_demand_planner/metrics.py",
        "src/bike_demand_planner/evaluation.py",
        "src/bike_demand_planner/regression.py",
        "src/bike_demand_planner/linear_evaluation.py",
        "src/bike_demand_planner/optimizers.py",
        "src/bike_demand_planner/optimization.py",
        "src/bike_demand_planner/optimizer_evaluation.py",
        "src/bike_demand_planner/objectives.py",
        "src/bike_demand_planner/objective_evaluation.py",
    )
    missing = [name for name in required if not (project_root / name).is_file()]
    nested_repository = (project_root / ".git").exists()
    passed = not missing and not nested_repository
    print(json.dumps({
        "status": "ok" if passed else "incomplete",
        "python": platform.python_version(),
        "package_version": version("bike-demand-planner"),
        "missing_files": missing,
        "nested_repository": nested_repository,
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    raise SystemExit(main(project_root=project_root))
