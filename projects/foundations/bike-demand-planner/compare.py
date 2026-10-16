"""Entry point for comparing saved evaluation results."""

from pathlib import Path

from bike_demand_planner.project import unavailable


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    run_directory = project_root / "runs"
    report_directory = project_root / "artifacts" / "comparison"
    raise SystemExit(unavailable(
        "compare", run_directory=run_directory, report_directory=report_directory,
    ))
