"""Entry point for hourly dataset preparation."""

from pathlib import Path

from bike_demand_planner.project import unavailable


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    source_url = "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"
    data_directory = project_root / "data"
    raise SystemExit(unavailable(
        "prepare_data", source_url=source_url, data_directory=data_directory,
    ))
