"""Display the current project status."""

import json

from bike_demand_planner.project import project_info


if __name__ == "__main__":
    print(json.dumps(project_info(), indent=2))
