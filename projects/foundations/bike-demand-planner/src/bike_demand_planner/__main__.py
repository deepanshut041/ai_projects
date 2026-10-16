"""Display project information."""

import json

from .project import project_info


if __name__ == "__main__":
    print(json.dumps(project_info(), indent=2))
