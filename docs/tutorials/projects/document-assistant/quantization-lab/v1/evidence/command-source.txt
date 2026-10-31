"""Read-only scaffold commands; no model frameworks or network clients."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from core.config import PROJECT_ID, load_config
from core.workflow import PENDING_ACTIONS, pending_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local Document Assistant Under a Memory Budget — project boilerplate")
    parser.add_argument("action", nargs="?", default="status",
                        choices=("status", "check", "show-config", *PENDING_ACTIONS))
    parser.add_argument("--config", type=Path, help="Use a project configuration TOML file")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.action in PENDING_ACTIONS:
            print(json.dumps(pending_result(args.action), indent=2), file=sys.stderr)
            return 2
        if args.action == "show-config":
            result = config
        else:
            if args.action == "check" and sys.version_info[:2] != (3, 12):
                raise ValueError("Use the project's Python 3.12 environment")
            result = {
                "project": PROJECT_ID,
                "status": "boilerplate",
                "model": config["model"],
                "experiment_defaults": config["experiment"],
                "pending_actions": list(PENDING_ACTIONS),
                "model_execution_verified": False,
            }
            if args.action == "check":
                result["setup_checks"] = {
                    "python": sys.version.split()[0],
                    "configuration": "valid",
                    "source_imports": "ok",
                }
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2), file=sys.stderr)
        return 1
