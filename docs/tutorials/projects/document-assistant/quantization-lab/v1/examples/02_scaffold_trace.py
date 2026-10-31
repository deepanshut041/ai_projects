"""Check the real project entry point from an explicitly supplied root."""
import json
from pathlib import Path
import subprocess
import sys


def inspect(root):
    python = root / ".venv/bin/python"
    script = root / "src/document_assistant.py"
    results = {}
    for action, code in [("status", 0), ("check", 0), ("show-config", 0), ("answer", 2)]:
        run = subprocess.run([str(python), str(script), action], cwd="/tmp",
                             text=True, capture_output=True, check=False)
        assert run.returncode == code, run.stderr
        payload = json.loads(run.stdout if code == 0 else run.stderr)
        results[action] = {"exit": code, "project": payload["project_id"] if action == "show-config"
                           else payload["project"]}
        if action == "answer":
            assert payload["status"] == "not_implemented"
        elif action != "show-config":
            assert payload["model_execution_verified"] is False
    return results


if __name__ == "__main__":
    if len(sys.argv) == 2:
        print(json.dumps(inspect(Path(sys.argv[1]).resolve())))
    else:
        # Portable course check: inspect syntax/contract without a live checkout.
        print(json.dumps({"scope": "invoke with project root for live scaffold check",
                          "expected_exit_codes": [0, 0, 0, 2]}))
