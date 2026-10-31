## 2. Know exactly what runs today

We have defined what an acceptable answer means. Before building one, inspect what the current project actually executes. A launcher that prints a model name can be useful setup tooling while providing no evidence that a model can load. We need that distinction to interpret every later command.

The owning project folder is `/mnt/projects/ai/ai_projects/projects/quantization-lab/document-assistant`. It is an ordinary directory in `ai_projects`, with its own Python environment and lockfile; it is not a nested Git repository. The shared controller lives independently in `/mnt/projects/ai/cli`. From workspace home use `./ai`; from Projects use `../ai`.

### Follow one action through its complete dependency chain

The direct script `src/document_assistant.py` calls `utils.commands.main()`. The command module imports `load_config()` and the explicitly pending workflow. To avoid a hidden-helper jump, the following captured listings show the definitions in dependency order: configuration, pending-result construction, command dispatch and the thin entry point. The listings are inspected working-source bytes, not a proposed full assistant.

**Existing `src/core/config.py`:**

```python
"""Load pinned configuration without loading a model or creating artifacts."""
from __future__ import annotations

from importlib import resources
from pathlib import Path
import re
import tomllib
from typing import Any


PROJECT_ID = "document-assistant"
POSITIVE_FIELDS = ("max_input_tokens","max_output_tokens",)


def load_config(path: Path | None = None) -> dict[str, Any]:
    text = (path.read_text(encoding="utf-8") if path is not None
            else resources.files("core").joinpath("project.toml").read_text(encoding="utf-8"))
    config = tomllib.loads(text)
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("Configuration requires schema_version = 1")
    if config.get("project_id") != PROJECT_ID:
        raise ValueError(f"Configuration must belong to {PROJECT_ID}")
    for section in ("model", "fallback_model"):
        if section == "fallback_model" and section not in config:
            continue
        model = config.get(section)
        if not isinstance(model, dict) or not isinstance(model.get("repo_id"), str) or not model["repo_id"].strip():
            raise ValueError(f"{section} requires a nonempty repo_id")
        revision = model.get("revision")
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError(f"{section} requires an exact 40-character model revision")
    if config["model"].get("compute_dtype") not in ("float32", "bfloat16"):
        raise ValueError("The source model uses float32 or bfloat16 inference")
    experiment = config.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("Configuration requires an experiment table")
    seed = experiment.get("seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("Experiment seed must be a nonnegative integer")
    for field in POSITIVE_FIELDS:
        value = experiment.get(field)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    return config
```

**Existing `src/core/workflow.py`:**

```python
"""Declare future entry points without simulating their completion."""

PENDING_ACTIONS = ("prepare-data","quantize","evaluate","answer",)


def pending_result(action: str) -> dict:
    if action not in PENDING_ACTIONS:
        raise ValueError(f"Unknown pending action: {action}")
    return {
        "project": "document-assistant",
        "action": action,
        "status": "not_implemented",
        "message": "Document ingestion, quantization, evaluation and answering remain pending. No model or output was created.",
    }
```

**Existing `src/utils/commands.py`:**

```python
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
```

**Existing `src/document_assistant.py`:**

```python
"""Run Local Document Assistant Under a Memory Budget scaffold commands."""
from utils.commands import main

if __name__ == "__main__":
    raise SystemExit(main())
```

`load_config()` reads packaged `core/project.toml` through `importlib.resources` when no explicit path is supplied. This is why an installed wheel can find the same defaults without relying on the current directory. TOML parsing produces ordinary Python values; the loader validates schema, owning project, exact model revision, source compute dtype, seed and token budgets. `type(value) is int` deliberately rejects a Boolean where an integer budget is required, since Python otherwise treats `True` as integer-like.

`pending_result()` does not create a fake output. The command layer prints it to standard error and returns 2. Successful setup actions print JSON to standard output and return 0; configuration/read failures return 1. Those codes distinguish “setup succeeded,” “configuration failed” and “workflow not implemented.” An empty results directory is consistent with a successful status command.

### Use the existing CLI and TUI

From `/mnt/projects/ai`, these commands use the shared controller and project-owned launcher profile. They inspect/run the scaffold; they do not acquire weights.

```bash
./ai run document-assistant
./ai test document-assistant
./ai tui
```

In the TUI, expand **Projects → Quantization Lab → Document Assistant**. Select **Project status**, **Check setup** or **Show configuration**, then choose **Run**. **Checks** runs scaffold tests. Setup environment and Build package are separate tooling actions. The profile deliberately does not offer an “Answer” button that implies working inference. Inspect [the captured launcher profile](evidence/launcher-profile.txt).

For direct execution, run these from the project folder:

```bash
.venv/bin/python src/document_assistant.py status
.venv/bin/python src/check_setup.py
.venv/bin/python src/show_config.py
.venv/bin/python src/document_assistant.py answer
```

The last command is a deliberate negative check: expected exit 2 and `status: not_implemented`. The environment is pinned to Python 3.12.13; the setup check enforces the 3.12 minor line. Frozen installation uses `uv sync --frozen` from this project folder when its environment needs recreation. That operation installs scaffold/dev tools, not a chosen inference stack.

### Verify without relying on the working directory

The reference below has an optional live mode. With no argument it states its portable contract. With the actual project path it invokes four real actions from `/tmp`, parses the correct output stream and checks exit codes. The live mode runs the project interpreter and actual script.

[Download the complete reference](examples/02_scaffold_trace.py)

```python
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
```

From the scratch directory containing that file, run:

```bash
python3 02_scaffold_trace.py
```

Expected checked output (JSON formatting may differ):

```json
{
  "scope": "invoke with project root for live scaffold check",
  "expected_exit_codes": [
    0,
    0,
    0,
    2
  ]
}
```

For the live check, from the scratch directory use:

```bash
python3 02_scaffold_trace.py /mnt/projects/ai/ai_projects/projects/quantization-lab/document-assistant
```

If imports work only from `src/`, inspect packaging and entry points. If `check` rejects the interpreter, recreate the pinned environment instead of disabling validation. If `answer` returns a plausible response here, investigate a source change before treating it as model evidence. The [reference-run record](project/reference-runs.md) states which checks were actually executed during tutorial preparation.

[[practice:scaffold-transfer]]

### Where later work belongs

Future canonicalization belongs in `src/data/`, artifact recipes in `src/quantization/`, and aligned scoring/measurement in `src/evaluation/`. `src/core/` owns configuration and the answer lifecycle; `src/utils/` owns shared boundaries. These are destination contracts, not existing implementations. We will not paste teaching fixtures into production packages and call the pipeline complete. Next we establish what a backend must prove before these planned actions can be enabled.
