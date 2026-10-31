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
