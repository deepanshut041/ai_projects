"""Read and validate one compatible practical-model comparison receipt."""

import json
from pathlib import Path

from .data import sha256
from .features import canonical_json


def summarize_tree_study(run_directory: Path):
    study = json.loads((run_directory/"study.json").read_text())
    protocol = json.loads((run_directory/"protocol.json").read_text())
    selection = json.loads((run_directory/"selection.json").read_text())
    if study["version"] != "calendar-tree-study-v1" or study["status"] != "completed" or study["test_evaluated"]:
        raise ValueError("Expected a completed development tree study with the final test reserved.")
    if sha256(canonical_json(protocol)) != study["protocol_sha256"] or selection != study["selection"]:
        raise ValueError("Comparison protocol or selection identity changed.")
    for relative in study["trial_directories"].values():
        directory = (run_directory/relative).resolve()
        if not directory.is_relative_to(run_directory.resolve()):
            raise ValueError("Trial directory escapes its study.")
        record = json.loads((directory/"run.json").read_text())
        if record["protocol_sha256"] != study["protocol_sha256"] or record["dataset"] != study["dataset"]:
            raise ValueError("Trial task or protocol is incompatible.")
        for name,digest in record["artifacts_sha256"].items():
            artifact = (directory/name).resolve()
            if not artifact.is_relative_to(directory) or sha256(artifact.read_bytes()) != digest:
                raise ValueError("Comparison artifact identity changed.")
    representative = selection["representative_trial"]
    if sha256((run_directory/study["trial_directories"][representative]/"model.json").read_bytes()) != selection["model_sha256"]:
        raise ValueError("Selected model identity changed.")
    return {"version": "calendar-comparison-summary-v1", "run": run_directory.name,
            "protocol_sha256": study["protocol_sha256"], "selection": selection,
            "test_evaluated": False, "failed_trials": study["failed_trials"],
            "trial_count": len(study["trial_directories"]),
            "timing_scope": "local fit times and transparent snapshot prediction replay; no deployment latency claim"}
