"""Export only prerequisite sources and generate a thin, output-cleared Colab notebook."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from study_notebook import notebook as study_notebook, notes_text

SOURCE_FILES = (
    "prerequisite_mlp.py", "prerequisite_checkpoint.py", "prerequisite_mlp_checks.py", "prerequisite_export.py",
    "prerequisite_math.py", "prerequisite_math.cpp", "test_prerequisite_math.py",
    "prerequisite_tensors.py", "test_prerequisite_tensors.py",
    "prerequisite_text.py", "test_prerequisite_text.py", "byte_bpe.py", "first_session.py",
    "deep_learning_math.py", "test_deep_learning_math.py", "study_notebook.py",
    "transformer_foundation_probes.py", "test_transformer_foundation_probes.py",
)


def notebook():
    cells = []

    def markdown(cell_id, text):
        cells.append({"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": text.splitlines(keepends=True)})

    def code(cell_id, text):
        cells.append({"cell_type": "code", "id": cell_id, "metadata": {}, "source": text.splitlines(keepends=True), "execution_count": None, "outputs": []})

    markdown("intro", "# Tutorial 0: prerequisites before Transformer models\n\nRun the same FP32 CPU two-layer classifier as the local source. This notebook contains setup and calls only. Hosted execution and Drive survival are unverified until you perform and record the fresh-runtime exercise. Keep the experiment's total goal fixed while slicing sessions.\n")
    code("source", '''import hashlib, json, sys, zipfile
from pathlib import Path
from google.colab import files

uploaded = files.upload()
if len(uploaded) != 1:
    raise ValueError("Upload exactly one prerequisite source ZIP")
archive_name, archive_bytes = next(iter(uploaded.items()))
print("Uploaded source ZIP SHA256:", hashlib.sha256(archive_bytes).hexdigest())
source_root = Path.cwd() / "prerequisite_source"
source_root.mkdir(exist_ok=True)
with zipfile.ZipFile(archive_name) as archive:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("Duplicate ZIP member")
    if any(Path(name).name != name or name in (".", "..") for name in names):
        raise ValueError("Source ZIP members must be safe flat filenames")
    manifest = json.loads(archive.read("manifest.json"))
    if manifest.get("format") != "tutorial0-prerequisite-source-v1":
        raise ValueError("Wrong source-export format")
    expected = set(manifest["files"]) | {"manifest.json"}
    if set(names) != expected:
        raise ValueError("ZIP members differ from the manifest")
    for name, record in manifest["files"].items():
        content = archive.read(name)
        if len(content) != record["bytes"] or hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError("Source digest differs: " + name)
        destination = source_root / name
        if destination.exists() and destination.read_bytes() != content:
            raise ValueError("Existing source differs; use a fresh runtime")
        destination.write_bytes(content)
print("Source revision:", manifest["source_revision"])
print("Working-tree snapshot:", manifest["working_tree_snapshot"])
print("Source and notebook files verified:", sorted(manifest["files"]))
''')
    code("dependencies", '''import subprocess

subprocess.run([sys.executable, "-m", "pip", "install", "numpy==2.5.4"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "torch==2.14.1+cpu",
                "--index-url", "https://download.pytorch.org/whl/cpu"], check=True)
import numpy as np
import torch
torch.set_num_threads(1)
if str(torch.__version__) != "2.14.1+cpu" or str(np.__version__) != "2.5.4":
    raise RuntimeError("Dependency versions differ; restart the runtime before training")
print(json.dumps({"python": sys.version, "torch": str(torch.__version__),
                  "numpy": str(np.__version__), "device": "cpu", "threads": 1}, indent=2))
''')
    code("correctness", '''subprocess.run([sys.executable, "-m", "unittest",
    "test_prerequisite_math.py", "test_prerequisite_tensors.py", "test_prerequisite_text.py",
    "test_deep_learning_math.py", "test_transformer_foundation_probes.py"],
    cwd=source_root, check=True)
subprocess.run([sys.executable, "prerequisite_mlp_checks.py"], cwd=source_root, check=True)
''')
    code("drive", '''from google.colab import drive
import shutil

drive.mount("/content/drive")
persistent_root = Path("/content/drive/MyDrive/ai-runs/tutorial0-prerequisites")
persistent_root.mkdir(parents=True, exist_ok=True)
probe = persistent_root / ".write-check"
probe.write_text("read-back")
if probe.read_text() != "read-back":
    raise OSError("Drive write/read verification failed")
probe.unlink()
print("Persistent location:", persistent_root)
print("Available bytes:", shutil.disk_usage(persistent_root).free)
''')
    markdown("selection", "## Select a run deliberately\n\nChoose `mode='new'` for an empty run, then use `mode='resume'` with the same run ID after deleting the runtime. Changing source/config/data/dependency identity is a new experiment. This initial slice is only 25 updates so you can practise recovery before a longer session.\n")
    code("run", '''run_id = "tutorial0-mlp-001"
mode = "new"  # Change explicitly to "resume" for the SAME compatible run.
if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
    raise ValueError("Use one simple run-directory name")
run_directory = persistent_root / run_id
command = [sys.executable, "prerequisite_mlp.py", "--mode", mode,
    "--run-dir", str(run_directory), "--total-steps", "1000",
    "--max-updates", "25", "--max-seconds", "3600",
    "--checkpoint-updates", "100", "--checkpoint-seconds", "300"]
process = subprocess.Popen(command, cwd=source_root, stdout=subprocess.PIPE, text=True)
output, _ = process.communicate()
if process.returncode != 0:
    raise subprocess.CalledProcessError(process.returncode, command)
report = json.loads(output)
print(json.dumps(report, indent=2))
print("Record restored update, completed update, source/runtime identity and last verified generation.")
''')
    markdown("fresh-runtime", "## Demonstrate persistence in a fresh runtime\n\nRecord the verified Drive generation and completed update. Choose **Disconnect and delete runtime** in Colab, reconnect, reopen this output-cleared notebook and repeat source/dependency/check cells. Mount the same Drive location, retain the same run ID, set `mode='resume'`, and run the same 25-update slice. It must print the restored update and advance it. Record actual versions, resource measurements, checkpoint bytes/latency and any rollback. Local temporary-directory tests do not prove Drive survived this exercise.\n\nAfter this recovery drill, you may set `--max-updates` to at most 1000 per session, keeping the 60-minute cap and saves every 100 completed updates or five minutes. The experiment still stops at its fixed total goal. Persistence failure raises an error and stops; do not continue training under the failed-save condition.\n")
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "cells": cells}


def export_source(destination, notebook_destination=None, study_destination=None):
    if study_destination is not None and Path(study_destination).exists():
        raise FileExistsError("Preserve existing study notes; choose a new notebook filename")
    root = Path(__file__).resolve().parent
    missing = [name for name in SOURCE_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError("Complete prerequisite sources before exporting: " + ", ".join(missing))
    payload = {name: (root / name).read_bytes() for name in SOURCE_FILES}
    payload["colab.ipynb"] = json.dumps(notebook(), indent=2).encode()
    payload["study-notes.md"] = notes_text().encode()
    payload["study.ipynb"] = json.dumps(study_notebook(payload["study-notes.md"].decode()), indent=2).encode()
    payload["requirements.txt"] = b"numpy==2.5.4\ntorch==2.14.1+cpu\n"
    try:
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        source_revision = revision.stdout.strip() if revision.returncode == 0 else "unavailable"
    except FileNotFoundError:
        source_revision = "unavailable"
    manifest = {"format": "tutorial0-prerequisite-source-v1", "source_revision": source_revision, "working_tree_snapshot": True, "files": {name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()} for name, content in payload.items()}}
    payload["manifest.json"] = json.dumps(manifest, sort_keys=True, indent=2).encode()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    if notebook_destination is not None:
        Path(notebook_destination).parent.mkdir(parents=True, exist_ok=True)
        Path(notebook_destination).write_bytes(payload["colab.ipynb"])
    if study_destination is not None:
        target = Path(study_destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload["study.ipynb"])
    return {"archive": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "members": sorted(payload), "notebook": str(notebook_destination) if notebook_destination else None, "study_notebook": str(study_destination) if study_destination else None, "hosted_execution": "unverified"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--study-notebook", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_source(args.destination, args.notebook, args.study_notebook), indent=2))
