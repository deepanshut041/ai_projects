"""Export only the self-contained reference source, with per-file hashes."""

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys
import zipfile


def colab_notebook(zip_name, archive_hash, runtime):
    def markdown(source):
        return {"cell_type": "markdown", "metadata": {}, "source": source}

    def code(source):
        return {"cell_type": "code", "metadata": {}, "source": source,
                "execution_count": None, "outputs": []}

    cells = [
        markdown("# Transformer course — bounded training and recovery\n\n"
                 "Use the exported Python source for every model and update. This notebook prepares "
                 "a CPU/FP32 Colab Free drill; no hosted result is claimed. Begin with translation. "
                 "After the RoBERTa lessons, choose mlm or qa. Save to Drive, delete the runtime, "
                 "then upload the same ZIP and choose resume with the same run ID/total schedule."),
        markdown("## Upload and verify the source\n\n"
                 "Upload the ZIP generated beside this notebook. The embedded archive hash pins "
                 "these source bytes. Members must be flat source files; every member hash is "
                 "checked before extraction. Rerunning accepts only an already identical source directory."),
        code("from google.colab import files\n"
             "from pathlib import Path, PurePosixPath\n"
             "import hashlib, io, json, stat, sys, zipfile\n\n"
             f"ZIP_NAME = {zip_name!r}\nEXPECTED_ARCHIVE_SHA256 = {archive_hash!r}\n"
             "uploaded = files.upload()\n"
             "if ZIP_NAME not in uploaded:\n    raise ValueError('Upload the exact exported ZIP filename')\n"
             "blob = uploaded[ZIP_NAME]\n"
             "if hashlib.sha256(blob).hexdigest() != EXPECTED_ARCHIVE_SHA256:\n"
             "    raise ValueError('Source ZIP differs from the pinned export')\n"
             "SOURCE = Path('/content/transformer-course-source')\n"
             "with zipfile.ZipFile(io.BytesIO(blob)) as archive:\n"
             "    members = archive.infolist()\n"
             "    names = [item.filename for item in members]\n"
             "    if len(names) != len(set(names)) or sum(item.file_size for item in members) > 20 * 1024 * 1024:\n"
             "        raise ValueError('Duplicate source names or oversized archive')\n"
             "    for item in members:\n"
             "        member = PurePosixPath(item.filename)\n"
             "        if member.is_absolute() or len(member.parts) != 1 or '..' in member.parts or item.is_dir() or stat.S_ISLNK(item.external_attr >> 16):\n"
             "            raise ValueError('Archive member is not a flat regular source file')\n"
             "    manifest = json.loads(archive.read('source-manifest.json'))\n"
             "    if set(names) != set(manifest['files']) | {'source-manifest.json'}:\n"
             "        raise ValueError('Archive members differ from the manifest')\n"
             "    contents = {name: archive.read(name) for name in names}\n"
             "    for name, expected in manifest['files'].items():\n"
             "        if hashlib.sha256(contents[name]).hexdigest() != expected:\n"
             "            raise ValueError('Source member failed its hash check: ' + name)\n"
             "    if SOURCE.exists() and any(SOURCE.iterdir()):\n"
             "        present = {path.name for path in SOURCE.iterdir() if path.name != '__pycache__'}\n"
             "        if present != set(names) or any((SOURCE / name).read_bytes() != data for name, data in contents.items()):\n"
             "            raise FileExistsError('Source directory contains different files; use a fresh directory')\n"
             "    else:\n"
             "        SOURCE.mkdir(parents=True, exist_ok=True)\n"
             "        for name, data in contents.items():\n            (SOURCE / name).write_bytes(data)\n"
             "sys.path.insert(0, str(SOURCE))\n"
             "print('Verified source:', SOURCE, 'files:', len(manifest['files']))"),
        markdown("## Install the matching CPU dependencies\n\n"
                 "These exact Torch/NumPy versions come from the exporting environment. The CPU wheel "
                 "keeps the first recovery comparison independent of a GPU allocation. If a previously "
                 "imported package retains its old version, restart the kernel and rerun these setup cells. "
                 "An unavailable wheel is a setup failure, not evidence of Colab compatibility."),
        code("import subprocess\n"
             f"EXPECTED_TORCH = {runtime['torch']!r}\nEXPECTED_NUMPY = {runtime['numpy']!r}\n"
             "subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch==' + EXPECTED_TORCH, "
             "'--index-url', 'https://download.pytorch.org/whl/cpu'], check=True)\n"
             "subprocess.run([sys.executable, '-m', 'pip', 'install', 'numpy==' + EXPECTED_NUMPY], check=True)"),
        code("import numpy as np\nimport torch\n"
             "if str(torch.__version__) != EXPECTED_TORCH or str(np.__version__) != EXPECTED_NUMPY:\n"
             "    raise RuntimeError('Loaded versions differ: restart the kernel after installation, then rerun setup')\n"
             "torch.set_num_threads(1)\n"
             "print(json.dumps({'python': sys.version, 'torch': str(torch.__version__), "
             "'numpy': str(np.__version__), 'device': 'cpu', 'dtype': 'float32', 'threads': torch.get_num_threads()}))"),
        markdown("## Check the chosen project before training\n\n"
                 "Translation checks include bounded fixture training. The RoBERTa checks are for use "
                 "after the relevant MLM or QA topics. Recovery and session checks use temporary directories; Drive persistence "
                 "is verified separately by the deleted-runtime drill."),
        code('TASK = "translation"  # Later choose "mlm" or "qa".\n'
             "subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(SOURCE), "
             "'-p', 'test_foundation_math.py'], cwd=SOURCE, check=True)\n"
             "subprocess.run([sys.executable, str(SOURCE / ('translation_checks.py' if TASK == 'translation' else 'roberta_checks.py'))], cwd=SOURCE, check=True)\n"
             "subprocess.run([sys.executable, str(SOURCE / 'course_recovery_checks.py'), '--project', "
             "'translation' if TASK == 'translation' else 'roberta'], cwd=SOURCE, check=True)\n"
             "subprocess.run([sys.executable, str(SOURCE / 'course_session_checks.py')], cwd=SOURCE, check=True)"),
        markdown("## Mount persistent storage and choose the run explicitly\n\n"
                 "New refuses an existing run directory. Resume requires its run manifest and a compatible "
                 "verified generation; it never becomes new automatically. Preserve TOTAL_STEPS, seed and "
                 "the source ZIP across slices. Change only MODE to resume after deleting the runtime."),
        code("from google.colab import drive\ndrive.mount('/content/drive')\n"
             "from course_session import run_session\n"
             'MODE = "new"\nRUN_ID = "run-001"\n'
             "TOTAL_STEPS = 400 if TASK == 'translation' else 120\n"
             "RUN_DIRECTORY = Path('/content/drive/MyDrive/ai-runs/transformer-course') / TASK / RUN_ID\n"
             "# For a new QA run, point BACKBONE at an exported compatible MLM backbone.\n"
             "BACKBONE = None\n"
             "summary = run_session(TASK, RUN_DIRECTORY, mode=MODE, total_steps=TOTAL_STEPS, "
             "seed=7, device='cpu', additional_updates=10, max_seconds=3600, "
             "save_updates=5, save_seconds=300, keep=2, "
             "backbone=BACKBONE if MODE == 'new' else None, export_backbone=(TASK == 'mlm'))\n"
             "print(json.dumps(summary, indent=2))"),
        markdown("## Delete the runtime and prove continuation\n\n"
                 "Record the generation/update, checkpoint bytes and save timing. Use Disconnect and delete "
                 "runtime. In the fresh runtime, upload this same ZIP, reinstall matching dependencies, "
                 "rerun setup/checks and remount Drive. Choose MODE='resume' with the same task, run ID and "
                 "TOTAL_STEPS. Require a restored update/generation, further completed updates and another "
                 "verified save. Inspect any skipped incomplete/corrupt generations. An exact local CPU "
                 "test and this output-cleared notebook do not establish that hosted evidence. For the "
                 "normal session budget, use at most 1,000 additional updates/3,600 seconds and save every 100 updates "
                 "or 300 seconds; the shorter 10-update/5-update drill above is deliberate."),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"course-{index:02d}"
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}}, "cells": cells}


def export_sources(destination):
    source = Path(__file__).resolve().parent
    files = sorted(p for p in source.iterdir() if p.is_file() and p.name != "source-manifest.json" and p.suffix in {".py", ".cpp", ".json", ".txt"})
    runtime = {"python": list(sys.version_info[:3]), "torch": metadata.version("torch"),
               "numpy": metadata.version("numpy")}
    from course_study_notebook import notes_text, notebook as study_notebook
    payload = {p.name: p.read_bytes() for p in files}
    payload['study-notes.md'] = notes_text().encode('utf-8')
    payload['study.ipynb'] = (json.dumps(study_notebook(), indent=2) + "\n").encode('utf-8')
    manifest = {"schema": 1, "runtime": runtime,
                "files": {name: hashlib.sha256(data).hexdigest() for name,data in payload.items()}}
    destination = Path(destination)
    colab_path = destination.with_suffix('.colab.ipynb')
    if destination.exists() or colab_path.exists():
        raise FileExistsError('Preserve existing exports and notebooks; use a fresh destination')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in payload.items():
            archive.writestr(name,data)
        archive.writestr("source-manifest.json", json.dumps(manifest, indent=2))

    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        for name, expected in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    archive_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
    notebook = destination.with_suffix(".colab.ipynb")
    notebook.write_text(json.dumps(colab_notebook(destination.name, archive_hash, runtime), indent=2) + "\n")
    return {"files": len(payload), "sha256": archive_hash, "notebook": str(notebook), "study_member": "study.ipynb",
            "runtime": runtime}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(export_sources(arguments.destination)))
