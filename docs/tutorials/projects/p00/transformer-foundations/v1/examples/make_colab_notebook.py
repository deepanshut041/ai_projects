import json
from pathlib import Path

CELLS = [
    ('markdown', """# P00: one common trainer, persistent recovery
Use after local numerical/model/recovery tests and retention/failure drills pass.
Run each cell in order. Choose new/resume explicitly. Hosted results are recorded only after execution.
"""),
    ('code', """import sys, platform
print(platform.python_version())
assert sys.version_info[:2] == (3, 12), "This child pins Python 3.12; resolve and record a different runtime deliberately before continuing"
"""),
    ('code', """from google.colab import files
uploaded = files.upload()  # Select the exact p00-source.zip you exported locally.
assert len(uploaded) == 1
archive_name, archive_bytes = next(iter(uploaded.items()))
import hashlib
print("Uploaded archive SHA256:", hashlib.sha256(archive_bytes).hexdigest())
# Compare this value with the local export before running the next cell.
"""),
    ('code', """from pathlib import Path
import hashlib, io, json, stat, zipfile
project = Path('/content/transformer-foundations-lab')
assert not project.exists(), "Source folder exists; inspect it instead of overwriting"
with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
    members = archive.infolist()
    for member in members:
        path = Path(member.filename)
        mode = member.external_attr >> 16
        assert not path.is_absolute() and '..' not in path.parts
        assert not stat.S_ISLNK(mode) and not member.is_dir()
    manifest = json.loads(archive.read('source-manifest.json'))
    assert {m.filename for m in members} == set(manifest['files']) | {'source-manifest.json'}
    assert len({m.filename for m in members}) == len(members)
    for relative, expected in manifest['files'].items():
        assert hashlib.sha256(archive.read(relative)).hexdigest() == expected, relative
    project.mkdir()
    archive.extractall(project)
print(manifest['revision'], 'dirty:', manifest['dirty'])
"""),
    ('code', """import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', str(project)], check=True)
# pip uses the child's pinned dependency versions; it does not interpret uv's CPU index table.
import torch
print('torch:', torch.__version__, 'CUDA wheel:', torch.version.cuda)
print('GPU available:', torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_model.py', '-v'], cwd=project, check=True)
"""),
    ('code', """from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
persistent = Path('/content/drive/MyDrive/ai-runs/p00')
persistent.mkdir(parents=True, exist_ok=True)
probe = persistent / 'write-probe.txt'
probe.write_text('verified')
assert probe.read_text() == 'verified'
probe.unlink()
import shutil
print('Persistent free bytes:', shutil.disk_usage(persistent).free)
"""),
    ('code', """# Edit these four choices before running. Keep the same ID/config/device when resuming.
mode = 'new'                 # Change to 'resume' after deleting/recreating the runtime.
run_id = 'copy-free-001'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
target_step = 20             # Use 40 on the recovery session, then a predeclared quality budget.
run_dir = persistent / run_id
assert mode in {'new', 'resume'}
if device == 'cpu':
    print('CPU fallback smoke only; GPU availability is not assumed.')
config = project / 'configs' / 'colab-free.json'
command = [sys.executable, '-m', 'transformer_foundations_lab.train',
           '--mode', mode, '--run-dir', str(run_dir), '--config', str(config),
           '--device', device, '--target-step', str(target_step),
           '--session-updates', '20', '--seconds', '120', '--save-every', '10']
print(command)
subprocess.run(command, cwd=project, check=True)
"""),
    ('markdown', """## Required recovery drill
Record the saved global step/generation, archive hash, environment, config/device and metrics.
Then choose Runtime → Disconnect and delete runtime. Reconnect, upload the same archive, install the same runtime dependencies, mount the same Drive, and run these cells again with mode='resume', the same run_id/device/config and a larger global target.
Confirm the log restores the recorded step and advances it. A missing GPU or changed identity must stop or use an explicitly different new run, not silently restart or weaken the identity check.
Save the output-cleared notebook to notebooks/colab.ipynb in the child. Put measured evidence in docs/colab-recovery.md, leaving credentials and data/weights out of Git.
"""),
]
root = Path(__file__).resolve().parents[1]
path = root / "notebooks/colab.ipynb"
path.parent.mkdir(exist_ok=True)
if path.exists():
    raise SystemExit("Notebook exists; edit it instead of overwriting")
notebook = {"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"cells":[]}
for index,(kind,source) in enumerate(CELLS):
    cell={"cell_type":kind,"id":f"p00-{index:02d}","metadata":{},"source":source.splitlines(keepends=True)}
    if kind == "code": cell.update(execution_count=None,outputs=[])
    notebook["cells"].append(cell)
path.write_text(json.dumps(notebook,indent=2)+"\n")
print(path)
