"""Export an explicit source allowlist, a hash manifest and an output-free notebook."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

FILES = ["gpt_config.py", "gpt_tokenizer.py", "gpt_data.py", "gpt_attention.py",
         "gpt_model.py", "gpt_objective.py", "gpt_checkpoint.py", "gpt_trainer.py",
         "gpt_train.py", "gpt_sample.py", "test_gpt_data.py",
         "test_gpt_model.py", "test_gpt_training.py", "cpu.json", "colab-free.json"]


def export(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new export directory")
    blobs = {name:(root/name).read_bytes() for name in FILES}
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],stderr=subprocess.DEVNULL).decode().strip()
    except subprocess.CalledProcessError:
        revision = "uncommitted-source"
    manifest = {"revision":revision, "files":{name:hashlib.sha256(raw).hexdigest() for name,raw in blobs.items()},
                "dependency":"torch==2.14.1", "note":"File digests identify actual source, including dirty edits"}
    output.mkdir(parents=True)
    archive = output/"p01-source.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as bundle:
        for name,raw in blobs.items(): bundle.writestr(name,raw)
        bundle.writestr("source_manifest.json",json.dumps(manifest,indent=2))
    archive_sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    cells=[]
    def markdown(text): cells.append({"cell_type":"markdown","metadata":{},"source":text.splitlines(True)})
    def code(text): cells.append({"cell_type":"code","metadata":{},"source":text.splitlines(True),"outputs":[],"execution_count":None})
    markdown("# P01: one trainer, a persistent run\nUse Colab Free. First run tests, then choose new or resume explicitly. After a verified save, disconnect AND delete the runtime, repeat setup, choose resume, and record the restored/advanced update. This notebook has no hosted proof until you do that drill.\n")
    code('''from google.colab import files
uploaded = files.upload()  # Upload the exact p01-source.zip accompanying this notebook.
''')
    code(f'''from pathlib import Path
import hashlib, json, zipfile, stat
archive = Path("p01-source.zip")
assert hashlib.sha256(archive.read_bytes()).hexdigest() == {archive_sha!r}, "Wrong source archive"
source = Path("/content/p01-source")
source.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as bundle:
    names = bundle.namelist()
    assert len(names) == len(set(names))
    assert sum(info.file_size for info in bundle.infolist()) < 5_000_000
    assert all(Path(name).name == name and name not in (".", "..") for name in names)
    assert all(not stat.S_ISLNK(info.external_attr >> 16) for info in bundle.infolist())
    manifest = json.loads(bundle.read("source_manifest.json"))
    assert set(names) == set(manifest["files"]) | {{"source_manifest.json"}}
    for name, expected in manifest["files"].items():
        raw = bundle.read(name)
        assert hashlib.sha256(raw).hexdigest() == expected
        (source/name).write_bytes(raw)
print(manifest)
''')
    code('''import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "torch==2.14.1"], check=True)
import torch, platform
print({"python":platform.python_version(), "torch":torch.__version__, "cuda":torch.version.cuda,
       "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
subprocess.run([sys.executable, "-m", "unittest", "test_gpt_data", "test_gpt_model", "test_gpt_training"], cwd=source, check=True)
''')
    code('''from google.colab import drive
drive.mount("/content/drive")
run = Path("/content/drive/MyDrive/ai-runs/p01/first-text-run")
run.parent.mkdir(parents=True, exist_ok=True)
probe = run.parent / "write-probe.txt"
probe.write_text("p01"); assert probe.read_text() == "p01"; probe.unlink()
data = Path("/content/p01-data")
subprocess.run([sys.executable, "gpt_data.py", str(data)], cwd=source, check=True)
''')
    code('''mode = "new"  # After deleting the runtime, explicitly change to "resume".
device = "cuda" if torch.cuda.is_available() else "cpu"
# Strict resume requires the same recorded runtime/device identity. Never silently restart.
subprocess.run([sys.executable, "gpt_train.py", "--data", str(data), "--run", str(run),
                "--config", "colab-free.json", "--mode", mode, "--device", device,
                "--updates", "100", "--seconds", "3600", "--save-every", "25",
                "--save-seconds", "300", "--storage-mib", "64"], cwd=source, check=True)
''')
    code('''subprocess.run([sys.executable, "gpt_sample.py", "--data", str(data), "--run", str(run),
                "--prompt", "Alice ", "--tokens", "80"], cwd=source, check=True)
print("Recovery location:", run)
''')
    markdown("## Record the recovery drill\nRecord runtime/version/device, source/data/tokenizer identities, before/after update and token counts, rejected generations/rollback, checkpoint bytes, save latency and peak RAM/VRAM. A changed device is a separately labelled continuation protocol, not deterministic parity. Do not mark hosted recovery complete from local tests.\n")
    notebook={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3"}},"cells":cells}
    (output/"colab.ipynb").write_text(json.dumps(notebook,indent=2))
    (output/"source_manifest.json").write_text(json.dumps({**manifest,"archive_sha256":archive_sha},indent=2))
    return {"archive":str(archive),"sha256":archive_sha,"files":len(blobs),"notebook":str(output/"colab.ipynb")}


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("output"); args=parser.parse_args()
    print(json.dumps(export(Path(__file__).parent,args.output),indent=2))
