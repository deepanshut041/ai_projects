"""Replace the teaching notebook with an operational Google Colab runner."""
import json
from pathlib import Path
import shutil
import textwrap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PROJECT = ROOT / 'ai_projects/projects/transformer-labs/attention-is-all-you-need'
COURSE = ROOT / 'ai_projects/docs/tutorials/projects/attention-is-all-you-need/attention-lab/v1'
RECOVERY = ROOT / '.ai/attention-layout/2026-10-23'
old = PROJECT / 'notebooks/attention_mechanism.ipynb'
if old.exists():
    shutil.copy2(old, RECOVERY / 'before-colab-attention_mechanism.ipynb')
    old.unlink()

cells = []
def cell(kind, body):
    source = textwrap.dedent(body).strip() + '\n'
    item = {'cell_type': kind, 'id': f'colab-{len(cells):02}', 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        item.update(execution_count=None, outputs=[])
    cells.append(item)

cell('markdown', '''
    # Attention Is All You Need — Google Colab runner

    Run the existing Python project on Colab. Read the **Attention Lab** tutorial
    in Projects Docs for the worked explanations, maths and figures.

    1. Open this notebook using **File → Upload notebook** in [Google Colab](https://colab.research.google.com/).
    2. Use a **CPU runtime** and run the cells in order.
    3. When prompted, select **both** `attention-is-all-you-need-source.zip` and
       its matching `source-manifest.json` from the local workspace export.
    4. Download the result archive at the end.

    The export is in `.ai/exports/attention-is-all-you-need/` at workspace home.
    The notebook creates a separate locked Python 3.12 environment; it does not
    import the project into Colab's own Python kernel. The Linux lock uses CPU
    PyTorch even when a GPU runtime is selected. Setup needs internet access.

    This runs the implemented attention component and its tests. Tokenizers,
    translation data, the full Transformer, training, BLEU and persistent recovery
    are still pending. No Drive mount is needed. Runtime files are temporary:
    [Colab's FAQ](https://research.google.com/colaboratory/faq.html) explains the limits.
''')
cell('markdown', '''
    ## 1. Upload and verify the source

    Select the source ZIP and manifest together. Their archive/file hashes must
    agree before extraction. Each setup uses a fresh directory under `/content`.
    Keep the pair from your own local export; matching hashes check consistency,
    not the identity of an unknown sender.
''')
cell('code', '''
    from pathlib import Path, PurePosixPath
    import hashlib
    import io
    import json
    import subprocess
    import sys
    import tempfile
    import zipfile
    from google.colab import files

    uploaded = files.upload()
    archive_name = "attention-is-all-you-need-source.zip"
    manifest_name = "source-manifest.json"
    if set(uploaded) != {archive_name, manifest_name}:
        raise ValueError("Select exactly the source ZIP and its matching source-manifest.json together.")
    manifest = json.loads(uploaded[manifest_name])
    archive_bytes = uploaded[archive_name]
    archive_hash = hashlib.sha256(archive_bytes).hexdigest()
    if archive_hash != manifest["archive_sha256"]:
        raise ValueError("ZIP hash mismatch. Export/upload a matching ZIP and manifest.")

    verified = {}
    prefix = "attention-is-all-you-need/"
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        for entry in archive.infolist():
            name = entry.filename
            relative = name.removeprefix(prefix)
            parts = PurePosixPath(relative).parts
            if (not name.startswith(prefix) or entry.is_dir() or not parts
                    or relative.startswith("/") or ".." in parts or "\\\\" in name
                    or relative != PurePosixPath(relative).as_posix() or relative in verified):
                raise ValueError(f"Unexpected or duplicate ZIP path: {name}")
            payload = archive.read(entry)
            if hashlib.sha256(payload).hexdigest() != manifest["files"].get(relative):
                raise ValueError(f"Source file hash mismatch: {relative}")
            verified[relative] = payload
    if set(verified) != set(manifest["files"]):
        raise ValueError("ZIP members do not match the manifest.")

    WORK = Path(tempfile.mkdtemp(prefix="attention-colab-", dir="/content"))
    SOURCE = WORK / "attention-is-all-you-need"
    for relative, payload in verified.items():
        target = SOURCE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    RESULTS = WORK / "results"
    RESULTS.mkdir()
    (RESULTS / manifest_name).write_bytes(uploaded[manifest_name])
    print("Source directory:", SOURCE)
    print("Verified source files:", len(verified))
    print("Archive SHA-256:", archive_hash)
    print("Source revision:", manifest["source_revision"])
    print("Includes local source changes:", manifest["working_tree"])
''')
cell('markdown', '''
    ## 2. Install the locked project environment

    Install pinned uv into a separate tools environment, then let it install
    Python 3.12.13 and the versions in `uv.lock`. The `notebook` extra includes
    visualization dependencies so the ten matrix-helper tests run as well as
    the twenty-five attention tests. Installation can take several minutes.
''')
cell('code', '''
    TOOLS = WORK / "tools"
    subprocess.run([sys.executable, "-m", "venv", str(TOOLS)], check=True)
    subprocess.run([str(TOOLS / "bin/python"), "-m", "pip", "install", "uv==0.10.10"], check=True)
    UV = TOOLS / "bin/uv"

    def run(*args):
        completed = subprocess.run([str(arg) for arg in args], cwd=SOURCE,
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(completed.stdout)
        completed.check_returncode()
        return completed.stdout

    run(UV, "sync", "--frozen", "--extra", "notebook", "--python", "3.12.13")
    PYTHON = SOURCE / ".venv/bin/python"
''')
cell('markdown', '''
    ## 3. Record the actual runtime

    This probes the project's installed Python, rather than Colab's notebook kernel.
''')
cell('code', '''
    environment_text = run(PYTHON, "-c", """
    import json, platform, sys
    import torch, numpy, matplotlib
    print(json.dumps({"python": sys.version, "platform": platform.platform(),
                      "torch": torch.__version__, "numpy": numpy.__version__,
                      "matplotlib": matplotlib.__version__, "demo_device": "cpu",
                      "cuda_available": torch.cuda.is_available()}, indent=2))
    """)
    environment = json.loads(environment_text)
    (RESULTS / "environment.json").write_text(json.dumps(environment, indent=2) + "\\n")
''')
cell('markdown', '''
    ## 4. Run the attention demo

    Expected: weighted values, attention matrices, encoder/decoder outputs shaped
    `(3,5,9)`, cross-attention shaped `(3,7,9)`, and a final `PASS`.
    These are component outputs from initialized parameters, not translations.
''')
cell('code', '''
    demo_output = run(PYTHON, "src/attention.py")
    if "PASS:" not in demo_output:
        raise RuntimeError("The demo did not report PASS.")
    (RESULTS / "attention-demo.txt").write_text(demo_output)
''')
cell('markdown', '''
    ## 5. Run all component and matrix checks

    The current source has **35 tests**. A missing plotting dependency must not
    quietly turn the matrix checks into a skipped module. Stop on failures or skips.
''')
cell('code', '''
    import xml.etree.ElementTree as ET

    test_output = run(PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                      "--junitxml=" + str(RESULTS / "tests.xml"))
    suites = ET.parse(RESULTS / "tests.xml").getroot()
    cases = list(suites.iter("testcase"))
    if not cases or any(case.find(tag) is not None for case in cases
                        for tag in ("failure", "error", "skipped")):
        raise RuntimeError("Tests failed, were skipped, or no tests ran.")
    (RESULTS / "tests.txt").write_text(test_output)
    print("Passing tests:", len(cases))
''')
cell('markdown', '''
    ## 6. Download the run evidence

    Download this small ZIP before the runtime is deleted. It contains the source
    manifest, environment versions, demo output and test results. It is a component
    execution record; training checkpoints and persistent recovery come later.
''')
cell('code', '''
    from datetime import datetime, timezone

    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "source_revision": manifest["source_revision"],
               "working_tree": manifest["working_tree"],
               "archive_sha256": archive_hash, "environment": environment,
               "demo_passed": True, "tests_passed": len(cases), "tests_skipped": 0,
               "scope": "attention component; no translation training or recovery"}
    (RESULTS / "run-receipt.json").write_text(json.dumps(receipt, indent=2) + "\\n")
    results_zip = WORK / "attention-colab-results.zip"
    with zipfile.ZipFile(results_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(RESULTS.iterdir()):
            archive.write(path, arcname=path.name)
    print(json.dumps(receipt, indent=2))
    files.download(str(results_zip))
''')
notebook = {'cells': cells, 'metadata': {'colab': {'name': 'attention_colab.ipynb', 'provenance': []},
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
(PROJECT / 'notebooks/attention_colab.ipynb').write_text(json.dumps(notebook, indent=1) + '\n')

# Current authored sources; captured/generated code and local notes are refreshed by 08-assemble.
for path in [PROJECT / '.ai-project.toml', PROJECT / '.gitignore',
             *[p for p in COURSE.rglob('*.md') if 'code' not in p.relative_to(COURSE).parts and p.name != 'lesson.md'],
             HERE / '11-verify-layout.py', HERE / '13-browser-review.mjs',
             ROOT / 'ai_projects/docs/implementation/2026-10/attention-is-all-you-need-source-backed-tutorial/08-assemble.py']:
    text = path.read_text().replace('attention_mechanism.ipynb', 'attention_colab.ipynb')
    text = text.replace('Attention concept notebook', 'Google Colab runner')
    path.write_text(text)

print(json.dumps({'notebook': 'notebooks/attention_colab.ipynb',
                  'code_cells': sum(c['cell_type'] == 'code' for c in cells), 'outputs_cleared': True}))
