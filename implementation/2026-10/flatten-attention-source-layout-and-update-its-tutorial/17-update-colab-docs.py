"""Update current project/docs surfaces for direct scripts and the Colab runner."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PROJECT = ROOT / 'ai_projects/projects/transformer-labs/attention-is-all-you-need'
DOCS = ROOT / 'ai_projects/docs'
COURSE = DOCS / 'tutorials/projects/attention-is-all-you-need/attention-lab/v1'

def change(path, old, new):
    text = path.read_text()
    if old not in text:
        raise ValueError(f'Missing expected prose in {path}: {old[:70]}')
    path.write_text(text.replace(old, new))

readme = PROJECT / 'README.md'
text = readme.read_text().replace('attention_mechanism.ipynb', 'attention_colab.ipynb')
text = text.replace('The notebook imports the same Python implementation; it does not contain a second model or plotting implementation.', 'The tutorial explains the implementation. The notebook runs the Python project on Google Colab.')
start, rest = text.split('## Open the concept notebook', 1)
_, end = rest.split('## Study the implemented project', 1)
text = start + '''## Run on Google Colab

Open the [notebook](notebooks/attention_colab.ipynb) using **File → Upload notebook** in [Google Colab](https://colab.research.google.com/). Run its six code cells in order. Upload the matching source ZIP and manifest when prompted; both are under `.ai/exports/attention-is-all-you-need/` at workspace home.

The runner verifies the archive and every file, installs pinned uv and locked Python 3.12.13 dependencies in an isolated environment, records runtime versions, runs `src/attention.py`, runs all 35 component/matrix checks, and downloads a result archive. CPU is sufficient. It stops on a failed command or skipped test. [Colab guidance](docs/COLAB.md) gives the complete transfer and execution steps.

The former concept notebook has been removed; the twelve-lesson tutorial owns the explanations, worked maths and figures. [Graph helpers](src/utils/attention_visuals.py) remain reusable Python utilities for those figures and matrix checks. To run all 35 checks locally, including the visualization cases:

```bash
uv sync --frozen --extra notebook
uv run --frozen --extra notebook pytest -q
```

The replacement runner is checked locally with uploaded source and a simulated Colab upload/download interface. Actual Google-hosted execution is still unverified.

## Study the implemented project''' + end
text = text.replace('Implemented attention component and notebook', 'Implemented attention component; Colab execution notebook')
readme.write_text(text)
(PROJECT / 'notebooks/README.md').write_text('''# Google Colab runner

[attention_colab.ipynb](attention_colab.ipynb) runs the existing Python project on Google Colab. The Attention Lab tutorial in Projects Docs owns the explanations, maths and figures; the former concept notebook has been removed.

1. Open [Google Colab](https://colab.research.google.com/) and choose **File → Upload notebook**.
2. Upload `attention_colab.ipynb`, select a CPU runtime, and run its cells in order.
3. At the upload prompt, select both `attention-is-all-you-need-source.zip` and its matching `source-manifest.json` from workspace `.ai/exports/attention-is-all-you-need/`.
4. Let the runner verify/extract source, install the locked environment, run the attention demo and all 35 checks, then download `attention-colab-results.zip`.

See [the detailed Colab instructions](../docs/COLAB.md). The runner uses a separate Python 3.12.13 environment and CPU PyTorch; Colab's own kernel does not need project imports. Every upload/extraction uses a fresh runtime directory. After a runtime reset, rerun from the top and upload the pair again.

The notebook is saved without outputs. Local execution with a simulated Colab file interface checks its operational cells; an actual hosted Colab run remains pending. It does not train a translator or implement persistent recovery.
''')

colab = PROJECT / 'docs/COLAB.md'
text = colab.read_text()
_, later = text.split('## Model dependencies', 1)
later = later.split('\nThe visual notebook imports', 1)[0]
later = later.replace('The current local evidence covers the attention component and concept notebook.', 'The current local evidence covers the attention component and the Colab runner executed with a simulated file interface. Actual hosted execution remains unverified.')
colab.write_text('''# Running in Google Colab

[attention_colab.ipynb](../notebooks/attention_colab.ipynb) is the project runner. It transfers the current source into Colab and executes the Python files. The Attention Lab tutorial in Projects Docs contains the worked explanations and original figures; the old concept notebook has been removed.

## Open the notebook and source export

The current local export at workspace home contains:

```text
.ai/exports/attention-is-all-you-need/
  attention_colab.ipynb
  attention-is-all-you-need-source.zip
  source-manifest.json
```

The notebook is also in the project's `notebooks/` folder and available from the project reader. The ZIP includes current source, tests, configs, dependency files and public reference/Colab docs. It excludes captions, Git history, environments, private study records, data, weights and run output. The matching manifest identifies the source revision, uncommitted changes and hashes of the archive and every file. Use the ZIP and manifest from the same export.

1. Open [Google Colab](https://colab.research.google.com/) and choose **File → Upload notebook**. Upload `attention_colab.ipynb`.
2. Connect to a CPU runtime. Run the notebook from its first cell.
3. When the upload picker appears, select **both** `attention-is-all-you-need-source.zip` and `source-manifest.json` together.
4. Wait for dependency installation, then run the environment, demo and test cells in order.
5. Run the final cell and save `attention-colab-results.zip` before ending the runtime.

Colab supports uploading notebooks and runs code in temporary hosted machines; see its [official FAQ](https://research.google.com/colaboratory/faq.html). There is no public repository URL for this local working tree, so the notebook uses the reviewed source export.

## What the six code cells do

| Cell | Operation | Expected result |
| --- | --- | --- |
| Upload and verify | Match ZIP SHA-256, all file hashes and safe member paths before extraction | Fresh source directory under `/content/attention-colab-…/` |
| Install | Isolated uv 0.10.10 tools environment; `uv sync --frozen --extra notebook --python 3.12.13` | Project environment built from the supplied lock |
| Runtime | Query the installed project's Python, PyTorch, NumPy and Matplotlib | Actual environment versions recorded |
| Demo | Execute `src/attention.py` | Component tensors and `PASS` |
| Checks | Execute pytest with a machine-readable report | All 35 current tests pass; failures/skips stop the run |
| Download | Save source identity, environment, demo/test logs and receipt | `attention-colab-results.zip` |

The `notebook` dependency extra is retained for the visualization/math checks; it includes Matplotlib, ipywidgets and local Jupyter tools. The Colab kernel only coordinates subprocesses. It never imports the project's Python modules, so its preinstalled PyTorch does not substitute for the locked environment. uv's [locking/sync documentation](https://docs.astral.sh/uv/concepts/projects/sync/) explains frozen synchronization.

The Linux lock uses CPU PyTorch. Selecting a GPU does not change it. The expected demo shapes are `(3,5,9)` for encoder/decoder and `(3,7,9)` for cross-attention. These are initialized component outputs; they are not translated text or a task score.

## Reruns and failures

Run cells in order. If the source hashes disagree, obtain a matching ZIP/manifest pair and repeat the first cell. Each upload creates a new directory, so a changed source version is not mixed with a prior extraction. If installation fails, read the command's printed output and rerun that cell after correcting the runtime/network issue. A failed or skipped test stops before the success receipt is written.

After **Disconnect and delete runtime**, reconnect and rerun from the top, uploading the pair again. No Drive mount is required for this component check; download the logs before deleting a runtime. Local execution of these cells with a simulated upload/download interface is verified. I have not executed this notebook on a Google-hosted runtime, and no training/recovery result is claimed.

## Model dependencies''' + later)

sources = PROJECT / 'docs/SOURCES.md'
text = sources.read_text().replace('`notebooks/attention_mechanism.ipynb` explains and runs these stages, importing the implementations from the Python files.', 'The Attention Lab tutorial explains these stages. `notebooks/attention_colab.ipynb` runs the existing Python implementation and tests on Colab.')
text = text.replace('The notebook has thirty-seven executable cells and twenty-eight original figures.', 'The tutorial retains the original figures; the execution notebook contains six operational code cells.')
text = text.replace('## Visual notebook', '## Original tutorial figures')
text = text.replace('The notebook contains short display calls; plotting helpers, widget construction and auxiliary visualization calculations stay in that separate module.', 'Plotting helpers, widget construction and auxiliary visualization calculations stay in that separate module. The tutorial uses the authored figures; the Colab notebook handles execution only.')
start, _ = text.split('The original outer project namespace has been replaced', 1)
text = start + 'The original outer project namespace has been replaced with responsibility packages under `src/`: `core` (attention/model/config), `data` (tokenizers/datasets), `training` (trainer/checkpoints) and `utils` (figures/controls). Six runnable scripts live directly under `src/`, with no `cli/` folder. The execution notebook is `notebooks/attention_colab.ipynb`. Imports and launchers use this structure; upstream repository paths in sources.json remain unchanged because they identify the original reference. The attention calculation is unchanged; Videos 2–3 remain pending.\n'
sources.write_text(text)

setup = COURSE / 'lessons/02-project-setup.md'
text = setup.read_text()
text = text.replace('Each group is importable after `uv sync`; notebooks import that same source instead of defining a second implementation.', 'Each group is importable after `uv sync`. The Colab notebook runs the Python entry point and tests in their own locked environment.')
text = text.replace('| `notebooks/attention_colab.ipynb` | Explains and calls the Python implementation | Saved examples, figures and live controls |', '| `notebooks/attention_colab.ipynb` | Runs the Python project on Google Colab | Source upload/verification, locked setup, demo, tests and downloadable logs |')
start, rest = text.split('### Add notebook tools only for the notebook', 1)
_, end = rest.split('### Diagnose a setup problem at its source', 1)
text = start + '''### Run the project on Google Colab

The [notebook](code/notebooks/README.md) is an execution runner. Open `notebooks/attention_colab.ipynb` using **File → Upload notebook** in [Google Colab](https://colab.research.google.com/), select CPU, and run the cells in order. At the prompt, upload the matching source ZIP and manifest from workspace `.ai/exports/attention-is-all-you-need/`. The export also contains a standalone copy of the notebook.

Its six code cells verify source hashes and paths, create a fresh extraction, install uv 0.10.10 and the locked Python 3.12.13 environment, record actual versions, launch `src/attention.py`, run all component/matrix checks, and download a result archive. The notebook kernel coordinates subprocesses; it does not import Colab's preinstalled PyTorch as the project runtime. The current lock selects CPU PyTorch. Actual hosted execution is still unverified; local checks exercise the operational cells with a simulated Colab file interface.

The former concept notebook has been removed. Read this tutorial for the worked explanations, maths and original figures. Graph helpers remain under `src/utils/`; there is one numerical implementation under `src/core/`.

For a complete local test run, include the existing visualization dependencies:

```bash
uv sync --frozen --extra notebook
uv run --frozen --extra notebook pytest -q
```

With that extra, all 35 tests should run: 25 attention cases and 10 visualization-maths cases. A run reporting a skipped visualization module has not checked those ten cases. The extra retains local Jupyter tools as well as Matplotlib and ipywidgets, but opening a local notebook is no longer part of the study route.

### Diagnose a setup problem at its source''' + end
setup.write_text(text)

change(COURSE / 'project/learning-points/00-setup.md',
       'Implementation stays in Python modules. The notebook imports those same classes, and the separate visualization module handles drawing and controls. This division lets the explanatory notebook evolve without introducing a second model implementation. Saved figures remain readable without executing cells; live widget interaction requires a kernel.',
       'Implementation stays in Python modules. The tutorial owns explanations and figures; the notebook uploads source and runs the Python project on Colab. Separate visualization utilities supply the original drawings and matrix fixtures. The runner records the environment actually used, and its logs distinguish component execution from later training or recovery.')
change(COURSE / 'project/comparisons/00-setup.md', '| Concept notebook | Inspect calculations, figures and controls | Saved output is separate from a fresh execution |', '| Colab execution notebook | Transfer verified source, run demo/tests and download logs | A local runner check is separate from actual hosted execution |')
change(COURSE / 'project/comparisons/00-setup.md', 'notebook for controlled exploration', 'Colab notebook for project execution')
change(COURSE / 'project/implementation.md', '| Figures and notebook | 28 saved figures, 37 freshly executed code cells | Fresh root/notebooks executions; no hosted execution |', '| Tutorial figures and Colab runner | Original figures in the tutorial; six operational notebook cells | Local runner verification; actual hosted execution pending |')
change(COURSE / 'project/readiness.md', 'Source modules are grouped by responsibility, and the notebook is relocated and freshly executed.', 'Source modules are grouped by responsibility, entry scripts are directly under src, and the notebook is a Colab execution runner. Its operational cells are locally verified; actual hosted execution remains pending.')
ref = COURSE / 'project/reference-runs.md'
text = ref.read_text()
start, rest = text.split('## Saved notebook evidence', 1)
_, end = rest.split('## Source preservation and limits', 1)
ref.write_text(start + '''## Colab runner evidence

The former 37-cell concept notebook has been removed. `notebooks/attention_colab.ipynb` contains six code cells for source verification, environment setup, runtime recording, demo execution, tests and downloadable logs. It is saved with cleared outputs. The tutorial retains the explanations and original figures.

Local verification exercises the exact cells with a simulated `google.colab.files` upload/download interface, a fresh extracted export and real uv synchronization. It runs the demo and all 35 tests, reads the result ZIP and probes rejection of corrupt archives and unsafe paths. That evidence covers the runner logic locally; it does not establish actual Google-hosted execution or Drive recovery.

## Source preservation and limits''' + end)

assembler = DOCS / 'implementation/2026-10/attention-is-all-you-need-source-backed-tutorial/08-assemble.py'
text = assembler.read_text()
text = text.replace('The notebook now lives in notebooks/, outside the responsibility packages core/data/training/utils, with entry scripts directly under src under src/. Its 37 code cells have been freshly executed from both root and notebooks working directories, producing 28 figures without cell errors.', 'The Colab execution notebook lives in notebooks/ outside the core/data/training/utils packages, with entry scripts directly under src/. Its six operational code cells verify/extract source, install the locked environment, record runtime identity, run demo/tests and download logs. Tutorial prose and figures replace the former concept notebook. Local runner verification is separate from an actual hosted run.')
text = text.replace('The notebook paths/imports were updated and all cells reexecuted after the source reorganization.', 'The former concept notebook was replaced with a Colab project runner; its operational cells are checked locally with simulated upload/download and real commands.')
text = text.replace('The notebook lives in notebooks/ and all 37 code cells execute from both project-root and notebooks working directories, producing 28 figures.', 'The six-cell Colab runner lives in notebooks/attention_colab.ipynb; local verification runs the exported source, demo and tests through its operational cells.')
assembler.write_text(text)

browser = HERE / '13-browser-review.mjs'
text = browser.read_text().replace(".length,37)", ".length,6)")
text = text.replace("assert.ok(JSON.stringify(notebook.cells).includes('from core.attention import'));", "assert.ok(JSON.stringify(notebook.cells).includes('from google.colab import files'));\n  assert.ok(JSON.stringify(notebook.cells).includes('src/attention.py'));\n  assert.ok(notebook.cells.filter(c=>c.cell_type==='code').every(c=>c.execution_count===null&&c.outputs.length===0));")
browser.write_text(text)
export = HERE / '12-export-source.py'
text = export.read_text().replace("'src/core,data,training,cli,utils; notebooks outside src'", "'src/core,data,training,utils; entry scripts directly in src; Colab runner outside src'")
text = text.replace("archive=EXPORT/'attention-is-all-you-need-source.zip'", "shutil.copyfile(PROJECT/'notebooks/attention_colab.ipynb',EXPORT/'attention_colab.ipynb')\narchive=EXPORT/'attention-is-all-you-need-source.zip'")
export.write_text(text)

# Refresh compact current snapshots without rewriting the historical dated ledger.
for path in [ROOT/'README.md', ROOT/'AGENTS.md', ROOT/'ai_projects/README.md', ROOT/'ai_projects/AGENTS.md', DOCS/'README.md', DOCS/'AGENTS.md']:
    text = path.read_text()
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith('Attention source organization (2026-10-23):'):
            link = line.split('[Setup/tutorial]',1)[1].split(')',1)[0] + ')' if '[Setup/tutorial]' in line else ''
            lines[index] = 'Attention source organization (2026-10-23): the four responsibility folders are src/core, src/data, src/training and src/utils. Six runnable scripts live directly in src; src/cli and the redundant outer package are removed. notebooks/attention_colab.ipynb replaces the concept notebook and runs verified source setup, the demo and all 35 tests on Colab, with downloadable logs. The tutorial retains explanations and figures. Local runner checks are separate from actual hosted execution; Parts 2-3, learner review and hosted recovery remain pending.'
    path.write_text('\n'.join(lines)+'\n')

for name in ('resume', 'progress', 'roadmap'):
    path = DOCS / f'guide/{name}.md'
    text = path.read_text()
    first, remainder = text.split('\n',1)
    note = '''
## Attention runner and source layout — current, 2026-10-23

Six runnable scripts now live directly under src; core/data/training/utils remain grouped below it. src/cli is removed. The concept notebook is replaced by notebooks/attention_colab.ipynb, which uploads verified source, installs the locked Python environment, runs the demo and all 35 checks, and downloads logs. The Attention Lab tutorial retains explanations, worked maths and original figures. Current docs, source downloads and launcher paths follow this structure. Local operational-cell verification is distinct from an actual Google-hosted run. Later translation implementation, learner review and hosted recovery remain pending. This supersedes the earlier grouped-source notebook/cli layout below.
'''
    path.write_text(first+'\n'+note+remainder)
with (DOCS / 'guide/decisions.md').open('a') as file:
    file.write('''
## 2026-10-23 — direct Attention entry scripts and Colab execution notebook

The user moved runnable commands out of src/cli directly into src while retaining core/data/training/utils responsibility folders. Replace the concept notebook with notebooks/attention_colab.ipynb for source upload/verification, locked setup, demo/tests and downloadable logs. The tutorial owns conceptual teaching and original figures. Update current paths, resource metadata, source bundles and docs together. Keep the notebook output-free; distinguish local runner verification from actual hosted execution and future training/recovery. This supersedes the cli/concept-notebook portions of the earlier organization decision.
''')
print('Updated active project/tutorial documentation and export/browser checks.')
