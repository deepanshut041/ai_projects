## Reproduce the existing project and read its boundaries

We have a concrete result to reproduce: equal attention over two values gives `[4,6]`. Now we need to know where that function lives, which environment runs it, and which apparent project entry points are still empty. This is a walkthrough of the existing setup, not an instruction to create another project.

### Use the existing ordinary project folder

From workspace home, the folder is `ai_projects/projects/transformer-labs/attention-is-all-you-need/`. It is tracked by the `ai_projects` repository. Continue the existing project. Its implementation is grouped directly below `src/`: `core`, `data`, `training` and `utils`, alongside direct runnable scripts. The former outer project-named namespace has been removed; installation exposes these responsibility packages.

```bash
cd /mnt/projects/ai/ai_projects/projects/transformer-labs/attention-is-all-you-need
uv sync --frozen
uv run --frozen python src/attention.py
uv run --frozen pytest -q
```

The first command selects the project as the working directory. `uv sync --frozen` installs the versions already recorded in `uv.lock` without resolving a new dependency set. It can create or reconcile the ignored `.venv/`; it does not train a model. The next command runs the demo. The last runs the numerical and interface tests available in the environment. The locked Python requirement is 3.12; `.python-version`, `pyproject.toml` and `uv.lock` travel together.

Linux currently uses a CPU PyTorch build. A machine having a GPU does not make this environment a GPU training environment. The checked runtime is PyTorch 2.14.1+cpu. Later training needs an explicit device/environment decision and measurement.

### Keep code grouped by responsibility

The original project-named directory under src was a conventional namespace for installation. This application now uses responsibility packages directly below src, as requested. `core` owns attention/model/configuration; `data` owns tokenizers and dataset handling; `training` owns fitting/recovery; `utils` owns graph/display helpers. Runnable commands live directly under `src/`, with no `cli/` folder. Each group is importable after `uv sync`. The Colab notebook runs the Python entry point and tests in their own locked environment.

```text
src/
  attention.py              Runnable attention demo
  tokenizer.py              Future tokenizer preparation command
  prepare_data.py           Future dataset preparation command
  train.py                  Future training command
  inference.py              Future translation command
  compute_bleu.py           Future BLEU command
  core/                     attention.py, model.py, config.py
  data/                     tokenizer.py, dataset.py
  training/                 trainer.py, checkpoint.py
  utils/                    attention_visuals.py
notebooks/
  attention_colab.ipynb
```

Hatch's source mapping installs these groups without an extra outer project namespace. The console command `uv run --frozen attention-demo` and the file command `uv run --frozen python src/attention.py` call the same demo. Plot helpers keep generated images in the source project's ignored `runs/attention-visuals/` directory; `configure(output_dir=...)` selects an explicit destination for an installed standalone copy.

### Know which file owns which responsibility

| File | What it does now | What to inspect |
| --- | --- | --- |
| `src/attention.py` | Exposes the attention classes and runs the demonstration | `run_demo`, its fixed seed, shape checks and path comparison |
| `src/core/attention.py` | Implements the reusable numerical component | Projections, head packing, masks, attention and output projection |
| `src/utils/attention_visuals.py` | Builds figures and interactive displays | Hand-set numerical examples and plotting helpers |
| `notebooks/attention_colab.ipynb` | Runs the Python project on Google Colab | Source upload/verification, locked setup, demo, tests and downloadable logs |
| `tests/test_attention.py` | Checks values, gradients, masks and interfaces | Independent PyTorch reference and perturbation cases |
| `tests/test_attention_visuals.py` | Checks the matrix examples | Known answers and preservation of random state |
| `src/tokenizer.py`, `src/prepare_data.py`, `src/train.py`, `src/inference.py`, `src/compute_bleu.py` | Comment-only destinations for later videos | No tokenizer, data processing, training, translation or BLEU execution yet |

The modules `src/core/model.py`, `src/core/config.py`, `src/data/dataset.py`, `src/data/tokenizer.py`, `src/training/trainer.py` and `src/training/checkpoint.py` are also placeholders. A successful exit from an empty Python script does not establish that its named function works. Read its source before interpreting a launcher result.

### Read the demo's output as evidence

With seed 7, the demo prints the weighted average, attention on raw inputs, separate/vectorized head outputs, causal weights and cross-attention. Encoder and decoder attention outputs have shape `(3,5,9)`; the cross-attention output is `(3,7,9)`. Its largest reported explicit/SDPA difference in this CPU run is approximately `1.19e-7`.

`PASS` means the selected numerical comparisons and perturbation assertion succeeded. The printed floating-point vectors come from randomly initialized attention parameters. They are not translated tokens or a measured task score. We have a working component demonstration and independent tests, not a fitted language system.

### Run the project on Google Colab

The Colab notebook is an execution runner. Open `notebooks/attention_colab.ipynb` using **File → Upload notebook** in [Google Colab](https://colab.research.google.com/), select CPU, and run the cells in order. At the prompt, upload the matching source ZIP and manifest from workspace `.ai/exports/attention-is-all-you-need/`. The export also contains a standalone copy of the notebook.

Its six code cells verify source hashes and paths, create a fresh extraction, install uv 0.10.10 and the locked Python 3.12.13 environment, record actual versions, launch `src/attention.py`, run all component/matrix checks, and download a result archive. The notebook kernel coordinates subprocesses; it does not import Colab's preinstalled PyTorch as the project runtime. The current lock selects CPU PyTorch. Actual hosted execution is still unverified; local checks exercise the operational cells with a simulated Colab file interface.

The former concept notebook has been removed. Read this tutorial for the worked explanations, maths and original figures. Graph helpers remain under `src/utils/`; there is one numerical implementation under `src/core/`.

For a complete local test run, include the existing visualization dependencies:

```bash
uv sync --frozen --extra notebook
uv run --frozen --extra notebook pytest -q
```

With that extra, all 35 tests should run: 25 attention cases and 10 visualization-maths cases. A run reporting a skipped visualization module has not checked those ten cases. The extra retains local Jupyter tools as well as Matplotlib and ipywidgets, but opening a local notebook is no longer part of the study route.

### Diagnose a setup problem at its source

If `torch` cannot be imported, verify that the command ran through the project's `uv` environment. If the package cannot be imported, check the working directory and installation before adding arbitrary `sys.path` edits. If a matrix shape is rejected, the later lessons explain the accepted tensor contract. Do not resolve a numerical failure by changing expected values until the calculation is understood.

The [reference-run record](project/reference-runs.md) distinguishes commands actually checked from proposed commands. The [setup learning points](project/learning-points/00-setup.md) and [setup comparisons](project/comparisons/00-setup.md) explain environment versus model readiness.

[[practice:setup-check]]

Next we will read the tensor axes, because most attention mistakes are understandable once we can say which axis represents queries, keys and features.
