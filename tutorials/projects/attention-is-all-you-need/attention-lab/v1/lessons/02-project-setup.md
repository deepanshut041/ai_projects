## Reference — run and inspect the existing attention project

Use this page when you want to execute the calculation from [the first lesson](lessons/01-weighted-information.md). The conceptual route is readable without environment setup.

### Run the implemented component and all existing checks

```bash
cd /mnt/projects/ai/ai_projects/projects/transformer-labs/attention-is-all-you-need
uv sync --frozen --extra notebook
uv run --frozen python src/attention.py
uv run --frozen --extra notebook pytest -q
```

The project is an existing ordinary folder tracked by `ai_projects`. `uv sync --frozen` uses its lockfile rather than resolving a new dependency set. The notebook extra supplies the visualization dependencies so the full set includes all 25 attention and 10 visualization-maths cases. A skipped visualization module does not check those ten cases.

The locked environment uses Python 3.12.13 and CPU PyTorch 2.14.1. A GPU being installed does not change the selected runtime. This command runs a component demo, not a translator.

### Run one complete teaching program

The ten displayed Python programs include their own imports, inputs and called helper definitions. The local copies live in `docs/tutorial-assets/walkthrough/`:

```bash
uv run --frozen python docs/tutorial-assets/walkthrough/01_weighted.py
uv run --frozen python docs/tutorial-assets/walkthrough/10_interface.py
```

The first prints the weighted contributions and context `[1.5,1]`. The last reproduces the all-position attention result through the complete actual class and checks its explicit/library paths.

For an exported tutorial package, [the restore script](examples/restore_source.py) rebuilds the captured project described by [the source bundle](code/source-bundle.json). From the tutorial package root:

```bash
python3 examples/restore_source.py restored-project
uv run --project restored-project --frozen python examples/walkthrough/01_weighted.py
```

### Find the owner of a behavior

| Location | Responsibility |
| --- | --- |
| `src/core/attention.py` | Actual attention formula, projections, head layout, masks and interface |
| `src/attention.py` | Runnable component demo; also available as `attention-demo` |
| `src/utils/attention_visuals.py` | Original matrix examples and display helpers |
| `tests/test_attention.py` | Values, gradients, masks, interface and independent PyTorch-reference checks |
| `tests/test_attention_visuals.py` | Known answers for numerical visualization fixtures |
| `notebooks/attention_colab.ipynb` | Source upload/verification, locked environment, demo/tests and downloadable logs |

Runnable scripts live directly under src; reusable groups are `core`, `data`, `training` and `utils`. The tokenizer/data/model/trainer/checkpoint destinations are placeholders for later implementation. An empty named script exiting successfully does not establish that its advertised model action exists.

### Interpret the demo output

The demo uses a fixed seed and prints weighted sums, head calculations, causal weights and cross-attention. Its random-parameter encoder/decoder outputs have shape `(3,5,9)` and cross-attention has `(3,7,9)`. Printed vectors are feature activations, not translated words. Path-agreement checks compare identical computations; they do not measure translation quality or hardware speed.

### Use the existing Colab runner

Upload `notebooks/attention_colab.ipynb` into [Google Colab](https://colab.research.google.com/) and follow its source-archive/manifest upload instructions. Matching local exports live in workspace `.ai/exports/attention-is-all-you-need/`. Its six operational code cells verify files, extract source, install the locked runtime, record versions, run the demo/tests and download logs. The notebook kernel coordinates subprocesses rather than silently substituting Colab's preinstalled PyTorch.

Previous checks exercised those operational cells locally with a simulated Colab file interface. Actual Google-hosted execution and recovery after runtime deletion remain unverified.

### Diagnose a failure at the responsible boundary

If torch cannot import, inspect the selected project environment. If the package cannot import, check the working directory and installation. If dimensions fail, use the tensor contracts in chapters 3 and 10. If a numerical assertion fails, trace the intermediate inputs and operation before changing expected values. Do not treat a successful setup command as a training result.

The [reference record](project/reference-runs.md) identifies author checks, including [independent scalar arithmetic](examples/arithmetic.py) and [actual-component probes](examples/verify_component.py). The [setup learning points](project/learning-points/00-setup.md) and [setup comparisons](project/comparisons/00-setup.md) remain available for environment review. These optional checks support the written route; they are not prerequisite lessons.

[[practice:setup-check]]
