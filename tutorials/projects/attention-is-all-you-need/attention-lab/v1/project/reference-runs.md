# Current component commands and reference evidence

The working directory for project commands is `ai_projects/projects/transformer-labs/attention-is-all-you-need/` from workspace home. `uv sync --frozen --extra notebook` reconciles the locked local environment; the installed environment used for this audit already contained the optional visualization dependencies.

## Existing tests and demo checked in this conversation

```bash
uv run --frozen --no-sync pytest -q
uv run --frozen --no-sync python src/attention.py
```

Observed: **35 passed**, with no skipped visualization module. The CPU demo prints `PASS`. With seed 7 and PyTorch 2.14.1+cpu, its encoder/decoder maximum explicit/SDPA difference is `1.19e-7`; cross-attention is `8.94e-8`. These are differences in the component outputs, not timing or translation metrics. `--no-sync` reuses the existing installed environment; use the documented sync first on a fresh checkout.

## Authored arithmetic and module probes

[arithmetic.py](examples/arithmetic.py) is a standard-library-only, independently written calculation of the course's weighted sums, matrix products, softmax identities, controlled projections, head merge and scalar derivative/update. The course publisher executes it and freezes its declared output in `example-checks.json`.

[verify_component.py](examples/verify_component.py) compares the independent arithmetic with the frozen actual module using PyTorch, checks finite differences, invalid interfaces, mask perturbations, shape contracts and selected parameter counts. Run it with the project environment and the tutorial file path as documented in that script's header. It is a component audit; it does not train a model or call the placeholder entry points.

## Colab runner evidence

The former 37-cell concept notebook has been removed. `notebooks/attention_colab.ipynb` contains six code cells for source verification, environment setup, runtime recording, demo execution, tests and downloadable logs. It is saved with cleared outputs. The tutorial retains the explanations and original figures.

Local verification exercises the exact cells with a simulated `google.colab.files` upload/download interface, a fresh extracted export and real uv synchronization. It runs the demo and all 35 tests, reads the result ZIP and probes rejection of corrupt archives and unsafe paths. That evidence covers the runner logic locally; it does not establish actual Google-hosted execution or Drive recovery.

## Source preservation and limits

Source and relevant input hashes are recorded in the authoring work package. The frozen source copy permits inspection without depending on live source files. Random layer outputs depend on source/runtime/seed; tight numerical agreement is tested with controlled parameters, and exact cross-device or cross-version reproducibility is not promised. Model quality, learned alignments, GPU speed and hosted recovery remain unmeasured.
