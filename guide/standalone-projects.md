# Run projects independently or through the workspace

Each collection is an independent repository. Every runnable child project owns its dependencies, lock, configuration and native entry points. The workspace CLI/TUI calls those same entry points and is optional. A standalone checkout needs Git, uv and the project's Python dependencies; it does not need `ai_projects`, sibling repositories, `AI_WORKSPACE` or an injected `PYTHONPATH`.

## Native commands

Enter the project folder and run `uv sync --frozen` first. The commands below run without the workspace CLI. `uv build` builds each runnable project. Model weights and datasets must still be acquired or prepared through that project's own documented steps before model/application actions that need them.

| Collection / project | Direct run from the project folder | Optional workspace command from workspace home |
| --- | --- | --- |
| Inference Labs / tinyserve | `uv run --frozen python -m tinyserve status` | `./ai run tinyserve --entrypoint status` |
| Attention Labs / attention-foundations | `uv run --frozen attention-foundations status` | `./ai run attention-foundations --entrypoint status` |
| Quantization Lab / document-assistant | `uv run --frozen document-assistant status` | `./ai run document-assistant --entrypoint status` |
| Quantization Lab / compressed-semantic-search | `uv run --frozen compressed-semantic-search status` | `./ai run compressed-semantic-search --entrypoint status` |
| Transformer Labs / transformer-foundations-lab | `uv run --frozen python first_prediction.py` | `./ai run transformer-foundations-lab --entrypoint prediction` |
| Transformer Labs / attention-is-all-you-need | `uv run --frozen attention-demo` | `./ai run attention-is-all-you-need --entrypoint attention` |
| Transformer Labs / support-ticket-router | `uv run --frozen python demo.py` | `./ai run support-ticket-router --entrypoint demo` |
| Foundations / bike-demand-planner | `uv run --frozen python demo.py` | `./ai run bike-demand-planner` |

All except Bike use `uv run --frozen python -m pytest -q` for their current tests. Bike uses `uv run --frozen python -m unittest discover -s tests -v`. The foundations learner scaffold has no authored tests yet; its saved arithmetic program runs, while pytest correctly reports no tests. The CLI uses each project's declared native test arguments when configured; Bike declares unittest in `pyproject.toml` without adding pytest as a dependency.

## Inference Labs outside this workspace

Clone the Inference Labs repository, enter `tinyserve`, then run:

```bash
uv sync --frozen
uv run --frozen python src/check_setup.py
uv run --frozen python -m tinyserve status
uv run --frozen tinyserve config
uv run --frozen pytest -q
uv build
```

TinyServe's model, cache, scheduler, API and kernel implementation remain learner-owned. Setup commands describe that pending state rather than simulate inference.

## Verification — 2026-10-31

All eight runnable project roots passed frozen installation, a safe native run and package build in clean source copies with workspace environment variables removed. The seven authored native suites passed **237 tests**, with one optional notebook-visual module skipped because notebook extras were not installed. The foundations learner scaffold has no authored suite. All eight safe launch checks also passed through the workspace adapter from `/tmp`.

A separate fresh clone of the Inference Labs repository outside the workspace passed TinyServe setup, source/module/console commands, its 10 tests and package build, with no parent umbrella or CLI present. The CLI working-tree suite passed **185 tests** and TUI smoke passed. A new regression runs a generated project through the workspace adapter and from an independent project copy.

Detailed local receipts are in workspace `.ai/verification/standalone-projects/`; the [setup continuation record](../../../cli/docs/implementation/2026-10/tinyserve-boilerplate-and-inference-labs-tui/00-resume.md) records scope and limits.

## Projects still at the planning stage

Attention Heads, Semantic Notes Search, Tiny Vision Transformer, E-commerce Purchase Predictor, Waste Material Classifier and Air Quality Forecaster currently contain plans rather than runnable project boilerplates. Their runtime/model implementation remains pending. Preparing any of these projects must include both native and optional workspace launch paths before its setup is called ready.
