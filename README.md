# AI Projects

Start in [docs/README.md](docs/README.md). All shared Projects documentation lives under `docs/`: the roadmap, checkpoints, setup guides, tutorials, implementation records and verification evidence.

## Current project context

Checkpoint: **2026-10-09 — P00 Tokenization and Transformer Foundations, P00-M1 not started.** Re-read [docs/resume.md](docs/resume.md) and [docs/progress.md](docs/progress.md) at session start; they override an older summary here. Local repository preparation and the P00 revision are covered by the [commit review](docs/implementation/2026-10/ai-workspace-setup/07-pre-p01-commit-review.md).

- Completed preparation: consolidated roadmap/docs, independent CLI, seven skills, the preserved [GPT-1 reference tutorial](docs/tutorials/projects/p01/gpt-1-foundations/v1/lesson.md), and the new [source-backed P00/P01 plan](docs/implementation/2026-10/project-zero-foundations-and-small-gpt-roadmap/02-implementation-plan.md). Learner understanding remains unassessed.
- Model progress: no foundation/GPT implementation repository or training run. Seed labs are infrastructure examples and do not complete P00.
- Next: at implementation kickoff create/resume `projects/transformer-foundations-lab/` and complete P00-M1 tensor/autograd/loss/optimizer checks plus source/runtime/CPU/Free budget contracts. Follow [START_HERE.md](docs/START_HERE.md).
- Learning path: **P00 tensors/tokenizer/attention/full encoder–decoder → P01 Small GPT Language Model → GPT-2 → Llama → Qwen Dense → MoE**, then the [remaining branches](docs/roadmap.md#learning-path-and-prerequisites). Exact GPT-1/classification is optional historical study. Work through understand → implement → train → inspect/debug → explain → profile → verify.
- Compute: [Colab Free training/recovery requirements](docs/COLAB.md) cover P00–P04; thin notebooks, full persistent checkpoints and actual hosted proof remain pending.

Implement and train established models in independent repositories under `projects/`. Use C++ and Python. Focused improvement studies belong in [AI Research](../ai_research/README.md); the shared workspace CLI lives in [../cli/](../cli/README.md).

From this directory run `../ai roadmap` to resume or `../ai start project-docs` to open the documentation site. The shared launcher belongs at workspace home. Read [AGENTS.md](AGENTS.md) before editing. README.md and AGENTS.md are the only visible root files; all other documentation belongs under `docs/`.

At each session end update docs/progress.md and docs/resume.md. Refresh the six short README/AGENTS snapshots when the current project, milestone/status, completed-work summary or next action changes; keep detailed tracking and evidence under docs/.
