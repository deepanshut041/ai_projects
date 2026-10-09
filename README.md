# AI Projects

Start in [docs/README.md](docs/README.md). All shared Projects documentation lives under `docs/`: the roadmap, checkpoints, setup guides, tutorials, implementation records and verification evidence.

## Current project context

Checkpoint: **2026-10-09 — P01 GPT-1, P01-M1 not started.** Re-read [docs/resume.md](docs/resume.md) and [docs/progress.md](docs/progress.md) at session start; they override an older summary here.

- Completed preparation: consolidated roadmap/docs, independent shared CLI, seven validated skills, the [GPT-1 foundations tutorial](docs/tutorials/projects/p01/gpt-1-foundations/v1/lesson.md) with checked C++/Python examples, and [Colab Free training/recovery requirements](docs/COLAB.md) for P01–P04. Adaptive tutorial/live-teaching guidance and the P01 prerequisite/transfer route are implemented; learner understanding remains unassessed. The [pre-P01 local commit review](docs/implementation/2026-10/ai-workspace-setup/07-pre-p01-commit-review.md) records repository preparation and current checks.
- Model progress: no GPT implementation repository or training run yet. The retained seed labs are infrastructure examples, not completed roadmap projects.
- Next: create/resume `projects/gpt-lab/`, then complete M1's source/architecture map, tiny config/tokenizer/data and Colab Free setup/storage/budget contracts. M3 includes the common-trainer notebook and full persistent resume; no hosted checks are complete. Follow [the kickoff guide](docs/START_HERE.md).
- Learning path: GPT-1 → GPT-2 → Llama → Qwen Dense → MoE, then the [roadmap's remaining branches](docs/roadmap.md#learning-path-and-prerequisites). Work through understand → implement → train → inspect/debug → explain → profile → verify.

Implement and train established models in independent repositories under `projects/`. Use C++ and Python. Focused improvement studies belong in [AI Research](../ai_research/README.md); the shared workspace CLI lives in [../cli/](../cli/README.md).

From this directory run `../ai roadmap` to resume or `../ai start project-docs` to open the documentation site. The shared launcher belongs at workspace home. Read [AGENTS.md](AGENTS.md) before editing. README.md and AGENTS.md are the only visible root files; all other documentation belongs under `docs/`.

At each session end update docs/progress.md and docs/resume.md. Refresh the six short README/AGENTS snapshots when the current project, milestone/status, completed-work summary or next action changes; keep detailed tracking and evidence under docs/.
