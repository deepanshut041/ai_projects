# AI Projects

All shared Projects documentation belongs under `docs/`. Begin with docs/README.md, docs/resume.md, docs/progress.md, docs/roadmap.md and docs/decisions.md. README.md and AGENTS.md are the only visible regular files permitted in this root. Do not create root-level tracking/setup/tutorial files, legacy roadmap copies or a duplicate launcher. Use the shared ../ai launcher from this directory, or ./ai from workspace home. Preserve the hidden Git/skill configuration and independent repository directories.

## Roadmap and continuation context

Paths in this file are relative to the Projects root. Current embedded checkpoint: **2026-10-09 — P00 Tokenization and Transformer Foundations, P00-M1 not started**. The source-backed plan adds tokenizer/attention/complete encoder–decoder foundations before **P01 Small GPT Language Model**; exact GPT-1/classification is optional historical study. Workspace/docs/CLI/seven-skill preparation and the earlier reference tutorial remain available. Colab Free training/recovery requirements now cover P00–P04. No foundation/GPT repository, model, notebook, training or learner mastery exists; seed labs do not complete P00. Local repository preparation and the P00 revision are covered by the [commit review](docs/implementation/2026-10/ai-workspace-setup/07-pre-p01-commit-review.md).

At the start of every project session:

1. Read [docs/resume.md](docs/resume.md) for the current project, milestone and next action, then [docs/progress.md](docs/progress.md) for completed work, evidence and pending learning gates.
2. Read the relevant part of [docs/roadmap.md](docs/roadmap.md) and [docs/decisions.md](docs/decisions.md). Follow its sequence/prerequisites and the complete-model learning cycle; keep narrow Research separate.
3. Inspect current Git state, the target child's README/AGENTS, and its existing package's resume/tracker. Reconcile those with the canonical checkpoint and the latest user instruction. Continue existing work; do not recreate a repository/package or redo setup just because this chat is new.
4. State briefly which project/milestone is active, what relevant work is already done and the concrete next action. Read the linked evidence before treating a gate as passed; separate implemented/tested/trained work from learner understanding.

For current P00-M1, use [docs/START_HERE.md](docs/START_HERE.md): create/resume projects/transformer-foundations-lab when implementation starts; pin sources/runtime, check tensor/autograd/loss/optimizer basics and declare CPU/Free data/storage/budgets. The roadmap then covers tokenizer, attention and full conditional-model training. Earlier GPT-1 tutorial availability does not complete P00 or P01.

Develop/check locally; P00–P04 must provide complete miniature-model training on Colab Free. Read [docs/COLAB.md](docs/COLAB.md) and carry its thin common-trainer notebook, complete persistent training-state checkpoints, interrupted-save fallback and recovery after runtime deletion into the child's package. Actual notebooks and hosted proof remain pending; later tiers are chosen from evidence, with Pro later when requested.

Before ending meaningful work, update the child package plus docs/progress.md and docs/resume.md with actual changes, checks, learner observations, unresolved questions and one next action. Append durable changes to docs/decisions.md. When the current project, milestone/status, completed-work summary or next action changes, refresh the compact snapshots in Projects README/AGENTS, docs README/AGENTS and workspace-home README/AGENTS. Canonical docs prevail over stale embedded snapshots; keep detailed history/evidence under docs/.

The independent `docs/` submodule owns the documentation tree and website. Independent model implementations are submodules under `projects/`. Research belongs in ../ai_research/; shared CLI tooling belongs in ../cli/. Preserve existing edits and repository history, and read each child's AGENTS.md before editing.

C++ and Python are the project and tutorial programming choices. Preserve hands-on implementation, small-model training, inspection/debugging and explanation as separate learning gates. Numerical changes need analytic answers or an independent reference, failure checks and stated tolerances; training loss alone does not establish model quality or understanding.

Use the shared planning skill at .agents/skills/planning/SKILL.md for substantial work. The collection links to ../ai_research/research_docs/.agents/skills/; do not duplicate it. Create/resume shared documentation work under docs/implementation/ with ../ai task new TITLE --owner projects or --owner project-docs. CLI and child work use their own owners. Update docs/progress.md and docs/resume.md after meaningful work.

Tutorials belong in docs/tutorials/ and are Git-eligible. Follow the shared developer-learning-tutorials and feature-whiteboard-defense skills. Runtime state, environments, weights, datasets and generated runs stay ignored. Local work is authorized; remote publication and deployment need a user instruction.
