# Start your first project

Updated: 2026-10-09
Current position: **P01 GPT-1 → P01-M1: sources, architecture, configuration and data contracts.**

The Projects workspace is prepared for kickoff. GPT-1 implementation and training have not started. Begin with a complete tiny CPU model, then run bounded training on Colab Free. Learn by implementing, training, inspecting/debugging and explaining each step. P01–P04 require the [Colab training and recovery contract](COLAB.md); notebooks and hosted checks are pending implementation.

## Read these first

1. [resume.md](resume.md): current milestone and one next action.
2. [roadmap.md](roadmap.md): sequence, prerequisites and completion gates.
3. [progress.md](progress.md): status and evidence; keep it current.
4. [GPT-1 starting plan](implementation/2026-10/llm-roadmap-review-and-gpt-1-starting-plan/02-implementation-plan.md): concrete model/data contracts and bounded learning checks.
5. [COLAB.md](COLAB.md): common trainer/notebook, Free profile, persistent full-state checkpoints and fresh-runtime resume gates.

[decisions.md](decisions.md) records the durable choices. Known-model learning belongs here; narrow improvement/reproduction studies belong in [Research](../../ai_research/roadmap.md). Shared tooling remains in ../../cli/.

## Kick off when you start

Run these commands from workspace home. If your current directory is the Projects root, use ../ai in place of ./ai:

```bash
./ai roadmap
./ai new project gpt-lab --title "GPT Learning Lab"
./ai task new "GPT-1 implementation and learning" --owner gpt-lab --domains project,experiment
```

These creation commands are for first kickoff only. If gpt-lab or its topic package already exists, read its README/AGENTS, inspect its Git state and resume the existing package. Do not recreate/reset it. The CLI preserves existing destinations and packages.

The generic scaffold has no model dependencies or meaningful tests yet. During M1, add and lock the CPU PyTorch runtime and needed test dependencies in the child, then verify a small tensor/autograd operation. The scaffold's test command intentionally fails until meaningful tests are added; that is not evidence of a completed model. Do not install model dependencies into the shared CLI.

Read the [P01 tutorial](tutorials/projects/p01/gpt-1-foundations/v1/lesson.md) before the first model session. Work through the examples and draw the architecture; this reading does not complete M1. Tutorials and whiteboard notes are versioned here under tutorials/, using C++ and Python.

## First learning milestone

- Pin the GPT-1 primary source already identified in the starting plan.
- Draw the model and annotate token/position/attention/logit shapes.
- Declare the tiny config: 2 layers, width 64, 4 heads, FFN 256, context 64 and 260 byte/special token IDs.
- Work through a short next-token target example before implementing the tokenizer/batch contract.
- Record simplifications, numerical checks and bounded run settings in the child-owned package.
- Declare a Colab Free profile, dependency setup, exact child-source ZIP delivery (current remotes are local), Drive run root and session/checkpoint budgets. Keep CPU correctness first; do not require the workspace CLI in Colab.

During M3 implement the thin `notebooks/colab.ipynb` and the common trainer's explicit new/resume path. Save model, optimizer, scheduler/scaler when used, RNG and next-data position to verified persistent checkpoints. Test interrupted saves and resume after deleting the Colab runtime before calling the project Colab-ready. See COLAB.md for acceptance evidence; no hosted behavior is verified yet.

Build one component at a time. Keep model code and learning exercises for the project sessions; a finished generated implementation is not a substitute for understanding.

## Keep the place in the roadmap

At each session end update the child package, progress.md and resume.md with actual evidence, what you understood, unresolved questions and one next step. Passing tests, learner understanding and publication remain separate states. Store data in data/, checkpoints in checkpoints/ or runs/, and weights in weights/; these stay ignored.

The [readiness audit](implementation/2026-10/llm-roadmap-review-and-gpt-1-starting-plan/09-readiness.md) records the workspace checks. Preserved setup/review packages are historical evidence; resume.md governs the current position.
