# Start your first project

Updated: 2026-10-09
Current position: **P00 Tokenization and Transformer Foundations → P00-M1: tensors, gradients and runtime/source contracts.**

The starting route is **tensors/autograd → character/byte/BPE tokenizer → attention → full tiny encoder–decoder → P01 Small GPT Language Model**. P01 means a decoder-only text model; exact historical GPT-1 is optional reference study. Implementation and training of both projects are not started. P00 ends in a trained sequence model, rather than isolated snippets.

## Read these first

1. [resume.md](resume.md): current milestone and next action.
2. [roadmap.md](roadmap.md#project-0-tokenization-and-transformer-foundations): P00/P01 contracts, prerequisites and gates.
3. [progress.md](progress.md): actual completed/pending evidence.
4. [P00/P01 plan](implementation/2026-10/project-zero-foundations-and-small-gpt-roadmap/02-implementation-plan.md): source review, shapes, phases and verification scope.
5. [COLAB.md](COLAB.md): common trainer/notebook, Free profiles, persistent full-state checkpoints and fresh-runtime recovery.

[decisions.md](decisions.md) records the change of sequence. Known-model learning belongs in Projects; narrow improvements/reproductions belong in [Research](../../ai_research/roadmap.md). Shared tooling remains in ../../cli/.

## Kick off P00 when implementation starts

Run from workspace home; from Projects use ../ai:

```bash
./ai roadmap
./ai new project transformer-foundations-lab --title "Tokenization and Transformer Foundations"
./ai task new "P00 foundations implementation and learning" --owner transformer-foundations-lab --domains project,experiment
```

These are proposed first-kickoff commands; the model child has not been created. If the child or topic package exists, read its README/AGENTS and package checkpoint, inspect Git state and resume it. Do not recreate/reset it. Only create gpt-lab when P00 gates permit P01.

The generic scaffold has no model runtime or meaningful model tests. During M1 add/lock CPU PyTorch and needed test dependencies in the child, not the shared CLI. Its placeholder test failure is expected until meaningful tests are implemented. Use Python/PyTorch for the first complete training path, with selected C++/Python traces rather than a mandatory full native port.

## First learning milestone: P00-M1

- Pin the paper/reference revisions recorded in the current package and declare deliberate simplifications. Draw the intended encoder/decoder/data path before implementing it.
- Check tensor dimensions, broadcasting and matrix multiplication; work through stable softmax/cross-entropy and a tiny hand-computed gradient.
- Compare one analytic/finite-difference gradient with autograd; perform an optimizer update and fit a tiny MLP on generated labels. Inspect finite gradients and explain loss changes. This is an introductory learning check, not Transformer completion.
- Record seed/data provenance, local lock/config and selected C++/Python exercise results in the child-owned package.
- Declare CPU and Colab Free profiles, exact source ZIP delivery, persistent Drive run root and bounded update/time/checkpoint/storage budgets under COLAB.md. Measure before expanding.

Next build byte tokenization and deterministic BPE in M2, attention/block primitives in M3, the complete post-norm/sinusoidal/ReLU encoder–decoder in M4, and copy-task training plus thin notebook/full recovery in M5. M6 covers error diagnosis, parameters/memory/profile and the tokenizer/trainer handoff to P01. Follow the roadmap for conditional masks and task gates; do not treat falling teacher-forced loss as successful generated copy behavior.

During P00-M5 verify interrupted saves and recovery after **Disconnect and delete runtime** before calling P00 Colab-ready. No hosted training/recovery has occurred in this roadmap update.

## Continue to P01 after the gates

P01 consumes a versioned frozen tokenizer and independently runnable training/recovery plumbing, removes the encoder/cross-attention and implements a small pre-norm decoder with learned positions/GELU. Use licensed text with disjoint splits, shifted next-token targets, held-out loss and text sampling. Required milestones are M1–M3 and M5; historical GPT-1 classification and early caching are optional, with required cache coverage in P02.

The preserved [GPT-1 tutorial](tutorials/projects/p01/gpt-1-foundations/v1/lesson.md) and [older starting plan](implementation/2026-10/llm-roadmap-review-and-gpt-1-starting-plan/02-implementation-plan.md) are historical references. P00 tutorials are planned for actual milestone work; lesson availability is separate from demonstrated mastery.

At each session end update the child package, progress.md and resume.md with actual implementation/checks, learner observations, unresolved questions and one next step. Data, checkpoints, weights, environments and generated runs remain ignored; keep provenance and acquisition instructions tracked. Local checks, learner understanding, hosted recovery and publication remain distinct.
