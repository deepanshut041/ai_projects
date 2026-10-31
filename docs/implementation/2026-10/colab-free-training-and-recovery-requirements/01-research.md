# Research — Colab Free training and recovery requirements

Updated: 2026-10-09
Lead: workspace planning coordinator with AI project workflow. Scope: Projects documentation, not model implementation or hosted training.

## Inspected workspace evidence

Read workspace, Projects and Docs AGENTS/README; canonical resume/progress/roadmap/decisions; START_HERE, SETUP, the saved GPT-1 starting plan and Docs website scripts/checks. P01 GPT-1 / M1 is not-started; projects/gpt-lab does not exist. Seed labs are infrastructure examples, not roadmap completion. The P01 plan has CPU checks but no notebook or persistent-storage recovery gate. Docs owns the common guidance; workspace.mjs renders selected canonical files into the existing site.

Existing modifications, moved submodule state and untracked planning/tutorial files predate this task. Preserve them; no commits, resets, pointer updates or publication. Targeted pre-edit contents are captured outside the workspace for scoped comparison.

## Primary sources inspected on 2026-10-09

| Source | Sections | Implication and limits |
| --- | --- | --- |
| [Google Colab FAQ](https://research.google.com/colaboratory/faq.html) | Temporary runtime storage, Drive I/O, resource limits, duration | VMs are temporary; resources/timeouts vary. Free sessions may last up to 12 hours depending on usage/availability, not a guaranteed budget. Bound Drive I/O and verify recovery copies. Pro also needs recovery. |
| [PyTorch saving/loading tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) | General checkpoints, cross-device loading | Resume needs optimizer state as well as weights; define a fuller workspace schema and device mapping. This does not supply our Drive publication protocol or prove Colab compatibility. |
| [PyTorch reproducibility source](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/randomness.md) | Reproducibility and randomness | Exact reproducibility across releases/devices is not guaranteed. Separate same-environment deterministic resume from cross-device portability. |

## Decisions and assumptions

Interpret first few as P01–P04: GPT-1, GPT-2, Llama variants and Qwen Dense. This is an initial planning boundary, not measured feasibility. Reassess P05 onward at kickoff. Keep CPU correctness and C++/Python learning; a thin Python notebook calls the child trainer. Create the actual notebook with P01 model work, rather than a disconnected pretend trainer. Budgets are proposed defaults to measure in the child; no GPU type, VRAM, throughput, quality or successful hosted run is established.

Deliver COLAB.md, integrate canonical plan/state and six snapshots, amend the P01 starting contract, expose the guide in the local website and run required content/build/browser checks. Actual notebook/training/fresh-runtime proof stays pending P01.
