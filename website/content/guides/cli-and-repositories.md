---
title: "Using the AI workspace"
description: "The command workflow, repository boundaries and how to add independent work."
updated: "2026-10-09"
status: "working guide"
tags: ["guides", "cli-and-repositories"]
sources: [{"title": "uv: Locking and syncing", "url": "https://docs.astral.sh/uv/concepts/projects/sync/"}, {"title": "Textual: DataTable", "url": "https://textual.textualize.io/widgets/data_table/"}]
---

## Start at home

```bash
./ai
./ai roadmap
./ai roadmap projects --section progress
./ai list
./ai doctor --json
./ai check
```

Projects and Research are the only terminal groups. The shared CLI lives in the independent workspace cli/ repository, outside both umbrellas. Projects owns complete known-model implementation/training for understanding; Research owns narrow reproductions and improvements. Each independent implementation is a Git submodule; the umbrellas own membership and project pointers. Projects Documentation uses port 4273 and Research Documentation uses 4274.

## Start Project 0

The TUI selects **P00 · Transformer Foundations** by default. Press **g** or click **Run** for the matrix and gradient walkthrough. **Test** checks the arithmetic and numerical gradient. **Open** (or **o**) starts Projects Documentation if needed and opens chapter 1. The output appears under Job logs; predict the numbers yourself before running it.

This entry runs the tutorial's existing Python references. It needs no PyTorch installation or GPU. Follow chapter 1, then chapters 2–3; create the independent foundations repository at implementation kickoff. The four setup demo repositories have been retired. Later roadmap projects become launcher entries when they have an implementation.

## Sites, jobs and logs

```bash
./ai start project-docs
./ai start research-docs
./ai logs research-docs
./ai run p00
./ai test p00
./ai open p00
./ai logs p00 --job
./ai status --json
./ai stop-all
```

Closing the TUI leaves managed jobs/sites running. Stop commands verify PID start time, launch token, command line and process group. A port conflict is reported without killing another listener. Change a site's port while it is stopped with `./ai port research-docs 4294`.

## Follow the current learning checkpoint

Use ./ai roadmap to read the canonical resume, then roadmap/progress/decisions. Implement, train, inspect/debug and explain each small model; update evidence and the next action at session end. Focused research protocols/results stay in Research rather than this learning ledger. See [the roadmap workflow](roadmap-workflow.md).

## Add work with documentation first

```bash
./ai new project my-project --title "My AI project"
./ai new research my-experiment --title "My experiment"
./ai task new "Define the experiment" --owner my-experiment --domains experiment,project
./ai research new question evaluation-gap --title "Where does evaluation fail?"
```

Repository scaffolding creates a local initial snapshot so Git can register a real submodule. The parent entry is staged for inspection; it is not committed or pushed. Scaffold tests intentionally fail until meaningful tests exist. A task package is an unassessed scaffold until you fill it.

## Move or back up the workspace

`./ai snapshot /tmp/ai-workspace.tar.gz` saves editable sources and Git histories without environments, raw archives, weights or generated runs. Extract into a new location and run `./ai relink` to rebuild local Git caches. Runtime state and active processes do not move. Remote publication remains a separate action.
