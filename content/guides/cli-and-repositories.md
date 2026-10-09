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
./ai list
./ai doctor --json
./ai check
```

Projects and Research are the only terminal groups. Each independent implementation is a Git submodule; the umbrellas own membership and project pointers. Projects Documentation uses port 4273 and Research Documentation uses 4274.

## Sites, jobs and logs

```bash
./ai start project-docs
./ai start research-docs
./ai logs research-docs
./ai run attention-lab --detach
./ai logs attention-lab --job
./ai status --json
./ai stop-all
```

Closing the TUI leaves managed jobs/sites running. Stop commands verify PID start time, launch token, command line and process group. A port conflict is reported without killing another listener. Change a site's port while it is stopped with `./ai port research-docs 4294`.

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
