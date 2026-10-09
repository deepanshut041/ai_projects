---
title: "Follow the roadmap and resume safely"
description: "Use the canonical plan, progress, checkpoint and decisions without losing the learning sequence."
updated: "2026-10-09"
status: "working guide"
tags: ["guides", "roadmap", "workflow"]
---

## Begin with the checkpoint

The consolidated docs/ repository owns roadmap.md, progress.md, resume.md and decisions.md. Their website views are generated from those source files on each build; edit the canonical files, not a second site copy. The sidebar's Roadmap link opens the learning path and its linked progress/checkpoint records.

```bash
./ai roadmap
./ai roadmap projects --section roadmap
./ai roadmap projects --section progress
./ai roadmap projects --section decisions
```

The default shows the current checkpoint and all four paths. Read the owner's AGENTS.md/Git state and topic package before editing. Project 1 is GPT-1; existing seed labs are supporting context, not completion of the full model.

When starting the first model, open the Start your first project record in the complete workspace, or read the owning START_HERE.md file. It explains the one-time repository/package creation and which runtime setup belongs in the child.

## Learn and record evidence

Implement and train small complete known models, inspect/debug losses and gradients, explain observations, then profile and verify established execution techniques. Tests and learner understanding are separate milestones. Update the progress ledger with evidence and the checkpoint with one next action before ending a substantial session.

## Keep Research separate

```bash
./ai roadmap research
```

Research has its own canonical records and owns narrow questions, focused reproductions and improvements. A research implementation can contain only the relevant mechanism. Broad complete-model learning remains in Projects. Research ideas saved as intake are not active experiments or measured findings.

## Workspace tooling

The shared CLI lives in workspace cli/, outside both umbrellas. Use --owner cli for its implementation packages, --owner projects for the Projects umbrella, --owner research for Research Docs, and a registered project ID for an independent implementation.

## Tutorials alongside project work

All shared documentation lives under ai_projects/docs/. Tutorials live under docs/tutorials/ and are eligible for Git. Use C++ and Python, complementary diagrams, worked traces, declarative scenarios, retrieval checks and whiteboard preparation. The first GPT-1 orientation is in the Project tutorials section.

Run ./ai learn list, validate, examples and render from workspace home. Model code, training evidence and demonstrated learner understanding remain separate from the lesson being available. Extend each project’s tutorials from actual implementation work. Shared planning and tutorial skills point to the canonical collection in Research Docs.
