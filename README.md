# AI Projects documentation

Start with [START_HERE.md](START_HERE.md), then [resume.md](resume.md), [roadmap.md](roadmap.md), [progress.md](progress.md) and [decisions.md](decisions.md). Project 1 is GPT-1; implementation and training have not started.

Current checkpoint: **2026-10-09 — P01 GPT-1 / P01-M1 not started.** Preparation completed: roadmap/docs consolidation, shared CLI, seven validated skills and the [P01 reference tutorial](tutorials/projects/p01/gpt-1-foundations/v1/lesson.md); [Colab Free training/recovery requirements](COLAB.md) are recorded for P01–P04. Next: source/shape/config/tokenizer/data and Free setup/storage/budget contracts in gpt-lab. Model notebooks/training/hosted recovery are pending. [resume.md](resume.md) and [progress.md](progress.md) carry live state and evidence. Adaptive tutorial/live-teaching guidance and the P01 prerequisite/transfer route are implemented; learner understanding remains unassessed. The [pre-P01 local commit review](implementation/2026-10/ai-workspace-setup/07-pre-p01-commit-review.md) records repository preparation and current checks.

The [Projects README](../README.md) and [Projects agent instructions](../AGENTS.md) embed a compact snapshot for new sessions. Read canonical state, inspect the child package and continue the actual next action. Update all six workspace/Projects/Docs snapshots when the current project, milestone/status, completed-work summary or next action changes; keep the detailed history here.

This is the single documentation home for Projects. Keep shared documentation here instead of creating files beside it in the Projects root.

| Material | Location |
| --- | --- |
| First-project kickoff and current direction | START_HERE.md, roadmap.md, resume.md |
| Status and durable choices | progress.md, decisions.md |
| Setup and project catalogue | SETUP.md, PROJECT_OVERVIEW.md |
| Colab Free profiles, notebooks and full persistent resume gates | [COLAB.md](COLAB.md) |
| Detailed model plans, source links and sizing guidance | [roadmap.md](roadmap.md#detailed-project-plans-and-retained-source-notes) |
| Project tutorials, diagrams, examples and whiteboard preparation | [tutorials/](tutorials/README.md) |
| Plans and implementation records | implementation/ |
| Verification evidence | verification/ |
| Authored website records and source provenance | content/, catalog/ |

Tutorials use C++ and Python and remain eligible for Git. Keep editable diagrams, teaching examples and selected safe source evidence with their lesson. Model implementations remain in independent sibling repositories under `../projects/`; checkpoints, weights, datasets and generated runs stay ignored.

The existing documentation repository and its history have been moved here from `project_docs/`. It also owns the local website, so there is one documentation tree and one reader. Run these commands from this directory:

```bash
npm ci
npm test
npm run build
npm run test:browser
npm run dev
```

Node 22+ and npm are required. Browser checks use Playwright Chromium. From workspace home run `./ai start project-docs`; the existing local URL remains http://127.0.0.1:4273/project_docs/. `SITE_BASE=/` supports root hosting. The site renders the canonical files above directly.

`node_modules/`, `dist/` and browser output are ignored. Optional raw `sources/` archives are ignored and unnecessary for a build. This local repository has no external publication configured.
