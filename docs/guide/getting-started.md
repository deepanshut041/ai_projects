# Getting started

Updated: 2026-10-12
One guide for the P00 kickoff, everyday setup and repository map. Read [the live checkpoint](resume.md), then [the roadmap](roadmap.md).

## Reading order

Start [Tutorial 0 — Maths and PyTorch foundations](../tutorials/projects/p00/transformer-foundations/v1/lessons/01-prediction.md), then [Course 1 — Translation and RoBERTa](../tutorials/projects/p00/transformer-course/v1/lessons/01-start.md), then [Course 2 — Small GPT](../tutorials/projects/p01/small-gpt/v1/lessons/01-next-token.md). Tutorial 0 assumes advanced Python and teaches the shared maths/PyTorch/data/training prerequisites. Existing child setup and its Lesson 3 continuation remain intact.

## First project

Updated: 2026-10-09
Current position: **P00 Tokenization and Transformer Foundations → P00-M1: tensors, gradients and runtime/source contracts.**

The starting route is **tensors/autograd → character/byte/BPE tokenizer → attention → full tiny encoder–decoder → P01 Small GPT Language Model**. P01 means a decoder-only text model; exact historical GPT-1 is optional reference study. P00 Lesson 2 setup is complete in the independent foundations child; P00-M1 is in-progress. Model training and P01 implementation are not started. P00 ends in a trained sequence model, rather than isolated snippets.

For the existing foundations lab, [its first exercise](../tutorials/projects/p00/transformer-foundations/v1/lessons/01-prediction.md) and current Lesson 3 continuation remain available. Tutorial 0 is the prerequisite reading route, followed by Course 1 above. The records below organize that existing lab's implementation work.

### Workspace records for implementation kickoff

1. [resume.md](resume.md): current milestone and next action.
2. [roadmap.md](roadmap.md#project-0-tokenization-and-transformer-foundations): P00/P01 contracts, prerequisites and gates.
3. [progress.md](progress.md): actual completed/pending evidence.
4. [P00/P01 plan](../implementation/2026-10/project-zero-foundations-and-small-gpt-roadmap/02-implementation-plan.md): source review, shapes, phases and verification scope.
5. [COLAB.md](COLAB.md): common trainer/notebook, Free profiles, persistent full-state checkpoints and fresh-runtime recovery.

[decisions.md](decisions.md) records the change of sequence. Known-model learning belongs in Projects; narrow improvements/reproductions belong in [Research](../../../../ai_research/roadmap.md). Shared tooling remains in ../../cli/.

### Kick off P00 when implementation starts

For learning, [begin with Lesson 1 — make one prediction](../tutorials/projects/p00/transformer-foundations/v1/lessons/01-prediction.md). It teaches input, weight, prediction and target before setup. Continue to Lesson 2 to give that same program a home, then Lesson 3 to extend it one function at a time. The reader has one eleven-lesson Learning path; each next lesson solves a problem raised by the previous one. The [progress checklist](../tutorials/projects/p00/transformer-foundations/v1/project/implementation.md) tracks this route, and each lesson contains the relevant chapter explanations, derivations and implementation.

Run from workspace home; from Projects use ../ai:

```bash
./ai roadmap
./ai new project transformer-foundations-lab --title "Tokenization and Transformer Foundations"
./ai task new "P00 foundations implementation and learning" --owner transformer-foundations-lab --domains project,experiment
```

These are first-kickoff commands for a new workspace. In this workspace they were executed on 2026-10-10: the foundations child and package exist, the local Python environment/import and first prediction are checked, and the next action is Lesson 3 part A. Do not run creation again here. If the child or topic package exists, read its README/AGENTS and package checkpoint, inspect Git state and resume it. Do not recreate/reset it. Only create gpt-lab when P00 gates permit P01.

The generic scaffold has no model runtime or meaningful model tests. During M1 add/lock CPU PyTorch and needed test dependencies in the child, not the shared CLI. Its placeholder test failure is expected until meaningful tests are implemented. Use Python/PyTorch for the first complete training path, with selected C++/Python traces rather than a mandatory full native port.

### First learning milestone: P00-M1

- Pin the paper/reference revisions recorded in the current package and declare deliberate simplifications. Draw the intended encoder/decoder/data path before implementing it.
- Check tensor dimensions, broadcasting and matrix multiplication; work through stable softmax/cross-entropy and a tiny hand-computed gradient.
- Compare one analytic/finite-difference gradient with autograd; perform an optimizer update and fit a tiny MLP on generated labels. Inspect finite gradients and explain loss changes. This is an introductory learning check, not Transformer completion.
- Record seed/data provenance, local lock/config and selected C++/Python exercise results in the child-owned package.
- Declare CPU and Colab Free profiles, exact source ZIP delivery, persistent Drive run root and bounded update/time/checkpoint/storage budgets under COLAB.md. Measure before expanding.

Next build byte tokenization and deterministic BPE in M2, attention/block primitives in M3, the complete post-norm/sinusoidal/ReLU encoder–decoder in M4, and copy-task training plus thin notebook/full recovery in M5. M6 covers error diagnosis, parameters/memory/profile and the tokenizer/trainer handoff to P01. Follow the roadmap for conditional masks and task gates; do not treat falling teacher-forced loss as successful generated copy behavior.

During P00-M5 verify interrupted saves and recovery after **Disconnect and delete runtime** before calling P00 Colab-ready. No hosted training/recovery has occurred in this roadmap update.

### Continue to P01 after the gates

Use [P01 Lesson 1 — predict the next token](../tutorials/projects/p01/small-gpt/v1/lessons/01-next-token.md), then follow its single twelve-lesson route. The [project area](../tutorials/projects/p01/small-gpt/v1/project/implementation.md) tracks implementation and required exit evidence.

P01 consumes a versioned frozen tokenizer and independently runnable training/recovery plumbing, removes the encoder/cross-attention and implements a small pre-norm decoder with learned positions/GELU. Use licensed text with disjoint splits, shifted next-token targets, held-out loss and text sampling. Required milestones are M1–M3 and M5; historical GPT-1 classification and early caching are optional, with required cache coverage in P02.

The [older GPT-1 starting plan](../implementation/2026-10/llm-roadmap-review-and-gpt-1-starting-plan/02-implementation-plan.md) remains a historical record. The integrated Small GPT course above replaces the retired GPT-1 tutorial. The [P00 practical tutorial](../tutorials/projects/p00/transformer-foundations/v1/lesson.md) and [workbook](../tutorials/projects/p00/transformer-foundations/v1/project/implementation.md) now prepare shared prerequisites; Course 1 owns the complete model builds. Their educational reference checks do not complete learner implementation, successful task training or hosted recovery; lesson availability is separate from demonstrated mastery.

At each session end update the child package, progress.md and resume.md with actual implementation/checks, learner observations, unresolved questions and one next step. Data, checkpoints, weights, environments and generated runs remain ignored; keep provenance and acquisition instructions tracked. Local checks, learner understanding, hosted recovery and publication remain distinct.

## Workspace setup

Start with [resume.md](resume.md), [progress.md](progress.md) and [roadmap.md](roadmap.md). The workspace CLI lives independently at [../cli](../../../cli/README.md), outside Projects and Research.

For the first model follow [START_HERE.md](getting-started.md#first-project). The generic repository scaffold has no model runtime; add and lock CPU PyTorch in the independent P00 foundations project during kickoff, not in the shared CLI environment. Its placeholder tests intentionally fail until actual implementation tests exist.

This workspace supports Linux process supervision through `/proc`, process groups and `fcntl`. The CLI uses Python 3.12 and the committed uv lock; documentation requires Node 22+ and npm. A C++20 compiler is used for tutorial examples and future native projects. C++ and Python are the learning languages. No GPU, model credentials or extra Python packages are required for the P00 first-session exercise.

### Start

```bash
cd /path/to/ai
./ai doctor --json
./ai roadmap
./ai
```

The launcher resolves paths from its own location and synchronizes its locked CLI environment. It works when called by absolute path from another current directory. Install uv from its [official installation guide](https://docs.astral.sh/uv/getting-started/installation/) if it is missing.

### Verify and run

```bash
./ai check --action setup
./ai check
./ai check --action build
./ai check --action run
./ai tui --smoke
```

Aggregate commands operate on entries supporting that action, summarize failures, and return a nonzero exit code for failed checks. See `./ai logs PROJECT --job` for each job's persistent output. For one project use `./ai setup|test|build|run PROJECT`. Python project environments/wheels are kept in `.ai/envs/` and `.ai/builds/`; sites retain their own ignored `node_modules/` and `dist/`.

### Colab for model training

Develop and verify locally, then train complete miniature models on Colab Free for P00–P04. [COLAB.md](COLAB.md) defines required child notebooks/configs, dependency setup, persistent full-state checkpoints, bounded sessions and recovery after runtime deletion. Later projects decide their compute tier from evidence; Pro remains a later user-directed choice.

Current remotes are local snapshots. The first notebook needs an exact, reproducible child-source ZIP upload path until hosted source publication is separately requested. Keep notebook outputs clear and datasets/weights/checkpoints ignored. Use Drive for verified run recovery and runtime-local disk for data access; the notebook calls the child's trainer without the workspace's Linux controller. Notebook implementation/training is pending P00/P01, not established by these documentation checks.

### Websites

```bash
./ai start project-docs
./ai start research-docs
./ai open project-docs
./ai logs research-docs
./ai status --json
./ai stop-all
```

Default URLs are http://127.0.0.1:4273/project_docs/ and http://127.0.0.1:4274/research_docs/. Their dev servers watch authored content, catalog and assets. Stop a site before managed build/setup commands. Change a port while stopped with `./ai port research-docs 4294`. The UI/CLI survives separate invocations; quitting leaves owned servers and jobs running. Stop commands verify ownership and never kill an unrelated listener just because it uses a port.

Browser checks run from Projects docs with `npm --prefix website run test:browser`, or from Research docs with `npm run test:browser`; install the declared Playwright Chromium with `npm --prefix website exec playwright install chromium` from Projects docs (or `npx playwright install chromium` from Research docs) if needed. `SITE_BASE=/ npm --prefix website run build` from Projects docs verifies root hosting; rebuild normally before using the configured prefixed local URL.

### New work and context recovery

```bash
./ai new project my-project --title "My AI project"
./ai new research my-study --title "My research study"
./ai task new "Design the study" --owner my-study --domains experiment,project
./ai task new "Workspace tooling" --owner cli --domains project
./ai task list --owner my-study
./ai research new experiment my-study --title "Study protocol"
./ai research sources
```

Projects scaffolds are ordinary folders owned by ai_projects: the default group is language-models, with --collection transformer-labs for mechanism/task labs. They do not create a nested Git repository, submodule or initial commit. Research scaffolds retain independent submodules and local initial snapshots. Scaffold tests intentionally fail until meaningful tests are added. Task packages are initially unassessed; fill and update them as work proceeds. Continue from `00-resume.md`, `05-tracker.md`, the current source and the latest evidence. Shared skills live in `ai_research/research_docs/.agents/skills/` and are exposed inside each umbrella as `.agents/skills/`. Use `$planning` or read `.agents/skills/planning/SKILL.md` in an owning umbrella; CLI instructions link the same canonical collection. The coordinator, helper and eight templates use the same saved-package workflow. There is no root `.agents` link.

### Local Git and relocation

The root is a lightweight home folder; `cli` is an independent shared tooling repository and `ai_projects`/`ai_research` are separate umbrellas. Their documentation children have their own Git histories; ordinary Projects implementation folders are created in their selected group at roadmap kickoff; Research implementations retain independent repositories. Initial submodule URLs point at local bare snapshots, with no GitHub repository or publication assumed.

```bash
./ai snapshot /tmp/ai-workspace.tar.gz
# Extract to a new destination, keeping the complete directory structure.
tar -xzf /tmp/ai-workspace.tar.gz -C /path/to/new-parent
/path/to/new-parent/ai/ai relink
/path/to/new-parent/ai/ai doctor --json
```

Snapshots include editable source and Git histories and exclude runtime state, environments, raw sources, datasets, model weights, generated runs, dependencies and secrets named `.env*`. Preserve needed ignored research artifacts separately. An existing snapshot is never overwritten. `relink` restores local cache/config paths and preserves declared external repository URLs. It does not push, pull or publish anything.

For later publication, create the actual requested child and umbrella remotes, publish child commits first, then replace local `.gitmodules` URLs and update parent pointers. Deployment, published-site URLs and research acceptance are separate work.


### Implementation layout, reports and focused-study handoff


Use consistent report/configuration conventions inside each independent implementation, while keeping model-specific state and behavior explicit. The owning layout is below; optional child folders are added when actual work needs them. Tracking and shared documentation stay under docs/:

```text
ai/
  ai                         # Shared workspace launcher
  cli/                       # Independent CLI repository
  ai_projects/
    README.md, AGENTS.md      # Root entry points only
    docs/                    # Roadmap, tracking, tutorials, plans and evidence
    projects/<group>/<project>/ # Ordinary folders tracked by ai_projects
      configs/               # Local correctness, Free and reference; Pro later
      notebooks/             # Thin child-owned Colab setup/train/resume entry
      references/            # Source links and inspected revisions
      tests/                 # Numerical, causal, state and checkpoint checks
      experiments/           # Bounded learning/optimization work
      benchmarks/            # Reproducible local profiling
      reports/               # Project-owned results and interpretation
  ai_research/
    research_docs/           # Sources, protocols and reports
    <study>/                 # Independent focused research implementations
```

Each project's report should contain:

```markdown
# Model / Experiment

### Question and hypothesis
### Architecture implemented and deliberate simplifications
### Sources and inspected revisions
### Local configuration and exact parameter count
### Correctness checks and numerical errors
### Learning check, if performed
### Compute profile, persistent checkpoint and fresh-runtime recovery evidence
### Optimization and benchmark method
### Results
### Interpretation and unresolved questions
### Next experiment
```

For controlled comparisons, hold the relevant data, token budget, objective, and compute settings constant. Report multiple runs when a result depends on training randomness. Small-model results can differ from large-model results; formulate conclusions around the conditions actually tested.

Useful research questions include:

- Does cache preallocation improve local decoding at these dimensions?
- Does a chunked recurrence match sequential gradients and reduce execution overhead?
- How does expert imbalance change with the routing or balancing mechanism?
- How much memory does cache sharing save at a fixed context length?
- What is the output error introduced by a particular state-quantization scheme?
- Can a tiny model learn a task requiring recurrent memory or visual input?


### Using the workspace skills

The canonical collection is in ai_research/research_docs/.agents/skills/. Projects and Research expose it through their existing .agents/skills links. These are repository-specific workflows; they keep this workspace's tracked tutorial boundary and C++/Python preference. From workspace home, follow the explicit paths in AGENTS.md. The root `.agents/skills/tutorial-video` link exposes the CLI-owned video coordinator; the existing umbrella links retain the shared skill collection.

| Skill | Use it for | Entry point |
| --- | --- | --- |
| `$planning` | Select the needed workflows and resume one saved package | [planning](../../../ai_research/research_docs/.agents/skills/planning/SKILL.md) |
| `$ai-project-workflow` | Implement and verify a model or workspace tool | [project workflow](../../../ai_research/research_docs/.agents/skills/ai-project-workflow/SKILL.md) |
| `$ai-paper-review` | Inspect paper evidence or write a formal review | [paper review](../../../ai_research/research_docs/.agents/skills/ai-paper-review/SKILL.md) |
| `$ai-experiment-planning` | Declare and run bounded learning checks or focused studies | [experiments](../../../ai_research/research_docs/.agents/skills/ai-experiment-planning/SKILL.md) |
| `$ai-reproduction-planning` | Reproduce a specific algorithm, figure or result | [reproduction](../../../ai_research/research_docs/.agents/skills/ai-reproduction-planning/SKILL.md) |
| `$developer-learning-tutorials` | Create diagram-rich lessons, worked examples and practice | [tutorials](../../../ai_research/research_docs/.agents/skills/developer-learning-tutorials/SKILL.md) |
| `$tutorial-video` | Explain any existing docs tutorial and attach verified playback to Workbench | [video coordinator](../../../cli/skills/tutorial-video/SKILL.md) |
| `$feature-whiteboard-defense` | Build an evidence-backed explanation or rehearse when requested | [whiteboard defense](../../../ai_research/research_docs/.agents/skills/feature-whiteboard-defense/SKILL.md) |

For the first project, ask to use the workspace-local developer-learning-tutorials skill to create/extend P00 at its current milestone; the integrated P01 Small GPT course follows after the P00 gates. Use --level introductory for a new introductory lesson. Ask for a whiteboard rehearsal when you want live questions; it presents one question at a time. Ordinary project learning remains in Projects even when it uses papers or bounded experiments.

From workspace home, the reusable commands are:

```bash
./ai task list --owner projects
./ai learn list
./ai learn validate --lesson projects/p01/small-gpt/v1
./ai learn examples --lesson projects/p01/small-gpt/v1
./ai learn render --lesson projects/p01/small-gpt/v1
```

After adding a lesson under a separately authorized tutorial task, update tutorials/index.json, build the existing project-docs site and inspect it. Authoring and rendering never imply learner mastery or completion of a model.

## Repository map

Read [roadmap.md](roadmap.md), [progress.md](progress.md) and [resume.md](resume.md) for the learning sequence. The shared CLI is independently owned at [shared CLI](../../../../cli/README.md); focused Research has [its own plan](../../../../ai_research/roadmap.md).

| CLI id | Owner | Checks | Current boundary |
| --- | --- | --- | --- |
| `p00` | Existing P00 tutorial in Projects docs | Standard-library arithmetic and numerical gradient | Run the first-session exercise; Open Lesson 1 |
| `project-docs` | `ai_projects/docs` | npm content/build/browser | Projects site and canonical roadmap |
| `research-docs` | `ai_research/research_docs` | npm content/build/browser | Research site and preserved evidence |

```bash
./ai run p00
./ai test p00
./ai open p00
```

The TUI selects P00 when it opens. Press **g** for Run, **t** for Test, **o** for Open. Open starts the documentation service if needed and follows its saved port. Test checks the first-session arithmetic and a finite-difference gradient; later model and recovery checks follow the tutorial. P00's badge says Learn because this is a tutorial entry. It does not claim that the independent model exists or that P00-M1 is complete.

The four original setup-only labs were retired from the active codebase on 2026-10-09. Verified Git bundles, checkout archives (including ignored working artifacts) and a recovery manifest are stored at workspace home under `.ai/retired-projects/2026-10-09/`. Historical work records and the Research synthetic-pilot result remain evidence. The current path is P00, then P01; later repositories are created when their roadmap work begins.

### Tutorial videos in the local Workbench

Ask `$tutorial-video` to explain a selected docs tutorial or exact section. `./ai video sources --owner projects --query TOPIC` finds sources/headings; the skill plans and authors the explanation through the shared CLI. After verification, `video attach OWNER/SLUG --document PATH [--heading HEADING]` adds an inline player without changing the prose, and `video open OWNER/SLUG` opens its Videos page. Plain Markdown gets a local source preview when needed. Captions, transcripts and downloads accompany the native controls. The [video runbook](../../../cli/docs/tutorial-video.md) covers authoring, reviewed legacy reuse and migration. Local tooling/media checks do not advance learning checkpoints.
