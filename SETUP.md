# Setup and everyday use

Start with [resume.md](resume.md), [progress.md](progress.md) and [roadmap.md](roadmap.md). The workspace CLI lives independently at [../cli](../../cli/README.md), outside Projects and Research.

For the first model follow [START_HERE.md](START_HERE.md). The generic repository scaffold has no model runtime; add and lock CPU PyTorch in the independent GPT project during kickoff, not in the shared CLI environment. Its placeholder tests intentionally fail until actual implementation tests exist.

This workspace supports Linux process supervision through `/proc`, process groups and `fcntl`. The CLI uses Python 3.12 and the committed uv lock; documentation requires Node 22+ and npm. A C++20 compiler is used for tutorial examples and future native projects. C++ and Python are the learning languages. No GPU or model credentials are required for the seeded examples.

## Start

```bash
cd /path/to/ai
./ai doctor --json
./ai roadmap
./ai
```

The launcher resolves paths from its own location and synchronizes its locked CLI environment. It works when called by absolute path from another current directory. Install uv from its [official installation guide](https://docs.astral.sh/uv/getting-started/installation/) if it is missing.

## Verify and run

```bash
./ai check --action setup
./ai check
./ai check --action build
./ai check --action run
./ai tui --smoke
```

Aggregate commands operate on entries supporting that action, summarize failures, and return a nonzero exit code for failed checks. See `./ai logs PROJECT --job` for each job's persistent output. For one project use `./ai setup|test|build|run PROJECT`. Python project environments/wheels are kept in `.ai/envs/` and `.ai/builds/`; sites retain their own ignored `node_modules/` and `dist/`.

## Colab for model training

Develop and verify locally, then train complete miniature models on Colab Free for P01–P04. [COLAB.md](COLAB.md) defines required child notebooks/configs, dependency setup, persistent full-state checkpoints, bounded sessions and recovery after runtime deletion. Later projects decide their compute tier from evidence; Pro remains a later user-directed choice.

Current remotes are local snapshots. The first notebook needs an exact, reproducible child-source ZIP upload path until hosted source publication is separately requested. Keep notebook outputs clear and datasets/weights/checkpoints ignored. Use Drive for verified run recovery and runtime-local disk for data access; the notebook calls the child's trainer without the workspace's Linux controller. Notebook implementation/training is pending P01, not established by these documentation checks.

## Websites

```bash
./ai start project-docs
./ai start research-docs
./ai open project-docs
./ai logs research-docs
./ai status --json
./ai stop-all
```

Default URLs are http://127.0.0.1:4273/project_docs/ and http://127.0.0.1:4274/research_docs/. Their dev servers watch authored content, catalog and assets. Stop a site before managed build/setup commands. Change a port while stopped with `./ai port research-docs 4294`. The UI/CLI survives separate invocations; quitting leaves owned servers and jobs running. Stop commands verify ownership and never kill an unrelated listener just because it uses a port.

Browser checks run separately inside each documentation repository with `npm run test:browser`; install the declared Playwright Chromium with `npx playwright install chromium` if needed. `SITE_BASE=/ npm run build` verifies root hosting; rebuild normally before using the configured prefixed local URL.

## New work and context recovery

```bash
./ai new project my-project --title "My AI project"
./ai new research my-study --title "My research study"
./ai task new "Design the study" --owner my-study --domains experiment,project
./ai task new "Workspace tooling" --owner cli --domains project
./ai task list --owner my-study
./ai research new experiment my-study --title "Study protocol"
./ai research sources
```

Repository scaffolds are real independent submodules with local initial snapshots. Their parent changes are staged for inspection. Scaffold tests intentionally fail until meaningful tests are added. Task packages are initially unassessed; fill and update them as work proceeds. Continue from `00-resume.md`, `05-tracker.md`, the current source and the latest evidence. Shared skills live in `ai_research/research_docs/.agents/skills/` and are exposed inside each umbrella as `.agents/skills/`. Use `$planning` or read `.agents/skills/planning/SKILL.md` in an owning umbrella; CLI instructions link the same canonical collection. The coordinator, helper and eight templates use the same saved-package workflow. There is no root `.agents` link.

## Local Git and relocation

The root is a lightweight home folder; `cli` is an independent shared tooling repository and `ai_projects`/`ai_research` are separate umbrellas. Their six children have their own Git histories. Initial submodule URLs point at local bare snapshots, with no GitHub repository or publication assumed.

```bash
./ai snapshot /tmp/ai-workspace.tar.gz
# Extract to a new destination, keeping the complete directory structure.
tar -xzf /tmp/ai-workspace.tar.gz -C /path/to/new-parent
/path/to/new-parent/ai/ai relink
/path/to/new-parent/ai/ai doctor --json
```

Snapshots include editable source and Git histories and exclude runtime state, environments, raw sources, datasets, model weights, generated runs, dependencies and secrets named `.env*`. Preserve needed ignored research artifacts separately. An existing snapshot is never overwritten. `relink` restores local cache/config paths and preserves declared external repository URLs. It does not push, pull or publish anything.

For later publication, create the actual requested child and umbrella remotes, publish child commits first, then replace local `.gitmodules` URLs and update parent pointers. Deployment, published-site URLs and research acceptance are separate work.


## Implementation layout, reports and focused-study handoff


Use consistent report/configuration conventions inside each independent implementation, while keeping model-specific state and behavior explicit. The owning layout is below; optional child folders are added when actual work needs them. Tracking and shared documentation stay under docs/:

```text
ai/
  ai                         # Shared workspace launcher
  cli/                       # Independent CLI repository
  ai_projects/
    README.md, AGENTS.md      # Root entry points only
    docs/                    # Roadmap, tracking, tutorials, plans and evidence
    projects/<project>/      # Independent model repositories
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

## Question and hypothesis
## Architecture implemented and deliberate simplifications
## Sources and inspected revisions
## Local configuration and exact parameter count
## Correctness checks and numerical errors
## Learning check, if performed
## Compute profile, persistent checkpoint and fresh-runtime recovery evidence
## Optimization and benchmark method
## Results
## Interpretation and unresolved questions
## Next experiment
```

For controlled comparisons, hold the relevant data, token budget, objective, and compute settings constant. Report multiple runs when a result depends on training randomness. Small-model results can differ from large-model results; formulate conclusions around the conditions actually tested.

Useful research questions include:

- Does cache preallocation improve local decoding at these dimensions?
- Does a chunked recurrence match sequential gradients and reduce execution overhead?
- How does expert imbalance change with the routing or balancing mechanism?
- How much memory does cache sharing save at a fixed context length?
- What is the output error introduced by a particular state-quantization scheme?
- Can a tiny model learn a task requiring recurrent memory or visual input?


## Using the workspace skills

The canonical collection is in ai_research/research_docs/.agents/skills/. Projects and Research expose it through their existing .agents/skills links. These are repository-specific workflows; they keep this workspace's tracked tutorial boundary and C++/Python preference. From workspace home, follow the explicit paths in AGENTS.md; there is no root .agents directory.

| Skill | Use it for | Entry point |
| --- | --- | --- |
| `$planning` | Select the needed workflows and resume one saved package | [planning](../../ai_research/research_docs/.agents/skills/planning/SKILL.md) |
| `$ai-project-workflow` | Implement and verify a model or workspace tool | [project workflow](../../ai_research/research_docs/.agents/skills/ai-project-workflow/SKILL.md) |
| `$ai-paper-review` | Inspect paper evidence or write a formal review | [paper review](../../ai_research/research_docs/.agents/skills/ai-paper-review/SKILL.md) |
| `$ai-experiment-planning` | Declare and run bounded learning checks or focused studies | [experiments](../../ai_research/research_docs/.agents/skills/ai-experiment-planning/SKILL.md) |
| `$ai-reproduction-planning` | Reproduce a specific algorithm, figure or result | [reproduction](../../ai_research/research_docs/.agents/skills/ai-reproduction-planning/SKILL.md) |
| `$developer-learning-tutorials` | Create diagram-rich lessons, worked examples and practice | [tutorials](../../ai_research/research_docs/.agents/skills/developer-learning-tutorials/SKILL.md) |
| `$feature-whiteboard-defense` | Build an evidence-backed explanation or rehearse when requested | [whiteboard defense](../../ai_research/research_docs/.agents/skills/feature-whiteboard-defense/SKILL.md) |

For the first project, ask to use the workspace-local developer-learning-tutorials skill to extend P01 at its current milestone. Use --level introductory for a new introductory lesson. Ask for a whiteboard rehearsal when you want live questions; it presents one question at a time. Ordinary project learning remains in Projects even when it uses papers or bounded experiments.

From workspace home, the reusable commands are:

```bash
./ai task list --owner projects
./ai learn list
./ai learn validate --lesson projects/p01/gpt-1-foundations/v1
./ai learn examples --lesson projects/p01/gpt-1-foundations/v1
./ai learn render --lesson projects/p01/gpt-1-foundations/v1
```

After adding a lesson, update tutorials/index.json, build the existing project-docs site and inspect it. Authoring and rendering never imply learner mastery or completion of a model.
