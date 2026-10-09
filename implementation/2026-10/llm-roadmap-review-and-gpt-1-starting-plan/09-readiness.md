# Projects readiness audit

Updated: 2026-10-09. Lead: AI project workflow.

## Requested outcome

Make ai_projects ready before the learner starts the first project. Verify canonical guidance/status, ownership, launchers, planning/scaffolding, retained examples, documentation rendering and artifact boundaries; fix concrete gaps. Preserve local edits/history and the separately authored architecture diagrams.

This prepares and checks kickoff readiness. The actual gpt-lab repository, model implementation and training remain the learner's subsequent P01 work; do not mark those done or create an already-finished model.

## Checks planned

- Canonical roadmap/progress IDs, P01 milestone/checkpoint and local links agree.
- Owner instructions and canonical planning link resolve; CLI remains outside Projects.
- Doctor, TUI and portable root/Projects launchers work.
- Retained Projects example tests/demos and documentation checks pass.
- Real CLI project creation plus child-owned planning works in a disposable workspace, without modifying real Projects membership or starting P01.
- Runtime/secrets/generated-data ignore policy and Git/submodule histories remain correct.
- Clear first-step guide and a dated evidence record leave one next action.

Status: passed; ready for P01-M1 kickoff. No actual GPT repository/model/training or new Research experiment started.

## Concrete gap found

The generic repository scaffold directs new Projects implementations to the research workflow and does not link their canonical area checkpoint. Correct the owner-specific README/AGENTS routing in the CLI before kickoff. Add explicit checkpoint/weight-file exclusions so saved training artifacts do not accidentally enter child Git history. This is scaffolding/tooling work, not model implementation.

## Delivered fixes and guide

The scaffold now links its own area's roadmap/progress/resume/decisions and owner instructions. Projects guidance preserves complete known-model learning/training; Research guidance preserves narrow study scope. Explicit checkpoints/ and common weight/checkpoint formats are ignored in new children and the Projects umbrella. Existing unrelated edits and histories are preserved.

START_HERE.md provides the one-time GPT-1 kickoff, first learning milestone, child-runtime setup boundary and session update workflow. It is linked from README/resume/roadmap/setup and rendered from its canonical source in the Projects site.

The older starter-lab page asking to choose a next project is retained as deferred extension ideas. It now points back to the selected P01 path, avoiding a competing current checkpoint.

## Verification

| Check | Result | Evidence |
| --- | --- | --- |
| CLI/controller/authoring suite | passed: 48 tests | uv run --frozen pytest -q; includes owner-guidance/artifact exclusion coverage |
| Existing Projects tests | all 4 entries passed | readiness-test.json |
| Existing Projects builds | all 4 entries passed | readiness-build.json |
| Existing project demos | all 3 passed | readiness-run.json |
| Documentation | 6 content checks, build and 13-record browser checks passed | native npm commands; links/downloads/desktop/mobile |
| Environment/TUI/portable launcher | passed | readiness-preflight.json |
| Canonical IDs, checkpoint and links | 19 stable IDs, P01-M1, 54 local links passed | readiness-preflight.json |
| Planning link, identity, ignore boundaries and original catalogue | passed | readiness-preflight.json |
| One-time GPT repository + child package creation | passed in disposable workspace | readiness-kickoff.json; actual membership/history unchanged |

Aggregate evidence: readiness-summary.json. The initial generic scaffold intentionally has no model dependencies/tests; CPU PyTorch selection/lock and meaningful tests belong in actual P01 kickoff. Readiness does not mean a GPT model or training environment has already been produced.

Next action: START_HERE.md → P01-M1. Resume the canonical checkpoint instead of this historical audit when project work begins.
