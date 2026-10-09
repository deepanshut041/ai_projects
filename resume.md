# Projects resume checkpoint

Updated: 2026-10-09
Read this first on continuation, followed by [progress.md](progress.md), [roadmap.md](roadmap.md) and [decisions.md](decisions.md).

- Current project: P01 GPT-1.
- Current milestone: P01-M1, not started.
- First-session guide: [START_HERE.md](START_HERE.md). Workspace readiness checks passed; begin P01-M1 when starting the model project.
- Objective: learn deeply by implementing and training known small models; eventual focused inference research belongs in Research.
- Documentation home: [README.md](README.md); all shared documentation, tutorials, checkpoints and work records are consolidated in this docs/ repository. C++ and Python are the learning languages.
- Tutorial: [P01 foundations](tutorials/projects/p01/gpt-1-foundations/v1/lesson.md), with a proposed prerequisite route, seven focused diagrams, six practice checks, C++/Python examples and whiteboard preparation; learner knowledge remains unassessed. Adaptive author/live/direct teaching guidance is implemented in the shared skill; [plan and verification](implementation/2026-10/learn-inspired-adaptive-tutorial-improvements/00-resume.md) record the exact scope. Availability is separate from model implementation and mastery.
- Completed: roadmap review and first-model plan; workspace roadmap/CLI integration is implemented and locally verified. [Colab Free training/recovery requirements](COLAB.md) are recorded for P01–P04; notebooks and hosted training/recovery remain pending.
- Next action: create/resume independent gpt-lab and its child-owned package, then complete M1 source/config/tokenizer/data contracts plus the Colab Free source/setup/storage/budget plan. Implement the common-trainer notebook and persistent resume in M3.
- Proposed commands (kickoff, not already executed):

```bash
./ai new project gpt-lab --title "GPT Learning Lab"
./ai task new "GPT-1 implementation and learning" --owner gpt-lab --domains project,experiment
```

Run commands from workspace home; from the Projects root use ../ai. Review the [detailed starting plan](implementation/2026-10/llm-roadmap-review-and-gpt-1-starting-plan/02-implementation-plan.md) and copy/adapt implementation-specific scope into the child package.

- Proposed first config: 2 layers, width 64, 4 heads, context 64, FFN 256, 260 byte/special IDs; FP32 CPU. Exact model count/runtime are not measured.
- Training path: CPU correctness first, then bounded Colab Free training; Pro later when requested. Follow COLAB.md for full-state Drive checkpoints and recovery after runtime deletion. Actual Free hardware fit and fresh-runtime recovery are unverified.
- Current documentation checks: CLI 64 tests; Docs 7 content tests, build and browser checks for 14 records including the adaptive tutorial controls; C++/Python examples and shared skill validators pass.
- Colab requirements delivery checks: Docs 7/7 content tests, build and browser checks for 14 records; new guide inspected at desktop/mobile sizes. [Review/evidence](implementation/2026-10/colab-free-training-and-recovery-requirements/06-review.md). These verify documentation, not a hosted training run.
- Earlier integration/readiness: CLI 48 tests, TUI/doctor and portable launchers; all existing Projects tests/builds/demos; Projects content/build/browser checks; canonical links/IDs and source/history preservation. Evidence: the readiness audit and ../../cli/docs/implementation/2026-10/workspace-cli-relocation/.
- Skills: all seven canonical entries and links validated; complete tutorial/planning/record workflows exercised. Follow [SETUP.md](SETUP.md#using-the-workspace-skills). Evidence: [readiness report](implementation/2026-10/tracked-project-tutorials-and-whiteboard-learning/skill-readiness.json).
- Preserve: current local changes/history, consolidated model/source notes in roadmap.md and previous review/setup evidence.
- Learning gate: the learner implements selected components, runs/trains/inspects and explains behavior; do not substitute a finished generated model for this cycle.
- Research state: no active SSD/KV research experiment; [Research intake](../../ai_research/roadmap.md) is separate.
- Blockers: none for kickoff; any future GPU/SSD runtime experiment requires its own hardware/protocol checks.
- Readiness evidence: [audit](implementation/2026-10/llm-roadmap-review-and-gpt-1-starting-plan/09-readiness.md). Project creation/planning was verified in a disposable workspace; the actual GPT repository and child CPU runtime are first-kickoff setup, not already installed/implemented work.
- Pre-P01 repository baseline: the user authorized local commits of prepared workspace work. [Review and current checks](implementation/2026-10/ai-workspace-setup/07-pre-p01-commit-review.md); preserve independent histories and commit child revisions before parent pointers. P01-M1 remains not-started.
- External publication: none.

At session end update this checkpoint with the real milestone, what was verified, unresolved questions and one concrete next action. Older package resumes are historical/topic-specific and do not supersede this checkpoint.

Entry-point context: workspace and Projects README/AGENTS contain a dated compact snapshot. Read this checkpoint/progress first and refresh those snapshots when the active project or milestone changes; detailed tracking remains here. A new session continues existing verified work instead of rerunning setup.
