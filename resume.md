# Projects resume checkpoint

Updated: 2026-10-09
Read this first on continuation, followed by [progress.md](progress.md), [roadmap.md](roadmap.md) and [decisions.md](decisions.md).

- Current project: **P00 Tokenization and Transformer Foundations**.
- Current milestone: **P00-M1, not started**.
- Objective: build tensor/gradient, tokenizer and attention foundations, train a complete tiny encoder–decoder, then enter **P01 Small GPT Language Model**. Exact GPT-1/post-norm/classification is optional historical study.
- First-session guide: [START_HERE.md](START_HERE.md). No foundation/GPT repository, model implementation, notebook or training exists. Seed labs do not complete P00.
- Next action: at implementation kickoff create/resume independent transformer-foundations-lab and its child-owned package, pin sources/CPU runtime, work through tensor/autograd/loss/optimizer checks and finalize CPU/Free data/storage/budget contracts for P00-M1.
- Proposed kickoff commands (not executed against a model repository):

```bash
./ai new project transformer-foundations-lab --title "Tokenization and Transformer Foundations"
./ai task new "P00 foundations implementation and learning" --owner transformer-foundations-lab --domains project,experiment
```

Run from workspace home; from Projects use ../ai. If the child/package exists, inspect its README/AGENTS, Git state and checkpoint and resume it. Copy implementation-specific contracts from the [current plan](implementation/2026-10/project-zero-foundations-and-small-gpt-roadmap/02-implementation-plan.md) into the child package before deep edits. Creating the scaffold will not complete M1.

- Proposed P00 CPU model (M4): 1 encoder + 1 decoder layer, width 32, 4 heads, FFN 128, source/target limit 32, V260, FP32. Proposed Free profile: up to 2+2 layers/width 64/context 64/V≤512. Parameter counts/resource fit/training outcomes are unmeasured. M1 starts with a tiny MLP/gradient exercise before attention.
- Compute: [COLAB.md](COLAB.md) requires local correctness plus Colab Free training/full-state persistent recovery for P00–P04. P00-M5 supplies the notebook and recovery evidence. Pro is later when requested. No hosted run/recovery is verified.
- Completed preparation: workspace/roadmap/docs consolidation, independent CLI, seven skills, adaptive teaching guidance and the preserved [GPT-1 tutorial](tutorials/projects/p01/gpt-1-foundations/v1/lesson.md). The [P00/P01 source review](implementation/2026-10/project-zero-foundations-and-small-gpt-roadmap/01-research.md) and revised six-milestone route are saved. Learner knowledge remains unassessed; no model milestone advances from documentation work.
- Tutorial state: P00 lessons are planned in [tutorials/index.json](tutorials/index.json), not authored. The earlier GPT-1 v1 remains an immutable historical orientation; do not present its post-norm/classifier scope as required P01 pre-norm implementation.
- Historical checks/evidence: [readiness audit](implementation/2026-10/llm-roadmap-review-and-gpt-1-starting-plan/09-readiness.md), [skill readiness](implementation/2026-10/tracked-project-tutorials-and-whiteboard-learning/skill-readiness.json), [adaptive tutorial checks](implementation/2026-10/learn-inspired-adaptive-tutorial-improvements/06-review.md), [Colab documentation review](implementation/2026-10/colab-free-training-and-recovery-requirements/06-review.md) and ../../cli/docs/implementation/2026-10/workspace-cli-relocation/ record previous local verification. They are not model/hosted capability proof.
- Current roadmap-delivery verification: recorded in the [current review](implementation/2026-10/project-zero-foundations-and-small-gpt-roadmap/06-review.md) as checks finish; prior test counts are historical.
- Preserve current local edits/history and later roadmap notes. Earlier topic packages retain dated checkpoints; this live checkpoint supersedes the GPT-1-first next action.
- Learning gate: the learner predicts, implements selected components, trains/inspects/debugs and explains behavior. Tests, task success, understanding and publication are separate states.
- Research remains deferred intake with no active new inference experiment. No kickoff blocker is known; child runtime/hardware and task budgets are still to be measured.
- External publication: none.

At session end update the child package, progress.md and this checkpoint with actual evidence, unresolved questions and one next action. Refresh all six workspace/Projects/Docs README/AGENTS snapshots when current state changes; detailed history remains here.

Local repository preparation and the completed P00 roadmap revision are saved in focused commits under the user’s explicit instruction. See the [commit review](implementation/2026-10/ai-workspace-setup/07-pre-p01-commit-review.md) and Git history. The next learning action remains P00-M1; no model work began during the commit review.
