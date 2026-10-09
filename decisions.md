# Projects decisions

Append-only decision log. Current plan: [roadmap.md](roadmap.md). Record a superseding decision when a choice changes.

| ID | Date | Decision | Reason / consequence |
| --- | --- | --- | --- |
| D001 | 2026-10-09 | Keep the full architecture-learning roadmap and complete small-model builds. | Deep mechanism understanding is needed before informed optimization. |
| D002 | 2026-10-09 | Implement and train small models; inspect gradients/losses and explain observations. | Reading/remembering alone is insufficient for this learner. |
| D003 | 2026-10-09 | Projects implements and studies established models/techniques. | Hands-on learning and known-method exercises stay broad and complete here. |
| D004 | 2026-10-09 | Research owns narrow questions, focused implementations/reproductions and improvements. | SSD/weight/KV optimization research is not mixed into Projects progress. |
| D005 | 2026-10-09 | Add memory/state/access study to each learning project. | It connects model understanding to later focused research without claiming novelty. |
| D006 | 2026-10-09 | Keep implementations as independent submodules; share reusable plumbing where useful. | Preserve ownership/history and avoid repeating harness work. |
| D007 | 2026-10-09 | Move the workspace CLI to independent top-level ../cli/. | It serves both umbrellas and does not belong to Projects. |
| D008 | 2026-10-09 | roadmap.md/progress.md/resume.md are canonical direction/status/checkpoint. | Avoid losing the sequence or inheriting stale package checkpoints. |
| D009 | 2026-10-09 | Preserve the original detailed catalogue and review/setup packages as reference/history. | Authored material and evidence remain available without competing as current status. |
| D010 | 2026-10-09 | Start P01 GPT-1 with a complete tiny CPU model; defer large allocations and SSD trials. | Establish correctness, training behavior and cache semantics first. |
| D011 | 2026-10-09 | Track code tests, learner understanding, measured results and publication separately. | Seeded examples or passing builds do not prove mastery/research outcomes. |
| D012 | 2026-10-09 | Verify kickoff in a disposable workspace and leave actual P01 implementation/training for the learner session. | Readiness means the workspace/creation path is tested; the child model runtime is selected/locked during first kickoff. |
| D013 | 2026-10-09 | Consolidate all shared Projects documentation in docs/, including the existing website repository. | Keep the Projects root limited to entry points, controls and implementations; no competing project_docs tree. |
| D014 | 2026-10-09 | Version project tutorials and whiteboard preparation under docs/tutorials/. | User overrides Dashboard's ignored/private tutorial boundary in this workspace. |
| D015 | 2026-10-09 | Use C++ and Python for Projects implementations and tutorial examples. | Build understanding using the learner's known languages. |
| D016 | 2026-10-09 | Merge legacy roadmap content into the existing roadmap/setup/progress docs and remove both old files. | Preserve all model plans and links with one canonical plan; this supersedes D009’s separate-file retention. |
| D017 | 2026-10-09 | README.md and AGENTS.md are the only visible Projects-root files. | Tracking/setup/tutorial work stays in docs; use the single workspace launcher ../ai and preserve hidden repository controls. |
| D018 | 2026-10-09 | Embed dated project/milestone/prior-work context in existing README/AGENTS entry points and reconcile it with canonical docs each session. | Agents resume the actual next action; update snapshots when milestones change while keeping detailed tracking under docs. |
| D019 | 2026-10-09 | Develop/check locally; require Colab Free training profiles and thin common-trainer notebooks for P01–P04. Decide later tiers at kickoff; Pro later when requested. | Initial interpretation of the first few projects; complete miniature architectures, variable Free resources and measured budgets. Extends D010's CPU-first starting scope; local correctness remains first. |
| D020 | 2026-10-09 | Full training checkpoints must persist outside the runtime and pass interrupted-save and fresh-runtime recovery gates. | Preserve optimizer/scheduler/scaler/RNG/data-position state, verified generations and older valid fallbacks. Weight-only saves or local round trips do not prove training recovery in Colab. See COLAB.md. |
| D021 | 2026-10-09 | Adapt connected, motivated and adaptive teaching from pinned amosblomqvist/learn while keeping authoring, live tutoring and direct explanation distinct. | Bounded checks and scoped premises; artifact work needs no human diagnostic replies, live feedback waits for answers, existing reader/schema remain, human answer retention is opt-in. No pi extensions or efficacy claim. |
| D022 | 2026-10-09 | Insert P00 Tokenization and Transformer Foundations before P01; keep all existing project IDs stable. | Ground-up tensors/gradients, byte/BPE, attention and complete tiny encoder–decoder training form one bounded prerequisite project. Supersedes D010/D012's immediate GPT-1-first kickoff. |
| D023 | 2026-10-09 | P01 means Small GPT Language Model, with pre-norm/learned positions/GELU; exact GPT-1/post-norm/classification becomes optional reference study. | The user wants a small GPT-style model, not exact GPT-1. Preserve older plans/tutorials; M1–M3/M5 are required, M4/M6 optional. P02 strengthens GPT-2-specific coverage and cache behavior. |
| D024 | 2026-10-09 | Extend the Colab Free/full-state persistent recovery requirement to P00–P04. | P00 ends in a complete trained conditional model and must survive runtime loss. Extends D019; no notebook/training/hosted capability is established by roadmap checks. |
| D025 | 2026-10-09 | Transfer versioned tokenizer/data/trainer contracts from P00 into an independently runnable P01; verify again for the new objective. | Avoid duplicate foundations and fragile workspace-relative imports. Encoder–decoder weights are not a GPT checkpoint. Synthetic task success and learner mastery remain separate. |
