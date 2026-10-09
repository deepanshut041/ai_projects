# Projects progress

Updated: 2026-10-09
Authoritative learning-project status and evidence ledger. Plan: [roadmap.md](roadmap.md). Current checkpoint: [resume.md](resume.md).

Statuses: not-started, in-progress, blocked, complete, deferred. Complete requires the roadmap gates and evidence. Track model verification, learner understanding and publication independently. “Not started” does not imply a blocker.

## Current project

P00 Tokenization and Transformer Foundations / P00-M1 is the next kickoff. P01 is now the Small GPT Language Model, entered after P00. The source-backed roadmap update is preparation: no foundation/GPT repository, model code, notebook or training exists. The three seed labs do not complete P00. Historical GPT-1 plans/tutorials remain optional references; earlier session rows describe the previous sequence.

Compute requirement: P00–P04 develop/check locally and train complete miniature models on Colab Free under [COLAB.md](COLAB.md). Shared requirements are recorded; child notebooks, full persistent checkpoints and hosted recovery proof are not implemented/tested yet. Decide later project tiers from measured evidence, with Pro when requested.

| ID | Project | Status | Evidence / coverage |
| --- | --- | --- | --- |
| P00 | Tokenization and Transformer Foundations | not-started | Source-backed six-milestone plan saved; M1 tensor/gradient/runtime/source/budgets next; implementation/training/recovery/mastery pending. |
| P01 | Small GPT Language Model | not-started | P00 prerequisite; pre-norm text LM required; exact GPT-1/classifier optional; Free run/recovery pending. |
| P02 | GPT-2 | not-started | Colab Free profile/notebook and recovery required; unverified. |
| P03 | Llama 2 → 3 | not-started | Colab Free profile/notebook and recovery required per variant; unverified. |
| P04 | Qwen3 Dense | not-started | Colab Free profile/notebook and recovery required; unverified. |
| P05 | Switch-style routing → Qwen3 MoE | not-started | — |
| P06 | Qwen3-Next | not-started | — |
| P07 | PaliGemma | not-started | — |
| P08 | Qwen3.5 | not-started | — |
| P09 | Qwen3.8 dense VLM | not-started | — |
| P10 | Gemma 4 E2B text | not-started | — |
| P11 | DeepSeek V3 → V3.2 | not-started | — |
| P12 | DeepSeek V4-Flash-style | not-started | — |
| P13 | Qwen3.8-Flash-Next-style | not-started | — |
| P14 | Kimi Linear → K3-style | not-started | — |
| P15 | DeepSeek V4.1-Flash-style | not-started | — |
| P16 | Mamba 2 → 3 | not-started | — |
| P17 | RWKV-7 | not-started | — |
| P18 | LLaDA → iLLaDA | not-started | — |
| P19 | Gemma 4 unified multimodal | not-started | — |

## P00 milestones

| ID | Status | Implementation / verification | Training and learner explanation |
| --- | --- | --- | --- |
| P00-M1 | not-started | Tensor/loss/gradient/optimizer checks and runtime/source/CPU/Free contracts pending | Hand-computed answers and tiny MLP fit/explanation pending |
| P00-M2 | not-started | Character/byte/BPE, artifact determinism/round-trip and leakage checks pending | Merge trace and token/byte interpretation pending |
| P00-M3 | not-started | Embeddings/positions/attention/masks/norm/FFN numerical checks pending | Attention/gradient/mask explanation pending |
| P00-M4 | not-started | Complete post-norm encoder–decoder, loss and conditional generation pending | Source/cross-attention/teacher-forcing explanation pending |
| P00-M5 | not-started | Common trainer/notebook/full persistent resume and fresh Free recovery pending | Copy fitting/held-out generation and controlled reversal pending |
| P00-M6 | not-started | Parameter/memory/profile/departure and P01 handoff records pending | Diagnosis, mechanism explanation and transfer assessment pending |

## P01 milestones

P01 begins after the P00 gates. M1–M3 and M5 are required; M4/M6 are optional extensions, and leaving them unstarted does not imply an incomplete required text-model scope.

| ID | Status | Implementation / verification | Training and learner explanation |
| --- | --- | --- | --- |
| P01-M1 | not-started | P00 handoff, source/config/frozen tokenizer/text-data and Free contracts pending | Decoder-only architecture and shifted-target explanation pending |
| P01-M2 | not-started | Complete pre-norm decoder/reference/causal/gradient checks pending | Differences from P00's post-norm/sinusoidal/ReLU model pending |
| P01-M3 | not-started | Text LM train/generate/common notebook/full resume/hosted recovery pending | Loss/gradient inspection, held-out LM evaluation and sampling pending |
| P01-M4 | not-started (optional) | Historical GPT-1 post-norm/pooling/classifier pending | Optional downstream objective/explanation pending |
| P01-M5 | not-started | Parameter/memory/profile evidence pending | Diagnosis/lifetime/access map and limitations pending |
| P01-M6 | not-started (optional) | Early cache parity/comparison pending; P02 requires cache coverage | Optional cache tradeoff explanation pending |

## Tutorial status

[P01 foundations](tutorials/projects/p01/gpt-1-foundations/v1/lesson.md) is available as a reference-design orientation: seven diagrams, six practice checks, a proposed prerequisite route, motivated masking/target derivations, C++/Python examples and whiteboard preparation. The two toy examples pass. Model implementation/training and learner mastery remain pending. [tutorials/index.json](tutorials/index.json) tracks P00–P19. The existing GPT-1 lesson is a preserved historical reference, not the P00 starting guide or a faithful description of required P01 pre-norm scope. No P00 lesson exists yet; author it from actual milestone work.

## Session log

| Date | Project | Work and evidence | Next action |
| --- | --- | --- | --- |
| 2026-10-09 | Planning | Architecture roadmap reviewed; hands-on learning and Projects/Research separation recorded in decisions.md | Start P01-M1 |
| 2026-10-09 | Workspace | Canonical plan/tracker/checkpoint created; CLI moved to workspace cli/ with history and local edits preserved | Integration verified; begin P01-M1 |
| 2026-10-09 | Readiness | CLI 48 tests; all 4 existing Projects test/build entries and 3 demos passed; documentation content/browser and isolated GPT kickoff passed. START_HERE.md added; scaffolds now follow their area owner and ignore checkpoints. | Begin P01-M1; no model milestone completed |
| 2026-10-09 | Documentation | All shared files consolidated under docs/; existing documentation history and 139 source hashes preserved. Tutorial/whiteboard skills adapted; P01 orientation and two language examples added. CLI 54 and Docs 7 tests/build/browser passed. | Start P01-M1; tutorial availability is separate from mastery |
| 2026-10-09 | Root cleanup | Retired both identical old roadmap copies after preserving 12 sections, 19 model plans and 60 resource URLs in roadmap.md, SETUP.md and this checklist. Removed the Projects-root ai wrapper; root visible files are README.md and AGENTS.md. | CLI 54, Docs 7, build/browser 13 records and shared launcher pass; start P01-M1 |
| 2026-10-09 | Skill readiness | All 7 entries/references/links validate; tutorial and planning/record workflows exercised in isolation. Fixed coverage/render/snapshot/proof handling and Project/Research routing. CLI 64 and Docs 7 tests/build/browser pass. | Start P01-M1 with workspace-local skills |
| 2026-10-09 | Agent context | Existing workspace/Projects/Docs README and AGENTS now expose P01-M1, completed preparation and the next action; session instructions require reading/reconciling canonical progress and updating snapshots on milestone changes. | Continue P01-M1; no model gate advanced |
| 2026-10-09 | Tutorial improvement | Adapted teaching ideas from pinned amosblomqvist/learn into bounded prerequisite checks, concept dependencies, motivated steps and attempt/feedback/transfer guidance. P01 pilot and composing skill validators/examples pass; site/instruction checks are recorded in the [owning package](implementation/2026-10/learn-inspired-adaptive-tutorial-improvements/06-review.md). | Begin P01-M1; no human mastery or model gate advanced |
| 2026-10-09 | Colab requirements | [Free training/recovery contract](COLAB.md), roadmap/kickoff/setup/P01 plan and six snapshots updated. [Verification](implementation/2026-10/colab-free-training-and-recovery-requirements/06-review.md): 7 content tests, build/browser across 14 records, desktop/mobile guide inspection passed. | gpt-lab M1 plus Free setup/storage/budgets; notebook, training and hosted recovery pending M3 |
| 2026-10-09 | Commit baseline | Reviewed the nine independent repositories; local commit preparation covers six dirty owners, preserves history and updates umbrella pointers after child commits. CLI 64, Projects Docs 7, Research Docs 5 and Generalization 4 tests pass; both browser suites, skills/examples and TUI/doctor pass. [Review](implementation/2026-10/ai-workspace-setup/07-pre-p01-commit-review.md). | Begin P01-M1 after the final clean-tree/pointer audit; no model gate advanced |
| 2026-10-09 | P00 roadmap revision | Source-backed P00 tokenizer/attention/full encoder–decoder plan added; P01 clarified as small GPT, historical scope optional. Six snapshots, kickoff/state/compute/tutorial membership/source catalog/site/architecture updated. [Review](implementation/2026-10/project-zero-foundations-and-small-gpt-roadmap/06-review.md): Docs 7 tests/build/browser across 15 records/13 sources, CLI 64 tests/TUI, preservation/link checks and desktop/mobile inspection pass. | Begin P00-M1; no foundation/model/training/hosted/mastery gate advanced |
| 2026-10-09 | Commit baseline completion | Included the completed P00 roadmap revision under the user’s explicit instruction. Preparation is saved in focused local commits, with child revisions recorded before umbrella pointers; [commit review](implementation/2026-10/ai-workspace-setup/07-pre-p01-commit-review.md) and Git history retain scope/checks. | Begin P00-M1 after the final clean-tree/pointer audit; no model gate advanced |

Workspace integration evidence: [CLI relocation tracker](../../cli/docs/implementation/2026-10/workspace-cli-relocation/05-tracker.md). CLI 46 tests, Projects content/build/browser checks, canonical links/IDs and original catalogue/history preservation passed. This completes infrastructure work, not P01.

Append a session row after meaningful work. Update status/coverage only with actual evidence; link the child-owned package, commands, source revisions, configuration, run results and learner explanation. Keep bulky/generated runs ignored. No roadmap models are externally published.


## Detailed completion checklist

Apply this inherited checklist alongside the [current roadmap gates](roadmap.md). These unchecked items are requirements, not evidence of completed work.


Advance when you can:

- [ ] Draw the full model and annotate important tensor/state shapes.
- [ ] Explain each model-specific equation and implement it in a small reference case.
- [ ] Run a complete forward pass, loss, backward pass, and generation path.
- [ ] Demonstrate causal or conditional-generation correctness.
- [ ] Demonstrate equivalent state/cache execution where expected.
- [ ] Save/reload the model and count its exact parameters.
- [ ] For P00–P04, run the thin common-trainer notebook on Colab Free; verify full-state persistent saves, interrupted-write fallback and recovery after runtime deletion. Record actual compute/budget evidence.
- [ ] Compare at least one optimized path with the reference numerically.
- [ ] Produce a reproducible local profile and interpret its limits.
- [ ] Record the implementation's coverage and simplifications.
- [ ] Propose the next experiment from observed evidence.

Reading a tutorial or obtaining plausible-looking output is not the completion criterion. A working, verified implementation and an evidence-backed explanation are the required deliverables.
