# P00 foundations and P01 small GPT plan

Updated: 2026-10-09
Lead: ai-project-workflow; supporting workflow: ai-paper-review. Status: selected design; model execution not started. The [canonical roadmap](../../../roadmap.md#project-0-tokenization-and-transformer-foundations) owns the live milestone and tokenizer/task gates; [research](01-research.md) and [pinned source provenance](source-provenance.json) support the selection.

## Ownership and scope

Documentation work belongs to ai_projects/docs/; six compact snapshots also live at Projects/workspace entry points. The site renders canonical files directly. Refresh current labels in the existing CLI-owned architecture package rather than changing CLI runtime behavior. Preserve current Git edits, historical packages/immutable tutorial revision, source catalog and later model notes.

Future implementation belongs to independent `projects/transformer-foundations-lab/`, registered at kickoff; no model child is created by this request. Its own AGENTS/README/Git/checkpoint and implementation package must be inspected/created before deep code edits. P01's proposed gpt-lab is a separate future child. Research remains deferred intake.

## P00 implementation contracts to copy into the child

| Milestone | Input → output | Required checks |
| --- | --- | --- |
| M1 | Sources/runtime + generated labels → tensor/gradient/optimizer primer and tiny MLP fit | Hand-computed answer, selected finite-difference/autograd agreement, finite gradients, loss inspection; CPU/Free profiles and bounded storage/setup plan. |
| M2 | Train-only text records → character/byte/BPE implementation and frozen artifact | Exact Unicode/empty/repeated text round trips, deterministic tie/merge ranks, save/load identity and split-before-fit checks. |
| M3 | Tiny numerical cases → embeddings/positions/attention/masks/norm/FFN | Independent tensor math, gradients and invalid-row behavior; causal/padding perturbation checks. |
| M4 | Verified components → complete post-norm encoder–decoder | Full forward/loss/backward, source conditioning, target causality, PAD-aware loss, EOS/bounds and request reset. |
| M5 | Complete model + disjoint synthetic pairs → bounded train/evaluate/generate and recoverable run | Fitting versus held-out exact/token metrics, copy and controlled reversal; common trainer/notebook; all COLAB.md recovery gates. |
| M6 | Code/runs/errors → profile/coverage/explanation and P01 handoff | Count actual parameters, memory/lifetimes, diagnose one failure/change, explain attention/conditioning/teacher forcing and version transfer. |

### Tensor, loss and mask conventions

Use batch-first tokens `[B,S]` / target inputs `[B,T]`; embeddings/memory `[B,S,D]` and decoder states `[B,T,D]`; Q/K/V `[B,H,L,D/H]`; self-attention score matrices `[B,H,S,S]` or `[B,H,T,T]`; cross-attention `[B,H,T,S]`; logits `[B,T,V]`. FP32 CPU is the first model path; FP64 tiny cases support selected gradient checks. Assert head divisibility/config and finite outputs/expected gradients.

A project-level boolean mask means **true = visible**; adapt deliberately when calling an independent library reference. Encoder/cross-attention hide source padding; decoder self-attention combines target padding and causality. Valid queries need at least one key. Reject empty/all-padding sources; define padded query rows safely rather than propagating softmax NaNs. Encoder attention is bidirectional, not GPT-causal. Hold the source fixed and perturb future decoder inputs for causality checks with dropout off; change only padded source IDs for padding isolation; perturb/shuffle the real source for conditioning checks.

Teacher-forced decoder inputs are BOS plus target prefix; labels are target plus EOS. Compute stable cross-entropy over non-PAD labels and report that denominator. Greedy evaluation receives only source+BOS and generated prefix, stops at EOS or declared maximum, and reports exact full-target matches separately from token accuracy/teacher-forced loss. Source-conditioned task success is not language capability.

### Tokenizer and provenance

Use bytes 0–255, PAD/BOS/EOS/SEP 256–259, merges ≥260. Declare lexicographic token-pair tie breaks, train-only records, no cross-record/special merges, fixed merge ranks at encode time and explicit special-token insertion. Persist schema/vocabulary/merges/policies/split/source hashes; freeze tokenizer hash into run identity. Character OOV handling is explicit; byte encoding covers valid unseen Unicode. Invalid generated UTF-8 has a declared rendering policy. No GPT tokenizer compatibility claim follows from implementing simple byte-BPE.

Generated digit/text copy pairs are the first required model task, using V260 byte tokenization. Optional bounded BPE run and controlled reversal reinforce transfer. Deduplicate/split source examples first; seed/generator/length/split identity is tracked, generated bulk data remains ignored. Proposed held-out gate: ≥90% exact matches on 100+ unseen examples in the trained length range, reported token accuracy and source-shuffle degradation. Finalize gates/run budgets before training; failed runs remain failed and changes are new protocols.

### Architecture, compute and recovery

Local proposal: 1 encoder + 1 decoder, D32/H4/FFN128/source-target32/V260, FP32/dropout0. Free proposal: at most 2+2/D64/H4/FFN256/source-target64/V≤512, sized from measurements. Exact parameter/time/memory fit is unmeasured. Preserve paper post-norm, sinusoidal positions, embedding scaling, ReLU and cross-attention; document tying/regularization/scheduler/data deviations. Annotated Transformer's convenience pre-norm is not paper post-norm authority.

Copy [COLAB.md](../../../COLAB.md) into the child package's requirements: thin common-trainer notebook, reproducible source ZIP/dependencies and CPU fallback, explicit new/resume, model/optimizer/scheduler/precision/RNG/data/progress/config identity, verified persistent checkpoint generations, at least two valid retained full states, corruption/interrupted-write fallback and recovery after deleting a Free runtime. Initial sessions cap at 60 minutes or 1,000 additional updates; saves requested at 5 minutes or 100 completed updates, with smaller drills. Final budgets/save latency/storage are measured; docs cannot prove hosted recovery.

### P01 transfer

Required P01 is a pre-norm decoder-only text LM with learned positions/GELU and final LayerNorm. Transfer versioned tokenizer contracts and tested trainer/data/recovery patterns into an independently installable/runnable child. No live sibling-path import or encoder–decoder-weight transfer. Source revision/license, artifact hash and differences are recorded. Use licensed text with disjoint splits and fixed next-token alignment; compare held-out loss within one tokenizer/objective. Reverify training/recovery for P01; P00 success alone cannot establish LM behavior. GPT-1 post-norm/classification and early cache are optional P01-M4/M6; P02 owns required cache and GPT-2 coverage.

## Documentation execution and validation

T01 inspect state/primary sources and pin versions → T02 update canonical roadmap/status/kickoff/compute/decisions → T03 synchronize six snapshots/tutorial index/source catalog/site readers and architecture → T04 existing content/build/browser checks, current CLI roadmap/list smoke and desktop/mobile visual inspection → T05 review and final ledger/checkpoint.

Acceptance includes P00-first current labels, P00–P04 compute scope, stable P01–P19 IDs, unchanged existing tutorial bytes/later model sections, independent ownership and no new model repository/training. Keep local filenames/links valid. Record commands, pass/fail outcomes and screenshot paths in verification/review as checks finish; no tests/hosted results are assumed from this plan.
