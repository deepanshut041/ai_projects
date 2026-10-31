# Document Assistant — implementation and return-to-work record

The written course covers every planned phase. The model project remains runnable boilerplate. Use the reader's previous/next lesson route for teaching; use this checklist when implementation actually begins. All model-phase boxes below remain unchecked until receipts exist.

## Current executable boundary

From workspace home: `./ai run document-assistant`, `./ai test document-assistant`, `./ai tui`. TUI: Projects → Quantization Lab → Document Assistant → status/check/configuration → Run. The project uses Python 3.12.13 and frozen uv tooling. The existing answer/prepare-data/quantize/evaluate actions return exit 2 and create no model output. Keep this honest boundary while preparing each real action.

The tutorial's `examples/` are standalone standard-library Python/C++ teaching files. Copy/download them into a new scratch directory. They do not need the model environment and do not alter production source. Their assertions, exact outputs and separate audit are author evidence, not learner answers.

## Phase 00 — compatibility and measured budget

- [ ] Recheck current releases/support and select one exact source/checkpoint/runtime. Keep the current E4B revision as the proposed baseline until deliberately revised.
- [ ] Freeze runtime checkout, compiler/build flags, dependencies, tokenizer/template/config/source hashes and intended CPU/device placement. Inspect model license/access conditions before acquisition.
- [ ] Add a bounded backend probe under `src/core/` using the accepted runtime's actual API. Record load, rendered prompt tokens, EOS/output-cap behavior, placement, failure and fresh-process reload. Do not accept configuration inspection as generation.
- [ ] Measure host/device load, post-load, prefill and decode peaks for a tiny prompt. Freeze input/output/capacity/reserve and record units and allocation semantics.
- [ ] Exit only with a working floating reference and complete compatibility receipt; otherwise retain failure and revise the runtime/model decision separately.

## Phase 01 — canonical data and labels

- [ ] Create `src/data/documents.py`: canonicalize original UTF-8 text under an explicit version, preserve raw/canonical digests and offsets; round-trip spans and repeated paragraphs.
- [ ] Create `src/data/passages.py`: deterministic tokenizer-bounded chunking, parent section/spans, declared overlap and oversized-input policy. Test non-ASCII, repeated text, long paragraphs and number/unit boundaries.
- [ ] Create `src/data/membership.py`: document-group calibration/development/final manifest; check duplicates/related versions before splitting.
- [ ] Create `src/data/prompts.py`: render complete prompts with the pinned template, count actual tokens, pack fixed evidence in declared order and retain drops. Verify labeled support was actually supplied.
- [ ] Review answerability, answer facts/aliases and alternative evidence sets. Record public source/license and reviewers/adjudication. Freeze all identities before comparing weights.

## Phase 02 — floating reference and first Q8 artifact

- [ ] Preserve source-framework and converted floating agreement probe with explicit tolerance/token contract.
- [ ] Create `src/quantization/artifacts.py`: immutable manifests, exact conversion/recipe commands, tensor inventory and output hashes. Keep source and generated weights ignored.
- [ ] Produce Q8 from the same converted floating source; inspect actual composition and requested/applied operations.
- [ ] Create `src/evaluation/predictions.py`: append per-question raw/parsed outputs, tokens, finish reason, error status and identity. Check exact membership and retain partial results.
- [ ] Create `src/evaluation/quality.py` and `timing.py`: explicit rubric/denominators/paired outcomes and declared monotonic timing/synchronization boundaries. Test fixtures before a pilot.
- [ ] Run a tiny pilot, freeze the main development protocol, execute matched reference/Q8 quality/resources and verify fresh-process reload.

## Phase 03 — Q4 and diagnosis

- [ ] Add one supported four-bit recipe derived from the same floating artifact. Hold all matched factors fixed.
- [ ] Inspect code/scale/precision composition, disk bytes, live host/device memory and workspace; do not infer peak memory from a filename.
- [ ] Review all changed development answers, citations, refusals and truncations, including failures by length/evidence/numerical facts. Preserve reference outputs.
- [ ] Exit with an attributable comparison, even if Q4 fails quality or saves no latency. Do not select a winner from file size alone.

## Phase 04 — calibration and one advanced method

- [ ] Create a bounded activation collector with channel axes, token masks/counts, layer coverage, data hashes and memory/time receipts.
- [ ] Compare one calibration-distribution change at a fixed token/method budget, then a separately supported advanced recipe. Keep runtime changes visibly classified.
- [ ] Inspect whether requested AWQ/GPTQ/importance operations ran on intended tensors. Preserve skipped/unsupported warnings. Toy derivations do not establish backend support.
- [ ] Compare grouping/selective precision under a declared budget and held-out reconstruction/task evaluation. Freeze search space and selection rule before choosing.

## Phase 05 — cache/context study

- [ ] Freeze one weight hash and inspect the runtime's actual local/full/shared cache-owner map.
- [ ] Count logical/physical used/reserved cache storage with actual types, groups, residuals and allocation rounding.
- [ ] Compare only supported cache policies over bounded complete-prompt lengths, with fixed output allowance. Hold prompts matched within each length comparison.
- [ ] Retain quality, dropped evidence, position/distractor cases, TTFT/prefill/decode, memory and OOM/fallback receipts. Unsupported requested policy is not a successful cache-quantization run.

## Phase 06 — saved application and final assessment

- [ ] Implement `src/core/manifests.py` and `answering.py`: immutable generations, verified loading/fallback, request ownership, bounded generation, raw output retention, schema/citation checks and cleanup.
- [ ] Add a real adapter, then a local application entry point using that same boundary. Make backend/artifact/fallback/context state visible. Fixture demos stay explicitly labeled.
- [ ] Add an application TUI runner only after it serves real accepted requests; keep Checks separate. Test start/stop/error state through the shared controller.
- [ ] Exercise supported/unavailable questions, citations, document changes, over-budget input, cancellation, malformed response and fresh-process restart replay.
- [ ] Freeze development selection or report no feasible candidate. Seal hashes/rubric/protocol before final labels; retain final reference/candidate failures and paired uncertainty.
- [ ] Complete the readiness matrix with actual source/test/run identities and limitations. Learner explanation and hosted execution remain separate gates.

## Ownership and recovery

Production source belongs under this existing ordinary project folder. Shared controller changes belong in the independent `cli/` repository; shared teaching files belong in Projects Docs. Detailed personal implementation notes remain in ignored project `docs/`. Never overwrite authored work or large artifacts from a teaching extraction. Runtime selection, framework installation, model acquisition and full experiments are separate implementation actions, not hidden effects of reading or rendering this course.
