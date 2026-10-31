# P00 tutorial research and evidence

Updated: 2026-10-09. Owner: Projects Docs. Request: thorough practical Project 0 tutorial. Lead: workspace developer-learning-tutorials; supporting planning/project workflow and feature-whiteboard-defense. No delegation requested.

## Workspace inspection

Read workspace/Projects/Docs instructions and canonical checkpoints, roadmap, decisions and COLAB.md. Both Projects and Docs Git status were clean. P00-M1 remains not-started. No transformer-foundations-lab exists. Preserve P01 historical tutorial. Use the existing lesson schema/semantic SVG renderer/site; source at tutorials/projects/p00/transformer-foundations/v1/.

## Primary sources accessed 2026-10-09

| Source | Locator | Implication and limit |
| --- | --- | --- |
| https://arxiv.org/html/1706.03762v7 | sections 3.1–3.5, 5 | Post-norm encoder/decoder, three attention contexts, sinusoidal positions, ReLU, scaled embeddings. Authored arithmetic is independent educational work, not translation reproduction. |
| https://arxiv.org/html/1508.07909v5 | section 3.2 | Merge-based subword motivation. Document-bounded bytes and structural specials are educational departures. |
| https://raw.githubusercontent.com/karpathy/minbpe/master/minbpe/basic.py | BasicTokenizer train/encode/decode | Byte merges/rank encoding reference. Branch source read, not pinned compatibility specification. |
| https://raw.githubusercontent.com/harvardnlp/annotated-transformer/master/the_annotated_transformer.py | SublayerConnection.forward, lines 312–324 | Code uses pre-norm for convenience; preserve paper post-norm in P00. |
| https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html | mask/dropout arguments | Independent reference. True allows a key in this API. Manual tensor implementation is learning target. |
| https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html | index targets, ignore_index | Feed logits, exclude PAD targets; check metric denominators separately. |
| https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html | formula/normalized_shape | Last-dimension population variance and affine transform. |
| https://docs.pytorch.org/docs/2.14/notes/randomness.html | reproducibility limits | Same-environment resume parity is bounded; cross-device identity not promised. |
| https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html | general checkpoint | Optimizer plus weights necessary; workspace adds RNG/data/config/identity/persistence. |
| https://research.google.com/colaboratory/faq.html | resource limits/runtime | Free resources vary; teach budget/deleted-runtime recovery without hosted proof. |

## Authored scope

Twelve chapters: route/setup; tensors/layout; numerical loss/autograd/MLP; Unicode/bytes; deterministic BPE; data/shift/metrics; embeddings/positions; attention/heads/masks; complete model/generation; training/inspection; checkpoint/Free recovery; profiling/applications/P01 handoff. Each major concept connects motivation, derivation, trace, implementation, failure and transfer practice.

Standard-library Python/C++ examples and separately checked complete PyTorch educational reference. Bounded CPU runs are tutorial verification, not learner milestone completion. No child repository, hosted training, paid service, commit or publication. Scratch/data/weights stay ignored under .ai/.

## Live findings

System Python had no torch. Created ignored .ai/envs/p00-tutorial with Python 3.12.13, torch 2.14.1+cpu from official CPU wheels. Child dependency lock remains future kickoff work.
Existing browser suite hard-codes the first tutorial's P01 scenario. Adapt it to validate every lesson and all authored cases.
Acceptance: independent arithmetic/gradient/tokenizer/mask/model/state checks; lesson validate/examples/render; native site tests/build/browser; inspect all P00 boards and desktop/mobile controls.

## Final source/QA updates

Pinned and re-read minbpe 1acefe89412b20245db5a22d2a02001e547dc602 and Annotated Transformer debc9fd747bb2123160a98046ad1c2d4da44a567. References in lesson.json use these exact revisions. A local captured evidence excerpt binds the selected reference-check/failed-task report.

Fresh-process CPU resume also passed: N=8, K=3, dropout 0.1; all model/optimizer/scheduler/sampler/RNG/progress state and loss sequence identical. Standard-library/C++ lesson examples passed; structural render succeeded. Seven native content tests and full browser checks passed before concurrent reader changes (16 records, 13 catalog sources). Every scenario combination and answer reveal,downloads/print/mobile plus standalone SVGs were checked. Dependency board refined to two independent prerequisite strands, avoiding a false tensor-calculus→byte dependency.

A concurrent user-owned chat, “Support multi-file tutorials”, completed chapter/support navigation in the same Docs repository. Preserved its code/test edits. Native content test assumptions changed from two lesson records to 26 overview/chapter/support/complete records. Final author content tests pass (8); the reader owner's verification records builds/browser checks at root and /project_docs/. Independently confirmed the running P00 overview/chapter/workbook/complete routes return HTTP 200 and inspected final overview/mobile chapter screenshots. Final fingerprint and integration evidence are in 06-review.md and integration-checks.json. No authority to message that chat was inferred.

## Learner-readiness revision — 2026-10-09, in progress

User now requests reviewing and improving P00 so learning can start. This new instruction authorizes editing tutorials after the earlier cleanup freeze. Reuse this package and the existing v1 reader URLs; preserve the subsequent guide/website/implementation-evidence organization and all unrelated edits. Active learner state is P00-M1 not-started.

Audit: all twelve chapters and the workbook inspected. Coverage is broad, but the opening diagnostic assumes undefined BOS/EOS, setup mixes first reading with model-wide checks, and several tensor/PyTorch/gradient steps are too compressed for a first pass. The Colab recipe also invokes persistent_run before the mount/path cell defines it. Existing reference code/checkpoint/task outcomes stay historical and unchanged.

Plan: (1) rewrite the entry around one concrete first session and bounded learning checkpoints; (2) add a dependency-free first-session walkthrough and C++→Python/PyTorch syntax bridge; (3) expand worked data, embedding, mask, gradient and generation traces with hidden-answer transfer questions; (4) make the workbook a sequence of runnable learner sessions with stop/resume guidance; (5) fix recipe ordering and polish source/claim terminology; (6) validate arithmetic/examples, render the tutorial, run native/browser checks, inspect the revised entry and diagrams, and refresh canonical preparation snapshots. No learner child or training milestone is created.

Primary sources re-opened: Transformer arXiv v7 (architecture), PyTorch 2.14 broadcasting semantics (trailing-axis rules), PyTorch autograd beginner guide (requires_grad, backward and accumulation). New mathematics/examples are original teaching derivations; no learning-effectiveness claim.

Learner revision verification: all four declared numerical/tokenizer/C++ examples passed; the tensor/MLP reference now checks intermediate head ownership and passes. Five standalone new chapter snippets match their displayed results, and independent broadcasting, merge, FFN and mask cases pass. A bounded author smoke reached update 20, then explicitly restored checkpoint 20 and advanced to 40 with unchanged model/source identity; exact held-out generation remained 0%, as expected for a smoke rather than a quality claim.

Browser QA found a real harness limit: the complete P00 page is 64,957 CSS pixels tall at desktop size and whole-page PNG capture timed out in both prefix/root hosting. Chapter traversal, control checks and links reached that capture successfully. Updated only the browser test capture helper to retain full-page images up to 16,000 pixels and readable top/middle/end viewport images above that; every chapter/control assertion remains. Root verification uses a disposable copy so the live /project_docs/ service is not disturbed.

## Page rendering and reading review — 2026-10-09, completed

User requests a wider page-by-page rendering check and human-readable tutorials, with additional snippets where useful. Project workflow leads; tutorial/whiteboard guidance supplies the teaching review. Reuse this package. Inspect all published record routes at desktop/mobile sizes and the P00 chapters in reading order; inspect historical P01 rendering while preserving its authored content. Fix shared reader defects, awkward P00 prose and incomplete teaching snippets. Keep local organization, previous edits and learning state intact.

Initial observations: chapter pages repeat the whole-course question and all external references rather than orienting the current chapter; sections skip directly from h1 to h3; code is rendered at 12px; first-column table cells cannot wrap. P00 positional and mask examples omit their input/setup definitions, and attention arithmetic has no adjacent runnable weighted-mixture example. These are candidates to verify and repair, not claims that the rest of the site has passed.

Acceptance: no page-wide horizontal overflow at desktop/tablet/narrow mobile widths; readable code/tables/math/boards; stable chapter/support/download links and keyboard controls; execute new self-contained snippets and verify their printed outputs; all native content/build/browser checks at root and prefix; selected visual inspection evidence. No learner repository, hosted training or remote action.

Final findings: the baseline 160-view sweep had no broken images/math or page overflow. Larger type made existing table containment important; the managed development process retained old renderer imports and could overwrite a manual build with old table HTML. Root isolation passed while the running preview exposed this mismatch on questions/next-project-directions at 320px (372px scroll width). Restarted only the managed project-docs service with ./ai restart project-docs --json. Its table wrapper and 320px scroll width now match the updated renderer. The final 160-view live sweep and full prefix suite pass. Root suite also passes from a disposable copy.

Executed nine selected code blocks verbatim; outputs match head ownership, BPE replacement, six valid labels/log(260)/zero PAD-position gradient, embedding accumulation, positions and attention/masks. All twelve P00 desktop/mobile chapter views inspected, plus focused new examples, workbook, roadmap, source register and historical P01. Tables keep code intact and scroll locally; keyboard code/table/equation scrolling is verified. A new h1-to-section check caught the earlier P01 practice heading gap, fixed in the reader without changing P01 source. Four declared examples and eight native content tests pass. Final evidence and fingerprint are in 06-review.md.

## Implementation-guide inspection — 2026-10-09

The workbook supplies broad M1–M6 deliverables but only suggests src/foundations/, whereas new_repository creates src/transformer_foundations_lab/. Most instructions omit concrete file/test names and executable end conditions. The launcher entry p00 executes reference examples, not an independent learner repository. The reference tiny_transformer.py provides Config, LayerNorm, Attention, Block, TinyTransformer, records/batch/objective, Sampler/Trainer and publish/recover. Reuse its behavior and map component extraction precisely; leave heavyweight monolithic reference as a comparison. The reader already publishes frozen supporting Markdown with stable paths; separate build pages require no new rendering framework. Official uv PyTorch CPU-index and PyTorch checkpoint guides rechecked; explicit CPU dependency source and full optimizer/state recovery remain required.

Implementation-guide verification complete: the actual scaffold/import and documented CPU lock/install pass in a disposable workspace. Twenty-five authored files produce 23 passing cumulative/combined tests; new/resume reaches 20→40. The notebook is output-cleared and syntax-valid; source export/extraction hashes pass locally. No hosted execution. All 46 record routes pass at root and prefix with four-width checks; nine sequential build links and builder download pass live. Nineteen original reference/evidence assets remain byte-identical. Final scope and fingerprint are in 06-review.md and integration-checks.json.

## Simpler connected teaching — in progress, 2026-10-09

Revise P00 for a reader who cannot connect the chapters and build steps. Lead: project workflow; supporting workspace tutorial/adaptive/whiteboard instructions. Preserve technical depth, existing executable contracts and independent learner status. Inspection found circular reading directions, an early four-function/large-file jump, and terminology introduced before a concrete use. Make build pages a self-contained main route: plain goal, tiny trace, vocabulary/code reading, exact file work, check and next step. Concept chapters gain approachable openings and explicit connections; existing derivations and acceptance gates remain. Add a section index to supporting build pages if needed. Verify worked files/new examples, rendering, root/prefix browsing and selected desktop/mobile views. No learner child or hosted training.

Final simpler-guide results: 23 cumulative/combined checks plus four new mini examples and four partial-function stages pass. Root/prefix 46-record browser suites and native eight-test suite pass; live overview/build/section links inspected. No efficacy/mastery claim. Detailed findings, preserved-content boundaries and test repairs are in 06-review.md.
