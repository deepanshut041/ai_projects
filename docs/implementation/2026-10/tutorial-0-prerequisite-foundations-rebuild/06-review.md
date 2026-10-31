# Tutorial 0 review

Completed locally on 2026-10-12. Tutorial 0 — Maths and PyTorch foundations is an eleven-lesson prerequisite route before translation/RoBERTa and Small GPT. Python programming is assumed. Syntax, function/collection/import primers and C++/Python syntax comparisons are removed. PyTorch runtime, tensor metadata/storage, autograd, registered neural state, data and training remain explained with maths at first use.

The route includes worked affine and embedding matrices, probabilities/NLL, derivatives and optimizer history; text IDs/original coordinates; split-safe batching/masks/denominators; modules, buffers, activation/normalization/dropout/mode; and a complete tiny classifier with compatible full-state recovery and a thin Colab export. Fifteen original diagrams, fourteen native practice reveals and three interactive cases are bound to the lessons. The prerequisite map explains where each concept is used in the later builds.

## Verification

- 74 source checks pass: 15 maths, 10 text, 26 tensor and 23 training/recovery/storage checks. Python/C++ reports match; independent finite differences, known answers, changed fixtures and library comparisons cover outputs and gradients.
- Final review fixed cancellation in huge-common-offset NLL: shifted-target arithmetic preserves ln(2) and the correct gradient. Added the view/reshape/contiguity trace; corrected embedding matrix row separators and the distinction between one buffer entry and two scalar values. In-lesson probes and cleared notebook schema also pass.
- The 42-parameter FP32 CPU classifier trains for 1,000 updates: held-out mean NLL 0.768718 → 0.000129996, correct 18/36 → 36/36. These are the declared structured grid and label rule, not an arbitrary-data result. Measured run: approximately 0.447 seconds, peak process RSS 313,974,784 bytes including imported runtime, parameter bytes 168 and checkpoint payload 26,040 bytes. Exact timings/RSS vary. The persistent-run budget is 8 MiB with 1 MiB free-space reserve.
- Fresh-process full-state parity passes in the suite; the final extracted source also saves 25, restores 25 and advances 50. Verified generations, corruption/incomplete fallback, source/config/runtime rejection, capacity failure and current-identity-only retention are checked. Separately assembled lesson snippets recover 4→5 under their own recorded identity.
- The final 16-member ZIP passes CRC, member-byte/hash checks and extracted-source execution. Its five notebook code cells contain setup/calls only, are output-cleared and syntax/schema checked. The final ZIP is bound in source-verification.json, superseding earlier temporary exports.
- Nine content tests pass. Root and /project_docs/ builds/browser suites pass across 85 records with search, navigation, math, controls, downloads, print and mobile layout checks. All 15 diagrams and representative desktop/mobile matrix/layout pages were visually reviewed. Browser screenshot lookup now uses declared diagram IDs rather than assuming manifest order matches lesson order; chapter search checks its owning tutorial while allowing guides that also refer to it.

## Preservation and limits

A verified ignored archive retains all 44 prior tutorial files. All 133 later-course files, both children's HEAD/status and 49 child source files are unchanged; all 9 original tutorial source files are retained. Stable eleven lesson paths/IDs, P00/P01 and the current linear() contract remain. Canonical guidance and six README/AGENTS snapshots show Tutorial 0 → Course 1 → Course 2.

The actual learner checkpoint is P00-M1 in-progress, Lesson 2 setup complete; resume the existing foundations child at Lesson 3 part A linear(). Author results do not complete learner implementation/explanation or Colab Free recovery. Hosted installation/resources/Drive/deleted-runtime recovery and GPU behavior remain unverified. No new commits or remote publication.

Selected evidence: source-verification.json, additional-pytorch-probes.json, lesson-assembly-verification.json, matrix-render-verification.json, reader-verification.json and preservation-verification.json. Temporary author notes/disposable runs are cleaned after these records are saved; ignored recovery archives and final source/notebook exports remain.

Further written maths and note-taking improvements are completed in [the subsequent review](07-written-maths-review.md); earlier counts and source snapshots above remain dated rebuild evidence.
