# Review — P01 small GPT practical tutorial

State: authored and checked locally on 2026-10-10; final visual evidence is linked below. No learner milestone or external publication follows from these checks.

Subsequent user-requested cleanup: [retirement review](07-retire-gpt1-tutorial.md). The preservation and 57-record checks below describe the original delivery before the old GPT-1 tutorial was removed.

## Delivered artifacts

The requested P01 tutorial is available at [Lesson 1](../../../tutorials/projects/p01/small-gpt/v1/lessons/01-next-token.md). Its twelve lessons integrate explanation, maths, code and checks, rather than splitting theory and implementation into separate reading routes. The project area contains an ordered implementation checklist, source/departure record, reference runs and a P02 readiness boundary. Seven diagrams, four interactive scenarios and 24 revealable practice answers accompany the same route.

The complete educational reference has two pre-norm blocks, learned positions, causal attention, GELU, final normalization and a tied vocabulary head: 120832 default parameters. Its pinned Alice text pipeline, token-weighted loss/evaluation, seeded generation, common trainer, verified persistent state and thin Colab export are downloadable and taught in dependency order. It makes no pretrained GPT-2 compatibility claim.

The tutorial index and canonical preparation records, including all six compact README/AGENTS snapshots, now link the new required P01 course. Existing P00 and optional historical GPT-1 source files were preserved. No independent learner repository was created.

## Checks actually performed

| Check and date | Result | Evidence | Skipped or remaining work |
| --- | --- | --- | --- |
| Code and arithmetic, 2026-10-10 | 21 data/model/training tests and declared Python/C++ arithmetic examples pass | Downloadable test files; example-checks.json in course; [course reconstruction](course-verification.json) | Tests do not establish learner understanding |
| Teaching order, 2026-10-10 | All 12 stages pass in a new disposable directory; 11 embedded files match reference ASTs; new/resume and source export work at their teaching point | [verify-course.py](verify-course.py), [report](course-verification.json) | Only declared source/download steps are reconstructed |
| Fixed-batch diagnostic, 2026-10-10 | Mean NLL 5.596348 → 0.082698 after 300 updates; target <0.1 passes | [reference verification](reference-verification.json) | Two-string fit is not held-out task performance |
| Actual bounded text run, 2026-10-10 | 150 new + 150 resumed updates; held-out validation NLL 5.585223 → 2.415295; final perplexity 11.193075 versus unigram NLL 3.347120 | [reference runs](../../../tutorials/projects/p01/small-gpt/v1/project/reference-runs.md), [raw selected results](../../../tutorials/projects/p01/small-gpt/v1/project/reference-results.json) | Untouched test chapters not evaluated; outputs still weak/repetitive; single small corpus/run |
| Restart and state, 2026-10-10 | Fresh processes: 8 uninterrupted versus 4 + exit + 4 with dropout 0.1; all saved state exactly equal | [verify-reference.py](verify-reference.py), [report](reference-verification.json) | Same CPU/runtime/source/config only; abrupt process/copy interruption and hosted recovery remain explicit learner fixtures |
| Export/profile, 2026-10-10 | Final allowlisted export and generated notebook cells checked locally; CPU forward/backward profiles at contexts 32/64 recorded with scope | [course report](course-verification.json), [reference report](reference-verification.json) | No hosted execution, GPU profile or general performance comparison |
| Native site tests, 2026-10-10 | 9/9 passed | `npm --prefix website test` | Local reader checks |
| Builds/browser, 2026-10-10 | Both `/project_docs/` and independent `/` builds: 57 records, 13 sources; every record at widths 1440/768/390/320; maths, images, links/downloads, controls, navigation and print action passed | `npm run build` and `npm run test:browser` in each build directory; [selected screenshots](../../evidence/2026-10-10-p01-tutorial/README.md) | Print action verified; not a separately reviewed exported PDF |
| Preservation, 2026-10-10 | All 64 preexisting P00/historical tutorial files match baseline; all 20 reference-source digests match the recorded run | [preservation report](preservation-verification.json) | Preserves preexisting working edits rather than resetting Git |

## Four-pass review

- Continuity: the first action is a three-row next-token calculation. Every later lesson explains the problem introduced by the previous result and ends with its next step; project records are resumable work records, not another course.
- Maths: probability/log loss, axes/head geometry, variance scaling, normalization/residual gradients, GELU, cross-entropy gradients, AdamW/clipping/scheduling, valid-token reductions, sampling and parameter/storage estimates are defined at first use with small calculations and transfer practice.
- Implementation: create/append boundaries, delayed imports and export prerequisites were checked by reconstructing the taught files. Trainer code is broken into explained stages. Tests and actual reference outcomes are distinct from learner exercises.
- Readiness: required P01 M1–M3/M5 and optional M4/M6 are explicit. The exit record preserves complete Colab Free training/deleted-runtime recovery and the GPT-2 fidelity/cache boundary.

Visual inspection caught colliding labels in the initial pre-norm/data boards. The corrected pre-norm grid retains both the direct residual and normalized sublayer paths; the data board retains train-only fitting and separate frozen encoding. The browser suites were rerun after these changes.

## Limitations and handoff

Actual learning remains P00-M1 not-started. This is completed tutorial/reference preparation, not a completed learner GPT project. No learner repository, remote publication, paid compute or actual hosted training/recovery was created. Read the explicit additional interruption/identity/storage fixtures in the implementation record; the author checks do not silently pass them.

Begin at Lesson 1 for a preview. Before P01 implementation kickoff, complete/review the real P00 exit evidence. Then follow the same lesson route and record each required result in the independent learner repository.
