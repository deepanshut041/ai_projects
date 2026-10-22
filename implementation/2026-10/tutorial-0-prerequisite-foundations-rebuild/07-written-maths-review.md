# Tutorial 0 written maths and notes review

Updated: 2026-10-12. Local preparation; no commits or publication. Learner P00-M1 remains Lesson 2 setup complete with Lesson 3 part A linear() next.

## Delivered material

The existing eleven-lesson route now connects affine offsets, shared-parameter contributions, local gradient descent, likelihood/NLL, output loss gradients and a complete two-layer Tanh backward pass. Four added revealable questions require changed-case calculations. Nielsen chapters 1–3 are optional written companions; D2L §5.3 supports reverse dependency order. Essential explanations and original calculations stay in the lessons. Source-domain indexed Nielsen excerpts were inspected after direct HTML retrieval failed; this does not claim a full current-book download.

Every lesson has a notes prompt and stopping point. The editable eleven-topic Markdown workbook and cleared study notebook follow the same route. Its five standard-library code cells import the checked source; the 42-parameter trainer and separate training/Colab notebook retain their roles. Existing personal notes are protected from overwrite. Notebook/source delivery remains local.

## Verification

- 57 numerical/tensor/text/new-maths tests and 23 classifier/export/recovery tests pass: 80 total. Six native publisher examples pass, including the new complete numerical report.
- The complete 2→2→2 and unequal-width 2→3→3 fixtures agree with CPU float64 autograd to maximum absolute differences 2.22e-16 and 1.67e-16. Every input, weight and bias has an independent forward finite-difference comparison. The changed-label gradients are independently checked as -3 times the original fixture's derivatives.
- Source ZIP member sizes/hashes verify. All five study code cells execute in a fresh process beside an isolated extracted source copy. Both notebooks have empty outputs/null execution counts and syntax-valid cells. Generator/exporter overwrite rejection preserves existing notes bytes.
- Nine reader content tests pass. Initial root and /project_docs/ builds/browser suites pass for 86 records, with 320/390/768/1440 widths, math, local/source links, practice/scenarios, downloads, navigation and print. Visual review found long new mobile equation rows; those were stacked, and post-correction isolated root and prefixed browser suites pass for the same 86 records. The last label-rule clarification is checked on root and the live prefixed reader at 390px: ready state true, no Katex errors, contained page.

Numerical/export evidence: [written-maths-verification.json](written-maths-verification.json). Exact preservation: [written-maths-preservation.json](written-maths-preservation.json), using [the baseline](before-written-maths.json). All 133 later-course files, both child heads/status/nonignored files, eleven lesson paths and twenty earlier tutorial references are retained. Only the original source exporter changed intentionally; new math and study sources are separate files.

## Four-pass teaching review

Continuity: prediction still starts before setup; affine maps/derivatives/loss/modules connect into the existing classifier and recovery route. External reading is optional. Each note prompt addresses its lesson, and the same notes workbook supports returning to a stopping point.

Maths: symbols, row/column orientation, reduction denominators, loss assumptions, activation derivatives and update timing are explicit. The output cross-entropy cancellation is distinguished from hidden saturation; the affine-corner contradiction concerns representability, not guaranteed training success. The twelve-parameter arithmetic fixture remains distinct from the 42-parameter trainer.

Implementation: new standalone snippets, complete imported math callable, current publisher examples and study notebook/source export execute. The backward listing is explicitly an interior fragment with a runnable complete callable beside it; it is not advertised as a standalone script. Existing learner files are preserved.

Readiness: notes/source checks, actual learner explanations, task fitting, local recovery and hosted deleted-runtime recovery have distinct evidence. No author verification advances learner or hosted gates. Desktop and mobile equations, whole-network tables, notes template and cross-entropy derivation are visually reviewed. Long new equations are stacked rather than requiring horizontal scrolling. Both isolated browser suites pass, and the live reader serves the revised course. All local work is complete; learner/hosted gates remain open.

A root attempt against the shared output timed out after a preview process rebuilt dist with the prefixed configuration. Console 404s identified the asset-prefix mismatch. Independent root/prefix source copies eliminated that output race; both full suites passed. No site-code change or unrelated process shutdown was required. See [reader evidence](written-maths-reader.json).
