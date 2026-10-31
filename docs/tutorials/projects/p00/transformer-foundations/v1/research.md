# Tutorial 0 research and teaching contract

## Complete topic rewrite — 2026-10-12

The user asks for a substantial mathematical deep-learning prerequisite course for Attention Is All You Need, organized by important concepts. Examples such as loss, optimizer, feedforward, encoding and embedding are scope indicators, not an exhaustive list. The [25-topic outline](project/course-outline.md) was written first, then lessons were authored in dependency order. This supersedes the prior eleven-page narrative structure.

The coverage starts with linear algebra, tensor storage/runtime, calculus, probability and information; develops nonlinear networks, initialization, full backpropagation, stable objectives and optimization; then explains data/text, embeddings, positions, attention/masks/heads, FFN/residual/LayerNorm, regularization and encoder–decoder learning. Evaluation, debugging, complete-state recovery and the paper assembly close the course. General Python primers are omitted for an advanced Python learner.

Every lesson has an essential explanation, defined axes/notation, worked numbers, a runnable probe or precise reference/implementation check, failure cases, a changed-condition practice and a specific personal-notes prompt. Native practice answers remain revealable; notes templates have no invented learner answers. Notebook/source assets supplement the written lessons without replacing implementation Python.

## Sources and departures

The original paper's arXiv1706.03762v7 architecture/training sections were inspected. Original post-norm/ReLU/Adam and sinusoidal/shared-embedding choices are explicit, and later pre-norm/GELU/AdamW comparisons do not silently replace them. Primary optimizer/initialization/normalization/residual/weight-tying papers, Unicode normalization and official PyTorch2.14 APIs support the contracts. D2L1.0.3 is the written supporting reference. [Source notes](project/source-notes.md) record implementation-convention differences and proof limits.

Nielsen's2015 book/siteDecember2019 Chapters1–3 were previously inspected via source-domain indexed excerpts after direct HTML returned502. Essential gradient/loss derivations are original and locally checked; no claim of a fresh full-book download is made. No source exercises, prose, illustration or old Python implementation is copied.

Numerical fixtures use row-vector weights, whereas PyTorch Linear stores `[out,in]`. The controlled2→2→2 Tanh fixture has12 parameters and a supplied arithmetic label; the actual2→8→2 classifier has42 parameters and a distinct opposite-sign labeling rule. Educational byte-BPE uses saved ranks, document boundaries and its own special IDs, not a pretrained model's tokenizer identity.

## Preservation and verification

An ignored verified archive preserves the pre-rewrite course. All133 later-course files and both implementation children have a byte/hash baseline. Existing chapter IDs/URLs and the learner's old Lesson3A continuation link remain reachable even though displayed topic numbers/order change. The prior study-note/notebook exports are not overwritten; new outputs use a separate directory. P00 reference-row removal from the TUI remains in effect.

Source mathematics, standalone lesson snippets, new attention/embedding/position probes, full classifier/recovery checks and source/notebook export are verified separately. Publisher examples/manifest/frozen assets and independent root/prefix browser/visual review complete local preparation. Selected results belong to the owning rebuild package. No authoring/reference check advances learner P00-M1, translation/model completion or hosted Free recovery. No commits/publication are requested.
