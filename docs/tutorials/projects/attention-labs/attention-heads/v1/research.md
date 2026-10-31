# Source and teaching review — 2026-10-30

Purpose: a self-contained course for a Python/C++ engineer learning attention mechanisms, with a whole-task overview followed by progressively deeper calculations. Explanations use original prose and executable references; the teacher's worked examples and presentation diagrams are retained alongside additional compact checks. The existing Attention Is All You Need tutorial and its accepted four-row fixture are not revised.

The user requested Chrome or Brave. Chrome verified the public playlist, video pages, original slide links and the full Slidev overview pages. The playlist lists Lectures 9 down to 1 plus one unavailable hidden entry. We map by the lecture number in each title, include Lectures 1–8, omit Lecture 9, and separate basic single-head foundations (1–7) from head/KV arrangements (8). No directory is created per basic attention type.

Read all slides in the eight included decks: 21, 20, 28, 17, 27, 25, 21 and 22 slides respectively (181 total). Their progressive motivation and single-operation-per-pass structure inform this route. The initial three-row-only rewrite omitted too much of the teaching route. The sequential correction restores the six-word and five-word examples, full diagrams, bar charts, table values and code demonstrations. The compact fixture remains for transfer checks, a projection derivative and loop/packed equivalence.

Also inspected Dive into Deep Learning 1.0.3, Attention Pooling by Similarity (11.2 introduction and pooling equation), Attention Scoring Functions (11.3 masked softmax, dot-product scaling, additive scoring), the Bahdanau alignment paper, Luong's score alternatives, and Attention Is All You Need v7 sections 3.2.1–3.2.3. D2L's explicit width contracts and distinction between score and normalization are useful; our route keeps those contracts beside the worked inputs rather than sending the learner through framework helper code. Primary papers bound the historical and architectural claims.

Departures from source simplifications: the category map is a useful checklist, not an exclusive partition of every method; attention does not always mean softmax; learned RNNs have no universal twenty-word collapse; a conventional decoder predicts vocabulary logits rather than nearest embeddings; scale normalization needs distributional assumptions; concentrated softmax does not imply every training gradient vanishes; raw embeddings can themselves be trainable; shared projections cannot distinguish identical position-free inputs with identical memory; zero-then-renormalize masking is equivalent in exact arithmetic but can fail numerically; dense causal code may still compute forbidden raw scores; dropout preserves a local expectation, not unit row sums in each draw or a whole nonlinear model's evaluation output; MHA allows but does not guarantee head specialization; packed execution has no hardware-independent timing guarantee.

Caption review: the full English automatic subtitle texts from DownSub were read alongside the presentations. Lecture 6 also has a matching Transcript.you copy. These captions supply the spoken state transitions, numerical reasoning, live code corrections, notebook demonstrations and practical explanations that slide text alone misses. Source recognition errors are checked against the visible mathematics and primary papers. Attribution belongs in references; raw acquisition is not a learner prerequisite.

The spoken review added explicit embedding lookup/training choices, the available-state-before-next-token sequence, rectangular attention and output-head widths, score-difference and random-distribution reasoning, negative/zero-sum normalization counterexamples, dropout's regularization purpose, all-input-features head projections and output-order bookkeeping. GQA version 3 was inspected only for future head-sharing vocabulary; no quality or speed result is reproduced.

The two routes preserve all requested topics. Positional Lecture 9 and unavailable video content are outside scope. Future GQA/MQA/MLA work is identified but not implemented. Source/math/render checks, full-model behavior, hosted runs and learner understanding are separate evidence categories.

Sequential teaching correction, 30 October 2026: complete each lecture before the next. Follow the teacher's problem → diagram → arithmetic → code progression, restore all meaningful diagrams and interactive bar/matrix states, and retain full-size access for small labels. Remove repeated provenance captions, JSON links and embedded presentation configuration from lesson prose. Covers, duplicate diagrams and interface icons are not additional technical concepts. The owning sequential review records actual per-lecture checks; it is not learner acceptance.

## MQA continuation — 2026-10-30

Read Vizuara Chapter 8 sections 8.1–8.4 and original figures 8.1–8.3, plus the public visual comparison's MHA/MQA panels. Read Shazeer v1 sections 2–3 including incremental state/performance analysis. Adopt the cache-cost → independent queries → shared memory → calculate → count → code order. All three original workshop figures are captured in Chrome. Correct the source's architecture-specific 32× shorthand: at fixed width the general ratio is Hq. Distinguish input-projection and full-layer parameter counts. The older MQA video page is a reference, not a claimed subtitle audit. Complete CPU float64 layer, head-loop output/all-gradient equivalence, future intervention and shared-state layout checks pass.

## GQA continuation — 2026-10-30

After MQA completed desktop/phone checks, read the GQA v3 method/conversion and experimental sections and Vizuara Chapter 8 sections 8.5–8.6/8.8–8.9. Capture all five matching workshop diagrams. Preserve group drawing → explicit mapping → scalar calculation → bytes → code → conversion/evaluation progression. Treat the quality/frontier plots as schematic and derive parameter/byte counts independently; do not claim a model choice is proved by a drawn plateau. CPU float64 grouped and expanded outputs/QKV gradients agree for G=1/2/4; multi-token cached chunks agree and invalid divisibility is rejected.

## Multi-head latent attention — 2026-10-30

Read public workshop Chapter 8.7–8.9 in the user Chrome session and the primary method. Retained two original workshop images and the primary architecture comparison. Corrected the omitted positional cache and ungrounded 80/10 GB comparison in adjacent prose. Complete CPU float64 program verifies outputs, all input/parameter gradients, chunk offsets and 400-byte payload; wrong rotary reset is detected. No full-model or GPU result.


## Original worked visual completion — 2026-10-30

The learner authorizes original figures for missing visual steps. Retain the source figures and add editable passive SVG boards that follow the complete executable references. Colors, axes, units, rows, arrows and intermediate values are explained in the same lesson beside each board. Figure review follows each topic before the next; no new lesson files, model code or source provenance JSON.

- `images/worked-gqa-bytes.svg`
- `images/worked-gqa-map.svg`
- `images/worked-mla-absorption.svg`
- `images/worked-mla-position-cache.svg`
- `images/worked-mqa-patterns.svg`

The additions explain worked states, controlled changes and failure cases rather than introduce new benchmark claims. The original public lessons/papers remain the external authorities. Component arithmetic, reader delivery and hosted/learner outcomes remain separate. See implementation/2026-10/attention-labs-playlist-tutorial/13-original-visual-completion.md for the collection audit.
