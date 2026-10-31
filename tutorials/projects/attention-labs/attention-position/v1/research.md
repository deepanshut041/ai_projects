# Source and teaching review

## RoPE: make the query-key comparison depend on distance — 2026-10-30

Read the complete Vizuara public RoPE article and RoFormer v5 Section 3. Follow its ordering problem → addition/rotation contrast → unit vector → positions (3,1) → coordinate products → common shift → inverse-rotation proof → multiple frequencies → layer placement → limitations. Capture the original thirteen figures in Chrome; animations are sampled as static states and retained at the original source. Qualify causal versus unmasked permutation arguments, content dependence, non-monotonic individual scores, nonzero compute cost and context-extension limits. CPU float64 norm, explicit-block/output-gradient, equal-gap, chunk-position and invalid-width checks pass. This is a separate new topic; Attention Family Lecture 9 remains excluded.

## ALiBi: put a distance penalty directly in the scores — 2026-10-30

Read ALiBi v2 Section 3 and the original authors' implementation explanation and figures 1–3. No accessible standalone Vizuara ALiBi diagram source was found after targeted searches, so preserve actual primary-source figures rather than inventing Vizuara slides. Teach the score decomposition and sign, a complete four-key value mixture, head slopes, absolute cached positions, then scoped original evaluations. CPU float64 arithmetic, cached/full, common-shift and future-intervention checks pass. This is a new lesson, not the omitted Attention Family Lecture 9.


## Original worked visual completion — 2026-10-30

The learner authorizes original figures for missing visual steps. Retain the source figures and add editable passive SVG boards that follow the complete executable references. Colors, axes, units, rows, arrows and intermediate values are explained in the same lesson beside each board. Figure review follows each topic before the next; no new lesson files, model code or source provenance JSON.

- `images/worked-alibi-row.svg`
- `images/worked-alibi-slopes.svg`
- `images/worked-rope-common-shift.svg`

The additions explain worked states, controlled changes and failure cases rather than introduce new benchmark claims. The original public lessons/papers remain the external authorities. Component arithmetic, reader delivery and hosted/learner outcomes remain separate. See implementation/2026-10/attention-labs-playlist-tutorial/13-original-visual-completion.md for the collection audit.
