# Cache source and teaching review — 2026-10-30

Read Vizuara's public Inference Engineering Chapters 5 and 7 in Chrome and its linked visual overview; inspect all 23 teaching figures. Retain the explanatory progression: whole generation loop → repeated Q/K/V rows → matrix cell/row/full score → one required query → append/reuse → persistent tensor dimensions → memory and bandwidth cost. Screenshot all 23 original figures in Chrome at their native 1376×768 rendered size. No video frames or substitute generated art are used.

Also read Hugging Face's complete How caching works explanation, especially cache append and absolute attention-mask dimensions. Adapt that implementation contract into a complete CPU float64 reference with multi-token chunks and request isolation. The older DeepSeek KV-cache video page/description and chapter list were inspected; its complete subtitles have not yet been obtained, and the text does not claim a full spoken-video audit. These are workshop figures, not asserted to be that video's slides. Source selection for the broader continuation was asked asynchronously.

Departures checked: avoid the book's incorrect blanket zero-logit masking wording; distinguish per-step FLOPs from elapsed-time speedup; derive cache bytes using KV-head count and binary units; qualify inconsistent scale bars; correct the bytes/FLOP dimensional mismatch and avoid mixing dense prefill and one-query decode roofline counts. Original prose, mathematical derivations and runnable fixtures explain these differences. The four-token Q/K table agrees numerically; the fifth-query/value fixture is explicitly newly declared.

The primary learning result remains component correctness. No full-model training, production benchmark, learner acceptance or hosted Colab run is asserted.

## PagedAttention — 2026-10-30

Read workshop Chapter11 and primary3.1/4.1–4.4 in user Chrome; all ten workshop figures included. Corrected inconsistent reservation totals, block-size timeline, invalid printed online-softmax equations and universal utilization claims. CPU noncontiguous retrieval, partial-tail COW, reference-count release, stale-entry-free reuse and failure-before-mutation checks pass. Gather reference is not a fused GPU paged kernel.


## Original worked visual completion — 2026-10-30

The learner authorizes original figures for missing visual steps. Retain the source figures and add editable passive SVG boards that follow the complete executable references. Colors, axes, units, rows, arrows and intermediate values are explained in the same lesson beside each board. Figure review follows each topic before the next; no new lesson files, model code or source provenance JSON.

- `images/worked-cache-mixture.svg`
- `images/worked-cache-offset.svg`
- `images/worked-paged-address.svg`
- `images/worked-paged-copy-on-write.svg`

The additions explain worked states, controlled changes and failure cases rather than introduce new benchmark claims. The original public lessons/papers remain the external authorities. Component arithmetic, reader delivery and hosted/learner outcomes remain separate. See implementation/2026-10/attention-labs-playlist-tutorial/13-original-visual-completion.md for the collection audit.
