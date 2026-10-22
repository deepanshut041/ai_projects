# Foundations mapped to Attention Is All You Need

The [topic outline](project/course-outline.md) is the full coverage map. The table below connects those topics to concrete paper operations so I can check whether I understand an implementation rather than only recognize its names.

| Paper operation or decision | Prerequisite lessons | What I should explain/check |
|---|---|---|
| Learned Q/K/V and output projections | 1–3, 5–7 | Matrix orientation, shared affine parameters, backward sums, nonlinearity and initialization assumptions |
| `softmax(QKᵀ/sqrt(dk))V` | 4, 6, 8, 15 | Score scale, row distribution over key positions, stable softmax and all backward paths |
| Padding and autoregressive visibility | 2, 11, 16, 22 | Separate valid keys, padded queries, ignored labels, target shift and future-token invariance |
| Parallel heads and concatenation | 2, 15–17 | B/H/T/S/d axes, independent projections, inverse transpose, output mixing and cost |
| Position-wise ReLU FFN | 5–7, 18 | Shared per-position D→F→D map, nonlinear feature capacity, parameter/compute count |
| Learned embeddings/shared output map | 11–13 | Exact vocabulary identity, one-hot equivalence, repeated lookup gradients, tied input/output paths and scaling |
| Sinusoidal positional addition | 1, 4, 13–14 | Radians/frequencies, equivariance without position, same-width addition and shift identity |
| Residuals followed by LayerNorm | 3, 19–20 | Shape/stream alignment, I+JF, feature statistics/derivatives and original post-norm order |
| Dropout and label smoothing | 8, 21 | Placement, expected scaling, mode versus graph recording, target convention and evaluation objective |
| Adam and warmup/inverse-square-root rate | 3, 8–10 | Gradient reduction, moments/bias correction, update index, schedule, clipping and precision state |
| Encoder memory and teacher-forced decoder | 11–17, 19–22 | Conditional factorization, source/prefix/labels, cross-attention and greedy/beam inference contracts |
| Token NLL, perplexity and translation quality | 4, 8, 11, 23, 25 | Valid-target totals, tokenizer-dependent units, held-out metrics, BLEU limits and corpus comparability |
| Reproducible complete-state training | 2, 9–11, 23–25 | Source/data/config identity, RNG/order/cursor, optimizer/schedule state and fresh-process/hosted boundaries |

## Continue into the model builds

[Course 1 — Transformer Foundations: Translation and RoBERTa](/records/tutorials/p00-transformer-course-v1/chapters/start/) applies these foundations to complete translation, MLM and extractive QA builds. This course explains the operations before those source recipes and model/task checks; it does not duplicate or modify their implementations.

[Course 2 — Small GPT](/records/tutorials/p01-small-gpt-v1/chapters/next-token/) follows the required P00 model gates. Its decoder-only, pre-norm and learned-position choices must be recorded as that architecture's contract. Knowing their differences from the original Transformer is preparation, not evidence that either model has been completed.
