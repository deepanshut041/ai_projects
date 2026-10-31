# My source notes and architecture choices

The original reference is [Attention Is All You Need](source:transformer-paper), inspected as arXiv1706.03762v7 on2026-10-12. I use its equations and architecture settings as reference facts; all lesson prose, arithmetic fixtures, diagrams and exercise wording are original. I am not reproducing its translation results in this prerequisite course.

## Original design versus comparisons

| Original design taught here | Comparison and its different contract |
|---|---|
| Post-norm residual order | Pre-norm has a different forward map and identity derivative path |
| ReLU position-wise FFN | GELU/gated FFNs are later choices with different formulas |
| Sinusoidal absolute positions | Learned absolute/relative/rotary alternatives need separate position rules |
| Adam with stated betas/epsilon and warmup schedule | Generic Adam defaults and AdamW are different update specifications |
| Shared vocabulary weights and sqrt(D) input scale | Untied/different-vocabulary models require explicit mapping and counts |
| Dropout and smoothing value in the paper | Placement, PAD treatment and the full smoothing distribution must be specified by an implementation |

The label-smoothing lesson distinguishes uniform-mixture targets from excluding the gold/PAD categories. The paper's reported epsilon alone does not establish equivalence of every modern API/recipe. Functional SDPA and MultiheadAttention invert Boolean mask meaning; the lessons name `allowed` then translate deliberately. Zero-empty-row attention is the explicit educational probe's tested policy, not an assertion about all kernels.

## Written companions

[Nielsen Chapters1–3](source:nielsen-foundations) support mathematical teaching. Their direct HTML fetch previously failed; source-domain indexed excerpts were inspected, rather than a fresh full-book download. [D2L1.0.3](source:d2l-transformer) supplies supporting context across probability, networks, attention and sequence learning. Essential explanations and checks are here; no external reading is required to fill a missing derivation. Older source code or prose is not copied into this course.

Primary optimizer, initialization, residual and normalization papers support named methods. PyTorch2.14 API docs define the runtime contracts; the local Torch2.14.1+cpu execution is separate behavioral evidence. Unicode's versioned normalization document defines representation policy, and the educational BPE format deliberately differs from pretrained tokenizer formats.

## Evidence I will keep distinct

A documented method, original known-answer calculation, independently executed check, author reference-training outcome, learner explanation and deleted hosted-runtime recovery prove different things. A finite-difference match says something about a smooth numerical fixture, not a translation benchmark. The classifier's64 training and36 held-out grid points test only its stated opposite-sign task.

The existing child implementations, later courses and personal notes exports are preserved. The current learner remains at `linear()`; hosted recovery is still pending. [Research record](research.md), [reference runs](project/reference-runs.md) and [readiness](project/p01-readiness.md) hold the supporting context.
