# Topic rewrite: numerical component review

Recorded 2026-10-12. The new educational source is `tutorials/projects/p00/transformer-foundations/v1/examples/transformer_foundation_probes.py`; its checks are in `test_transformer_foundation_probes.py`. These files do not import or change either implementation child. They provide small, inspectable operations and arithmetic probes for lessons 13–21 and the paper handoff, not a second full Transformer implementation.

## Runtime and result

From the course's `examples/` folder, run:

```bash
python -m unittest test_transformer_foundation_probes -v
python transformer_foundation_probes.py
```

The verified interpreter was the attention child's existing frozen environment: Python 3.12.13, PyTorch 2.14.1+cpu. All **34 checks pass** with CPU float64 arithmetic. `numerical-probe-result.json` saves the full command result, exact deterministic JSON stdout, runtime versions and source SHA256 identities. The source uses lazy Torch imports; importing the module itself needs only the standard library. Running the functions or report requires PyTorch.

## Meaningful checks

| Foundation | Independent evidence |
|---|---|
| Embeddings | One-hot lookup equivalence; exact repeated-ID gradient sum; `padding_idx` suppresses lookup gradient without forcing assigned values to zero; tied output projection contributes a nonzero PAD-row gradient; one truly shared Parameter reaches the optimizer once. |
| Positions | Known D=4 rows; odd D=5 with a correctly sized cosine slice; final unpaired sine; empty-length table; pairwise shift rotation; permutation-equivariant content attention changes when fixed positions are reassigned. |
| Attention forward | Q/K fixture gives exact probabilities 1/6,1/3,1/2; values change retrieval independently of weights; forbidden-key renormalization differs from zeroed values; stable softmax for large finite logits. |
| Attention backward | Independent hand dScores/dQ/dK/dV; manually derived matrix VJP matches float64 autograd on unequal query/key lengths and batches; central differences on separate Q/K/V coordinates; `gradcheck` includes masked entries and an empty row. |
| Masks and causality | Fully masked weights/output and input gradients are finite exact zero; changing padded K/V cannot affect real outputs; blocked K/V gradients are zero; changing future input rows cannot change earlier causal outputs; cached rectangular masks use absolute positions. |
| Library parity | Functional SDPA evaluation parity with explicit `dropout_p=0.0`, Boolean allowed-True mask and rectangular Q/K lengths. Empty-row parity is observed for this pinned CPU runtime, rather than assumed for every API/backend. |
| Multi-head attention | Learned Q/K/V/output projections with biases; B=2,T=3,S=5,D=8,H=2; vectorized and head-loop forward/weight/gradient parity; independently arranged SDPA projection/head path; final output bias changes the zero-output invariant. |
| Position-wise FFN | Exact two-position D=2,F=3 ReLU outputs; changing one position leaves another unchanged; a loss at one position gives zero input gradient at the other. |
| LayerNorm | Hand feature-axis mean/population variance including epsilon; affine scale/bias and a constant row; all trailing axes versus features alone; PyTorch parity; gradient check; train/eval invariance. |
| Residual connections | Pre-norm zero branch preserves x; post-norm zero branch normalizes x; residual Jacobian is I+J_F; shape mismatch is rejected. |
| Regularization | Enumerating all four keep masks proves inverted-dropout mean/variance at p=.5; deterministic supplied masks; eval identity; hand smoothed targets and softmax-minus-target loss gradient agree with PyTorch. |
| Schedule and counts | Exact linear warmup, peak and inverse-square-root values; reject step0; independent FFN/attention/vocabulary counts, including the original D=512,F=2048 FFN count2,099,712. |

No model training, learner mastery, complete translation behavior or hosted recovery is inferred from these component checks.

## Public interfaces

- `sinusoidal_positions(length, width, offset=0)` returns `[length,width]`. It accepts odd widths by ending with an unpaired sine. Its default dtype is float64 and the table is a CPU tensor.
- `masked_softmax(scores, allowed=None)` normalizes the last key axis. The mask is Boolean **True = allowed**, and broadcasts to the score shape. Fully masked rows have zero probabilities by explicit policy.
- `explicit_attention(q, k, v, allowed=None)` returns `(output, weights)` for `[...,T,d]`, `[...,S,d]`, `[...,S,dv]`. Leading batch/head axes must match explicitly. No attention dropout is hidden inside the operation.
- `attention_backward(q, k, v, upstream, allowed=None)` returns a dictionary of manual `q`, `k`, `v` and `scores` gradients. It runs the derived VJP under `no_grad()` to keep its result distinct from autograd's reference calculation.
- `causal_allowed(query_positions, key_positions)` compares absolute integer positions; a query at3 can see keys0–3 even in a two-query cached chunk.
- `multihead_attention(xq, xk, xv, wq, wk, wv, wo, heads=H, allowed=None, biases=None, implementation="vectorized")` returns `(output, weights)` for projected `[B,T,D]` streams. `implementation="loop"` uses matching output-column slices. Four optional `[D]` biases belong to Q/K/V/output. The supplied mask must broadcast to `[B,H,T,S]`; a key mask `[B,S]` must deliberately gain the head/query singleton axes.
- `positionwise_ffn(x, w1, b1, w2, b2)` is the original ReLU expansion/contraction. Weights use mathematical `[input,output]` orientation rather than PyTorch Linear's stored transpose.
- `layer_norm(x, normalized_shape, scale=None, bias=None, eps=1e-5)` computes population variance over the explicit trailing axes. `residual_norm(x, sublayer, placement="post"|"pre")` compares the two arrangements with final-feature LayerNorm.
- `inverted_dropout(x, keep, probability, training=True)` uses a supplied Boolean mask and therefore consumes no global random state. The caller must sample Bernoulli(1−p) masks when using it as a stochastic training operation.
- `smoothed_targets(targets, classes, smoothing=.1)` mixes one-hot targets with a uniform distribution over **all** classes, including the correct class. Ignored rows become all zero; the caller must still use the correct valid-target denominator.
- `transformer_learning_rate(step, width=512, warmup=4000)` uses a positive optimizer-update index and the paper's formula without an additional scale factor.
- `parameter_counts(...)`, `embedding_gradient_probe()` and `report()` supply small worked checks. `report()` emits compact stable JSON when the file is run directly.

## Source/API cautions checked live

The following official PyTorch 2.14 documentation was read on 2026-10-12:

- [Functional SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html): Boolean True participates in attention; callers must explicitly set dropout0 for evaluation. Kernel selection and arithmetic can differ by runtime/backend. An all-masked-zero contract in these probes is an authored policy, not a generic promise taken from the pseudocode.
- [Embedding](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html): padding suppresses lookup-gradient contributions, assigned padding values may be nonzero, and sparse/frequency-scaled lookup choices would change the gradient contract. The tied-output PAD update is independently demonstrated; a padding lookup flag cannot globally freeze another use of a shared Parameter.
- [LayerNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html): statistics use trailing `normalized_shape` dimensions, variance has correction0, and current-input statistics apply during both training and evaluation.
- [CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html): inputs are logits, built-in smoothing mixes with uniform targets, ignored index targets need a valid-target denominator. Zeroing a dense soft target alone does not change a generic mean reduction's denominator.

All source calculations, fixtures and tests are original. No textbook prose, figures or exercises were copied.
