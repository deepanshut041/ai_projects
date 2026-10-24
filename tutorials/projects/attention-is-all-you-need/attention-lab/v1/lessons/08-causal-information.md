## Stop a decoder query from reading its future

Padding asks whether a source position contains a real token. Causality asks whether a real position is allowed to influence the current query. The decoder needs both questions answered independently.

### Match the mask to the prediction task

In a future teacher-forced translation batch, a target prefix may be laid out as `[BOS, je, bois]` with labels `[je, bois, EOS]`. The representation at input position zero will predict `je`; it may read BOS but must not read later input positions that reveal the answer. This is an illustrative contract for the later data phase, not an implemented tokenizer or training batch.

For self-attention with equal query/key lengths L, query i can read key j when `j≤i`. With L=4:

```text
             key position
             0 1 2 3
query 0      1 0 0 0
query 1      1 1 0 0
query 2      1 1 1 0
query 3      1 1 1 1
```

The diagonal is allowed: the current input token is part of the available prefix. Whether that token is already a future answer depends on the data shift. A correct triangular attention mask cannot repair incorrectly aligned decoder inputs and labels.

### Combine padding and temporal visibility with AND

If source/key validity is `[1,1,0,0]`, intersect that row with each causal row:

```text
query 0      1 0 0 0
query 1      1 1 0 0
query 2      1 1 0 0
query 3      1 1 0 0
```

Using OR would allow keys that satisfy either condition, including future valid tokens or padded past tokens. The builder uses `allowed & triangle` for exactly this reason. Its causal triangle broadcasts across batch and heads, then intersects with each example's padding constraints.

![The explicit path shows scores and weights before and after visibility restrictions.](images/11-mask-score-weight-trace.png)

### Follow the source branch

`make_attention_mask` checks equal query/key lengths for this simple causal self-attention contract, creates a lower-triangular boolean tensor and intersects it with any padding mask. The public `Attention.forward` rejects supplying both cross-attention target queries and `causal=True`. It does not silently invent a rectangular prefix-alignment policy.

When the efficient path has no padding mask and weights are not requested, `_context` lets SDPA handle the causal triangle using `is_causal=True`. When padding is supplied, it constructs the combined explicit mask and sets the SDPA causal flag accordingly. These branches must represent the same allowed pairs, which is one reason both outputs and gradients are compared in tests.

### Prove something stronger than a triangular picture

A mask heatmap shows intent. A controlled perturbation tests effect. Keep the first three positions fixed and change positions 3 onward by a large value. With the layer in evaluation mode and identical parameters, outputs at positions 0–2 must remain unchanged under causal self-attention.

![Changing future tokens leaves protected earlier outputs unchanged.](images/12-future-token-isolation.png)

The existing tests run this isolation check for explicit and efficient paths. Without a causal mask, earlier positions are generally free to change because they can retrieve later keys/values. “Generally” is deliberate: a degenerate zero-parameter layer could ignore them even when visibility is allowed. Passing one accidental no-change case would not by itself prove masking.

### Left padding exposes a useful boundary case

A sequence beginning with invalid keys can create causal query rows with no allowed keys. For validity `[0,0,1,1,1]`, the first two causal rows are entirely blocked. The empty-row handling from lesson 7 matters here; merely avoiding an entirely empty sentence is insufficient.

Causality also does not supply positional information. If we permute source tokens in an unpositioned encoder, attention has no independently supplied notion of original order. Later positional embeddings/encodings and full model wiring must define that information. We have implemented the visibility mechanism, not the full target prediction system.

[[practice:causal-check]]

Next we let a target query retrieve from a different source sequence. Its score matrix becomes rectangular, and the source has a different visibility contract from the target prefix.
