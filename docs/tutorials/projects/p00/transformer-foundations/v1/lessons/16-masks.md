## Lesson 16 — Padding masks, causal masks and empty rows

**Goal:** define exactly which query can use which key, and test that forbidden information cannot influence an output. Prerequisites: attention and collation. Target shifting is previewed below and developed fully in Lesson 22. Masking is an information-flow contract, not a shape trick.

### Mask scores before normalization

Let allowed `[T,S]` be Boolean, true when a query may attend to a key. Set forbidden scores to −∞ before softmax. Allowed entries are then renormalized to sum1 and forbidden weights are0. Zeroing values instead leaves their scores in the denominator. Zeroing weights after softmax without renormalizing lowers the row sum and is another operation.

For scores `[0,log2,log3]`, allowing only keys0 and2 gives `[1/4,0,3/4]`. With the previous values, output is `[3,1.5]`. Zeroing V₁ while keeping all keys instead gives `[2,1]`. These differ because masked probability mass is removed from normalization.

### Three common mask patterns

For source padding with valid keys `[True,True,False]`, every real encoder query sees the first two keys. Shape `[B,1,1,S]` broadcasts across heads and queries. It does not by itself suppress a padded **query** output.

For target `[a,b,EOS]`, teacher-forced inputs are `[BOS,a,b]` and labels are `[a,b,EOS]`. At input position1 the token is `a`, but the answer being predicted is `b`: the mask may include its diagonal without revealing that answer. If inputs and labels both contained `[a,b,EOS]`, the diagonal would expose the answer. This is the target-shift contract used in the following mask; Lesson 22 develops training and inference in full.

For a decoder prefix of length3, causal allowed is

$$C=\begin{bmatrix}1&0&0\\1&1&0\\1&1&1\end{bmatrix}.$$

Query t may use keys s≤t in its input prefix. The supervised target at t is the next token because inputs are shifted. If labels are unshifted, this diagonal can reveal the answer itself. Combine causality with valid-key masking using Boolean AND, not addition of unrelated integer IDs.

Cross-attention uses T decoder queries and S encoder memory keys, often T≠S. It ordinarily masks source padding but allows each decoder query to inspect every real source token. A triangular T×S mask would incorrectly impose output-prefix causality on the source. When using a cached prefix, rectangular causal alignment requires actual absolute query/key positions; blindly applying a fresh top-left triangle can forbid valid previous keys.

[[diagram:mask-contracts]]

### All-masked rows need an explicit policy

Softmax of a row of all −∞ is undefined: subtracting its maximum involves −∞−(−∞), producing NaNs in a naive implementation. Replacing forbidden scores with an arbitrary finite negative number can make an empty row uniform, silently inventing attention to padding.

For these probes, the declared policy is **zero attention and output for an empty row**. Before softmax, temporarily replace an empty row's scores by zeros, then multiply its normalized row by a nonempty-row indicator. This avoids NaNs and gives zero score/value contributions from that row. Another implementation may reject empty rows; define and test its policy rather than assuming a backend's behavior.

```python
import torch
scores = torch.tensor([[0.,1.,2.],[0.,1.,2.]])
allowed = torch.tensor([[True,False,True],[False,False,False]])
nonempty = allowed.any(dim=-1, keepdim=True)
masked = scores.masked_fill(~allowed, -torch.inf)
safe = torch.where(nonempty, masked, torch.zeros_like(masked))
weights = safe.softmax(-1) * nonempty
assert weights[0,1].item() == 0
assert torch.equal(weights[1], torch.zeros(3))
assert torch.isfinite(weights).all()
```

If padded queries should remain zero, gate their outputs at the appropriate boundary too. An output projection bias can turn a zero attended vector into a nonzero result, and a residual input may remain nonzero. State where the invariant is required.

### Library Boolean conventions differ

In functional SDPA, Boolean attention-mask true means allowed. In `nn.MultiheadAttention` Boolean attention-mask true means blocked; its Boolean key-padding-mask true means padding to ignore. Name your own mask `allowed`, then deliberately invert it for the relevant API. `is_causal` is not a substitute for understanding query/key alignment. Functional SDPA's dropout probability must be explicitly0 for deterministic evaluation; it does not read an enclosing module's eval flag.

A causality check changes **future input tokens** while keeping parameters/prefix fixed, then compares earlier outputs in evaluation mode. A padding check changes padded keys/values and checks real-query invariance. Shape checks alone cannot prove either property. Compare a rectangular cross-attention case as well as square self-attention.

The reusable [foundation probes](examples/transformer_foundation_probes.py) implement these small calculations; [independent probe checks](examples/test_transformer_foundation_probes.py) test hand answers, gradients and changed conditions. Use the recorded Torch interpreter from Lesson 2.

[[practice:mask-transfer]]

### Notes and sources

Draw allowed matrices for encoder, decoder and cross-attention for S=4,T=3 with padding. Calculate one renormalized row and explain the empty-row policy. Predict which invariance check exposes an inverted mask.

Sources: [SDPA mask/dropout contracts](source:torch-sdpa), [MultiheadAttention masks](source:torch-mha). Continue with [multi-head attention](lessons/17-multihead.md).
