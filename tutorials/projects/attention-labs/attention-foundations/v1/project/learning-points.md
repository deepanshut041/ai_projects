# Learning points

No. The score defines compatibility, causality defines permitted source positions, and the head arrangement defines separate or shared projections. These choices can coexist. Attention weights are over source positions; vocabulary probabilities belong to a later prediction head.

The effect passes through two multiplications by 0.5, giving 0.25. A real RNN can have different matrices, nonlinearities and gates, so this is an illustrative sensitivity calculation rather than a universal memory law. A decoder still uses a separate vocabulary head and can produce a different number of tokens.

Not in this separable scorer: uᵀWₛs adds the same scalar to every source score for a fixed query. Softmax is invariant to that common offset, so the relative weights do not change. Nonlinear interaction makes the query contribution affect relative comparisons.

No. The scale uses the number of summed coordinate products in each dot product, d_k=2, rather than the number of keys. More keys changes the number of scores and the normalization denominator, not the coordinate count in a score.

No: rows can have different normalization denominators. In this fixture A_13≈0.422319 while A_31≈0.211942. A real zero query gives equal finite scores, hence uniform weights over allowed keys; it does not declare padding.

Q, K, scores and attention weights stay fixed. V and the returned contexts can change. With a task loss, downstream gradients can also change because the output being measured changed. Identical position-free queries with the same memory still cannot be distinguished by merely naming the roles differently.

A dropout sample need not sum to one, and surviving weights are scaled by 1/(1-p); values above one are possible. The local expectation matches the original weights. Zeroing forbidden weights and renormalizing is mathematically equivalent to masked softmax when the allowed denominator is nonzero, but it is numerically fragile if allowed probabilities have already underflowed. Simply zeroing without renormalizing is a different operation.
