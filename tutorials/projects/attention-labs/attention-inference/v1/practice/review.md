# Cache transfer review

Predict the permitted key indices when a four-token cache receives a three-token chunk. They are 0–4, 0–5 and 0–6 respectively. Explain why a triangle anchored at column zero is wrong.

A cached token of all zeros is still a token. Explain why removing it from softmax changes normalization even when its key coordinates are zero.

Change one layer's coefficients after prefill. The old cache is no longer generally valid: it encodes the old computation. Reset and recompute rather than attaching the stale arrays to a new model state.
