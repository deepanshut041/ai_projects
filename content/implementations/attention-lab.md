---
title: "Attention Lab"
description: "Inspect the forward computation and masking invariants of scaled dot-product attention."
updated: "2026-10-09"
status: "implemented example"
tags: ["implementations", "attention-lab"]
sources: [{"title": "Attention Is All You Need \u2014 arXiv v7, section 3.2.1", "url": "https://arxiv.org/html/1706.03762v7"}]
---

## Working contract

Queries and keys share a feature width; keys and values share a sequence length. The implementation computes

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V.$$

A boolean allowed-key mask can be combined with causal self-attention. Masked keys receive zero probability; fully masked query rows fail explicitly. Subtracting the maximum logit keeps softmax stable for large finite scores.

## Run and inspect

```bash
./ai test attention-lab
./ai run attention-lab
```

Tests verify the exact mean-value answer for uniform logits, row sums, causal prefix independence and permutation equivariance without positional information. Numerical overflow fails instead of emitting a silent nonfinite output.

## Scope

This implements a single-head forward operation based on equation 1 in the original paper. It has no learned projections, backward pass, positional encoding, language model, tokenizer or Transformer training. The paper's translation results have not been reproduced here.
