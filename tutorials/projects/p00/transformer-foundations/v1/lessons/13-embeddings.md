## Lesson 13 — Embeddings, lookup gradients and weight tying

**Goal:** understand the learned interface between discrete token IDs and continuous model features. Prerequisites: vocabulary identity, matrix maps and shared-gradient sums. An embedding is learned model state; tokenization chooses which row to use.

### Lookup is a special matrix multiplication

Let E have shape `[V,D]`: one learned row per vocabulary ID. For token i, its embedding is Eᵢ. Define one-hot row uᵢ of width V, with a 1 at i. Then uᵢE=Eᵢ. This proves the lookup's relationship to a linear map without suggesting we should allocate `[B,T,V]` one-hot tensors for ordinary training.

For

$$E=\begin{bmatrix}0&0\\1&2\\3&4\\-1&1\end{bmatrix},\qquad I=[2,1,2],$$

lookup yields `[[3,4],[1,2],[3,4]]`. IDs `[B,T]` produce features `[B,T,D]`, and there are VD parameters irrespective of sequence length. Learned geometric relations depend on the training objective; ID number differences do not establish semantic distance.

[[diagram:prereq-lookup]]

A static embedding gives the same vector to every occurrence of a token. A Transformer later produces **contextual representations** that depend on neighboring tokens and positions. Calling every contextual hidden vector “the embedding table row” loses this distinction. Pretrained vectors are optional; end-to-end learned embeddings can start from random values.

### Repeated IDs sum their backward contributions

If two occurrences select E₂ and receive upstream gradients `[1,1]` and `[3,3]`, then ∇E₂=`[4,4]`. An occurrence selecting E₁ with gradient `[2,2]` gives ∇E₁=`[2,2]`; an unselected ordinary row receives zero lookup contribution. Indices are integer decisions; autograd differentiates the selected values, not a continuous path through the ID numbers.

```python
import torch
E = torch.tensor([[0.,0.],[1.,2.],[3.,4.],[-1.,1.]], requires_grad=True)
ids = torch.tensor([2,1,2])
upstream = torch.tensor([[1.,1.],[2.,2.],[3.,3.]])
(E[ids]*upstream).sum().backward()
assert E.grad.tolist() == [[0.,0.],[2.,2.],[4.,4.],[0.,0.]]
```

`nn.Embedding(..., padding_idx=0)` suppresses lookup-gradient contributions to that padding row. Its default fresh row is zero, but assigning a custom weight can change its value. A tied output use or explicit regularization can also contribute updates to the same underlying row; padding_idx is not a global freeze on every use. Zero padding features do not replace attention masks.

### Output projection returns to the vocabulary

For decoder hidden h `[B,T,D]`, a separate output matrix U `[D,V]` gives logits hU+c. This is a categorical score for each possible next token. **Weight tying** reuses Eᵀ as U: logits=hEᵀ, optionally plus a separately defined output bias. Input lookup and output projection then send gradients to the same parameter through two distinct paths.

With V=4,D=2, separate source embedding, target embedding and output matrix use 3VD=24 matrix parameters; sharing all three uses VD=8. This requires matching vocabulary row meanings and widths. If source and target vocabularies differ, tying target/output can still be possible while sharing source/target requires a compatible joint mapping. Equal matrix shapes alone are insufficient.

A shared parameter must be genuinely shared, not a one-time copy of equal values. Registering two independent Parameters initialized from the same table will let them diverge. Check parameter identity and whether the optimizer sees the intended shared object once.

### Embedding scale and position

The original design multiplies lookup vectors by sqrt(D), then adds same-width positional features. At D=4, a row `[.1,.2,.3,.4]` becomes `[.2,.4,.6,.8]`. The output projection uses the shared weight table rather than automatically multiplying every output logit by that input scale. The embedding scale and attention's division by sqrt(dₖ) act at different locations and have different purposes.

A vocabulary artifact is part of the model's identity. Suppose IDs1 and2 are swapped while E is unchanged: the same text now selects the wrong rows. If both input/output rows and the tokenizer are permuted consistently, the representation can be preserved. This is why saving only “vocabulary size=4” is not enough.

The reusable [foundation probes](examples/transformer_foundation_probes.py) implement these small calculations; [independent probe checks](examples/test_transformer_foundation_probes.py) test hand answers, gradients and changed conditions. Use the recorded Torch interpreter from Lesson 2.

[[practice:embedding-transfer]]

### Notes and check

Write the one-hot equivalence, calculate the repeated-ID gradients and draw both gradient paths into a tied table. Count tied and untied parameters. Explain the distinction between token ID, lookup row, position-enriched input and contextual representation.

Sources: [Embedding API](source:torch-embedding), [original embedding interface](source:transformer-paper), [weight-tying paper](source:weight-tying). Continue with [position](lessons/14-position.md).
