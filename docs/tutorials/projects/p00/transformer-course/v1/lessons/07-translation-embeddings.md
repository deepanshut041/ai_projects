## Lesson 7 — Translation embeddings and vocabulary projection

**Goal:** explain how integer symbols become learned vectors and how a decoder vector becomes vocabulary logits. Prerequisites: token IDs, matrix derivatives and the translation loss. Input and output tables have different roles even when their dimensions happen to match.

### Lookup is a sparse linear operation

Let E be a vocabulary table with shape [V,D]. A token ID i retrieves row Eᵢ; a batch of IDs [B,T] produces [B,T,D]. The integer itself is an address, not a magnitude the model should compare. Equivalently, one-hot row uᵢ multiplied by E gives the same vector. A lookup avoids constructing the large one-hot tensor.

For E=`[[0,0],[1,2],[-1,3],[4,1]]` and IDs `[1,2,1]`, the vectors are `[[1,2],[-1,3],[1,2]]`. If upstream gradients at those positions are `[2,1]`, `[3,-2]`, `[4,5]`, the table gradient is zero at unused rows, `[6,6]` at row 1 and `[3,-2]` at row 2. Repeated IDs **sum** their contributions because they use the same parameter.

$$
\frac{\partial L}{\partial E_{i,:}}
=\sum_{b,t:\,\mathrm{id}_{bt}=i}\frac{\partial L}{\partial X_{bt,:}}.
$$

This sparsity describes the lookup gradient. Other uses of the same parameter, regularization or optimizer state can alter the update. In particular, a table tied to an output classifier receives a dense output gradient as well.

### Translation owns two symbol systems

The reference [Transformer](examples/translation_model.py) owns `src_embeddings` [Vₛ,D] and `tgt_embeddings` [Vₜ,D]. Source ID 4 and target ID 4 need not represent the same token. In the fixture, source PAD is 0 and target PAD is 1. Equal vocabulary sizes do not establish equal row meanings.

`padding_idx` suppresses the lookup gradient for the designated row. The model initializes that row to zero. Position vectors are then added, so a padded position's hidden vector need not remain zero. Key masks prevent valid queries from reading it; loss masks exclude padded target labels. Those are three separate contracts.

Run this exact gradient check from the reference `examples/` folder:

```python
import torch
embedding = torch.nn.Embedding(4, 2, padding_idx=0)
with torch.no_grad():
    embedding.weight.copy_(torch.tensor([[0.,0.],[1.,2.],[-1.,3.],[4.,1.]]))
ids = torch.tensor([0,1,2,1])
upstream = torch.tensor([[9.,9.],[2.,1.],[3.,-2.],[4.,5.]])
(embedding(ids) * upstream).sum().backward()
expected = torch.tensor([[0.,0.],[6.,6.],[3.,-2.],[0.,0.]])
assert torch.equal(embedding.weight.grad, expected)
```

### The output table scores every target symbol

For decoder hidden states H [B,T,D], the independent head has stored weight U [Vₜ,D] and bias c [Vₜ]: Z=HUᵀ+c. A vocabulary row acts as a class vector. All classes appear in the softmax denominator, including classes that are never a gold label in this batch.

With hidden row h=`[2,-1]`, U=`[[1,0],[0,1],[1,1]]`, and zero bias, logits are `[2,-1,1]`. Their probabilities are approximately `[.7054,.0351,.2595]`. For gold class 2, δ=p−one_hot(2); each output row receives δⱼh. Unlike a lookup, this generally changes every row. The hidden gradient is δU. Trace this signal back through decoder layers and cross-attention to the source; the output head does not perform that whole computation itself.

### Weight tying requires semantic compatibility

Tying input and output tables can save VₜD parameters, but it means assigning **the same Parameter object**, not copying equal values. Tied target input/output rows share vocabulary identity and width; source–target tying additionally requires a compatible shared tokenizer. A tied PAD row can receive output-head gradients even if its lookup gradient is suppressed. A policy that keeps that row fixed needs an additional explicit mechanism.

The miniature uses separate tables and a separate vocabulary head. Its default counts are source 10×32=320, target 12×32=384, and head 12×32+12=396: 1,100 parameters altogether. Target tying would save 384; it would also change parameter ownership and gradients. Count unique parameters rather than duplicated state-dict entries.

[[diagram:translation-embeddings]]

### Scaling and initialization are declared choices

The original Transformer multiplies token embeddings by √D before adding sinusoidal positions. This source adds unscaled embeddings, initialized with standard deviation .02, to those positions. It also has no embedding-plus-position dropout module. These choices affect the relative initial scale of content and position; attention's separate √dₖ score scaling does not compensate for them. An exact paper reproduction would explicitly change and test the recipe.

[[practice:translation-embeddings-transfer]]
[[practice:translation-embeddings-gradient]]

### Notes and implementation

Derive repeated-ID and output-table gradients in my notes. Draw separate source/target row meanings, mark the PAD lookup rule, and count tied versus untied unique parameters. Inspect `translation_model._init_weights` and the head declaration; verify the table and head identities without changing the learner repository. Continue to [sinusoidal positions](lessons/08-translation-positions.md). References: [Embedding API](source:torch-embedding), [original architecture](source:attention-paper).
