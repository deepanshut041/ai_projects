## Lesson 17 — Multi-head projections, splitting and concatenation

**Goal:** derive every axis and learned map in multi-head attention. Prerequisites: learned affine maps, attention and masks. Multiple heads are parallel learned retrieval operations, not a loop that averages arbitrary chunks.

### Each head has its own projection space

Let model width D, number of heads H and per-head key/value width d=D/H. Require D divisible by H for this common design. For head h, maps Wqʰ,Wkʰ,Wvʰ have shape `[D,d]`. Each head computes

$$O_h=\operatorname{softmax}\!\left(\frac{(X_qW_q^h)(X_kW_k^h)^T}{\sqrt d}\right)(X_vW_v^h).$$

Concatenate output features from all heads, shape `[B,T,Hd]=[B,T,D]`, and apply output matrix Wo `[D,D]`. Concatenation keeps the outputs available to the learned mixing map; averaging would discard their separate feature coordinates and define another architecture.

Projections can be represented by one `[D,D]` weight per Q/K/V, then split its output columns into heads. This is algebraically equivalent to separate per-head projections when the parameter values are matched. For self-attention, a fused `[D,3D]` map is another implementation choice; cross-attention Q and K/V come from different input streams.

### Trace unequal dimensions through the operation

Use B=2,T=3,S=5,D=8,H=2,d=4:

| Stage | Query stream | Key/value stream |
|---|---|---|
| inputs | `[2,3,8]` | `[2,5,8]` |
| projected | `[2,3,8]` | `[2,5,8]` |
| split before transpose | `[2,3,2,4]` | `[2,5,2,4]` |
| head-major | `[2,2,3,4]` | `[2,2,5,4]` |
| scores/weights | `[2,2,3,5]` | last axis indexes keys |
| weighted values | `[2,2,3,4]` | one result per query/head |
| merge heads | `[2,3,8]` | transpose before reshape |

[[diagram:head-shapes]]

```python
import torch
B,T,S,D,H = 2,3,5,8,2
d = D//H
q = torch.arange(B*T*D).reshape(B,T,D).double()
k = torch.arange(B*S*D).reshape(B,S,D).double()
v = k + 1
split = lambda x: x.reshape(x.shape[0], x.shape[1], H, d).transpose(1,2)
qh,kh,vh = split(q),split(k),split(v)
a = (qh @ kh.transpose(-2,-1) / d**.5).softmax(-1)
o = a @ vh
merged = o.transpose(1,2).contiguous().reshape(B,T,D)
assert a.shape == (B,H,T,S)
assert merged.shape == (B,T,D)
```

This probe omits learned projections to isolate axis mechanics. It does not demonstrate learned head specialization. Add projections in the full component; verify vectorized heads against a loop using the same weights and masks. Simply reshaping original features creates chunks, while independently learned maps can choose different combinations of all original features.

### Parameter and compute counts

With d=D/H, all Q matrices together have D² entries, likewise K and V. Wo adds D², total4D² without biases, or4D²+4D with one bias per output coordinate of each of the four maps. H does not multiply the D² total when D stays fixed. More heads means narrower per-head features, not automatically more projection parameters.

Projection cost scales with sequence length times D². For attention, query–key and weight–value products cost approximately proportional to BHTSd; summing heads gives BTSD. Score storage in an explicit implementation is BHTS elements. Self-attention has T=S, hence quadratic score storage in length. At fixed D, more heads can increase the explicit score storage even though projection parameter count stays unchanged. Optimized kernels can alter materialization and performance without changing the intended operation.

The original base choice is D=512,H=8,d=64. Our small exercises use smaller values while preserving the contracts. Each head is not guaranteed to correspond to a specific linguistic role; interpret trained heads using evidence.

The reusable [foundation probes](examples/transformer_foundation_probes.py) implement these small calculations; [independent probe checks](examples/test_transformer_foundation_probes.py) test hand answers, gradients and changed conditions. Use the recorded Torch interpreter from Lesson 2.

[[practice:multihead-transfer]]

### Notes and check

Draw the split/transpose/merge path with unequal T,S. Count parameters at D=8,H=2 and compare H=4. Explain why Wo is still needed after concatenation and why a reshape before the inverse transpose scrambles positions and heads.

Source: [original multi-head formulation](source:transformer-paper), [D2L multi-head attention](source:d2l-multihead). Continue with [position-wise FFN](lessons/18-positionwise-ffn.md).
