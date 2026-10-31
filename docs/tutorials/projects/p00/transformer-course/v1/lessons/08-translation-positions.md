## Lesson 8 — Sinusoidal position implementation

**Goal:** derive the position table, explain why the model needs it and check its indexing/state contract. Prerequisites: embeddings and basic trigonometry. Source and target sequences each start their own position coordinates.

### Attention alone does not assign an order

Without positions, shared token projections and pointwise FFNs treat permutations consistently. If P permutes rows, self-attention on PX produces P times the original output, provided the mask is permuted with the same rows and columns. This is permutation **equivariance**. The network can recognize the same set of vectors but lacks an explicit absolute order. A causal mask imposes a visibility order; it does not by itself supply the positional representation used in this encoder–decoder recipe.

Positions enter as addition Xᵢ=E[tokenᵢ]+PE(i). Addition keeps width D and permits learned projections to combine content with position. Concatenation would change the interface and parameter count. The source encoder and target decoder use different token tables but the same sinusoidal formula at their own indices.

### Derive the frequency pairs

For even feature width D and frequency-pair index j=0,…,D/2−1,

$$
\begin{aligned}
\omega_j&=10000^{-2j/D},\\
PE(p,2j)&=\sin(p\omega_j),\\
PE(p,2j+1)&=\cos(p\omega_j).
\end{aligned}
$$

Here p is a zero-based token position. With D=4, the two angular frequencies are 1 and .01. Position 0 is `[0,1,0,1]`; position 1 is approximately `[.841471,.540302,.00999983,.999950]`. Pair j=1 varies much more slowly. The numbers encode coordinates, not hand-assigned meanings for words.

For a fixed offset k, trigonometric addition gives a fixed rotation within each pair:

$$
\begin{bmatrix}\sin((p+k)\omega)\\\cos((p+k)\omega)\end{bmatrix}
=\begin{bmatrix}\cos(k\omega)&\sin(k\omega)\\-\sin(k\omega)&\cos(k\omega)\end{bmatrix}
\begin{bmatrix}\sin(p\omega)\\\cos(p\omega)\end{bmatrix}.
$$

That linear relation motivates learning relative shifts from the representation. It does not prove a trained model generalizes to arbitrary sequence lengths. The code's allocated table and training distribution remain real limits.

[[diagram:sinusoidal-positions]]

### Read the actual module and its state

[PositionalEncoding](examples/translation_model.py) builds the FP32 table once and registers `encodings` as a **buffer**. It is saved in `state_dict` and moves with the module's device, but is absent from `model.parameters()` and receives no optimizer updates. Forward slices the first T rows and converts them to the input dtype before adding them. A learned position embedding, used later by RoBERTa, is a Parameter table with a different indexing and gradient contract.

```python
import math, torch
from translation_model import PositionalEncoding
positions = PositionalEncoding(max_len=4, embed_dim=4)
actual = positions(torch.zeros(1,2,4))
expected = torch.tensor([[[0.,1.,0.,1.],
    [math.sin(1),math.cos(1),math.sin(.01),math.cos(.01)]]])
assert torch.allclose(actual, expected, atol=1e-7)
assert "encodings" in positions.state_dict()
assert list(positions.parameters()) == []
try:
    positions(torch.zeros(1,5,4))
except ValueError:
    pass
else:
    raise AssertionError("position capacity was not enforced")
```

The model validates even D because this implementation fills sine/cosine pairs. The mathematical idea can be adapted for odd widths, but this particular source does not implement that extension. A maximum source length of 32 and target length of 32 are separate capacities; a source length of 7 and decoder length of 4 both slice from zero.

### Incremental decoding needs an offset contract

The reference module has no position-offset argument. Its greedy decoder recomputes the complete target prefix each time, so the newest token receives its true prefix index. Feeding only the newest token through this module would repeatedly assign position 0. A KV cache extension must pass the absolute prefix offset, align causal key visibility with that offset and respect table capacity. A square lower-triangle mask alone cannot establish those rectangular cached semantics.

### Detect an indexing failure

Zero token embeddings do not produce zero hidden states: at position 0 the cosine coordinates are one. A padded source position also receives a position vector; masking keys, rather than expecting zero hidden rows, establishes padding invariance for valid queries. Reordering tokens while leaving positions fixed should usually change valid outputs, whereas changing a masked padded payload should not affect real positions under the declared mask.

[[practice:translation-positions-transfer]]
[[practice:translation-positions-offset]]

### Notes and implementation

Write the D=4 table and offset rotation in my notes. Mark buffer versus Parameter state, source versus target indices, capacity and the missing cache-offset interface. Predict the error from using only the newest token, then inspect the actual full-prefix generation path in Lesson 19. Continue to [learned Q/K/V](lessons/07-qkv.md). Source: [position representation](source:attention-paper).
