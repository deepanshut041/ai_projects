## 3. Keep one output per position and expose the matrix axes

One query gave position 0 a gathered vector. The encoder cannot stop with that one result: the next layer needs a sequence of representations, including an updated row for positions 1, 2 and 3. Repeat the same procedure with each input row supplying its own query.

All queries read the **same input** keys and values. Position 1 does not read position 0's newly computed output. This matters because it lets us calculate every row at once.

### Each score cell has an owner

Stack the queries into Q, keys into K and values into V. In our identity/doubling fixture, `Q=X`, `K=X`, `V=2X`. The product `Q @ K.T` is:

```text
                         source key
                        0    1    2    3
query 0, receives       [1    0    1    0]
query 1, receives       [0    1    1    0]
query 2, receives       [1    1    2    0]
query 3, receives       [0    0    0    0]
```

Cell `(i,j)` compares receiver i with source j. A row answers “how does this receiver score all available positions?” A column answers “how do different receivers score this one available position?” We need the first question to obtain a mixture for each receiver, so softmax normalizes **across columns within each row**.

![The score matrix, row-normalized weights and output rows with their dimensions](images/23-matrix-axes.svg)

After scaling and softmax, the weight rows are approximately:

```text
A = [[0.334881, 0.165119, 0.334881, 0.165119],
     [0.165119, 0.334881, 0.334881, 0.165119],
     [0.221181, 0.221181, 0.448581, 0.109057],
     [0.250000, 0.250000, 0.250000, 0.250000]]
```

The product `C = A @ V` sums over the available-position axis. It preserves the receiver axis and the value-feature axis:

$$C_{im}=\sum_j A_{ij}V_{jm}.$$

For example, `C[0,1]` is the **second feature received by query 0**. Its calculation is `A[0,0]×0 + A[0,1]×2 + A[0,2]×2 + A[0,3]×0`. No position disappeared merely because matrix multiplication was used.

### Put the repeated computation in one function

This function accepts queries and keys with the same comparison width, and paired keys/values with the same source-position count. All keys are available for now. `keepdim=True` retains a size-one column for each row's maximum and denominator, so each receiver uses its own normalization.

```python
import math
import torch

def attend(q, k, v):
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    shifted = scores - scores.max(dim=-1, keepdim=True).values
    exponentials = torch.exp(shifted)
    weights = exponentials / exponentials.sum(dim=-1, keepdim=True)
    context = weights @ v
    return context, weights

torch.set_printoptions(precision=6, sci_mode=False)
x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]], dtype=torch.float64)
q = x
k = x
v = 2 * x
context, weights = attend(q, k, v)
print(weights)
print(context)

batch_x = x.unsqueeze(0)
batch_context, batch_weights = attend(batch_x, batch_x, 2 * batch_x)
print(batch_context.shape)
assert torch.allclose(batch_context[0], context)
```

The output rows are `[1.339523,1.000000]`, `[1.000000,1.339523]`, `[1.339523,1.339523]` and `[1,1]`. The fourth row started at zero but receives a nonzero context by averaging the four available values. They have different mixtures even though they read the same source rows. The first row is exactly chapter 2's calculation, now embedded in the full matrix.

### Batch sequences without mixing them

`unsqueeze(0)` adds a batch axis: `(4,2)` becomes `(1,4,2)`. The assertion checks that this wrapper preserves the result. For a batch of B independent examples, use:

| Quantity | Shape | What the axes mean |
| --- | --- | --- |
| Q | `(B,T,d_k)` | example, receiving position, comparison feature |
| K | `(B,S,d_k)` | example, available position, comparison feature |
| V | `(B,S,d_v)` | example, available position, contributed feature |
| Scores and A | `(B,T,S)` | example, receiver, available position |
| C | `(B,T,d_v)` | example, receiver, received feature |

The batch axis is carried through independently. It must never be flattened into S, which would allow one sentence to read another sentence's tokens. Query count T and source count S can differ; cross-attention will use that freedom.

### Where does word order enter?

There is a missing capability if we feed only token embeddings into this operation: unrestricted self-attention has no inherent notion of first, second, third or fourth. Reorder X's rows and its output rows reorder in the same way. Pairwise scores depend on vector content, not an invisible row-number feature.

More precisely, for a permutation matrix P and no masks or positional signals, `Attention(PQ,PK,PV) = P Attention(Q,K,V)`. The matching keys and values move together. The operation treats a sequence as a collection of rows with corresponding outputs.

Translation needs order. The original Transformer adds positional encodings to token embeddings before the encoder/decoder stacks. Then the input rows carry position information, so reordering tokens while assigning them their new positions changes more than a simple row permutation. The [paper's section 3.5](https://arxiv.org/html/1706.03762v7) specifies that component. Our numerical fixture isolates attention; it does not calculate positional encodings.

The matrix calculation exposes another tradeoff: a length-n sequence has n² pair comparisons per head. Scores and weights can be computed in parallel, but parallelism does not remove that quadratic work or, in the explicit implementation, storage. This is why “can connect distant positions directly” should not be read as “free for arbitrarily long inputs”.

**Runnable copy:** [all positions and batch check](examples/walkthrough/03_shapes.py).

[[practice:shapes-check]]
