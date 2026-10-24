## Read attention as rows, columns and batches

The demo runs. Now we need to explain why its matrices have their observed shapes. We will use rows as positions and columns as features, then add batch and head axes without changing the underlying dot product.

### One dot product answers one comparison

For query `q=[1,2]` and key `k=[3,4]`, the dot product is `1×3 + 2×4 = 11`. Matching coordinates multiply and their products add. This requires the query and key to use the same coordinate dimension. The result is a scalar score, not a two-coordinate vector.

For several query rows and key rows, matrix multiplication evaluates every pair. If `Q` has shape `(T,d)` and `K` has shape `(S,d)`, transpose the last two key axes to get `(d,S)`. Then:

$$R=QK^\mathsf{T},\qquad R_{ij}=\sum_{r=0}^{d-1}Q_{ir}K_{jr},\qquad R\in\mathbb{R}^{T\times S}.$$

Here `i` selects a query position, `j` selects a key position and `r` selects a feature coordinate. Each row answers “how does this query compare with each key?” Each column collects comparisons against one key. Softmax will later run across each row's key axis.

### Work an unequal-shape multiplication

```text
left = [[1,2,3],       right = [[ 7, 8],
        [4,5,6]]               [ 9,10],
                               [11,12]]

(2×3) @ (3×2) -> (2×2)
result[0,0] = 1×7 + 2×9 + 3×11 = 58
result[0,1] = 1×8 + 2×10 + 3×12 = 64
result       = [[58,64], [139,154]]
```

The shared inner dimension is consumed; the outer dimensions remain. Unequal shapes make a useful independent check because a wrong transpose cannot hide behind a square example. Our matrix-display tests check these exact numbers.

![A highlighted row meets a highlighted column to form one result cell.](images/17-basic-matrix-product.png)

### Mix values using the key axis

Once a query has weights `A` of shape `(T,S)`, each row mixes `S` value vectors. If `V` has shape `(S,d_v)`, then `C=A@V` has shape `(T,d_v)`. It has one output row per query and one output coordinate per value feature.

This explains why query and key lengths can differ. The first contraction compares query/key features. The second contracts key positions with their corresponding values. Queries and keys must agree in feature width; keys and values must agree in position count. The low-level mathematical operation can use a distinct value width. This project's public module projects all three to the same model width before splitting heads.

### Add batches and heads deliberately

The public module receives `src` as `(B,S,D)`: batch count, source positions, model features. For self-attention the queries also have `S` positions. For cross-attention the target provides `(B,T,D)` queries.

| Tensor | After splitting heads | Role |
| --- | --- | --- |
| `q` | `(B,H,T,d)` | One query vector per batch/head/query position |
| `k` | `(B,H,S,d)` | One key vector per batch/head/source position |
| `v` | `(B,H,S,d)` | Values aligned with source keys |
| `q @ k.transpose(-2,-1)` | `(B,H,T,S)` | All query/key scores per head |
| `weights @ v` | `(B,H,T,d)` | Retrieved head contexts |

Here `H` is head count and `d=D/H`. Batch/head axes remain separate. The final two axes participate in each matrix multiplication; PyTorch applies that product across compatible leading axes. For a four-dimensional tensor, `.T` reverses all axes and is not the operation we want. `.transpose(-2,-1)` exchanges only key positions and key features.

### Trace a concrete shape through the demo

For `B=3`, `S=5`, `D=9`, `H=3`, each head has width 3. Queries/keys/values become `(3,3,5,3)`. Scores become `(3,3,5,5)`. Weighted values return `(3,3,5,3)`. Reassembling the three heads restores `(3,5,9)`.

The public interface rejects integer token IDs because it expects floating-point representations. An integer matrix `(B,S)` is not interchangeable with embeddings `(B,S,D)`. Later tokenization and embedding layers will bridge those forms; they have not been implemented here.

[[practice:shapes-check]]

At this point the contractions are defined, but raw dot products can be negative, unbounded and hard to compare. The next lesson explains the transformation into usable weights.
