## 5. Give every position a context from its own sequence

So far, we described a decoder query retrieving source information. Now supply queries and memory from the same sequence. Each input position receives its own mixture of the sequence's content. That is the input-origin relationship called self-attention; it does not by itself specify a score formula, a mask, or a number of heads.

### Compute all three queries

Use the same three hand-set rows,

$$X=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}.$$

Associate them with the positions in “Quiet cats nap” only as an indexing aid. The coordinates are synthetic and have no measured linguistic interpretation. For this first pass, use each row directly for all three roles: query, key, and returned value. Also leave scores unscaled to expose the raw dot-product calculation; scaled scores can be substituted without changing the self-attention relationship.

All pairwise scores fit into

$$S=XX^\top=\begin{bmatrix}1&0&1\\0&1&1\\1&1&2\end{bmatrix}.$$

Row $i$ asks from position $i$; column $j$ identifies the potential source position. The first score is $[1,0]\cdot[1,0]=1$, the second is $[1,0]\cdot[0,1]=0$, and the bottom-right entry is $[1,1]\cdot[1,1]=2$.

Normalize independently across each row and multiply by the value rows:

Dividing raw scores by their sum is not a suitable general replacement for softmax. Scores $[1,-1]$ have a zero sum, while $[-1,2]$ would give a negative mixing weight. Exponentiation followed by normalization gives positive weights for finite unmasked scores and preserves their ranking. It is a particular useful choice of mixture, not the only normalization ever used in attention research.

$$A=\operatorname{softmax}_{\mathrm{keys}}(S),\qquad Z=AX.$$

![The query and source axes of the weight matrix](images/05-self-matrix.svg)

The computed matrices, rounded only for display, are

$$A\approx\begin{bmatrix}
0.422319&0.155362&0.422319\\
0.155362&0.422319&0.422319\\
0.211942&0.211942&0.576117
\end{bmatrix},\qquad
Z\approx\begin{bmatrix}
0.844638&0.577681\\
0.577681&0.844638\\
0.788058&0.788058
\end{bmatrix}.$$

For example, the first output's first coordinate is $0.422319+0.422319$, because source rows 1 and 3 carry a one in that coordinate. Its second coordinate is $0.155362+0.422319$. Every input position gets an output row; there is no single pooled vector for the entire sentence unless another operation explicitly pools these rows.

### Shape and normalization are part of the meaning

$X$ is $3\times2$, $X^\top$ is $2\times3$, scores and weights are $3\times3$, and $Z$ is $3\times2$. Normalizing along the feature axis of $X$ would not produce distributions over source positions. Normalizing the columns of $S$ would answer a different question and would not ensure each query's weights sum to one.

Although $S$ is symmetric, $A$ need not be. The first and third score rows have different denominators, so $A_{13}=0.422319$ while $A_{31}=0.211942$. Softmax preserves a row's score ordering; it does not preserve arbitrary matrix symmetries.

### What happens when inputs change?

Change one source row and it can affect its own query, its key in every other query's comparisons, and its value in every mixture. That is why contextualization is a collective computation. It is not a rule that only the modified token's output changes.

For a zero query `[0,0]` and three finite allowed keys, all dot products are zero and the weights are `[1/3,1/3,1/3]`. A genuine zero row is therefore meaningful input in this toy computation. Padding is a storage convention that requires an explicit mask; zeros alone do not declare a row invalid.

If we permute all input rows using the same permutation matrix $P$, then unmasked, position-free self-attention obeys

$$Z(PX)=PZ(X).$$

The score matrix becomes $PSP^\top$, row-wise softmax permutes with it, and multiplication by $PX$ produces the permuted output. This is **permutation equivariance**, not a statement that ordered output arrays are unchanged. A causal mask or positional information changes the conditions. We identify the limitation here without teaching Lecture 9's positional methods.

<!-- example:05_self_attention.py:start -->

Complete file: [05_self_attention.py](examples/05_self_attention.py).

```python
import json
from row_math import X, dot, matmul, rounded, softmax, transpose


def self_attention(inputs):
    scores = matmul(inputs, transpose(inputs))
    weights = [softmax(row) for row in scores]
    return scores, weights, matmul(weights, inputs)


if __name__ == '__main__':
    scores, weights, output = self_attention(X)
    permutation = [2, 0, 1]
    permuted = [X[i] for i in permutation]
    _, _, changed = self_attention(permuted)
    error = max(abs(changed[i][j] - output[old][j])
                for i, old in enumerate(permutation) for j in range(2))
    zero_weights = [softmax([dot([0.0, 0.0], key) for key in X])]
    assert error < 1e-14
    assert zero_weights[0] == [1 / 3] * 3
    assert weights[0][2] != weights[2][0]
    print(json.dumps({'scores': scores, 'weights': rounded(weights),
                      'output': rounded(output), 'permutation_equivariant': error < 1e-14,
                      'zero_query_weights': rounded(zero_weights[0])}))
```

<!-- example:05_self_attention.py:end -->

Run `python3 examples/05_self_attention.py`. It prints the full matrices, tests the zero query and verifies the permutation relation. The equality checks use unrounded values; displayed six-decimal row sums can differ slightly from one due to rounding.

### What this version can learn

With $X$ held fixed, this calculation has no trainable projections, so it cannot change its routing by updating attention parameters. In a larger model, trainable input embeddings could still change $X$. “No trainable parameters in this isolated attention operation” is narrower than “nothing upstream could ever learn”.

The next lesson keeps the same input rows but learns separate transformations for comparing and delivering information. That makes the roles independently adjustable instead of forcing every role to use the raw coordinates.

[[practice:self-check]]

### The original lecture and slides

[Vizuara Lecture 5](https://www.youtube.com/watch?v=vQPCjniSfNU) develops the parameter-free pass and its matrix form. The 27-slide deck includes a different six-token example. Our three-row fixture is intentionally different and remains consistent through the later projections and causal mask.

```presentation
{"title":"Vizuara · Lecture 5 · Original presentation","url":"https://lecture5-attention-mechanisms.vercel.app/#/1"}
```
