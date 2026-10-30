## 1. Several independent mixtures, then one output

The Attention Foundations route ended with one causal attention head: projected queries and keys select a weighted mixture of projected values, with future positions excluded. Here we change the arrangement of those projections. Several heads build separate mixtures, then an output projection combines their results.

This folder owns head and key/value layouts. Multi-head attention is the implemented reference here. GQA, MQA and related arrangements can be added to this same folder later; this course does not implement them in advance.

### What a second head changes

For one head, every value-feature column is mixed using the same matrix $A$:

$$Z=AV.$$

A single query row can distribute weight over several source positions. It is therefore too strong to say that one head can attend to only one token or represent only one relationship. Its actual restriction is that one distribution is shared across the value-feature channels of that head.

For heads $r=1,\ldots,H$, use separate parameter matrices:

$$Q_r=XW_{Q,r},\quad K_r=XW_{K,r},\quad V_r=XW_{V,r},$$
$$Z_r=\operatorname{softmax}\left(Q_rK_r^\top/\sqrt{d_k}+M\right)V_r.$$

The different $A_r$ matrices can route information differently. Separate parameters permit this; they do not guarantee interpretable roles or prevent redundant heads. Labels such as “syntax head” require analysis of an actually trained model, not an assumption based on head number.

![Independent attention patterns are concatenated before output projection](images/01-heads.svg)

### A small example with two genuinely different patterns

Keep the three input rows from the foundations route:

$$X=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}.$$

Each of two heads now has comparison width one and value width one. Head 1 selects the first input coordinate for queries and keys, and returns the sum of the two coordinates as its scalar value. Head 2 selects the second coordinate for queries and keys, and returns the first coordinate minus the second as its value.

| Head | $W_Q$ | $W_K$ | $W_V$ |
| --- | --- | --- | --- |
| 1 | `[1,0]` as a column | `[1,0]` as a column | `[1,1]` as a column |
| 2 | `[0,1]` as a column | `[0,1]` as a column | `[1,-1]` as a column |

For the final position, Head 1 has query 1 and keys `[1,0,1]`, so its weight row is approximately `[0.422319,0.155362,0.422319]`. Head 2 has query 1 and keys `[0,1,1]`, yielding `[0.155362,0.422319,0.422319]`. Scaling divides by $\sqrt1$, so it makes no numerical change here.

The value lists are `[1,1,2]` and `[1,-1,0]`. Their final-position context scalars are about `1.422319` and `-0.266956`. Concatenation puts those two outputs next to each other:

$$z_{\mathrm{joined},3}=[1.422319,-0.266956].$$

At the first position only the first key is allowed. Its two head outputs are `[1]` and `[1]`, so concatenation gives `[1,1]`. The head arrangement changes neither the causal permission rule nor the meaning of the sequence axis.

### Concatenation preserves the separate channels; projection mixes them

Use the row-vector output matrix

$$W_O=\begin{bmatrix}1&0.5\\-0.5&1\end{bmatrix},\qquad
Y=\operatorname{Concat}(Z_1,Z_2)W_O.$$

The first position becomes $[1,1]W_O=[0.5,1.5]$. At the final position,

$$Y_3=[1.422319+0.133478,\ 0.711159-0.266956]
\approx[1.555797,0.444203].$$

The output projection acts on feature channels **within each token**. It does not create a new token-to-token attention distribution; the cross-token mixing already happened inside the heads. It can combine, rotate, suppress or preserve head features according to learned parameters.

Concatenation order is a consistent bookkeeping choice, not an automatic priority ranking. If we swap the two joined channels and swap the matching rows of $W_O$, the final output remains the same. The output projection supplies learned combinations of head features; it need not scramble every coordinate, and an identity projection would preserve their separation.

<!-- example:multihead.py:start -->

Complete file: [multihead.py](examples/multihead.py).

```python
"""Two independent scalar heads, with all helpers defined here."""
import json
import math


def dot(a, b):
    if len(a) != len(b):
        raise ValueError('Mismatched widths')
    return sum(x * y for x, y in zip(a, b))


def matmul(a, b):
    return [[dot(row, column) for column in zip(*b)] for row in a]


def head(x, wq, wk, wv):
    q, k, v = matmul(x, wq), matmul(x, wk), matmul(x, wv)
    outputs, patterns = [], []
    for i, query in enumerate(q):
        scores = [dot(query, key) / math.sqrt(len(key)) for key in k[:i + 1]]
        maximum = max(scores)
        numerators = [math.exp(s - maximum) for s in scores]
        weights = [s / sum(numerators) for s in numerators]
        outputs.append(matmul([weights], v[:i + 1])[0])
        patterns.append(weights + [0.0] * (len(x) - i - 1))
    return outputs, patterns


if __name__ == '__main__':
    x = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    h1, a1 = head(x, [[1.0], [0.0]], [[1.0], [0.0]], [[1.0], [1.0]])
    h2, a2 = head(x, [[0.0], [1.0]], [[0.0], [1.0]], [[1.0], [-1.0]])
    joined = [first + second for first, second in zip(h1, h2)]
    output = matmul(joined, [[1.0, 0.5], [-0.5, 1.0]])
    def rounded(matrix):
        return [[round(v, 6) for v in row] for row in matrix]
    assert a1[-1] != a2[-1]
    assert output[0] == [0.5, 1.5]
    print(json.dumps({'head1_weights': rounded(a1), 'head2_weights': rounded(a2),
                      'concatenated': rounded(joined), 'output': rounded(output)}))
```

<!-- example:multihead.py:end -->

From `ai_projects/docs/tutorials/projects/attention-labs/attention-heads/v1/`, run:

```bash
python3 examples/multihead.py
```

The script is self-contained and prints both full head-weight matrices, the concatenated outputs and the final outputs. It checks that the last-row distributions differ. The weights are hand-set snapshots, so no claim about learned linguistic specialization follows from these results.

### Dimensions express a design choice

In general, $X$ has width $D$, each head returns width $d_v$, concatenation has width $Hd_v$, and $W_O$ can have shape $Hd_v\times D_{\mathrm{out}}$. A common Transformer choice sets $D_{\mathrm{out}}=D$ and $d_k=d_v=D/H$. That produces the familiar relation $D=H d_k$.

This relation is a convention enforced by our packed implementation, not a universal mathematical requirement for attention. Other implementations can use different projection widths. At fixed $D$ with the conventional split, increasing $H$ makes each head narrower rather than multiplying the total projection width by $H$.

Each general head projection can use **all $D$ input features**. Its $D\times d_k$ query matrix produces a smaller set of comparison features from that entire input row. Our sparse hand-set matrices make the arithmetic small; they are not a rule to cut the raw input into disjoint slices. The packed implementation splits the outputs of learned projections into heads.

[[practice:heads-check]]

The optional [transfer review](practice/transfer-review.md) tests feature mixing, equal-size axis mistakes and the difference between packing and sharing.

### The original lecture and slides

The [caption provenance](project/source-provenance.json) records the English subtitle source and hash used in the lecture review.

This route develops [Vizuara Lecture 8](https://www.youtube.com/watch?v=IOmg660SvmM). Slides 8–11 introduce independent projections, concatenation, output mixing and dimensions. The next lesson opens its packed tensor implementation. We retain the original deck here while using new prose, diagrams and numerical parameters throughout our explanation.

```presentation
{"title":"Vizuara · Lecture 8 · Original presentation","url":"https://lecture8-attention-mechanisms.vercel.app/#/1"}
```

The [source map](project/source-map.md), [learning points](project/learning-points.md), [reference runs](project/reference-runs.md), [implementation record](project/implementation.md), [readiness record](project/readiness.md), and [research](research.md) retain the course's scope and return-to-work evidence. You can also download the [complete standalone Python reference](examples/multihead.py).
