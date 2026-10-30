## 3. Build a context that depends on the current query

We now retain every source state instead of handing off only the final one. The decoder's current need must influence which states contribute most. We represent that need by a query $s$, compare it with each available state $h_j$, normalize the comparisons, and use the resulting weights to combine the stored content.

### A context vector for each output step

![A source-state mixture weighted differently for each available token](images/original-slides/lecture-03-slide-09.png)

*Original diagram: [Vizuara, Lecture 3, slide 9](https://lecture3-attention-mechanisms.vercel.app/#/9). Captured from the linked presentation website. Open the image for full size.*

Each source state contributes its entire feature vector, multiplied by one scalar weight. The colored proportions describe contributions to one returned context, rather than probabilities that the decoder has already emitted those words. These are the source presentation’s four states; the worked calculation later uses our three retained rows.

For output step $t$ and source position $j$, write a scalar compatibility score as $e_{tj}=a(s_{t-1},h_j)$. The previous decoder state is the query in this convention. Another decoder implementation can order its internal updates differently; what matters here is that the query is available before this retrieval is computed.

Convert the source-position scores into weights and a context:

$$\alpha_{tj}=\frac{\exp(e_{tj})}{\sum_{r=1}^{N}\exp(e_{tr})},\qquad
c_t=\sum_{j=1}^{N}\alpha_{tj}h_j.$$

$N$ is the number of stored source positions. Each row of $\alpha$ belongs to one query and sums to one. The resulting $c_t$ has the same feature width as the stored states. We have not chosen an output word yet: a decoder update and vocabulary head still consume this context.

There is no circular requirement to know the next translated word before computing attention. The already available decoder state and previous output tokens form the query. Its retrieved context helps compute the next state and vocabulary logits; only then does the system choose the next token. During training, a reference next token supplies a target for the loss, rather than an input to this retrieval.

![A query produces scores, weights, and a returned context](images/03-additive-flow.svg)

### Why the score is called additive

![Separate query and source projections meet at an addition before a nonlinear scalar scorer](images/original-slides/lecture-03-slide-24.png)

*Original diagram: [Vizuara, Lecture 3, slide 24](https://lecture3-attention-mechanisms.vercel.app/#/24). Captured from the linked presentation website. Open the image for full size.*

The two branches first project the decoder query and one source state to compatible widths. Their addition happens inside the scoring network, before tanh and the final scalar projection. It is separate from the weighted sum of values that happens after normalization.

One possible learned scorer is

$$a(s,h)=u^\top\tanh(W_ss+W_hh+b).$$

For this formula only, the vectors are written as columns. If $s\in\mathbb R^{d_s}$, $h\in\mathbb R^{d_h}$ and the scorer's hidden width is $d_a$, the matrices have shapes $d_a\times d_s$ and $d_a\times d_h$, and $u,b\in\mathbb R^{d_a}$. The projections put query and source in a shared hidden space; they are added before the nonlinearity. That addition gives the scorer its name. Query and source widths need not be equal.

The tanh also matters. If it were removed, $u^\top W_ss$ would be the same scalar added to every source score for a fixed query. Softmax cancels a common additive offset, so that query contribution would not affect the relative weights. Nonlinear interaction prevents this simple cancellation.

The Bahdanau translation paper uses learned alignment and bidirectional encoder annotations. Our arithmetic isolates its scoring-and-retrieval idea; it does not implement that complete model or reproduce its training results. See the [original paper](https://arxiv.org/abs/1409.0473).

### A complete small calculation

![One decoding step from stored encoder states through scores, weights, context and vocabulary prediction](images/original-slides/lecture-03-slide-14.png)

*Original diagram: [Vizuara, Lecture 3, slide 14](https://lecture3-attention-mechanisms.vercel.app/#/14). Captured from the linked presentation website. Open the image for full size.*

Read the middle path first: compare the available query with every state, normalize the scores, and combine those states. Only afterward does the output-side computation use the context to score vocabulary entries. The source uses four English positions and rounded displayed values; our calculation below independently checks three states and keeps source-position weights separate from vocabulary probabilities.

We will reuse the hand-set memory

$$H=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}.$$

The rows will also serve as illustrative token features later. For now they are three available source states. Take scalar query $s=0$, query projection $[s,-s]$, identity source projection, scoring vector $u=[1,-1]$, and zero bias. The three hidden rows and scores are

| Source state | Hidden row before tanh | Score |
| --- | --- | --- |
| `[1,0]` | `[1,0]` | $\tanh(1)-\tanh(0)=0.761594$ |
| `[0,1]` | `[0,1]` | $\tanh(0)-\tanh(1)=-0.761594$ |
| `[1,1]` | `[1,1]` | $\tanh(1)-\tanh(1)=0$ |

Softmax gives approximately `[0.593494,0.129391,0.277115]`. The returned row is

$$c=0.593494[1,0]+0.129391[0,1]+0.277115[1,1]
\approx[0.870609,0.406506].$$

The first coordinate receives contributions from rows 1 and 3; the second receives contributions from rows 2 and 3. A negative score did not create a negative weight. Exponentiation makes all unmasked weights positive.

### Compute softmax without overflowing

For scores `[1000,1000]`, directly evaluating exponentials can overflow even though the answer is simply `[0.5,0.5]`. Subtract the maximum score $m$ before exponentiating:

$$\frac{e^{e_j-m}}{\sum_r e^{e_r-m}}=\frac{e^{-m}e^{e_j}}{e^{-m}\sum_r e^{e_r}}=\alpha_j.$$

The cancellation is exact in real arithmetic and makes numerical evaluation safer. This shared reference module supplies the row operations used from here onward. `dot` multiplies matching coordinates and sums them; `matmul` applies that dot product to every row/column pair. The shared input rows are the same hand-set memory used above; there are no trainable projection parameters in this helper module.

<!-- example:row_math.py:start -->

Complete file: [row_math.py](examples/row_math.py).

```python
"""Row-vector operations and the shared synthetic fixture; standard library only."""
import math


def dot(a, b):
    if len(a) != len(b):
        raise ValueError('Dot product widths must agree')
    return sum(x * y for x, y in zip(a, b))


def transpose(matrix):
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError('Expected a nonempty rectangular matrix')
    return [list(column) for column in zip(*matrix)]


def matmul(a, b):
    columns = transpose(b)
    return [[dot(row, column) for column in columns] for row in a]


def softmax(scores, allowed=None):
    if not scores:
        raise ValueError('Need at least one score')
    allowed = [True] * len(scores) if allowed is None else list(allowed)
    if len(allowed) != len(scores) or not any(allowed):
        raise ValueError('Every query needs at least one allowed key')
    if any(not math.isfinite(score) for score, use in zip(scores, allowed) if use):
        raise ValueError('Allowed scores must be finite')
    maximum = max(score for score, use in zip(scores, allowed) if use)
    numerators = [math.exp(score - maximum) if use else 0.0
                  for score, use in zip(scores, allowed)]
    denominator = sum(numerators)
    return [value / denominator for value in numerators]




def rounded(value):
    return [rounded(x) for x in value] if isinstance(value, list) else round(value, 6)


X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
```

<!-- example:row_math.py:end -->

The additive example then implements the particular scorer above. All imports refer to the displayed module in the same `examples/` folder.

<!-- example:03_additive.py:start -->

Complete file: [03_additive.py](examples/03_additive.py).

```python
import json
import math
from row_math import X, matmul, rounded, softmax


def additive(query, states):
    # Project a scalar query to [query, -query]; project each key identically.
    # The output scoring vector is [1, -1]. No biases in this fixture.
    scores = [math.tanh(query + h[0]) - math.tanh(-query + h[1]) for h in states]
    weights = softmax(scores)
    context = matmul([weights], states)[0]
    return scores, weights, context


if __name__ == '__main__':
    scores, weights, context = additive(0.0, X)
    other = additive(1.0, X)
    assert abs(sum(weights) - 1.0) < 1e-14
    assert other[1] != weights
    assert softmax([1000.0, 1000.0]) == [0.5, 0.5]
    print(json.dumps({'scores': rounded(scores), 'weights': rounded(weights),
                      'context': rounded(context), 'new_query_weights': rounded(other[1])}))
```

<!-- example:03_additive.py:end -->

Run `python3 examples/03_additive.py` from the package directory. The result includes the scores, weights and context above. It also changes the query to $s=1$, keeping the source states and parameters fixed, and obtains weights `[0.541045,0.206330,0.252626]`. The changed context comes from a new retrieval, not a new memory or a training step.

### Read an alignment matrix correctly

![An alignment matrix whose output-step rows can select reordered source columns](images/original-slides/lecture-03-slide-20.png)

*Original diagram: [Vizuara, Lecture 3, slide 20](https://lecture3-attention-mechanisms.vercel.app/#/20). Captured from the linked presentation website. Open the image for full size.*

Follow a row to see which source positions contribute to that output step; follow successive rows to see the retrieval change. The stronger cells need not form a diagonal when translation changes word order. This is a source illustration of alignment, not a learned result from our project or a requirement that attention always produces one-to-one matches.

Stack the weights for several output steps. Rows are output-step queries; columns are source positions. A bright entry can indicate that one source position contributes strongly to that step's context. It is not necessarily a one-to-one word alignment or a proof of a linguistic relationship. Source and target tokenization can differ, soft mixtures can distribute information, and a trained decoder uses more than this matrix.

The number of rows can differ from the number of columns. Generation still ends according to the decoder's output behavior, not when attention has used every column once. Nothing in row-wise normalization imposes a monotonic alignment or requires each source position to be selected exactly once.

For a concrete rectangular case, stack the two weight rows computed above for $s=0$ and $s=1$. The result is a $2\times3$ matrix: two output-step queries, three source positions. Its rows sum to one, but a column has no such constraint. Multiplying it by the $3\times2$ memory produces two width-two contexts. If a decoder state has width three, concatenating one context with that state produces width five; a $5\times V_{\mathrm{out}}$ matrix can then produce one logit for each output vocabulary token. Attention normalization and vocabulary normalization have different axes even inside the same decoding step.

[[practice:additive-check]]

### The original lecture and slides

[Vizuara Lecture 3](https://www.youtube.com/watch?v=6yIe4ekER14) supplies the additive-attention motivation and alignment discussion. The 28-slide presentation moves from retained source states to a full decoding step and then opens the scoring network. This lesson uses fresh coordinates and checks their arithmetic independently.

```presentation
{"title":"Vizuara · Lecture 3 · Original presentation","url":"https://lecture3-attention-mechanisms.vercel.app/#/1"}
```
