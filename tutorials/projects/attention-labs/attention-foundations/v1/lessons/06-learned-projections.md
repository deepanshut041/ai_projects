## 6. Learned self-attention: queries, keys and values

The six-word calculation returned one context vector per position. With those input vectors fixed, it had no parameter that could learn a more useful comparison or a different representation to deliver.

![Each starting embedding must become a representation informed by its context](images/original-slides/lecture-06-slide-02.png)

We still want $x^{(i)}\rightarrow z^{(i)}$. What changes is the numerical representation used for each job inside the operation.

![Scoring and self-attention connectivity now work together](images/original-slides/lecture-06-slide-03.png)

Scaled dot products answer how to compare; self-attention answers where the compared inputs come from. Now add learned projections to those choices.

![The staircase advances from raw-input self-attention to learned projections, before masks and heads](images/original-slides/lecture-06-slide-05.png)

We will follow the same three passes: first a concrete failure of fixed comparisons, then every projection and one complete query, then the formula, variance experiment and reusable code.

### A tie that a parameter can change

![The sentence has two occurrences of it, referring to different nouns](images/original-slides/lecture-06-slide-08.png)

“The dog chased the ball, but it couldn't catch it.” The first **it** refers to the dog; the second refers to the ball. This motivates two distinct questions. Can we learn to score **different source vectors** differently? And do the two occurrences of **it** have different information available to distinguish their requests?

Start with the first question. Choose illustrative embeddings

$$x_{\rm it}=[0.5,0.5,0.1],\quad x_{\rm dog}=[0.9,0.1,0.1],\quad
x_{\rm ball}=[0.1,0.9,0.1].$$

Raw dot products give $0.51$ for both dog and ball. With these embeddings held fixed, this isolated attention operation has no trainable comparison parameter to change the tie.

![A diagonal key projection doubles the second coordinate and breaks the dog/ball tie](images/original-slides/lecture-06-slide-09.png)

Keep the query fixed and choose

$$W_K=\operatorname{diag}(1,2,1).$$

The transformed keys are $k_{\rm dog}=[0.9,0.2,0.1]$ and $k_{\rm ball}=[0.1,1.8,0.1]$. Their query comparisons become

$$q_{\rm it}\cdot k_{\rm dog}=0.45+0.10+0.01=0.56,$$

$$q_{\rm it}\cdot k_{\rm ball}=0.05+0.90+0.01=0.96.$$

The same matrix changes both source vectors, but their different coordinates cause different effects. A model can learn such a transformation through a task loss; here we selected one manually to make its effect inspectable.

![With stretch one, the two key scores and weights are tied](images/original-slides/lecture-06-lab-tie.png)

![With stretch two, ball receives a larger score and softmax weight](images/original-slides/lecture-06-lab-stretch.png)

Read both levels of bars: the changed scores become unequal weights. The mouse control changes a parameter snapshot; it is not itself an optimization run.

This fixes a tie between **different keys**. It does not, by itself, distinguish the two identical it embeddings. If two positions have identical features, shared projections and the same allowed memory, they get identical queries and outputs. Context from earlier layers, position information or different allowed memory supplies additional distinctions. We will not claim the diagonal matrix alone solves the complete pronoun task.

### Give each input three separate jobs

![One input branches into query, key and value projections](images/original-slides/lecture-06-slide-10.png)

A **query** represents what the requesting position is looking for. A **key** supplies the comparison features of an available position. A **value** supplies the content to mix after matching. The database-lookup analogy helps remember the roles, but these are learned numerical features, not guaranteed human-readable questions or dictionary records.

![Raw input comparisons and mixtures become comparisons and mixtures of transformed vectors](images/original-slides/lecture-06-slide-11.png)

Follow the score path and content path separately. Query and key interact to determine routing weights; the value travels around that comparison and enters the final mixture. Changing only $W_V$ therefore changes content while leaving scores and weights fixed, provided $X,W_Q,W_K$ stay fixed.

![The routing branches and delivered-content branch have different responsibilities](images/06-projections.svg)

For row-vector inputs,

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.$$

For $X\in\mathbb R^{N\times d_{\rm in}}$, take $W_Q,W_K\in\mathbb R^{d_{\rm in}\times d_k}$ and $W_V\in\mathbb R^{d_{\rm in}\times d_v}$. Queries and keys need the same comparison width. Values may have a different width; our upcoming numerical example chooses both widths as two.

### Freeze a five-word example and calculate every projection

Use **The dog chased the ball**. This is a new five-word input, separate from the three-vector tie demonstration and the previous six-word example.

| Token | Input |
| --- | --- |
| The | $[0.5,1.2,0.8]$ |
| dog | $[1.1,0.3,0.7]$ |
| chased | $[0.6,1.5,0.9]$ |
| the | $[0.9,0.8,0.2]$ |
| ball | $[1.2,1.7,0.5]$ |

Choose one fixed snapshot of the trainable matrices:

$$W_Q=\begin{bmatrix}0.6&0.2\\0.1&0.7\\0.5&0.4\end{bmatrix},\quad
W_K=\begin{bmatrix}0.3&0.6\\0.4&0.2\\0.7&0.5\end{bmatrix},\quad
W_V=\begin{bmatrix}0.2&0.5\\0.8&0.1\\0.3&0.6\end{bmatrix}.$$

Each product has shape $(5\times3)(3\times2)=5\times2$.

![Three products project the five input positions into query, key and value coordinates](images/original-slides/lecture-06-slide-14.png)

Count five positions in each result, whatever the schematic grid's drawn subdivisions suggest. Projection changes feature width, not sequence length. For The,

$$q_{\rm The}=0.5[0.6,0.2]+1.2[0.1,0.7]+0.8[0.5,0.4]=[0.82,1.26].$$

The complete result is:

| Token | Query | Key | Value |
| --- | --- | --- | --- |
| The | $[0.82,1.26]$ | $[1.19,0.94]$ | $[1.30,0.85]$ |
| dog | $[1.04,0.71]$ | $[0.94,1.07]$ | $[0.67,1.00]$ |
| chased | $[0.96,1.53]$ | $[1.41,1.11]$ | $[1.59,0.99]$ |
| the | $[0.72,0.82]$ | $[0.73,0.80]$ | $[0.88,0.65]$ |
| ball | $[1.14,1.63]$ | $[1.39,1.31]$ | $[1.75,1.07]$ |

The table preserves the two The/the positions separately. Sequence position identifies each row; a token label is not a unique row identifier.

### Follow ball through score, scale and softmax

![The ball query compares against all five keys, then scales and normalizes the row](images/original-slides/lecture-06-slide-16.png)

Use $q_{\rm ball}=[1.14,1.63]$. Against the first key, the score is $1.14(1.19)+1.63(0.94)=2.8888$. Repeat for all five keys:

| Source | Raw score | Scaled score | Weight |
| --- | ---: | ---: | ---: |
| The | 2.8888 | 2.042690 | 0.172734 |
| dog | 2.8157 | 1.991001 | 0.164032 |
| chased | 3.4167 | 2.415972 | 0.250895 |
| the | 2.1362 | 1.510522 | 0.101452 |
| ball | 3.7199 | 2.630367 | 0.310887 |

The divisor is $\sqrt2$ because each query/key has two coordinates. Five counts source positions; three was the original input width. Neither is the scaling dimension here.

![The interactive five-position score and weight bars for ball](images/original-slides/lecture-06-lab-ball.png)

Ball and chased have the largest two weights in this selected snapshot. That is a property of these specified matrices, not evidence that training has discovered a verb/object relation. Learning can change the distribution and its usefulness without necessarily making it sharper.

### Mix values, not the original input embeddings

![Five value vectors contribute to ball's two-coordinate context](images/original-slides/lecture-06-slide-17.png)

Compute

$$z_{\rm ball}=\sum_{j=1}^{5}\alpha_{{\rm ball},j}v_j
\approx[1.366709,0.957835].$$

For its first coordinate, the contributions are the five weights multiplied by $1.30,0.67,1.59,0.88,1.75$, respectively. For its second, multiply by $0.85,1.00,0.99,0.65,1.07$. Keep unrounded weights until the final display. The two-decimal weights in the illustration sum to 0.99 due to rounding; do not renormalize or compute from them as if they were exact.

The output has width two because the values have width two. Mixing the three-coordinate $x_j$ instead would give another operation and discard the learned content projection.

### Check the scaling argument experimentally

![Unscaled dot-product variance grows with dimension while the scaled variance stays near one](images/original-slides/lecture-06-slide-19.png)

The horizontal axis changes comparison width. The vertical quantity here is **variance**, whose unscaled theoretical value is $d_k$ under independent, zero-mean, unit-variance coordinates. This differs from the typical-magnitude graph in Lecture 4, which grows like $\sqrt{d_k}$.

![A sample of random pairs at width five gives raw variance near five and scaled variance near one](images/original-slides/lecture-06-lab-variance5.png)

![At width one hundred the raw and scaled variances are on very different scales](images/original-slides/lecture-06-lab-variance100.png)

Read both panels. They summarize the **same random dot products**, before and after division by $\sqrt{d_k}$. The interactive sample uses 500 pairs, so the reported estimates fluctuate when resampled. Our deterministic program below uses 20,000 pairs and measures about 5.02197 versus 1.00439 at width five, and 102.19674 versus 1.02197 at width one hundred.

Finite samples need not equal the theoretical values. Learned or correlated features need not satisfy the sampling assumptions. Scaling controls one source of score magnitude; it does not guarantee a particular attention distribution or successful training.

### Read the whole formula through its data flow

$$Z=\operatorname{softmax}_{\rm keys}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.$$

![The input splits through three projections; queries and keys determine weights, which combine values](images/original-slides/lecture-06-slide-21.png)

$QK^\top$ scores every query against every key. Dividing controls the comparison scale. Row-wise softmax creates a distribution over source positions. Multiplication by $V$ returns one value mixture per query. In this example, $X$ is $5\times3$, all three projections are $5\times2$, weights are $5\times5$, and output is $5\times2$. Use those labels to interpret the schematic; every projection and output still has five rows.

### Put the tensor steps into reusable classes

The complete file below has both explicit `nn.Parameter` matrices and `nn.Linear` projections. Parameters registered on an `nn.Module` are discoverable by an optimizer; random initialization alone does not train them. The forward call computes activations without updating the matrices.

`nn.Linear` stores its coefficient matrix as output-by-input and applies its transpose. That is why matching our row-vector $XW$ snapshot requires copying `W.T` into its `.weight`. Both classes transpose only the last two key axes so the same code supports `[positions,features]` and `[batch,positions,features]` inputs.

Complete file: [learned_attention.py](examples/learned_attention.py). Paste the full file into one Colab cell, or upload it and run `!python learned_attention.py`. It needs PyTorch and no other tutorial helper files.

```python
"""Five-word projection snapshot, two equivalent classes and a variance probe."""
import math
import torch
from torch import nn


class ParameterAttention(nn.Module):
    def __init__(self, d_in, d_key, d_value):
        super().__init__()
        self.WQ = nn.Parameter(torch.rand(d_in, d_key, dtype=torch.float64))
        self.WK = nn.Parameter(torch.rand(d_in, d_key, dtype=torch.float64))
        self.WV = nn.Parameter(torch.rand(d_in, d_value, dtype=torch.float64))

    def forward(self, x):
        q, k, v = x @ self.WQ, x @ self.WK, x @ self.WV
        weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1]), dim=-1)
        return weights @ v, weights


class LinearLayerAttention(nn.Module):
    # This name means linear projection layers, not recurrent linear attention.
    def __init__(self, d_in, d_key, d_value):
        super().__init__()
        self.query = nn.Linear(d_in, d_key, bias=False, dtype=torch.float64)
        self.key = nn.Linear(d_in, d_key, bias=False, dtype=torch.float64)
        self.value = nn.Linear(d_in, d_value, bias=False, dtype=torch.float64)

    def forward(self, x):
        q, k, v = self.query(x), self.key(x), self.value(x)
        weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1]), dim=-1)
        return weights @ v, weights


if __name__ == '__main__':
    torch.manual_seed(123)
    x = torch.tensor([[.5, 1.2, .8], [1.1, .3, .7], [.6, 1.5, .9],
                      [.9, .8, .2], [1.2, 1.7, .5]], dtype=torch.float64)
    wq = torch.tensor([[.6, .2], [.1, .7], [.5, .4]], dtype=torch.float64)
    wk = torch.tensor([[.3, .6], [.4, .2], [.7, .5]], dtype=torch.float64)
    wv = torch.tensor([[.2, .5], [.8, .1], [.3, .6]], dtype=torch.float64)
    q, k, v = x @ wq, x @ wk, x @ wv
    scores = q @ k.T
    weights = torch.softmax(scores / math.sqrt(2), dim=-1)
    contexts = weights @ v
    for name, matrix in [('Q', q), ('K', k), ('V', v), ('scores', scores),
                         ('weights', weights), ('contexts', contexts)]:
        print(name, matrix.tolist())
    manual = sum((weights[-1, j] * v[j] for j in range(5)), torch.zeros(2, dtype=x.dtype))
    torch.testing.assert_close(manual, contexts[-1], atol=1e-14, rtol=0)
    explicit = ParameterAttention(3, 2, 2)
    linear = LinearLayerAttention(3, 2, 2)
    with torch.no_grad():
        for dest, source in [(explicit.WQ, wq), (explicit.WK, wk), (explicit.WV, wv),
                             (linear.query.weight, wq.T), (linear.key.weight, wk.T),
                             (linear.value.weight, wv.T)]:
            dest.copy_(source)
    torch.testing.assert_close(explicit(x)[0], contexts, atol=1e-14, rtol=0)
    torch.testing.assert_close(linear(x)[0], contexts, atol=1e-14, rtol=0)
    batch = torch.stack([x, x + .1])
    torch.testing.assert_close(explicit(batch)[0], linear(batch)[0], atol=1e-14, rtol=0)
    torch.manual_seed(123)
    for width in [5, 100]:
        queries = torch.randn(20000, width, dtype=torch.float64)
        keys = torch.randn(20000, width, dtype=torch.float64)
        raw = (queries * keys).sum(dim=-1)
        print('width', width, 'raw variance', raw.var(unbiased=False).item(),
              'scaled variance', (raw / math.sqrt(width)).var(unbiased=False).item())
    print('Both classes match the fixed snapshot, including the batch path.')
```

The fixed five-word outputs must match before comparing classes. The program copies the same snapshot into both classes, checks the tensor result, and checks a two-item batch. Independently initialized classes need not give equal values merely because they have the same shapes. `LinearLayerAttention` here means a class using linear projection layers; it is still **softmax attention**, not the later recurrent technique called linear attention.

The lecture's notebook also follows a six-word input with seed 123. You can make that same structural comparison by supplying the six rows from Lesson 5 to either class: $Q,K,V,Z$ have six rows instead of five, but the learned comparison width stays two. Changing a random seed changes initial matrix entries and outputs, while preserving their declared shapes. It does not measure model quality.

### Follow one actual gradient update on the compact fixture

To connect “trainable” to a measured change, the next calculation uses our earlier three-row transfer fixture and one declared scalar loss. It isolates one parameter rather than claiming to train a translator.

Return to the smaller three-row transfer fixture from Lesson 5 and use this parameter snapshot

$$W_Q=\begin{bmatrix}1&0\\0&1\end{bmatrix},\quad
W_K=\begin{bmatrix}1&1\\0&1\end{bmatrix},\quad
W_V=\begin{bmatrix}2&0\\0&-1\end{bmatrix}.$$

This gives

$$Q=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\quad
K=\begin{bmatrix}1&1\\0&1\\1&2\end{bmatrix},\quad
V=\begin{bmatrix}2&0\\0&-1\\2&-1\end{bmatrix}.$$

For source row 3, $[1,1]W_K=[1,2]$: its first output coordinate is $1\times1+1\times0$, and its second is $1\times1+1\times1$. The same row's value becomes `[2,-1]`, which shows explicitly that a negative value coordinate is allowed even though attention weights are nonnegative.

### Follow the last query all the way through




The last query `[1,1]` scores the three keys as `[2,1,3]`. Divide by $\sqrt2$ to get approximately `[1.414214,0.707107,2.121320]`. Stable softmax gives

$$a\approx[0.283995,0.140029,0.575975].$$

The output mixes **values**:

$$z_3=a_1[2,0]+a_2[0,-1]+a_3[2,-1]
\approx[1.719942,-0.716005].$$

Mixing $X$ instead of $V$ would return different numbers and discard the delivered-content projection. Using $\sqrt3$ would also be wrong: there are three source rows, but each query/key row has width two.

### One trainable entry, one actual update

Set $W_Q[0,0]=\theta$ and leave all other entries fixed. Because the last input row is `[1,1]`, its query becomes $[\theta,1]$. Use a deliberately small objective on that row alone:

$$L=\frac12(z_{3,1}-2)^2.$$

This target concerns one illustrative output feature, not a vocabulary token or translation score. At $\theta=1$, the loss is approximately `0.039216379`. Let $g_j=(z_{3,1}-2)V_{j,1}$ be the derivative with respect to weight $a_j$. Softmax backpropagates as

$$\frac{\partial L}{\partial e_j}=a_j\left(g_j-\sum_ra_rg_r\right).$$

Here the scaled score derivative is $\partial e_j/\partial\theta=K_{j,1}/\sqrt2$, so

$$\frac{\partial L}{\partial\theta}=\sum_j
a_j\left(g_j-\sum_ra_rg_r\right)\frac{K_{j,1}}{\sqrt2}
\approx-0.047694266.$$

A step with learning rate $\eta=0.2$ sets $\theta\leftarrow\theta-\eta\,\partial L/\partial\theta\approx1.009538853$. Re-evaluating reduces the loss to approximately `0.038763847`. The program checks the analytic derivative against a centered finite difference before applying that step.

The shared retrieval module below uses only the row operations already defined in Lesson 3. It introduces the projected single-head call and the parameter snapshot used in this lesson. Its optional causal path is opened in Lesson 7.

<!-- example:attention_math.py:start -->

Complete file: [attention_math.py](examples/attention_math.py).

```python
"""Projected single-head retrieval, using the earlier row operations."""
import math
from row_math import X, dot, matmul, rounded, softmax


def attention(q, k, v, causal=False, key_valid=None, scaled=True):
    if len(k) != len(v) or not k or not q:
        raise ValueError('Need queries and matching nonempty keys/values')
    if causal and len(q) != len(k):
        raise ValueError('This toy causal path assumes aligned self-attention')
    key_valid = [True] * len(k) if key_valid is None else list(key_valid)
    if len(key_valid) != len(k):
        raise ValueError('Key validity must match the key axis')
    scale = math.sqrt(len(k[0])) if scaled else 1.0
    scores = [[dot(query, key) / scale for key in k] for query in q]
    weights = [softmax(row, [valid and (not causal or j <= i)
                            for j, valid in enumerate(key_valid)])
               for i, row in enumerate(scores)]
    return scores, weights, matmul(weights, v)


WQ = [[1.0, 0.0], [0.0, 1.0]]
WK = [[1.0, 1.0], [0.0, 1.0]]
WV = [[2.0, 0.0], [0.0, -1.0]]
```

<!-- example:attention_math.py:end -->

<!-- example:06_projections.py:start -->

Complete file: [06_projections.py](examples/06_projections.py).

```python
import json
import math
from attention_math import X, WQ, WK, WV, attention, matmul, rounded


def forward(theta):
    wq = [[theta, 0.0], [0.0, 1.0]]
    q, k, v = matmul(X, wq), matmul(X, WK), matmul(X, WV)
    _, weights, output = attention(q, k, v)
    error = output[-1][0] - 2.0
    return error * error / 2, q, k, v, weights, output


if __name__ == '__main__':
    loss, q, k, v, weights, output = forward(1.0)
    a = weights[-1]
    d_output = output[-1][0] - 2.0
    d_weights = [d_output * value[0] for value in v]
    weighted_gradient = sum(p * g for p, g in zip(a, d_weights))
    d_scores = [p * (g - weighted_gradient) for p, g in zip(a, d_weights)]
    analytic = sum(g * key[0] / math.sqrt(2) for g, key in zip(d_scores, k))
    epsilon = 1e-5
    numeric = (forward(1 + epsilon)[0] - forward(1 - epsilon)[0]) / (2 * epsilon)
    assert abs(analytic - numeric) < 1e-9
    new_theta = 1 - 0.2 * analytic
    assert forward(new_theta)[0] < loss
    print(json.dumps({'q': q, 'k': k, 'v': v, 'last_weights': rounded(a),
                      'last_output': rounded(output[-1]), 'loss': round(loss, 9),
                      'gradient': round(analytic, 9), 'gradient_verified': True,
                      'updated_parameter': round(new_theta, 9),
                      'updated_loss': round(forward(new_theta)[0], 9)}))
```

<!-- example:06_projections.py:end -->

Run `python3 examples/06_projections.py`. The output contains every projection, the final weight row, output, loss, derivative and updated result. A finite-difference check tests this local derivative; it does not prove that a complete training system works or that every learning rate reduces every objective.

### The next restriction: information available during generation

![A future-position triangle will be removed from the attention matrix](images/original-slides/lecture-06-slide-25.png)

Today every position can inspect the complete supplied sentence. Next-token generation requires a stricter contract: features used to predict the next token cannot depend on that answer or later tokens. The next lesson adds a visibility rule to the scores before normalization.

[[practice:projection-check]]

### References

[Vizuara — The Attention Family, Lecture 6](https://www.youtube.com/watch?v=wshXBhGItkM) · [Presentation and diagrams](https://lecture6-attention-mechanisms.vercel.app/) · [Interactive projections and variance lab](https://lecture6-attention-lab.vercel.app/) · [Lecture Colab notebook](https://colab.research.google.com/drive/10DoLeN3FK9BqmCebRz3cdS5BrdEshbw2?usp=sharing).

[Vaswani et al. — Attention Is All You Need, section 3.2](https://arxiv.org/abs/1706.03762).
