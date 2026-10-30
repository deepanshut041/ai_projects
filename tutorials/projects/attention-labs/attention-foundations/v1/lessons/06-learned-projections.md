## 6. Separate comparison features from delivered content

The previous operation used the same row as a query, a key, and a value. That ties two decisions together: how useful a source appears in a comparison, and which information it delivers if selected. Three trainable linear projections let those jobs use different features.

### Name the roles after their jobs

![Separate learned projections create comparison requests, matching features and returned content](images/original-slides/lecture-06-slide-10.png)

*Original diagram: [Vizuara, Lecture 6, slide 10](https://lecture6-attention-mechanisms.vercel.app/#/10). Captured from the linked presentation website. Open the image for full size.*

Start at one token representation and follow its three branches. The query branch determines how that position asks, the key branch determines how a source can match, and the value branch determines the content delivered after matching. These are learned roles; the labels do not assign fixed linguistic meanings to coordinates.

A query is the feature row used to request information. A key is the feature row it is compared with. A value is the content row included in the weighted mixture. These are learned numerical roles, not literal database records or guaranteed human-readable questions.

For row-vector inputs,

$$Q=XW_Q,\quad K=XW_K,\quad V=XW_V,\qquad
Z=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.$$

If $X$ has shape $N\times d_{\mathrm{in}}$, then $W_Q,W_K$ have shape $d_{\mathrm{in}}\times d_k$ and $W_V$ has shape $d_{\mathrm{in}}\times d_v$. Queries and keys must share the comparison width $d_k$; values can have a different width $d_v$. Output shape is $N\times d_v$.

![Three projections separate the routing path and content path](images/06-projections.svg)

Changing $W_Q$ or $W_K$ can change routing weights. Changing only $W_V$ changes returned content while leaving the weights fixed. During training a task loss can update all of these parameters, but adding them does not automatically make attention sharp, linguistically correct, or different across repeated token identities.

### Project the same three rows

![The input matrix is multiplied by three distinct projection matrices](images/original-slides/lecture-06-slide-14.png)

*Original diagram: [Vizuara, Lecture 6, slide 14](https://lecture6-attention-mechanisms.vercel.app/#/14). Captured from the linked presentation website. Open the image for full size.*

The three products create Q, K and V before any query–key score is normalized. The source diagram uses a five-token input with its own parameter values. Below, we apply the same three-branch operation to our three rows and named matrices, so every resulting number can be checked independently.

Take the input from Lesson 5 and the parameter snapshot

$$W_Q=\begin{bmatrix}1&0\\0&1\end{bmatrix},\quad
W_K=\begin{bmatrix}1&1\\0&1\end{bmatrix},\quad
W_V=\begin{bmatrix}2&0\\0&-1\end{bmatrix}.$$

This gives

$$Q=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\quad
K=\begin{bmatrix}1&1\\0&1\\1&2\end{bmatrix},\quad
V=\begin{bmatrix}2&0\\0&-1\\2&-1\end{bmatrix}.$$

For source row 3, $[1,1]W_K=[1,2]$: its first output coordinate is $1\times1+1\times0$, and its second is $1\times1+1\times1$. The same row's value becomes `[2,-1]`, which shows explicitly that a negative value coordinate is allowed even though attention weights are nonnegative.

### Follow the last query all the way through

![A complete learned self-attention flow from input through Q/K/V to output](images/original-slides/lecture-06-slide-21.png)

*Original diagram: [Vizuara, Lecture 6, slide 21](https://lecture6-attention-mechanisms.vercel.app/#/21). Captured from the linked presentation website. Open the image for full size.*

Follow Q and K into the score matrix, then follow V around the comparison branch into the final value mixture. This bypass explains why changing a value projection can change delivered features without changing the attention weights. The matrix axes and widths below give the exact contract for our smaller worked example.

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

### A boundary the projections cannot remove

If two positions begin with identical features and have identical allowed memory, position-free shared projections give them identical queries and the same attention output. A learned key matrix can break a tie between **different** key vectors; it cannot manufacture information that is absent from identical inputs. Context from earlier layers or information about position can distinguish positions, but those are additional inputs or mechanisms.

For batch tensors, transpose only the last two axes of the key tensor: `k.transpose(-2,-1)`, not an ambiguous `.T` on a tensor with more than two dimensions. The head-layout course will make those axes explicit.

[[practice:projection-check]]

### The original lecture and slides

[Vizuara Lecture 6](https://www.youtube.com/watch?v=wshXBhGItkM) motivates the Q/K/V projections. Its slides 10–18 separate the roles and work through projection, scoring and value mixing. This lesson adds a checked local parameter update and explicitly states what learned projections cannot infer from identical inputs.

```presentation
{"title":"Vizuara · Lecture 6 · Original presentation","url":"https://lecture6-attention-mechanisms.vercel.app/#/1"}
```
