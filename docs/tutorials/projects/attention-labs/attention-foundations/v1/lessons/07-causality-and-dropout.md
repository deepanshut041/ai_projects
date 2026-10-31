## 7. Causal attention: hide the future, then apply dropout

A model predicting the next token must operate with the prefix available at that moment. During training we can process many positions together, but an output at an earlier position must not acquire information from a later input token. A causal mask imposes that restriction without changing the learned projections or the score function.

![The learned projections and value mixture from the previous lesson](images/original-slides/lecture-07-slide-02.png)

The projections, score and mixture keep their jobs. In the six-word example there are six queries, six keys, six values and six outputs; schematic subdivisions do not change those counts.

![Connectivity determines which source positions a query is allowed to use](images/original-slides/lecture-07-slide-03.png)

![The staircase adds a visibility rule before multiple heads](images/original-slides/lecture-07-slide-05.png)

We will follow three questions: why generation needs restricted visibility, where to impose that restriction, and how training-time dropout changes the resulting weights.

### Locate the target before drawing the triangle

![Only Dream big and is available when predicting work](images/original-slides/lecture-07-slide-08.png)

During training, the complete sentence **Dream big and work for it** may be stored in one tensor. During generation, a model given “Dream big and” has not produced work, for or it yet. Its context for the next prediction must therefore mix only the values of the available prefix. Having the training labels in memory does not authorize using them as input features.

![A causal attention matrix permits the diagonal and earlier source positions](images/original-slides/lecture-07-slide-09.png)


Use a row as the position making a prediction and a column as a possible source. The visible prefix grows one position at a time; entries strictly to the right are forbidden. The diagonal is allowed because that row predicts the following token from the current token and its prefix, as the shifted targets below make explicit.

For the six-word sentence followed by an end marker, the shifted prediction contract is

| Position | Input token | Prediction target | Allowed input positions |
| --- | --- | --- | --- |
| 1 | Dream | big | 1 |
| 2 | big | and | 1, 2 |
| 3 | and | work | 1, 2, 3 |
| 4 | work | for | 1, 2, 3, 4 |
| 5 | for | it | 1, 2, 3, 4, 5 |
| 6 | it | `<EOS>` | 1, 2, 3, 4, 5, 6 |

The diagonal is allowed because the current **input** token is known; its target is the next token. At row 1, allowing input position 2 would reveal “big”, the answer being predicted. A correct triangular mask does not repair incorrectly aligned inputs and labels, and a mask at only one layer does not prevent leakage through another path that has already mixed future information.

![Without a causal rule, every row can draw from all six source positions](images/original-slides/lecture-07-lab-unmasked.png)

![With the rule enabled, the upper triangle is zero and the last row retains full visibility](images/original-slides/lecture-07-lab-masked.png)

Compare the first row, second row and final row. First, only Dream is allowed, so its weight is one and $z_1=v_1$. Second, Dream and big share the mass, approximately 0.48 and 0.52 in this snapshot, so $z_2\approx0.48v_1+0.52v_2$. Third, the last query may see the whole supplied prefix, so its row agrees with the unmasked last row. The values themselves need not be deleted: exactly zero future weights remove their contribution.

### First attempt: zero weights and repair their row sums

![Ordinary softmax, zeroing future weights and renormalizing the retained entries](images/original-slides/lecture-07-slide-11.png)

Simply zeroing forbidden **weights** after ordinary softmax leaves the remaining weights with a sum below one. Renormalizing the remaining weights does recover the masked distribution in exact arithmetic:

$$\frac{e^{s_j}/D}{\sum_{r\in\mathcal A}e^{s_r}/D}
=\frac{e^{s_j}}{\sum_{r\in\mathcal A}e^{s_r}},\qquad j\in\mathcal A.$$

So it is incorrect to say that this renormalized version necessarily leaks future information mathematically. It is nevertheless less convenient and can underflow if a huge forbidden score makes every allowed probability numerically zero before renormalization. Masking before softmax directly normalizes over the permitted set.

The dense matrix product may still compute forbidden raw dot products. Causal masking is a rule about usable information and outputs; it is not a guarantee that a dense kernel never calculates a future pair internally.

### Standard method: apply visibility before normalization

![Forbidden score entries become negative infinity before exponentiation and normalization](images/original-slides/lecture-07-slide-12.png)


The blocked entries contribute zero after exponentiation, so the denominator includes only allowed sources. The order is essential to the direct masked-softmax calculation. A row with no allowed finite score still needs explicit handling; the picture does not define a valid distribution for that case.

For a first-position example, scores `[2,3,5]` become `[2,-inf,-inf]`. Their exponential contributions are $[e^2,0,0]$; the denominator is only $e^2$, so weights are exactly `[1,0,0]`. Replacing forbidden raw scores by **zero** would fail because $e^0=1$.

![The unmasked route uses all three scores; the masked route keeps only the permitted one](images/original-slides/lecture-07-lab-before.png)

![Raising a future score to twelve changes the unmasked bars while the masked bars remain 1, 0, 0](images/original-slides/lecture-07-lab-future.png)

Compare the lower bars across these two states. They stay fixed because both future entries are replaced by minus infinity. The upper bars show the ordinary, not-yet-repaired distribution. After zeroing and renormalizing it, the exact mathematical result would also be `[1,0,0]`; the cancellation above explains why.

In PyTorch, a concrete mask construction is:

```python
import torch

n = 6
future = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
scores = torch.zeros(n, n)  # Replace with your Q @ K.T in a real pass.
weights = torch.softmax(scores.masked_fill(future, -torch.inf), dim=-1)
print(weights)
```

`diagonal=1` starts strictly above the main diagonal. `True` marks positions to overwrite in this `masked_fill` call. This Boolean convention belongs to this operation; other APIs may define their masks differently.

With row $i$ as query and column $j$ as source, define

$$M_{ij}=\begin{cases}0&j\leq i,\\-\infty&j>i.\end{cases}\qquad
A=\operatorname{softmax}\left(QK^\top/\sqrt{d_k}+M\right).$$

Exponentials of forbidden entries are zero. Every row must retain at least one finite allowed score. If all entries are masked, subtracting the maximum is not defined in the usual way; do not silently present the result as a valid distribution. Our reference raises an error. A larger application must explicitly choose how to handle padded query positions and exclude them from the objective.

![Causal and padding permissions on the key axis](images/07-causal-mask.svg)

Using Lesson 6's smaller three-row transfer fixture gives

$$A\approx\begin{bmatrix}1&0&0\\0.5&0.5&0\\0.283995&0.140029&0.575975\end{bmatrix},\quad
Z\approx\begin{bmatrix}2&0\\1&-0.5\\1.719942&-0.716005\end{bmatrix}.$$

The second query has equal scores for its two allowed keys, so its weights are one half each. The last query can access the complete prefix, which here is the entire sequence, and therefore retains the same output as the unmasked last row.

### Combine a causal mask with padding validity

A source key is usable only if both conditions hold: it is in the allowed prefix, and it represents a real input token. The reference helper combines them as `valid_key[j] and j <= i`. Replacing a padded embedding with zeros does not impose this rule: a zero key can still receive a nonzero softmax weight and change the denominator.

Masking padded keys also does not automatically eliminate padded **query** outputs or their losses. An application normally tracks query validity separately and excludes those rows from the loss, or explicitly zeroes their outputs according to its contract. This tutorial's examples use three real queries and reject an empty attention support.

### Dropout changes a sample, not the intended mean

![Causal attention weights are selectively removed and survivors enlarged during training](images/original-slides/lecture-07-slide-15.png)

The causal zeros stay zero. A dropout coin flip acts on each attention entry independently; probability $p=0.5$ does not promise that exactly half of a particular row disappears. An entire row can be dropped. These weights are computed activations, not independent model parameters.

![A dropped contribution is either removed or rescaled when it survives](images/original-slides/lecture-07-slide-16.png)


The two outcomes illustrate the expectation calculation: a contribution sometimes becomes zero, and the survivors are enlarged by the reciprocal keep probability. Each sampled attention row need not sum to one. Our example below enumerates every keep/drop pattern and checks the immediate weighted mixture rather than claiming equality for a whole nonlinear model.

For the illustrated work-to-big weight $0.27$ at $p=0.5$, the two outcomes are zero and $0.54$. Their average is $0.27$. With $p=1/3$, the keep probability is $2/3$ and survivors are multiplied by $3/2$ instead. This preserves an immediate expectation, not the exact value of each sample.

![One sampled dropout matrix can contain zero rows and weights above one](images/original-slides/lecture-07-lab-dropout.png)

A first-row one may become zero or two. That is not a broken causal mask or a request to normalize the sampled row again. At inference, disable dropout and use the original causal weights. To inspect a particular random mask, generate it once and reuse that exact mask for printing and multiplication; making two dropout calls can generate different patterns. A fixed seed controls repeatability of a specified sequence of random calls.

The training motivation is regularization: randomly removing contributions makes the computation operate with varying subsets of its usual information paths. It can reduce reliance on particular contributions, but whether it improves held-out quality depends on the model, data and dropout probability. No generalization improvement is established by the arithmetic example alone.

During training, attention dropout can replace each normalized weight $a_j$ by

$$\widetilde a_j=\frac{m_j}{1-p}a_j,\qquad m_j\sim\operatorname{Bernoulli}(1-p),\quad0\leq p<1.$$

Taking the expectation gives $\mathbb E[\widetilde a_j]=a_j$. This rescales surviving weights; it does not renormalize each sampled row to sum to one. At $p=0.5$, keeping the first and third entries of our final weight row gives approximately `[0.567991,0,1.151951]`, with sum `1.719942`. A sampled weight can exceed one. In evaluation mode dropout is disabled.

The expectation claim is local to the dropped weights or their immediate linear mixture. It does not say that a whole nonlinear model's average training prediction equals its evaluation prediction. The example enumerates all eight independent keep/drop patterns for three weights, proving the immediate expectation without relying on a lucky random sample.

<!-- example:07_causal.py:start -->

Complete file: [07_causal.py](examples/07_causal.py).

```python
import itertools
import json
from attention_math import X, WQ, WK, WV, attention, matmul, rounded, softmax


def run(inputs):
    q, k, v = matmul(inputs, WQ), matmul(inputs, WK), matmul(inputs, WV)
    return attention(q, k, v, causal=True)


if __name__ == '__main__':
    scores, weights, output = run(X)
    changed = [X[0], X[1], [99.0, -37.0]]
    assert run(changed)[2][:2] == output[:2]
    assert all(weights[i][j] == 0 for i in range(3) for j in range(i + 1, 3))
    # Exact expectation over every independent dropout keep/drop choice, p=1/2.
    draws = [[2 * p * keep for p, keep in zip(weights[-1], mask)]
             for mask in itertools.product([0, 1], repeat=3)]
    average = [sum(row[j] for row in draws) / len(draws) for j in range(3)]
    assert max(abs(x - y) for x, y in zip(average, weights[-1])) < 1e-14
    chosen = [2 * p * keep for p, keep in zip(weights[-1], [1, 0, 1])]
    try:
        softmax([1.0, 2.0], [False, False])
    except ValueError:
        all_masked_rejected = True
    else:
        raise AssertionError('An empty attention support must not silently normalize')
    print(json.dumps({'weights': rounded(weights), 'output': rounded(output),
                      'future_invariant': True, 'dropout_sample': rounded(chosen),
                      'dropout_sample_sum': round(sum(chosen), 6),
                      'dropout_expectation': rounded(average),
                      'all_masked_rejected': all_masked_rejected}))
```

<!-- example:07_causal.py:end -->

Run `python3 examples/07_causal.py`. It checks exact zero future weights, changes the final input to `[99,-37]` and verifies that the first two outputs do not change. That intervention is stronger than checking only that a printed matrix looks triangular. It also tests the empty-support error and dropout expectation.

### Put the whole causal pipeline into one class

![The full path projects, scores, masks, scales, normalizes, drops training connections and mixes values](images/original-slides/lecture-07-slide-17.png)

Follow the value branch all the way to the final multiplication. Normalized weights are not yet context vectors. Dropout, when active, occurs after softmax and before multiplying by $V$. Scaling before the minus-infinity overwrite or afterward gives the same result for a positive finite scale; the forbidden entries remain minus infinity either way.

The class below stores a strictly upper-triangular Boolean mask with `register_buffer`. It travels with the model's device and checkpoint state, but an optimizer does not train it. Slice it to the actual sequence length before applying it. This class handles aligned full-sequence self-attention; cached generation with separate query/key lengths needs a specified position-offset mask.

Complete file: [causal_attention.py](examples/causal_attention.py). It is self-contained PyTorch and can be pasted into Colab or uploaded and run with `!python causal_attention.py`. It declares its input vectors and random initialization explicitly; do not infer exact coefficients from rounded figure entries.

```python
"""Causal weights, mask-order equivalence, dropout and prefix invariance."""
import math
import torch
from torch import nn


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout=0.5):
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError('Drop probability must be in [0, 1)')
        self.query = nn.Linear(d_in, d_out, bias=False, dtype=torch.float64)
        self.key = nn.Linear(d_in, d_out, bias=False, dtype=torch.float64)
        self.value = nn.Linear(d_in, d_out, bias=False, dtype=torch.float64)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('future', torch.triu(
            torch.ones(context_len, context_len, dtype=torch.bool), diagonal=1))

    def forward(self, x):
        length = x.shape[-2]
        if not 0 < length <= self.future.shape[0]:
            raise ValueError('Sequence length is outside the prepared context')
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
        scores = scores.masked_fill(self.future[:length, :length], -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        dropped = self.dropout(weights)
        return dropped @ v, weights, dropped


if __name__ == '__main__':
    torch.manual_seed(789)
    x = torch.tensor([[.72, .45, .31], [.75, .20, .55], [.30, .80, .40],
                      [.85, .35, .60], [.55, .15, .75], [.25, .20, .85]],
                     dtype=torch.float64)
    layer = CausalAttention(3, 2, 6, dropout=.5)
    layer.eval()
    output, weights, dropped = layer(x)
    torch.testing.assert_close(weights.sum(-1), torch.ones(6, dtype=x.dtype))
    assert torch.equal(weights.triu(1), torch.zeros_like(weights))
    changed = x.clone()
    changed[3:] = torch.tensor([99., -37., 12.], dtype=x.dtype)
    torch.testing.assert_close(layer(changed)[0][:3], output[:3], atol=1e-14, rtol=0)
    torch.testing.assert_close(layer(x[:3])[0], output[:3], atol=1e-14, rtol=0)
    q, k = layer.query(x), layer.key(x)
    scores = q @ k.T / math.sqrt(2)
    full = torch.softmax(scores, dim=-1)
    zeroed = full.masked_fill(layer.future, 0.0)
    repaired = zeroed / zeroed.sum(-1, keepdim=True)
    torch.testing.assert_close(repaired, weights, atol=1e-14, rtol=0)
    torch.testing.assert_close(full[-1], weights[-1], atol=1e-14, rtol=0)
    extreme = torch.tensor([0., 1000.], dtype=x.dtype)
    lost = torch.softmax(extreme, dim=-1)[0]
    assert lost.item() == 0  # Too late to recover this allowed probability.
    safe = torch.softmax(extreme.masked_fill(torch.tensor([False, True]), -torch.inf), dim=-1)
    assert safe.tolist() == [1., 0.]
    print('evaluation weights', weights.tolist())
    layer.train()
    torch.manual_seed(123)
    _, original, after_dropout = layer(x)
    print('dropout weights', after_dropout.tolist())
    assert torch.all((after_dropout == 0) | (after_dropout == 2 * original))
    layer.eval()
    assert torch.equal(layer(x)[1], layer(x)[2])
    print('Prefix interventions, short contexts, both mask orders and mode checks pass.')
```

The executable checks change all future input rows while preserving the earlier prefix outputs. They also compare a short prefix run against the matching part of the full masked run, verify the last row against unmasked attention, and demonstrate why post-softmax repair is fragile when allowed weights have already underflowed. These interventions check information flow more strongly than looking at a triangular picture alone.

`layer.train()` enables dropout; `layer.eval()` disables it. Evaluation mode does not itself disable autograd or prove model quality. The single-head component still needs embeddings/positions, surrounding blocks, a vocabulary head, an objective and training to become a language model.

### The same causal calculation in C++

The following complete program iterates only through allowed source indices `j <= i`, so it never constructs a forbidden-score array. The reduction and returned context are the same as the Python reference. It deliberately fixes this example's shapes rather than claiming to be a general tensor library.

<!-- example:attention.cpp:start -->

Complete file: [attention.cpp](examples/attention.cpp).

```cpp
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
    using Row = std::array<double, 2>;
    const std::array<Row, 3> q{{{1, 0}, {0, 1}, {1, 1}}};
    const std::array<Row, 3> k{{{1, 1}, {0, 1}, {1, 2}}};
    const std::array<Row, 3> v{{{2, 0}, {0, -1}, {2, -1}}};
    std::array<Row, 3> output{};
    for (std::size_t i = 0; i < 3; ++i) {
        std::array<double, 3> scores{};
        for (std::size_t j = 0; j <= i; ++j)
            scores[j] = (q[i][0] * k[j][0] + q[i][1] * k[j][1]) / std::sqrt(2.0);
        const double maximum = *std::max_element(scores.begin(), scores.begin() + i + 1);
        std::array<double, 3> weights{};
        double total = 0;
        for (std::size_t j = 0; j <= i; ++j) {
            weights[j] = std::exp(scores[j] - maximum);
            total += weights[j];
        }
        for (std::size_t j = 0; j <= i; ++j)
            for (std::size_t d = 0; d < 2; ++d)
                output[i][d] += (weights[j] / total) * v[j][d];
    }
    std::cout << std::fixed << std::setprecision(6) << "{\"output\":[";
    for (std::size_t i = 0; i < 3; ++i) {
        if (i) std::cout << ',';
        std::cout << '[' << output[i][0] << ',' << output[i][1] << ']';
    }
    std::cout << "]}\n";
}
```

<!-- example:attention.cpp:end -->

From the package directory:

```bash
c++ -std=c++20 -O2 -Wall -Wextra -Werror -pedantic examples/attention.cpp -o /tmp/attention-foundations
/tmp/attention-foundations
```

The output rows are `[2,0]`, `[1,-0.5]`, and `[1.719942,-0.716005]`. Python and C++ comparisons use a tolerance appropriate to the displayed precision; no claim of bitwise agreement across unrelated implementations is needed.




### Continue with multiple independently learned views

![Several attention heads will retrieve in parallel before their outputs are combined](images/original-slides/lecture-07-slide-21.png)

We now have one complete attention head with an explicit information contract. A head can mix many sources, but it produces one weight pattern per query. Multiple heads add separately learned comparison/content channels, then combine their outputs. They do not guarantee one human-interpretable grammatical job per head.

[[practice:causal-check]]

Continue in the separate **Attention Heads** tutorial for Lecture 8. GQA and other head/KV arrangements belong in that category when added later. Lecture 9 remains excluded.

For consolidation, use the [learning points](project/learning-points.md) and optional [transfer review](practice/transfer-review.md). The [readiness check](project/readiness.md), [implementation notes](project/implementation.md), [reference runs](project/reference-runs.md), and [lecture reference list](project/source-map.md) remain available from the course navigation.

### References

[Vizuara — The Attention Family, Lecture 7](https://www.youtube.com/watch?v=Dv8e0ayAikA) · [Presentation and diagrams](https://lecture7-attention-mechanisms.vercel.app/) · [Interactive masks and dropout lab](https://lecture7-attention-lab.vercel.app/) · [Lecture Colab notebook](https://colab.research.google.com/drive/1im86HwxSOUoF2fo93mpaUqRc7kfQRAJ4?usp=sharing).
