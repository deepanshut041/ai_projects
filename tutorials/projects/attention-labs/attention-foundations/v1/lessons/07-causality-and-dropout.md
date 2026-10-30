## 7. Make the training information match generation

A model predicting the next token must operate with the prefix available at that moment. During training we can process many positions together, but an output at an earlier position must not acquire information from a later input token. A causal mask imposes that restriction without changing the learned projections or the score function.

### Locate the target before drawing the triangle

For tokens `[Quiet,cats,nap,<EOS>]`, one shifted training pair is

| Position | Decoder input token | Prediction target | Allowed input positions |
| --- | --- | --- | --- |
| 1 | Quiet | cats | 1 |
| 2 | cats | nap | 1, 2 |
| 3 | nap | `<EOS>` | 1, 2, 3 |

The diagonal is allowed because the current **input** token is known; its target is the next token. At row 1, allowing input position 2 would reveal “cats”, the answer being predicted. A correct triangular mask does not repair incorrectly aligned inputs and labels, and a mask at only one layer does not prevent leakage through another path that has already mixed future information.

### Apply visibility before normalization

With row $i$ as query and column $j$ as source, define

$$M_{ij}=\begin{cases}0&j\leq i,\\-\infty&j>i.\end{cases}\qquad
A=\operatorname{softmax}\left(QK^\top/\sqrt{d_k}+M\right).$$

Exponentials of forbidden entries are zero. Every row must retain at least one finite allowed score. If all entries are masked, subtracting the maximum is not defined in the usual way; do not silently present the result as a valid distribution. Our reference raises an error. A larger application must explicitly choose how to handle padded query positions and exclude them from the objective.

![Causal and padding permissions on the key axis](images/07-causal-mask.svg)

Using Lesson 6's projections gives

$$A\approx\begin{bmatrix}1&0&0\\0.5&0.5&0\\0.283995&0.140029&0.575975\end{bmatrix},\quad
Z\approx\begin{bmatrix}2&0\\1&-0.5\\1.719942&-0.716005\end{bmatrix}.$$

The second query has equal scores for its two allowed keys, so its weights are one half each. The last query can access the complete prefix, which here is the entire sequence, and therefore retains the same output as the unmasked last row.

### Why zeroing after softmax needs care

Simply zeroing forbidden **weights** after ordinary softmax leaves the remaining weights with a sum below one. Renormalizing the remaining weights does recover the masked distribution in exact arithmetic:

$$\frac{e^{s_j}/D}{\sum_{r\in\mathcal A}e^{s_r}/D}
=\frac{e^{s_j}}{\sum_{r\in\mathcal A}e^{s_r}},\qquad j\in\mathcal A.$$

So it is incorrect to say that this renormalized version necessarily leaks future information mathematically. It is nevertheless less convenient and can underflow if a huge forbidden score makes every allowed probability numerically zero before renormalization. Masking before softmax directly normalizes over the permitted set.

The dense matrix product may still compute forbidden raw dot products. Causal masking is a rule about usable information and outputs; it is not a guarantee that a dense kernel never calculates a future pair internally.

### Combine a causal mask with padding validity

A source key is usable only if both conditions hold: it is in the allowed prefix, and it represents a real input token. The reference helper combines them as `valid_key[j] and j <= i`. Replacing a padded embedding with zeros does not impose this rule: a zero key can still receive a nonzero softmax weight and change the denominator.

Masking padded keys also does not automatically eliminate padded **query** outputs or their losses. An application normally tracks query validity separately and excludes those rows from the loss, or explicitly zeroes their outputs according to its contract. This tutorial's examples use three real queries and reject an empty attention support.

### Dropout changes a sample, not the intended mean

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

[[practice:causal-check]]

### The original lecture and slides

[Vizuara Lecture 7](https://www.youtube.com/watch?v=Dv8e0ayAikA) supplies the masking and dropout sequence. Original slides 10–14 discuss the triangle and normalization; slides 16–19 discuss dropout and the causal pipeline. Our explanation distinguishes exact masking equivalence, numerical failure, padding validity and shifted labels.

```presentation
{"title":"Vizuara · Lecture 7 · Original presentation","url":"https://lecture7-attention-mechanisms.vercel.app/#/1"}
```

You now have a checked single-head mechanism. The separate Attention Heads route begins with several independently parameterized copies and then proves their packed implementation equivalent. The [reference-run record](project/reference-runs.md) records author checks; the [readiness record](project/readiness.md) separates them from your own explanation and any full-model work.
