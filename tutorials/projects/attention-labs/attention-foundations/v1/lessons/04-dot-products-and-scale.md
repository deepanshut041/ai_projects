## 4. Change the score while keeping retrieval intact

We can preserve the source-position normalization and value mixture while replacing the additive network with a different compatibility function. A dot product is a particularly convenient choice when queries and keys have the same width: it scores a pair with coordinate multiplications and a sum, and many pairs fit naturally into a matrix product.

### Direction and magnitude both affect a dot product

For rows $q,k\in\mathbb R^{d_k}$,

$$q\cdot k=\sum_{r=1}^{d_k}q_rk_r=\lVert q\rVert\lVert k\rVert\cos\theta.$$

The coordinate formula defines the computation. The geometric formula explains why a larger magnitude can increase a score even without changing direction. Query `[1,0]` scores key `[2,0]` as 2 and key `[1,0]` as 1. Both keys point in the same direction. A dot-product ranking is therefore not a cosine-similarity ranking unless magnitudes are controlled.

Using query `[1,1]` with the three rows retained from Lesson 3 gives scores `[1,1,2]`. Plain softmax returns approximately `[0.211942,0.211942,0.576117]`. The third source receives most weight. This follows from our chosen coordinates; it is not evidence that a model has learned which English word matters.

Luong and colleagues studied several alignment score forms, including dot, general/bilinear, and concat forms. Dot-product scoring is a useful case from that work, not its entire contribution. A bilinear score $qMk^\top$ can also bridge unequal query/key widths. See the [original paper](https://aclanthology.org/D15-1166/).

### Why divide by a square root?

Assume, for this calculation, independent query and key coordinates with mean zero and variance one, with the coordinate-product terms mutually independent. For $r_i=q_ik_i$,

$$\mathbb E[r_i]=0,\qquad \operatorname{Var}(r_i)=\mathbb E[q_i^2]\mathbb E[k_i^2]=1.$$

There are $d_k$ such terms in the dot product, so

$$\operatorname{Var}(q\cdot k)=d_k,\qquad
\operatorname{Var}\left(\frac{q\cdot k}{\sqrt{d_k}}\right)=1.$$

This explains the conventional scaling factor. Actual learned coordinates can have different variances or correlations, so the division does not guarantee unit variance or a soft distribution in every layer. It is a scale-control choice motivated by those assumptions, not a theorem about all trained activations.

Softmax depends on **differences** between scores, since $a_i/a_j=\exp(e_i-e_j)$. Scores $[1000,1001]$ therefore have the same weights as $[0,1]$: approximately $[0.268941,0.731059]$. A large common offset alone does not make attention nearly one-hot. In contrast, increasing the difference from 1 to 20 does. Scaling divides those differences as well as the individual scores.

This distinction matters when experimenting with random vectors. Independent coordinates drawn uniformly from $[0,1]$ are not mean-zero, unit-variance coordinates: their products have mean $1/4$, so the expected raw dot product is $d_k/4$. That growing common mean is not itself the source of softmax concentration. Variation in score differences is what affects the normalized weights. State the sampling distribution when interpreting a dimension experiment, and compare many draws rather than treating one random sample as a universal trend.

For our width-two query, divide scores `[1,1,2]` by $\sqrt2$ before softmax. The weights become `[0.248255,0.248255,0.503490]`, less concentrated than the unscaled result. The key width is the scaling dimension; it is not the number of source positions or the value width.

![The same ranking with different concentration](images/04-scale.svg)

### Softmax derivatives explain the concern more precisely

Let $a_i=\operatorname{softmax}(e)_i$. Differentiating gives

$$\frac{\partial a_i}{\partial e_j}=a_i(\delta_{ij}-a_j).$$

Here $\delta_{ij}$ is 1 when the indices match and 0 otherwise. For a two-score vector `[0,2]`, the larger probability is about 0.880797, and its own derivative is about 0.104994. For `[0,20]`, that derivative is about $2.061\times10^{-9}$. Very concentrated softmax weights can make changes in the weights insensitive to small score changes.

This does **not** imply that every gradient in a model vanishes or that learning necessarily stops. For example, softmax followed by cross-entropy has logit gradient $p-y$; a confident wrong prediction can have a large gradient. The effect depends on the surrounding computation and objective.

### Temperature is a related operation, not the same setting

Writing $a=\operatorname{softmax}(e/\tau)$ with $\tau>0$, increasing $\tau$ makes score differences smaller and generally spreads probability more evenly. Decreasing it sharpens the distribution. Attention's $1/\sqrt{d_k}$ scales **query–key scores**; a generation temperature typically scales **vocabulary logits**. The algebra is related, but these are different distributions at different places in the model.

<!-- example:04_scoring.py:start -->

Complete file: [04_scoring.py](examples/04_scoring.py).

```python
import json
import math
from row_math import X, dot, rounded, softmax


if __name__ == '__main__':
    query = [1.0, 1.0]
    scores = [dot(query, key) for key in X]
    ordinary = softmax(scores)
    scaled = softmax([score / math.sqrt(2) for score in scores])
    sharp = softmax([0.0, 20.0])[-1]
    mild = softmax([0.0, 2.0])[-1]
    assert dot([1.0, 0.0], [2.0, 0.0]) > dot([1.0, 0.0], [1.0, 0.0])
    assert abs(sum(scaled) - 1.0) < 1e-14
    print(json.dumps({'scores': scores, 'unscaled': rounded(ordinary),
                      'scaled': rounded(scaled),
                      'mild_diagonal_derivative': round(mild * (1 - mild), 9),
                      'sharp_diagonal_derivative': round(sharp * (1 - sharp), 12)}))
```

<!-- example:04_scoring.py:end -->

Run `python3 examples/04_scoring.py`. It reproduces both weight rows and the two derivatives, and checks the magnitude counterexample. No parameters or embeddings are updated.

[[practice:scale-check]]

### The original lecture and slides

[Vizuara Lecture 4](https://www.youtube.com/watch?v=DDxMVJnjgps) compares the classical score forms. Its slides 8–13 motivate dot products, dimension-related scale and softmax concentration. Our derivation states the independence assumptions and separates a small softmax Jacobian from an unconditional claim that training stops.

```presentation
{"title":"Vizuara · Lecture 4 · Original presentation","url":"https://lecture4-attention-mechanisms.vercel.app/#/1"}
```
