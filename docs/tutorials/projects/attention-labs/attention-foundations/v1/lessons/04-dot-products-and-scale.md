## 4. Dot products, scale and the softmax bars

We can preserve the source-position normalization and value mixture while replacing the additive network with a different compatibility function. A dot product is a particularly convenient choice when queries and keys have the same width: it scores a pair with coordinate multiplications and a sum, and many pairs fit naturally into a matrix product.

![A decoder retrieves a weighted source mixture while writing chat](images/original-slides/lecture-04-slide-02.png)

The weights and context still have the same jobs as in additive attention. We are replacing the **scorer**, not the source memory or the weighted-sum operation.

![The scoring category is still highlighted](images/original-slides/lecture-04-slide-03.png)

We will do this in the same three passes: understand the dot product, inspect its scale as feature width grows, then divide by a carefully chosen factor before softmax.

![An additive scoring network is replaced by a dot product between two vectors](images/original-slides/lecture-04-slide-07.png)

The left branch projects, adds, applies a nonlinearity and reduces to a scalar. The right branch multiplies matching coordinates and sums them. The dot operation itself introduces no learned parameters. Queries and keys may still come from trainable networks; removing a separate scoring network does not make the whole model parameter-free.

### Direction and magnitude both affect a dot product

![Query and key vectors with aligned, perpendicular and opposite directions](images/original-slides/lecture-04-slide-08.png)


The three panels isolate the sign contributed by direction: aligned vectors can give a positive score, perpendicular vectors give zero, and opposite vectors can give a negative score. Length also matters through the product of the norms. Use the same-direction, different-length counterexample below to avoid reading this picture as a cosine-only comparison.

For rows $q,k\in\mathbb R^{d_k}$,

$$q\cdot k=\sum_{r=1}^{d_k}q_rk_r=\lVert q\rVert\lVert k\rVert\cos\theta.$$

The coordinate formula defines the computation. The geometric formula explains why a larger magnitude can increase a score even without changing direction. Query `[1,0]` scores key `[2,0]` as 2 and key `[1,0]` as 1. Both keys point in the same direction. A dot-product ranking is therefore not a cosine-similarity ranking unless magnitudes are controlled.

Using query `[1,1]` with the three rows retained from Lesson 3 gives scores `[1,1,2]`. Plain softmax returns approximately `[0.211942,0.211942,0.576117]`. The third source receives most weight. This follows from our chosen coordinates; it is not evidence that a model has learned which English word matters.

Luong and colleagues studied several alignment score forms, including dot, general/bilinear, and concat forms. Dot-product scoring is a useful case from that work, not its entire contribution. A bilinear score $qMk^\top$ can also bridge unequal query/key widths. See the [original paper](https://aclanthology.org/D15-1166/).

![The interactive vector plane shows q=(1,2), k=(2,1), and dot product 4](images/original-slides/lecture-04-lab-dot.png)

Use the vector plane to connect the geometry to arithmetic. For the displayed vectors, $q\cdot k=1\times2+2\times1=4$. Rotating one vector while keeping its length fixed changes the cosine term. Increasing its length without rotating also changes the dot product. A positive score is therefore compatibility in the model's learned comparison space, not a universal test of whether two words have the same meaning.

### Why divide by a square root?

![An illustration of typical random dot-product size increasing with key width](images/original-slides/lecture-04-slide-10.png)


The plotted scale is a typical score magnitude, approximately the square root of the width under the stated random-coordinate assumptions. It is not a plot of variance: the variance calculation below gives d before scaling. Learned or correlated coordinates need not follow this sketch.

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

### Reproduce the one-query, two-key experiment

The experiment uses one query and **two keys**, so there are two scores at every feature width. For width one, $Q$ has shape $1\times1$ and $K$ has shape $2\times1$. At width 1,000, their shapes become $1\times1000$ and $2\times1000$, while the score row remains $1\times2$. Do not confuse the number of keys with their width.

At $d_k=1$, dividing by $\sqrt{d_k}$ changes nothing. At width four it divides each score by two. At width 1,000 it divides by about 31.62; at 4,096 it divides by 64. Compare scores and weights before and after that operation, keeping the same query and keys within each comparison.

Complete file: [dimension_probe.py](examples/dimension_probe.py). Place it beside the already displayed `row_math.py`; it uses only the standard library and also runs in Colab with those two files uploaded.

```python
"""One query, two keys: repeat the dimensionality experiment without packages."""
import math
import random
from row_math import dot, softmax


def inspect(width, rng, positive=False):
    draw = rng.random if positive else lambda: rng.gauss(0.0, 1.0)
    query = [draw() for _ in range(width)]
    keys = [[draw() for _ in range(width)] for _ in range(2)]
    scores = [dot(query, key) for key in keys]
    scaled = [score / math.sqrt(width) for score in scores]
    return scores, softmax(scores), scaled, softmax(scaled)


if __name__ == '__main__':
    for positive in [False, True]:
        print('Uniform [0,1]' if positive else 'Normal mean 0, variance 1')
        rng = random.Random(17)
        for width in [1, 4, 64, 1000, 4096]:
            raw, weights, scaled, scaled_weights = inspect(width, rng, positive)
            assert abs(sum(weights) - 1.0) < 1e-12
            assert abs(sum(scaled_weights) - 1.0) < 1e-12
            print(f'd={width:4} Q=(1,{width}) K=(2,{width}) '
                  f'scores={[round(x, 3) for x in raw]} '
                  f'weights={[round(x, 4) for x in weights]} '
                  f'scaled={[round(x, 3) for x in scaled]} '
                  f'new_weights={[round(x, 4) for x in scaled_weights]}')
```

Run `python3 examples/dimension_probe.py` from this tutorial package. The fixed-seed normal example at width 1,000 gives raw scores about `[4.074,12.764]`: the unscaled weights are `[0.0002,0.9998]`, while the scaled weights are `[0.4317,0.5683]`. With positive uniform coordinates at width 4,096, raw scores are about `[1011.228,1028.973]`; scaling changes the weights from effectively `[0,1]` to `[0.4311,0.5689]`.

One sample need not become more concentrated at every successive width. In the normal experiment, the unscaled width-64 row happens to be more concentrated than the width-4,096 row. The variance argument concerns a distribution of draws, not a monotonic law for five individual samples. Scaling controls a typical scale; it does not promise uniform attention or an automatic quality improvement.

### Follow every bar as the scores change

![Four ordinary scores become a soft distribution over the four source positions](images/original-slides/lecture-04-lab-soft.png)

Read the upper bars as scores and the lower bars as weights, keeping their different units in mind. For `[2,4,1,3]`, the normalized row is approximately `[0.0871,0.6439,0.0321,0.2369]`. The second position has the largest score and largest weight, but other positions contribute too.

![At dimension 512, amplified score differences produce nearly one-hot weights](images/original-slides/lecture-04-lab-sharp.png)

The interactive demonstration raises its dimension control to 512 and amplifies the four baseline numbers to about `[22.6,45.3,11.3,33.9]`. The second lower bar now takes almost all the mass. Every lower bar matters: the small ones show how little the other values can contribute to the returned context.

![Enabling scale control restores the lab's baseline scores and soft weight bars](images/original-slides/lecture-04-lab-scaled.png)

With scale control enabled, this demonstration returns the bars to its baseline `[2,4,1,3]` and the soft weight row. This is a controlled visualization of amplification and its removal; it does not redraw random vectors at each width. The executable experiment above applies the literal $1/\sqrt{d_k}$ division to explicitly generated vectors.

### Softmax derivatives explain the concern more precisely

![Multiplying score differences produces a more concentrated softmax distribution](images/original-slides/lecture-04-slide-11.png)


The second panel multiplies the scores, enlarging their differences and concentrating the weights. The left printed weights are schematic: softmax([2,4,1,3]) is the recalculated row above. For the right scores [6,12,3,9], the weights are approximately [0.00236,0.95022,0.00012,0.04731]. A common offset alone would leave the distribution unchanged. To state the gradient consequence precisely, the derivation below concerns the softmax Jacobian, and does not imply that every gradient in a model disappears.

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

### Finish the scoring pass and change the next question

| Scorer | How the inputs meet | Main condition |
| --- | --- | --- |
| Additive | Separate projections are added inside a nonlinear scoring network | Projected widths must match |
| Dot product | Corresponding coordinates multiply, then sum | Query and key widths must match |
| Scaled dot product | The same sum is divided by $\sqrt{d_k}$ | $d_k$ is the comparison width |

![The scoring map now contains additive, dot-product and scaled dot-product attention](images/original-slides/lecture-04-slide-15.png)

These are three classical choices in the scoring category. They do not exhaust attention research, and scaled dot-product attention is not a claim about every layer in every modern hybrid architecture.

![Every token in one sentence receives a new contextual vector from that sentence](images/original-slides/lecture-04-slide-17.png)

The next picture changes who asks, rather than how compatibility is scored. Instead of a decoder looking into an encoder, each position in a sentence will retrieve from its own sentence. The arrows are information paths; the small output dots are new feature coordinates, not predicted vocabulary tokens.

[[practice:scale-check]]


### References

[Vizuara — The Attention Family, Lecture 4](https://www.youtube.com/watch?v=DDxMVJnjgps) · [Presentation and diagrams](https://lecture4-attention-mechanisms.vercel.app/) · [Interactive vector and softmax lab](https://lecture4-attention-lab.vercel.app/).

[Luong et al. — Effective Approaches to Attention-based Neural Machine Translation](https://aclanthology.org/D15-1166/) · [Vaswani et al. — Attention Is All You Need, section 3.2.1](https://arxiv.org/abs/1706.03762).
