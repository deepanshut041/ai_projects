## Turn comparisons into normalized attention weights

We can now form every query-key score. The next question is why the implementation scales those scores and uses softmax before multiplying by values. We will first keep Q, K and V equal to the input, so no learned projection hides the arithmetic.

### An input can fill all three roles

Take two input rows `X=[[1,0],[0,1]]`. Their dot-product matrix `X@X.T` is the identity: each row has score 1 against itself and 0 against the other. These numbers encode this chosen geometry; they are not a discovered relationship between words.

The implementation divides scores by the square root of query/key feature width. Width is 2, so the first score row becomes `[1/√2,0]`, approximately `[0.707107,0]`.

Softmax converts a row `s` into weights:

$$a_j=\frac{\exp(s_j)}{\sum_{\ell=0}^{S-1}\exp(s_\ell)}.$$

The exponential is positive. Dividing each positive number by the sum makes weights add to one, provided the row contains at least one finite allowed score. The subscript `j` selects a key; the denominator sums over the keys available to this query. Do not normalize across query rows or across independent batch examples.

### Calculate the whole raw-input example

For the first row, `exp(0.707107)≈2.028115` and `exp(0)=1`. The denominator is about `3.028115`. The weights are approximately `[0.669762,0.330238]`. The second row reverses them.

```text
A ≈ [[0.669762, 0.330238],
     [0.330238, 0.669762]]

C = A @ X = A
```

The last equality is a special property of choosing identity-valued inputs. With different values, the same weights produce a different context. Attention weights and retrieved information are related but distinct objects. The existing known-answer test checks both for this example.

![The two matrix products with Q, K and V all equal to X.](images/27-unprojected-scores.png)

### Why use an exponential rather than the raw scores?

Raw scores might be `[-2,0,2]`; they cannot directly serve as nonnegative normalized weights. Softmax yields approximately `[0.015876,0.117310,0.866813]`. Larger relative scores get more weight while smaller scores retain a contribution. Other weighting rules could be designed; softmax is the rule implemented here.

A score difference has a precise consequence: `a_i/a_j = exp(s_i-s_j)`. Adding the same constant to every score leaves all differences and weights unchanged. Multiplying all scores by a positive factor does change their differences and concentration.

For `[0,0]`, weights are `[0.5,0.5]`. For `[0,ln(3)]`, exponentials are `[1,3]`, so weights are `[0.25,0.75]`. For `[0,2ln(3)]`, the second exponential becomes 9 and weights become `[0.1,0.9]`. There is no threshold that turns these weights into a hard selected token.

### Keep normalization numerically stable

Exponentials of very large floating-point scores can overflow. Subtracting the row maximum before exponentiating gives the same weights because the common factor cancels:

$$\frac{e^{s_j-m}}{\sum_\ell e^{s_\ell-m}}=\frac{e^{s_j}}{\sum_\ell e^{s_\ell}}.$$

For `[1000,1000+ln(3)]`, a naive `exp` calculation can fail, but the maximum-shifted calculation still gives `[0.25,0.75]`. The code uses PyTorch's `softmax`, rather than writing an unstable raw exponential division in the model. This does not make NaN inputs meaningful or normalize an entirely blocked row; those need separate handling.

### Why scale by the head width?

A dot product adds `d` coordinate products. Under the explanatory assumption that coordinates have zero mean, unit variance and suitable independence, each product has variance one and the sum has variance `d`. Dividing by `√d` brings that variance to one. These assumptions motivate the rule; they are not a measurement that learned coordinates remain independent.

Large score differences push softmax toward saturation, where its derivative for a dominant probability approaches zero. Scaling reduces one source of dimension-dependent score growth. It is not cosine normalization: Q and K are not divided by their norms. It also does not guarantee stable training on arbitrary data.

![Head width changes the distribution of unscaled dot products.](images/05-head-width-scaling.png)

In the source, the scale is `q.shape[-1] ** -0.5`. After splitting heads this last dimension is `d=D/H`, not the whole model width D and not sequence length S. With `D=12,H=3`, the divisor is `√4=2`. Dividing by `√12` would change the weights even if the output shape still looked correct.

[[practice:softmax-check]]

Next we replace the fixed Q=K=V choice with learned affine projections. That gives the model parameters through which useful comparison and content representations can eventually be learned.
