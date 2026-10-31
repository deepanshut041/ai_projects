## Lesson 4 — Probability, statistics and information

**Goal:** interpret probabilities, expected values and log likelihood before using softmax or a loss. Prerequisites: sums/dot products and elementary derivatives. Probabilities represent uncertainty under a model; they are not proof that an answer is true.

### Distributions and expectations

For a categorical variable C with V possible outcomes, pᵢ=P(C=i) must be nonnegative and sum to one. A vector of raw scores such as `[0,log(3)]` is not a probability distribution. A model needs a transformation that enforces those conditions.

The expectation of a function is $\mathbb E[f(C)]=\sum_i p_i f(i)$. If values are `[2,6]` with probabilities `[.25,.75]`, their expected value is 5. That weighted sum is the arithmetic attention uses to combine vectors. A convex mixture with nonnegative normalized weights stays within the values' convex hull; subsequent learned projections need not.

For a numeric variable, variance is $\mathbb E[(X-\mu)^2]=\mathbb E[X^2]-\mu^2$, with μ=E[X]. Mean and variance describe location and spread, while norm measures distance from zero. Values `[1,3]` have population mean 2 and variance 1. Values `[101,103]` have mean 102 and the same variance. They have different norms. An unbiased sample-variance estimator divides its centered sum by N−1; feature normalization uses population variance, dividing by N.

When X and Y are independent, Var(X+Y)=Var(X)+Var(Y); otherwise add 2Cov(X,Y). Initialization arguments often assume independent, centered coordinates. Learned activations can violate those assumptions; the derivation is a guide to initial scale, not a theorem about every training step.

### Conditional probability and sequence likelihood

Conditional probability P(A|B)=P(A∩B)/P(B) is defined when P(B)>0. The chain rule of probability yields

$$P(y_1,y_2,y_3\mid x)=P(y_1\mid x)P(y_2\mid y_1,x)P(y_3\mid y_1,y_2,x).$$

This factorization does **not** assume the output tokens are independent. It explicitly conditions each factor on the prefix. An autoregressive decoder models these factors. Independent records are a separate training assumption used when multiplying their likelihoods.

For correct-token probabilities `.5,.25,.8`, the sequence probability is `.1`. Natural-log likelihood is `log(.5)+log(.25)+log(.8)=log(.1)`. Negative log likelihood is log(10)≈2.302585. Logs convert products to sums, avoid multiplying many tiny values, and preserve the ordering of positive likelihoods. A token mean NLL would be about .767528; it is a different reduction from the sequence total.

### Entropy, cross-entropy and KL

Let q be a target distribution and p the model distribution. Using natural logs gives units of nats:

$$\begin{aligned}
H(q)&=-\sum_iq_i\log q_i,\\
H(q,p)&=-\sum_iq_i\log p_i,\\
D_{KL}(q\Vert p)&=\sum_iq_i\log\frac{q_i}{p_i},\\
H(q,p)&=H(q)+D_{KL}(q\Vert p).
\end{aligned}$$

Use the limit `0 log 0=0`. If q assigns positive mass to an outcome for which p=0, cross-entropy and KL are infinite. KL is generally asymmetric, so it is not a distance metric. With fixed q, minimizing cross-entropy also minimizes this KL because H(q) does not depend on model parameters.

For q=`[.25,.75]`, entropy is about .562335. Against p=`[.5,.5]`, cross-entropy is log2≈.693147 and KL≈.130812. A one-hot target `[0,1]` has zero entropy, so its cross-entropy is simply −log p₁. This explains the connection between a categorical loss and likelihood. It does not mean every observed label perfectly describes the unknown population distribution.

### Softmax and its limits

Softmax maps real logits z to pᵢ=exp(zᵢ)/Σⱼexp(zⱼ). With `[0,log3]`, p is `[.25,.75]`. Adding the same constant to every logit changes neither p nor the categorical loss. Softmax is invariant to a shared shift, not to scaling; division by a positive temperature changes sharpness.

```python
import math
q = [.25, .75]
p = [.5, .5]
entropy = -sum(v*math.log(v) for v in q)
cross = -sum(v*math.log(w) for v, w in zip(q, p))
kl = sum(v*math.log(v/w) for v, w in zip(q, p))
assert abs(cross-entropy-kl) < 1e-12
assert abs(math.prod([.5,.25,.8])-.1) < 1e-12
```

Uniform predictions over V tokens have hard-target NLL log V and perplexity V. Perplexity is exp(mean token NLL) and depends on tokenization, conditioning and the evaluation corpus; comparing differently tokenized models directly can mislead.

[[practice:probability-transfer]]

### Notes and checks

Write the sequence factorization and state where an independence assumption is used. Compute an expectation, a variance and all three information quantities for the same distribution. Explain why an attention row and a vocabulary distribution can both be probabilities yet index different things.

Written reference: [D2L probability](source:d2l-probability). [prerequisite_math.py](examples/prerequisite_math.py) supplies stable arithmetic checks. Continue with [feedforward networks](lessons/08-model.md).
