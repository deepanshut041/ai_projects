## Lesson 6 — Initialization, symmetry and signal scale

**Goal:** explain why the starting parameter scale matters before the optimizer takes a step. Prerequisites: feedforward maps, derivatives, mean and variance. Initialization controls a starting distribution; training is still required.

### Symmetry can stop units from becoming different

If two hidden units start with identical weights and biases and are used identically, they produce the same activation and receive the same gradients. Deterministic updates keep them identical. A layer of many cloned units may therefore act like a much smaller layer. All-zero multilayer weights can also block signals through products of zero matrices. This does not mean every parameter must be random: zero biases and LayerNorm bias are common, and a simple convex linear regression can start at zero. State which symmetry matters.

Random initialization breaks identical-unit symmetry. Its scale also controls whether activations and gradients tend to grow or shrink. “Random” alone is not a scale prescription.

### Derive the fan-in variance argument

For one preactivation $a=\sum_{i=1}^D w_ix_i$, assume independent, centered weights, independent inputs, and weights independent of inputs. Let Var(wᵢ)=σ² and E[xᵢ²]=v. Cross terms vanish under these assumptions:

$$\begin{aligned}
\mathbb E[a]&=0,\\
\operatorname{Var}(a)&=\sum_i\mathbb E[w_i^2]\mathbb E[x_i^2]=D\sigma^2v.
\end{aligned}$$

Choosing σ²≈1/D preserves this second moment through the affine operation. With D=4 and v=1, weight standard deviation .5 gives output variance 1; standard deviation 1 gives variance 4. The number of inputs is **fan-in**. In backward propagation, a transposed matrix sums over the output width, bringing **fan-out** into the analysis.

Xavier/Glorot balances both directions with variance approximately `2/(fan_in+fan_out)` for its basic setting. Its uniform form has bounds ±sqrt(6/(fan_in+fan_out)). Activation-specific gains can change that scale. He initialization for ReLU uses variance approximately `2/fan_in`: for a symmetric preactivation, ReLU keeps half of the second moment.

The ReLU statement concerns E[ReLU(a)²], not its centered variance. ReLU's output mean is positive, so its centered variance is less than its second moment. Mixing those two quantities leads to a wrong derivation. Sigmoid/tanh saturation, correlation between inputs and deep architectural choices all limit these simple estimates.

### The same reasoning explains attention scaling

Let q and k have d independent centered coordinates, each with variance 1, and assume independence between q and k. Their dot product s=Σᵢqᵢkᵢ has mean 0 and variance d. Dividing by sqrt(d) gives variance 1. Dividing by d instead gives variance 1/d, which shrinks as width grows.

Without scale control, large score gaps make softmax nearly one-hot. Its Jacobian contains terms pᵢ(δᵢⱼ−pⱼ), so a saturated distribution often has small derivatives with respect to its scores. Scaling is an initial statistical motivation, not a guarantee: learned Q and K are not always independent unit-variance vectors.

### Check the assumptions with an exhaustive small distribution

Use independent ±1 coordinates. Each product qᵢkᵢ is also a centered ±1 variable. For d=2, the dot product distribution is −2 with probability .25, 0 with probability .5 and 2 with probability .25. Its variance is 2. After dividing by sqrt2, the variance is 1.

```python
import itertools, math
vectors = list(itertools.product([-1.,1.], repeat=2))
scores = [sum(a*b for a,b in zip(q,k)) for q in vectors for k in vectors]
assert sum(scores)/len(scores) == 0
assert sum(s*s for s in scores)/len(scores) == 2
assert abs(sum((s/math.sqrt(2))**2 for s in scores)/len(scores)-1) < 1e-12
```

For PyTorch, `nn.init.xavier_uniform_` and `nn.init.kaiming_normal_` make the fan/gain choices explicit. Know whether the stored weights are `[out,in]`; custom row-oriented matrices may need a transpose when applying a fan-based initializer. A library's default linear initialization is not automatically the Xavier/He choice you intended.

[[practice:initialization-transfer]]

### Notes and diagnostics

Derive the variance calculation and list every independence/centering assumption. Calculate scales for a 128→512 layer. Inspect activation standard deviations and gradient norms across layers instead of assuming an initializer proves healthy signals. Explain both identical-unit symmetry and attention's sqrt(d) denominator.

Sources: [Glorot and Bengio](source:glorot-paper), [He and colleagues](source:he-paper), [PyTorch initialization](source:torch-init). Continue with [backpropagation](lessons/04-autograd.md).
