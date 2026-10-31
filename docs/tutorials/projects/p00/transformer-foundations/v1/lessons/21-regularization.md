## Lesson 21 — regularization and generalization controls

**Goal:** explain what dropout, weight decay, label smoothing and early stopping change, calculate their effects, and apply them at explicit locations in a Transformer. Prerequisites: the loss, optimizer, residual and normalization lessons. A regularizer changes a training objective, computation or selection rule; it is not a substitute for a valid held-out split.

### Training loss and generalization answer different questions

For model parameters $\theta$ and a record $z$, write its loss as $\ell(\theta;z)$. Training minimizes an empirical average over the training records. The quantity we ultimately care about is expected loss on future records from the declared task distribution:

$$
\begin{aligned}
\widehat R_{\mathrm{train}}(\theta)&=\frac{1}{N}\sum_{i=1}^{N}\ell(\theta;z_i),\\
R(\theta)&=\mathbb E_{z\sim\mathcal D}[\ell(\theta;z)].
\end{aligned}
$$

We cannot calculate $R$ exactly from a small dataset. A separate validation set estimates behavior and supports choices such as dropout rate or stopping update. Repeatedly selecting settings using that set also adapts the experiment to it; a final untouched test set has a different role. A low training loss and high validation loss can indicate overfitting, but also a distribution mismatch, label problem or evaluation bug. Inspect those before increasing regularization blindly.

### Derive inverted dropout rather than memorizing its scaling

Let $a$ be one fixed activation and let $m\sim\mathrm{Bernoulli}(1-p)$, where $p$ is the probability of dropping it. For $0\leq p<1$, inverted dropout returns $\widetilde a=ma/(1-p)$ during training and returns $a$ during evaluation. The scaling preserves this activation's expectation:

$$
\begin{aligned}
\mathbb E[\widetilde a]&=\frac{a}{1-p}\mathbb E[m]=a,\\
\mathbb E[\widetilde a^2]&=\frac{a^2}{(1-p)^2}(1-p),\\
\operatorname{Var}(\widetilde a)&=a^2\frac{p}{1-p}.
\end{aligned}
$$

With $a=4$ and $p=0.5$, the output is 0 or 8 with equal probability. Its mean is 4 and variance is 16. With $p=0.1$, a retained activation becomes $4/0.9\approx4.4444$. These are random training computations, not a rule to multiply the evaluation output by 0.9. The endpoint $p=1$ must be handled separately; the formula has a zero denominator.

For a fixed sampled mask, the local derivative is also $m/(1-p)$. A dropped branch receives zero gradient through that operation; another residual path can still carry gradient. Expectation preservation at one operation does **not** imply $\mathbb E[f(\widetilde a)]=f(a)$ for a nonlinear network. For example, apply the square: the noisy activation has expected square 32, whereas the mean activation squared is 16. [PyTorch Dropout contract](source:dropout).

### Placement changes the computation you regularize

For a post-norm residual block, sublayer dropout gives

$$Y=\operatorname{LayerNorm}(X+\operatorname{Dropout}(F(X))).$$

Here $X,F(X),Y$ all have shape $[B,T,D]$. The identity branch remains present when elements of $F(X)$ are dropped. Dropping the whole sum instead gives different random behavior. Applying dropout to the initial embedding-plus-position tensor perturbs both contributions before the stack. The original paper uses those residual and input locations with base dropout 0.1. [Transformer regularization](source:transformer-paper).

An attention implementation may also drop entries of the probability tensor $A\in\mathbb R^{B\times H\times T_q\times T_k}$ before computing $AV$. Under inverted element dropout, a sampled row need not sum to one. Its expectation still equals the original row, conditional on that row. Do not silently renormalize it: renormalization creates a different operator. Keep the mask convention from Lesson 16, and distinguish random training dropout from permanently forbidden attention connections. Any additional FFN-internal dropout is another explicit architecture choice.

### Label smoothing changes the target distribution

For $V$ eligible classes, true class $y$ and smoothing $\varepsilon$, define the uniform-mixture convention

$$q_j=(1-\varepsilon)\mathbf 1[j=y]+\frac{\varepsilon}{V}.$$

This convention, used by PyTorch's label-smoothing option, includes the true class in the uniform part. For $V=4$, $\varepsilon=0.2$ and $y=2$, it gives $q=[0.05,0.05,0.85,0.05]$. Another convention puts the entire smoothing mass only on the incorrect classes; it would give $[0.0667,0.0667,0.8,0.0667]$. State which convention you use.

The loss and logit derivative become

$$L=-\sum_jq_j\log p_j,\qquad \frac{\partial L}{\partial z_j}=p_j-q_j.$$

For prediction $p=[0.1,0.1,0.7,0.1]$, ordinary one-hot NLL is $-\log0.7\approx0.356675$. The smoothed loss is $-0.85\log0.7-0.15\log0.1\approx0.648562$. Its derivative is $[0.05,0.05,-0.15,0.05]$. A prediction matching $q$ has zero logit gradient, so the training optimum does not request probability one for the recorded label. Report ordinary held-out NLL separately if you want a comparable perplexity measure. The original paper specifies smoothing 0.1; its main text does not fully specify every padding/vocabulary implementation detail. [Loss API](source:cross-entropy), [paper](source:transformer-paper).

Ignoring a PAD target removes that **position** from loss. It does not automatically remove the PAD output column from the smoothing distribution on other positions. If you exclude PAD, construct and normalize $q$ over the declared eligible classes, then keep that policy consistent with output probabilities and evaluation.

### Decay and stopping are different controls

An L2 objective adds $\lambda\lVert\theta\rVert^2/2$, contributing gradient $\lambda\theta$. Decoupled decay instead shrinks selected parameters by $1-\eta\lambda$ alongside the optimizer update. With $\theta=2$, $\eta=0.1$, $\lambda=0.01$ and zero data gradient, decoupled decay gives 1.998. With adaptive optimization, injecting $\lambda\theta$ into the moments is generally different from that shrinkage. Record parameter groups: excluding biases or normalization scales is a deliberate choice, not an automatic consequence of the formula. Lesson 9 derives this difference.

Early stopping selects a checkpoint using a declared validation metric and patience. If validation losses at updates 100, 200, 300 and 400 are $[0.8,0.5,0.6,0.7]$, patience two without ties selects update 200 and stops after two non-improving evaluations. Retain the **best** state, not merely the final weights. The existing small trainer uses a fixed total-update goal; it records best loss but does not implement automatic early stopping. Do not attribute a stopping rule to it.

### Check modes and the smoothing arithmetic

Run this complete probe from the course `examples/` directory with the frozen PyTorch interpreter. It adds no files to your implementation child.

```python
import math
import torch
from torch import nn

a, p = 4.0, 0.5
outcomes = torch.tensor([0.0, a / (1.0 - p)])
assert float(outcomes.mean()) == a
assert float(outcomes.var(unbiased=False)) == 16.0
q = torch.tensor([0.05, 0.05, 0.85, 0.05], dtype=torch.float64)
prob = torch.tensor([0.1, 0.1, 0.7, 0.1], dtype=torch.float64)
print('smoothed loss:', float(-(q * prob.log()).sum()))
assert math.isclose(float(-(q * prob.log()).sum()), 0.64856146624, abs_tol=1e-10)

layer = nn.Dropout(0.5)
x = torch.ones(32, requires_grad=True)
torch.manual_seed(7)
layer.train()
with torch.no_grad():
    noisy = layer(x)
assert not noisy.requires_grad
assert set(noisy.tolist()) <= {0.0, 2.0}
layer.eval()
identity = layer(x)
assert torch.equal(identity, x)
assert identity.requires_grad
print('evaluation identity and gradient-mode checks passed')
```

`eval()` changes module behavior and propagates to registered children; it does not turn off gradient recording. `no_grad()` disables recording in its context; it does not change the module's training flag. Evaluation usually needs both. A functional dropout call must receive `training=self.training` if it is to follow the enclosing module. [Gradient mode](source:torch-nograd).

### Exercise and notes

[[practice:regularization-transfer]]

In [my study notes](project/study-notes.md), derive dropout variance for $a=-3,p=0.2$, then explain which residual path remains when the sublayer is dropped. Write both smoothing conventions for a three-class target, and predict whether a `no_grad()` call alone makes a training-mode Transformer deterministic. Connect each proposed regularizer to the validation evidence that would justify it.

Continue with [encoder–decoder learning and inference](lessons/22-encoder-decoder.md), where these controls must operate on precisely shifted sequence streams.
