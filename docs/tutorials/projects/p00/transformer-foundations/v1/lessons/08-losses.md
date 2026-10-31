## Lesson 8 — Loss functions, logits and stable arithmetic

**Goal:** select the right objective and derive its learning signal, including stable evaluation and reduction. Prerequisites: probability, derivatives and backpropagation. A loss says what is rewarded; an optimizer decides how to update parameters using its gradients.

### Regression and binary classification

For a real-valued target y, squared error `(a−y)²` penalizes the size of the numeric error. Its derivative is 2(a−y). MSE averages those errors over the declared elements. Under a fixed-variance Gaussian observation model, negative log likelihood is proportional to squared error plus constants; this is a modeling assumption, not a reason to use MSE for every target.

For a binary label y∈{0,1}, let p=σ(z). Binary cross-entropy is

$$L=-y\log p-(1-y)\log(1-p).$$

Differentiate with respect to p and multiply by dp/dz=p(1−p), obtaining ∂L/∂z=p−y. At y=1,p=.01, the gradient is −.99. By comparison, half squared error on p has logit derivative `(p−y)p(1−p)=−.009801`. This distinction explains why a confident wrong sigmoid prediction can receive a stronger output signal under BCE. It does not remove hidden-layer saturation.

A stable logit formula is `max(z,0)−zy+log1p(exp(−abs(z)))`. It avoids explicitly taking log(0) after a rounded sigmoid. `BCEWithLogitsLoss` expects floating logits and floating targets of compatible shape. Multilabel classification uses independent binary outputs; it is a different contract from choosing exactly one categorical class.

### Categorical cross-entropy from likelihood

For V mutually exclusive classes and hard target t, let p=softmax(z). Then

$$L=-\log p_t=\log\sum_j e^{z_j}-z_t.$$

Differentiating log-sum-exp gives softmax, so

$$\frac{\partial L}{\partial z_j}=p_j-\mathbf1[j=t].$$

The gradients sum to zero, reflecting invariance to a shared logit shift. The loss does not choose or modify the target; the data supplies it. For `[0,log3]`, target1 gives L≈.287682 and derivative `[.25,−.25]`; target0 gives L≈1.386294 and derivative `[−.75,.75]`.

For a normalized soft target q, cross-entropy is −Σqⱼlog pⱼ and its gradient is p−q. A target that does not sum to one changes the derivation. Label smoothing deliberately replaces one-hot q; its convention is explicit in Lesson 21.

### Stabilize log-sum-exp before exponentiating

Let m=maxⱼzⱼ. Then

$$\log\sum_j e^{z_j}=m+\log\sum_j e^{z_j-m}.$$

Subtracting m keeps exponentials at most 1. With logits `[1000,999]`, the target0 NLL is `log(1+exp(−1))≈.313262`, not infinity. Calculating softmax and then logging a rounded zero still risks underflow; use log-softmax or the fused loss. A common shift cancels analytically even if naive exponentiation overflows.

```python
import torch
from torch.nn import functional as F
z = torch.tensor([[1000.,999.]], dtype=torch.float64, requires_grad=True)
t = torch.tensor([0], dtype=torch.long)
loss = F.cross_entropy(z, t)
assert abs(loss.item()-.31326168751822286) < 1e-12
loss.backward()
assert abs(z.grad.sum().item()) < 1e-12
```

### Shape and denominator are part of the objective

`CrossEntropyLoss` receives raw logits and class IDs. For token logits `[B,T,V]`, flatten to `[B*T,V]` and labels `[B*T]`, or move V to the class axis for the unflattened API. Calling the loss on `[B,T,V]` without rearranging can treat T as the class dimension. Applying softmax first changes the optimization problem by passing probabilities as if they were logits.

For unweighted hard labels and ignored sentinel −100, sum only valid losses and divide by the valid count. If one batch contains 3 valid tokens and another 7, averaging their two means gives equal weight to batches, not to tokens. A weighted hard-label loss uses a different denominator in the native mean: the sum of selected target weights. This course's hand denominator examples are unweighted.

When no valid labels exist, skip that objective/update or reject the batch explicitly; dividing zero by zero is not useful learning. Ignoring the loss at a padded target does not hide padded keys from attention. Conversely, an attention mask does not remove labels from the loss.

[[practice:loss-transfer]]

### Notes and runnable checks

Derive p−one_hot from log-sum-exp, compute both target changes and identify the exact denominator for a token batch. Compare BCE and categorical target contracts. The stable references are [prerequisite_tensors.py](examples/prerequisite_tensors.py) and [deep_learning_math.py](examples/deep_learning_math.py), with extreme-value and gradient checks.

Sources: [CrossEntropyLoss](source:cross-entropy), [BCEWithLogitsLoss](source:torch-bce), [Nielsen's output-loss discussion](source:nielsen-cross-entropy). Continue with [optimizers](lessons/09-optimizers.md).
