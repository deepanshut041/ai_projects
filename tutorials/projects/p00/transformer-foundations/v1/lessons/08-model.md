## Lesson 5 — Feedforward networks and nonlinear activations

**Goal:** understand what a hidden layer adds, how nonlinearities change expressiveness, and how to count a network's parameters. Prerequisites: affine maps, derivatives and probabilities. Backpropagation is developed separately in Lesson 7.

### Why two linear layers are not enough

An affine layer transforms feature coordinates with reusable weights. Compose two:

$$\begin{aligned}
a&=xW_1+b_1,\\
z&=aW_2+b_2\\
 &=x(W_1W_2)+(b_1W_2+b_2).
\end{aligned}$$

This is still a single affine map. More layers may change parameterization or constrain rank, but without nonlinearities they cannot represent a general nonlinear decision boundary. A hidden **activation** breaks that collapse:

$$h=\phi(xW_1+b_1),\qquad z=hW_2+b_2.$$

The hidden coordinates are learned features. They need not have names such as “noun” or “sentiment”; useful interpretation requires evidence. Depth and width are capacity choices, not guarantees of generalization.

A two-class affine classifier's decision is based on the sign of one affine logit difference. It cannot classify opposite-sign quadrants perfectly. The positive-class points `(1,−1),(−1,1)` would require their affine scores to be positive, whose average implies b>0. The negative-class points `(1,1),(−1,−1)` would require negative scores, whose average implies b<0. Contradiction. This is why the small trainer needs a nonlinear hidden layer.

### Activation functions and their derivatives

| Activation | Formula | Local derivative and consequence |
|---|---|---|
| ReLU | max(0,a) | 1 at positive a, 0 at negative a; nondifferentiable at zero; inactive units can receive no local signal. |
| Sigmoid | 1/(1+exp(−a)) | s(1−s); maps to (0,1), saturates at large magnitudes. |
| Tanh | (exp(a)−exp(−a))/(exp(a)+exp(−a)) | 1−tanh²(a); centered outputs, but can also saturate. |
| GELU | a Φ(a) | Φ(a)+a φ(a), using the normal CDF and density; smooth gating, not simply a probability output. |

Use a stable library implementation at extreme inputs. An activation's output range and derivative affect forward values and backward signals. Sigmoid is useful for a binary probability, but repeatedly applying saturating activations through a deep chain can shrink gradients. The original Transformer FFN uses ReLU. GELU is a later comparison here, not a substitution for that design.

### A complete two-layer forward calculation

Use row-vector convention with a controlled fixture:

$$\begin{aligned}
x&=[1,2],\quad
W_1=\begin{bmatrix}1&-1\\-\tfrac12&\tfrac12\end{bmatrix},\\
b_1&=[0,\tfrac12\log3],\quad
W_2=\begin{bmatrix}1&-1\\0&2\log3\end{bmatrix},\quad b_2=[0,0].
\end{aligned}$$

First, xW₁=`[0,0]`; adding b₁ gives a=`[0,.5 log3]`. Because tanh(.5 log3)=.5, hidden h=`[0,.5]`. Then z=hW₂=`[0,log3]`. Its softmax is `[.25,.75]`. For supplied target index 1, loss is −log(.75)=log(4/3)≈.287682. The target is chosen for an arithmetic fixture; `(1,2)` has label 0 under the trainer's opposite-sign task. Do not mix the fixture's supplied label with that separate task's rule.

[[diagram:mlp-shapes]]

The 2→2→2 fixture has `(2×2+2)+(2×2+2)=12` parameters. The reference trainer uses 2→8→2, giving `(2×8+8)+(8×2+2)=42`. Batch size changes activation storage, not either parameter count.

### PyTorch implements the same maps

```python
import torch
from torch import nn
m = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2))
x = torch.tensor([[1.,2.],[-1.,2.]])
logits = m(x)
assert logits.shape == (2,2)
assert sum(p.numel() for p in m.parameters()) == 42
```

This block uses random initialization, so it declares shapes/counts rather than a fixed output. The exact fixture is [deep_learning_math.py](examples/deep_learning_math.py); it has an independent Torch comparison and full backward values. Never add a final softmax before a loss that expects logits. A hidden activation and an output probability transform have different jobs.

[[practice:feedforward-transfer]]

### Notes and implementation

Draw the affine–activation–affine graph and annotate every shape. Derive the affine-collapse identity, reproduce the controlled hidden/logit values and explain the quadrant contradiction. For the existing lab, attempt selected numerical units rather than copying the entire reference network into your learner file.

Sources: [D2L MLP](source:d2l-mlp), [PyTorch Tanh](source:torch-tanh) and [Linear](source:torch-linear). Continue with [initialization](lessons/06-initialization.md).
