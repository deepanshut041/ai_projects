## Lesson 7 — Backpropagation and automatic differentiation

**Goal:** derive the entire backward pass of a network, then inspect how autograd implements it. Prerequisites: chain rule, affine maps, tanh, softmax and the forward fixture in [Lesson 5](lessons/08-model.md). The categorical loss gradient used below is derived in [Lesson 8](lessons/08-losses.md).

### Reverse the graph, not the program's meaning

A forward graph records intermediate values and dependencies. A backward pass starts with ∂L/∂L=1 and repeatedly multiplies an upstream gradient by each operation's local derivative. Contributions from all uses of a shared variable add. Gradients have the same coordinate shapes as the values they describe.

For $Y=XW+b$ with X `[N,D]`, W `[D,F]`, and upstream G=∂L/∂Y `[N,F]`, differentiating the indexed sum gives

$$\begin{aligned}
\frac{\partial L}{\partial W}&=X^TG\quad[D,F],\\
\frac{\partial L}{\partial b}&=\sum_nG_{n,:}\quad[F],\\
\frac{\partial L}{\partial X}&=GW^T\quad[N,D].
\end{aligned}$$

The bias is used by every row, so its gradient sums over rows. If Y also has a position axis, sum over batch and position. For h=tanh(a), the local derivative is `1−h²`, giving ∂L/∂a=(∂L/∂h)⊙(1−h²), where ⊙ means elementwise multiplication.

### Work every stage of the two-layer network

The 12-parameter fixture has x=`[1,2]`, h=`[0,.5]`, z=`[0,log3]`, p=`[.25,.75]`, target 1 and L=log(4/3). For softmax cross-entropy, output gradient δ₂=p−one_hot(1)=`[.25,−.25]`.

1. Output bias gradient is δ₂. The output weight gradient is outer product hᵀδ₂:

$$\nabla W_2=\begin{bmatrix}0&0\\.125&-.125\end{bmatrix},\qquad
\nabla b_2=[.25,-.25].$$

2. Hidden upstream gradient is δ₂W₂ᵀ=`[.5,−.5 log3]`. Tanh derivatives are `[1,.75]`, so

$$\delta_1=[.5,-.375\log3]\approx[.5,-.411979608].$$

3. Hidden weights again receive an outer product, and the hidden bias receives δ₁:

$$\begin{aligned}
\nabla W_1&=\begin{bmatrix}.5&-.375\log3\\1&-.75\log3\end{bmatrix},\\
\nabla b_1&=[.5,-.375\log3].
\end{aligned}$$

4. Even if the inputs are not learned parameters, we can differentiate them: ∇x=δ₁W₁ᵀ≈`[.911979608,−.455989804]`.

All twelve parameter coordinates and both input coordinates are checked in [test_deep_learning_math.py](examples/test_deep_learning_math.py) against independent forward perturbations. The reference [deep_learning_math.py](examples/deep_learning_math.py) also compares them to FP64 Torch autograd. Changing only the supplied target to 0 leaves h,z,p unchanged and multiplies every derivative in this fixture by −3; it changes which class the loss rewards.

### Shared parameters gather all paths

For L(w)=(2w−1)²+(3w−1)², at w=1 the first use contributes 4 and the second 12, total 16. Recording only the last branch is wrong. At w=.5, total derivative is 3; averaging the two losses gives 1.5. The reduction belongs to the objective, not to a mysterious automatic averaging rule in backpropagation.

[[diagram:prereq-gradient]]

### Autograd records the same dependency structure

```python
import torch
w = torch.tensor(1., dtype=torch.float64, requires_grad=True)
loss = (2*w-1)**2 + (3*w-1)**2
loss.backward()
assert w.grad.item() == 16
((2*w-1)**2 + (3*w-1)**2).backward()
assert w.grad.item() == 32  # leaf .grad accumulates
w.grad = None
```

A second forward pass builds a new graph. A backward call usually releases saved intermediates; reusing the same graph can require `retain_graph=True`, but retaining everything is not a normal training-loop fix. Leaf `.grad` accumulates until cleared. Nonleaf intermediates do not ordinarily retain `.grad`; call `retain_grad()` when deliberately inspecting them.

`detach()` breaks a gradient path while sharing underlying data storage; it is not a safe independent copy for arbitrary mutation. `item()` converts to a Python number, so arithmetic using that value is outside the differentiable graph. `no_grad()` avoids recording new graph operations. In-place changes to saved values can invalidate backward. Prefer direct, explicit forward expressions while learning.

[[practice:backprop-transfer]]

### Notes and stopping check

Derive all four parameter gradients and the input gradient, annotate upstream versus local derivatives, and explain the zero first row of ∇W₂. Compare one coordinate with a central difference before inspecting Torch. Explain how repeated embedding IDs will use the same “sum all paths” rule.

Sources: [Nielsen Chapter 2](source:nielsen-backprop), [D2L computation graphs](source:d2l-backprop), [PyTorch autograd](source:autograd-intro). Continue with [loss functions](lessons/08-losses.md).
