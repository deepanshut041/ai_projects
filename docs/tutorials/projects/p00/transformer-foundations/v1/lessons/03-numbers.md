## Lesson 3 — Derivatives, gradients and local descent

**Goal:** calculate how a small parameter change affects a scalar loss, and understand what backpropagation must compute. Prerequisites: [neural maps](lessons/01-prediction.md) and [tensor axes](lessons/02-workspace.md).

**Existing learner continuation:** the old “Lesson 3 part A” destination is preserved. Your next implementation remains `linear()` in `src/transformer_foundations_lab/numerics.py`; its full contract, rectangular fixture and stopping point now appear in [Lesson 1's implementation exercise](lessons/01-prediction.md). Resume that exercise before treating these derivative checks as learner completion.

### A derivative is a local rate of change

For a scalar function f, the derivative at w is

$$f'(w)=\lim_{h\to0}\frac{f(w+h)-f(w)}{h}.$$

The numerator is a change in output and the denominator a change in input. It is not the ratio f(w)/w. For $f(w)=w^2$, expand $(w+h)^2$ to obtain `2w+h`; the limit is 2w. This is a local statement. It does not say the function remains linear for a large step.

In learning, define prediction a=wx+b, target y, error e=a−y and squared loss L=e². Hold x and y fixed; w and b are the adjustable parameters. Applying the chain rule gives

$$\begin{aligned}
\frac{\partial L}{\partial w}&=2e\,x,\\
\frac{\partial L}{\partial b}&=2e.
\end{aligned}$$

At x=3,w=2,b=0,y=1, prediction is 6, error 5, loss 25, and the gradients are 30 and 10. The factor x appears in the weight derivative because changing w changes the prediction by x times the change. Bias changes it one-for-one. If the loss were e²/2, both derivatives would be half as large; define the objective before comparing gradients.

### Chain rule and the functions we will differentiate

If L=f(a) and a=g(w), a small change obeys Δa≈g′(w)Δw, then ΔL≈f′(a)Δa. Combining the two rates gives dL/dw=f′(a)g′(w). The intermediate a must be evaluated at the actual forward value. For a square loss, the local derivative is with respect to the **error**, and the error then depends on the prediction and parameter. Multiplying the rates explains the factors above; shared uses require adding all paths afterward.

| Scalar function | Derivative where defined | First later use |
|---|---|---|
| exp(a)=eᵃ | exp(a) | softmax/BCE |
| log(a), a>0 | 1/a | negative log likelihood |
| sqrt(a), a>0 | 1/(2sqrt(a)) | normalization and scale |
| sin(a) | cos(a) | positions, with angles in radians |
| cos(a) | −sin(a) | positions |

Natural log is the inverse of exp: log(exp(a))=a, and log(xy)=log(x)+log(y) for positive x,y. These identities and derivatives supply the steps used in the loss and normalization derivations; they are not extra learned parameters.

### Partials, gradients and directional change

A partial derivative changes one coordinate while keeping the others fixed. The gradient collects them in a vector, here `[30,10]`. For a small change Δθ, first-order Taylor expansion gives

$$L(\theta+\Delta\theta)\approx L(\theta)+\nabla L(\theta)^T\Delta\theta.$$

Choose Δθ=−η∇L for positive η. The predicted change is $-\eta\|\nabla L\|_2^2$, which is nonpositive. Higher-order curvature terms can invalidate that approximation for a large η. The negative gradient is the steepest local descent direction under a Euclidean step-length constraint; another geometry can give another direction.

Keeping b fixed, η=.01 gives w=1.7 and loss `(5.1−1)²=16.81`. At η=.2, w=−4 and loss `(−12−1)²=169`: correct derivative, bad step size. If both w and b are updated at η=.01, they become `[1.7,−.1]` and loss is 16. These are different updates.

[[scenario:gradient-step]]

For a direction v, the directional derivative is ∇L·v. At our fixture, direction `[1,−3]` has derivative `30−30=0`; a first-order zero change does not prove the loss stays constant for a finite move.

### Vector outputs require a Jacobian

For a vector-valued map a(θ), Jacobian J has entries Jᵢⱼ=∂aᵢ/∂θⱼ. If a scalar loss sends upstream gradient g=∂L/∂a, the chain rule gives ∂L/∂θ=Jᵀg. This vector–Jacobian product is what reverse-mode differentiation computes efficiently; it need not materialize the whole Jacobian.

For a=wx+b, J with respect to `[w,b]` is the row `[x,1]`. Multiplying its transpose by upstream 2e gives `[2ex,2e]`. For a batch, the same parameter influences many outputs, so the contributions add. A mean loss divides by the chosen number of supervised terms; a sum does not. Broadcasting a shared bias creates many uses and therefore a backward sum.

### Finite differences are an independent check

A central difference estimates the derivative by evaluating the forward function twice:

```python
from math import isclose
w, x, b, y, h = 2., 3., 0., 1., 1e-5
f = lambda weight: (weight*x + b-y)**2
measured = (f(w+h)-f(w-h))/(2*h)
assert isclose(measured, 30., rel_tol=1e-9)
```

This checks the formula independently of an autograd graph. Too-large h measures curvature; too-small h amplifies rounding error. Check several sizes using FP64 and smooth points. At a ReLU kink, an ordinary two-sided derivative does not exist; disagreement there is not necessarily a backward implementation bug.

[[practice:calculus-transfer]]

### Notes and source

Derive both partials from intermediate quantities, label what is held fixed and calculate the two different SGD updates. Explain how Jᵀg becomes a shared parameter's gradient. Use [prerequisite_math.py](examples/prerequisite_math.py) for central differences, not as a substitute for the derivation.

Written reference: [Nielsen's backpropagation chapter](source:nielsen-backprop); original derivations and numerical fixtures above. Continue with [probability and information](lessons/04-probability.md).
