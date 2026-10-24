## Follow an error backward through attention

A component used in training needs more than plausible forward outputs. Its parameters and inputs must receive the derivatives of the chosen loss. We will derive one scalar query example, verify it with autograd and finite differences, then connect that evidence to the existing parameter-gradient tests.

### Define a tiny loss and all of its inputs

Use one scalar query q, keys `[1,-1]` and values `[2,6]`. Head width is one, so the scale is one. The scores are `[q,-q]`. Let a be the weight on the first key; the other weight is `1-a`:

$$a=\frac{e^q}{e^q+e^{-q}},\qquad c=2a+6(1-a)=6-4a.$$

Choose an illustrative desired context `y=5` and half squared error `L=(c-y)^2/2`. This loss is a numerical probe for the attention computation. The eventual translation task will use vocabulary logits and token loss, not an asserted target representation of five.

At q=0, both scores are zero, a=0.5 and c=4. The loss is 0.5. We want the next small step to move the context upward toward five, which means placing more weight on value six and less on value two.

### Derive the query gradient in pieces

For a general softmax row, `∂a_j/∂s_l = a_j(δ_jl-a_l)`, where δ is one when indices match and zero otherwise. This follows by differentiating the exponential numerator and the shared sum denominator. It captures both effects: increasing one score changes its own numerator and every probability's denominator.

Here the two scores depend on q with slopes +1 and -1. Therefore:

$$\frac{da}{dq}=a(1-a)+(-a(1-a))(-1)=2a(1-a).$$

At q=0 this is 0.5. Because `c=6-4a`, `dc/dq=-4×0.5=-2`. The loss slope with respect to context is `dL/dc=c-y=-1`. The chain rule multiplies the local slopes:

$$\frac{dL}{dq}=\frac{dL}{dc}\frac{dc}{dq}=(-1)(-2)=2.$$

A gradient tells us how the loss changes for a small increase in q. It is positive, so gradient descent decreases q. With step size 0.1, `q_new=0-0.1×2=-0.2`. The first weight becomes about 0.401312; context becomes about 4.394751 and loss about 0.183163. Moving the query in this direction raises the weight on the larger value and reduces this chosen loss.

### Verify the derivative using the actual helper

```python
import torch
from core.attention import scaled_dot_product_attention

q = torch.tensor([[[0.]]], dtype=torch.float64, requires_grad=True)
k = torch.tensor([[[1.], [-1.]]], dtype=torch.float64)
v = torch.tensor([[[2.], [6.]]], dtype=torch.float64, requires_grad=True)
context, weights = scaled_dot_product_attention(q, k, v)
loss = 0.5 * (context - 5).square().sum()
loss.backward()
print(context.item(), loss.item())  # 4.0, 0.5
print(q.grad.item())               # 2.0
print(v.grad.flatten().tolist())   # [-0.5, -0.5]
```

The value gradients follow directly: each value contributes to c with coefficient 0.5, so its loss derivative is `(-1)×0.5=-0.5`. Both values could move upward to improve this toy loss. In this particular zero-query fixture, some other gradients can be zero for a legitimate mathematical reason. “All gradients must be nonzero” is not a general correctness criterion.

![Gradient flow through the attention module and its learned projections.](images/16-gradient-flow.png)

### Check the slope independently of the backward program

For a small ε, estimate the derivative with central differences: `[L(q+ε)-L(q-ε)]/(2ε)`. Using ε=`1e-6` around zero gives approximately 2. This calculation reevaluates forward losses instead of using the graph's backward rule, so it can detect a plausible incorrect gradient implementation.

Too small an ε can amplify floating-point cancellation; too large an ε measures curvature over a wider interval. Use float64 and compare within a declared tolerance. A finite-difference agreement at one point is bounded evidence, so retain other fixtures rather than declaring every shape or dtype correct.

### Connect the probe to learned parameters

In the public module, Q/K/V depend on affine weights. For one row, `q=xW_Q+b_Q` in the displayed convention. The chain rule sends the query gradient through that map to x, W and b; shared parameter contributions from multiple positions/examples accumulate. Output projection, value mixing and the score path each contribute their own derivative routes.

The existing tests create explicit and SDPA layers with identical parameters, calculate the same squared-output objective, call backward, and compare input and parameter gradients. Cross-attention also checks target-query input gradients. Combined with PyTorch's independent forward reference and analytic probes, this is stronger than checking that `.backward()` merely returns without raising an exception.

No optimizer, real translation dataset, training loop or translation-quality experiment is implemented by this tutorial. The q update above is an isolated arithmetic check. It explains why a learnable attention module can receive a useful update; it does not establish that a full model has learned a language.

[[practice:gradient-check]]

The final lesson gathers the evidence and states the exact next data-phase handoff, keeping mathematical understanding, source correctness and task performance distinct.
