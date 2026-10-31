## Lesson 13 — Position-wise FFN and activation choice

**Goal:** derive the shared nonlinear feature map, its gradients and cost, then inspect the source's activation/dropout recipe. Prerequisites: affine maps, chain rule and attention. Attention mixes positions; this FFN mixes features independently within each receiving position.

### Expansion followed by nonlinear contraction

For row vector x of width D and hidden width F=rD,

$$
\begin{aligned}
a&=xW_1+b_1,\quad a\in\mathbb R^F,\\
z&=\operatorname{GELU}(a),\\
f(x)&=zW_2+b_2,\quad f(x)\in\mathbb R^D.
\end{aligned}
$$

W₁ has mathematical shape [D,F], W₂ [F,D]. PyTorch stores their transposes. Applying the same map to [B,T,D] retains B/T and shares all parameters across positions. Expansion creates a larger nonlinear feature basis; contraction returns to the residual width. Without a nonlinearity, `(xW₁+b₁)W₂+b₂` collapses to one affine map, although a narrow intermediate factor could still constrain rank.

### GELU supplies a smooth gate

For the standard normal cumulative distribution Φ and density φ,

$$
\operatorname{GELU}(a)=a\Phi(a),\qquad
\operatorname{GELU}'(a)=\Phi(a)+a\phi(a).
$$

Φ is a cumulative probability, not a distribution learned by this layer. GELU(1)≈.841345, GELU(−1)≈−.158655 and GELU(0)=0. Its derivative at 1 is approximately 1.083315, at −1 approximately −.083315, and at 0 exactly .5. A negative output or derivative is possible; this is not ReLU with a different name. The source uses the default exact GELU, rather than explicitly selecting its optional tanh approximation. [GELU contract](source:torch-gelu).

The original Transformer used ReLU, `max(0,a)`. Changing to GELU changes the function and gradient even if dimensions/parameters match. Neither is an automatic quality guarantee.

### Work one map and its backward signal

Choose x=`[1,-1]`, W₁=`[[1,0,1],[0,1,1]]`, b₁=`[0,0,1]`, W₂=`[[1,0],[0,1],[1,-1]]`, b₂ zero. The intermediate a is `[1,-1,1]`; activation is `[.841345,-.158655,.841345]`; output is approximately `[1.682689,-1]`.

If L is the first output coordinate, ∂L/∂z=`[1,0,1]`. Multiplying by GELU′ gives ∂L/∂a≈`[1.083315,0,1.083315]`. Consequently ∂L/∂x≈`[2.166631,1.083315]`. Parameter derivatives are outer products: ∇W₂=zᵀg, ∇W₁=xᵀ∇a; biases receive the summed corresponding gradients across all positions.

```python
import torch
x = torch.tensor([1.,-1.],dtype=torch.float64,requires_grad=True)
w1 = torch.tensor([[1.,0.,1.],[0.,1.,1.]],dtype=x.dtype)
w2 = torch.tensor([[1.,0.],[0.,1.],[1.,-1.]],dtype=x.dtype)
a = x@w1 + torch.tensor([0.,0.,1.],dtype=x.dtype)
y = torch.nn.functional.gelu(a)@w2
y[0].backward()
assert abs(y[0].item()-1.682689492) < 1e-8
assert abs(y[1].item()+1) < 1e-12
assert torch.allclose(x.grad,torch.tensor([2.16663094,1.08331547],dtype=x.dtype))
```

The worked hidden width 3 demonstrates the map; the reference configuration instead expresses F as an integer multiple of D. Its default D=32,r=4 gives F=128.

[[diagram:translation-ffn]]

### Source placement and parameter accounting

[FeedForward.forward](examples/translation_model.py) executes `intermediate_dense → GELU → intermediate_dropout → output_dense → output_dropout`. Both dropout modules use `hidden_dropout_p` and obey train/eval mode. The surrounding block adds the residual and normalizes. There is no second outer FFN dropout in that block: output dropout is already inside the FFN.

The parameter count is DF+F+FD+D=2DF+F+D. At D=32,F=128, weights contain 8,192 entries and biases 160, totaling 8,352. Approximately 2B T D F multiply-accumulates are needed per FFN, plus activation/dropout work. This scales linearly with token count, whereas self-attention's score/value work scales quadratically in sequence length. A long-sequence memory problem can coexist with most learned parameters residing in FFNs.

Initialization and normalization affect activation/gradient scale; expansion alone does not fix instability. The source initializes linear weights with std .02 and biases zero. Its mathematical choice is part of the saved configuration, not an explanation of trained feature meanings.

### Test the position-wise claim

Under evaluation, perturb only token row 1 before this FFN. Other token outputs must remain identical because no token-axis operation exists inside it. Perturbing a token before attention can affect other receiving positions. Use these two controls to distinguish feature transformation from communication. Random training dropout would need identical masks or a statistical comparison; a naïve equality check across two random calls is invalid.

[[practice:translation-ffn-transfer]]
[[practice:translation-ffn-gradient]]

### Notes and implementation

Derive the output and backward rows in my notes, count a changed expansion ratio and mark both dropout locations. Recreate the independent-token perturbation on `FeedForward`, then explain where the model mixes positions. Continue with [residuals and LayerNorm](lessons/11-transformer-blocks.md).
