## Lesson 18 — The position-wise feedforward sublayer

**Goal:** distinguish feature transformation from token mixing and calculate the FFN's capacity/cost. Prerequisites: feedforward networks, shared maps and multi-head attention. Attention and the FFN perform different kinds of computation inside each layer.

### Expand, activate and contract

For one position's row x `[D]`, the original sublayer is

$$\operatorname{FFN}(x)=\operatorname{ReLU}(xW_1+b_1)W_2+b_2,$$

with W₁ `[D,F]`, b₁ `[F]`, W₂ `[F,D]` and b₂ `[D]`. F is the hidden feature width, often called d_ff. The original base model uses D=512,F=2048. The output returns to width D so it can be added to a residual input of width D.

“Position-wise” means the same weights are applied to each token independently. It does not mean a different network is trained for every position. Different Transformer layers generally have their own FFN parameters; sharing across positions does not automatically imply sharing across depth.

### Calculate two positions with one shared network

Choose D=2,F=3:

$$\begin{aligned}
W_1&=\begin{bmatrix}1&-1&0\\0&1&1\end{bmatrix},\quad b_1=[0,0,0],\\
W_2&=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\quad b_2=[1,-1].
\end{aligned}$$

For x=`[2,−1]`, preactivation=`[2,−3,−1]`, ReLU=`[2,0,0]`, output=`[3,−1]`. For x=`[−1,2]`, preactivation=`[−1,3,2]`, ReLU=`[0,3,2]`, output=`[3,4]`. Modifying only the first input cannot alter the second FFN output. It can alter other token outputs if these inputs are used by an earlier attention operation.

```python
import torch
x = torch.tensor([[2.,-1.],[-1.,2.]])
w1 = torch.tensor([[1.,-1.,0.],[0.,1.,1.]])
w2 = torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
out = torch.relu(x @ w1) @ w2 + torch.tensor([1.,-1.])
assert out.tolist() == [[3.,-1.],[3.,4.]]
changed = x.clone(); changed[0] += 10
assert torch.equal((torch.relu(changed @ w1) @ w2)[1], (out-torch.tensor([1.,-1.]))[1])
```

The FFN supplies nonlinear feature capacity after attention has mixed information across tokens. Replacing it with one affine layer loses this particular nonlinearity even if dimensions match. A much wider hidden layer adds parameters and activation memory; “attention is all you need” does not mean the model contains only attention operations.

### Count parameters and compare work

With biases, the count is DF+F+FD+D=2DF+F+D. At D=2,F=3 it is17. At F=4D it is8D²+5D. For D=512,F=2048, that is2,099,712 parameters. A four-projection attention module has4D² matrix parameters, so the FFN can hold roughly twice the matrix parameters of attention in the same block.

Across B sequences and T positions, the two dense multiplies require about2BTDF multiply–accumulates, or4BTDF FLOPs under separate multiply/add counting. This grows linearly with T, while explicit self-attention's score/value mixing grows quadratically with T. The FFN can dominate compute at shorter lengths; attention can dominate memory at longer lengths. Measure an actual profile rather than assuming “attention” is always the bottleneck.

### Implementation choices and backward behavior

In PyTorch, `Linear(D,F)→ReLU→Linear(F,D)` automatically applies to the last axis of `[B,T,D]`. Do not flatten position into the feature width: a `Linear(T*D,...)` would learn dependence on the fixed sequence layout and mix tokens. A 1×1 convolution can implement an equivalent position-wise map under a compatible axis convention.

GELU and gated FFNs are later choices with different formulas/counts; they are not the original ReLU sublayer. For ReLU, the backward derivative masks nonpositive preactivations under the chosen implementation. A hidden unit inactive for all records may receive no useful gradient; inspect activation and gradient distributions before assuming an optimizer bug.

The reusable [foundation probes](examples/transformer_foundation_probes.py) implement these small calculations; [independent probe checks](examples/test_transformer_foundation_probes.py) test hand answers, gradients and changed conditions. Use the recorded Torch interpreter from Lesson 2.

[[practice:ffn-transfer]]

In my notes: work both positions by hand, show parameter sharing, calculate the count and compare the leading attention/FFN operations. Source: [original FFN equation](source:transformer-paper), [D2L architecture](source:d2l-transformer). Continue with [residual connections](lessons/19-residual.md).
