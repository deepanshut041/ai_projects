## Lesson 6 — assemble the whole GPT

Attention can mix an allowed prefix. We now need a complete path from token IDs to V next-token scores. Follow one row: BOS becomes a vector, receives position zero, passes through two blocks, and produces a score for every vocabulary ID. No encoder or cross-attention participates.

[[diagram:model]]

### Learned positions are parameters

Token ID 104 always selects row 104 of the token embedding matrix E[V,D]. That alone cannot tell the model whether `h` occurs first or later. A learned position matrix P[Tmax,D] adds a different trainable vector at each position:

$$X_{b,t,:}=E[x_{b,t},:]+P[t,:].$$

For a toy token vector `[1,2]` and position vector `[0.1,-0.2]`, their sum is `[1.1,1.8]`. It is addition of matching feature axes, not concatenation. Gradients update both selected rows. A token repeated in several positions accumulates gradient contributions into the same E row, whereas those positions may use different P rows.

P00's sinusoidal positions were a fixed formula stored as a buffer. These positions are learned parameters with an explicit maximum index. We do not scale the token embeddings by √D in this design. Both are recorded choices. If T exceeds 64, the model rejects the input; the generation policy later crops a context before calling the model.

[[practice:positions]]

### Put normalization before each sublayer

For each `[D]` feature vector, LayerNorm computes the mean and population variance across features, then applies learned scale γ and bias β:

$$\mu=\frac{1}{D}\sum_j x_j,\quad v=\frac{1}{D}\sum_j(x_j-\mu)^2,
\qquad LN(x)_j=\gamma_j\frac{x_j-\mu}{\sqrt{v+\epsilon}}+\beta_j.$$

It does not average over time or across examples. `[1,3]` has mean 2 and variance 1, giving approximately `[-1,1]` with unit γ and zero β. ε avoids division by zero for a constant vector; it is not a sample-variance correction.

A pre-norm block computes

$$U=X+Attention(LN_1(X)),\qquad Y=U+FFN(LN_2(U)).$$

Compare P00's `LN(X+Attention(X))`. If the sublayer outputs zero, the pre-norm residual returns X, while the post-norm form returns LN(X). The derivative of the pre-norm residual contains the direct identity path plus the derivative through the sublayer. This helps explain the ordering; it does not prove training can never have exploding/vanishing gradients. After all blocks we apply a final LayerNorm before the vocabulary projection.

[[diagram:prenorm]]
[[practice:prenorm]]

### Why the FFN uses GELU

The FFN transforms each position independently: D→F→D. Without a nonlinear activation, two affine layers collapse into one affine layer. Attention moves information between positions; the FFN transforms the features at each position.

P00 used ReLU, which sets negative inputs to zero. GELU weights the input by a smooth probability curve. For a standard normal random variable Z with mean zero and variance one, Φ(x)=P(Z≤x) is its cumulative probability. Exact GELU is $x\Phi(x)$. At x=0, Φ=0.5 and GELU=0. At x=−1, Φ≈0.158655, so GELU≈−0.158655; it keeps a small negative value.

Our declared tanh approximation is

$$GELU(x)\approx\frac{x}{2}\left[1+\tanh\left(\sqrt{\frac{2}{\pi}}(x+0.044715x^3)\right)\right].$$

Hyperbolic tangent is a smooth function bounded between −1 and 1: $\tanh z=(e^z-e^{-z})/(e^z+e^{-z})$. The bracket scales values smoothly between roughly zero and two. We use the library activation so autograd supplies its derivative; the standard-library trace compares it with the exact Gaussian-CDF result at −1. The [versioned GELU API](source:gelu) distinguishes these two variants. Do not change the approximation while claiming exact parity with a differently configured reference.

[[practice:gelu]]

### Write a block

Create `gpt_model.py` with its imports and Block. `nn.Module` registers child layers and parameters; the module list in the next section registers every repeated block. An ordinary Python list would not provide that registration automatically.

<!-- file:gpt_model.py mode:write -->
```python
"""Complete small pre-norm GPT. Explicit attention, learned positions, tied head."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from gpt_config import Config
from gpt_attention import CausalAttention
from gpt_tokenizer import PAD


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width)
        self.attention = CausalAttention(config)
        self.norm2 = nn.LayerNorm(config.width)
        self.ffn = nn.Sequential(nn.Linear(config.width, config.hidden),
                                 nn.GELU(approximate="tanh"),
                                 nn.Linear(config.hidden, config.width),
                                 nn.Dropout(config.dropout))

    def forward(self, x, valid, *, optimized=False):
        x = x + self.attention(self.norm1(x), valid, optimized=optimized)
        return x + self.ffn(self.norm2(x))
```

Read the two lines in `forward` against the equations. Both additions preserve `[B,T,D]`. The FFN's hidden width F=256 is temporary; its final layer returns D=64 so the residual shapes agree.

### Connect embeddings, blocks and scores

Append GPT and the parameter formula:

<!-- file:gpt_model.py mode:append -->
```python
class GPT(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab, config.width)
        self.position = nn.Embedding(config.context, config.width)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.final_norm = nn.LayerNorm(config.width)
        self.dropout = nn.Dropout(config.dropout)
        # F.linear below uses the SAME embedding parameter; no duplicate head.
        self.apply(self.initialize)
        for block in self.blocks:
            nn.init.normal_(block.attention.output.weight, std=0.02 / math.sqrt(2*config.layers))
            nn.init.normal_(block.ffn[2].weight, std=0.02 / math.sqrt(2*config.layers))

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward(self, ids, *, optimized=False):
        if ids.ndim != 2 or ids.dtype != torch.long:
            raise ValueError("Token IDs must be a [B,T] int64 tensor")
        batch, time = ids.shape
        if batch < 1 or not 1 <= time <= self.config.context:
            raise ValueError("Empty input or context limit exceeded")
        if bool(((ids < 0) | (ids >= self.config.vocab)).any()):
            raise ValueError("Token ID outside frozen vocabulary")
        valid = ids != PAD
        if not bool(valid[:, 0].all()) or bool(((~valid[:, :-1]) & valid[:, 1:]).any()):
            raise ValueError("Use nonempty, right-padded sequences")
        positions = torch.arange(time, device=ids.device)
        x = self.dropout(self.token(ids) + self.position(positions))
        for block in self.blocks:
            x = block(x, valid, optimized=optimized)
        return F.linear(self.final_norm(x), self.token.weight)


def parameter_formula(c):
    # Per block: QKV/output + two FFN matrices + all biases + two norms.
    return c.vocab*c.width + c.context*c.width + c.layers * (
        4*c.width*c.width + 2*c.width*c.hidden + 9*c.width + c.hidden) + 2*c.width
```

`F.linear(features, token.weight)` computes the vocabulary scores using the same E matrix used for input lookups. This is **weight tying**. The input lookup and output projection both contribute gradients to E. There is no second independent V×D head matrix to count. The head has no bias; internal linear layers do have biases.

The code initializes linear/embedding weights near zero and scales each residual branch's final projection by $1/\sqrt{2L}$. This is a declared initialization choice for two residual branches per layer. LayerNorm begins with unit scales and zero biases. The small model is influenced by the inspected [pre-norm implementation](source:nanogpt), but uses our tokenizer, sizes, explicit attention, dropout and corpus; it is not a GPT-2 weight-compatible reproduction.

The model checks `[B,T]` int64 IDs, vocabulary range, context length and right-padding rules. These errors belong at the boundary. Avoid quietly clamping an unknown ID or truncating a training input: that would change the task without recording it.

### Run a complete forward pass

```bash
python - <<'PYCODE'
import torch
from gpt_model import GPT
from gpt_config import Config
m=GPT(Config())
x=torch.tensor([[257,104,105]])
print(m(x).shape)
print(sum(p.numel() for p in m.parameters()))
PYCODE
```

Expect `[1,3,260]` and 120832 parameters. These are logits, not probabilities yet. Each of the three positions has 260 scores. Generation will choose from the last position; training compares all valid positions with their shifted labels.

Save [test_gpt_model.py](examples/test_gpt_model.py) now, but run its tests after the first loss function in Lesson 7 exists. Its checks cover independent attention values/gradients/finite differences, the hand average, full-model causal perturbations and SDPA parity, padding/context errors, the pre-norm identity case, parameter count/tied gradients, loss gradients and unequal accumulation.

Do one causal probe now without the test file: evaluate `[BOS,1,2,3]`, change the last ID to 9, and compare logits through position 2 with `model.eval()`. They must agree. The last logits may change. If earlier logits change, inspect mask orientation before touching the optimizer.

The whole model computes scores, but its weights still express no learned text behavior. Next make those scores into a training signal and inspect one update.

[Continue to Lesson 7](lessons/07-loss-and-learning.md)

### Complete files from this lesson

[gpt_model.py](examples/gpt_model.py)
