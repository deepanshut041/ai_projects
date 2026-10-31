## Lesson 10 — Multi-head attention implementation

**Goal:** implement independent retrieval patterns with correct feature ownership, verify explicit/SDPA parity and account for cost. Prerequisite: single-head attention and its projection gradients. Head meanings emerge from training; the architecture does not assign a grammatical label to each one.

Use [translation_attention.py](examples/translation_attention.py). Its packed projections, shape rearrangements and output projection constitute the actual operation. [Source](source:attention-code), [SDPA contract](source:torch-sdpa).

### Keep the total width fixed

Let `B` be batch size, `T` query length, `S` source length, `D` total model width, `H` heads and `d=D/H` features per head. Choose `D` divisible by `H`. For `D=8,H=2`, each head has four features. Four heads at the same `D` would have two features each; it would not create four eight-feature heads.

Project the input to `[B,T,D]`, reshape it to `[B,T,H,d]`, then swap the token and head axes to `[B,H,T,d]`. This ensures the attention operation treats a whole head as one independent sequence computation. Multiplication produces `[B,H,T,S]` scores and `[B,H,T,d]` context.

[[diagram:head-geometry]]

For one batch with two tokens, make the feature identities impossible to confuse:

```text
projected [1,2,8]
token0:  0  1  2  3 |  4  5  6  7
token1:  8  9 10 11 | 12 13 14 15

head0 [2,4]: [[0,1,2,3], [8,9,10,11]]
head1 [2,4]: [[4,5,6,7], [12,13,14,15]]
```

Calling `reshape(B,H,T,d)` directly would group a different set of values. A tensor can have the right shape and still describe the wrong tokens. `transpose` changes the interpretation of axes; it is not an arbitrary cleanup step.

Open the two complete helper functions:

```python
def split_heads(x, num_heads):
    batch, length, width = x.shape
    if width % num_heads:
        raise ValueError("feature width must be divisible by num_heads")
    return x.reshape(batch, length, num_heads, width // num_heads).transpose(1, 2)

def merge_heads(x):
    return x.transpose(1, 2).contiguous().flatten(2)
```

The merge reverses the swap and combines only `H` and `d`. `contiguous()` gives a regular storage layout after the swap; `flatten(2)` joins all axes from axis two onward. Batch and token identity remain separate.

Run this inspection now:

```python
import torch
from translation_attention import split_heads, merge_heads

x = torch.arange(16).reshape(1, 2, 8)
heads = split_heads(x, 2)
print(heads.shape, heads[0, 1].tolist())
print(torch.equal(merge_heads(heads), x))
```

It prints `torch.Size([1,2,2,4])`, head one's two rows `[[4,5,6,7],[12,13,14,15]]`, and `True`. The values establish correctness more strongly than a shape assertion alone.

### Each head is learned through the projections

It is possible to write a loop with a separate `Linear(D,d)` for every head. The packed implementation instead uses one `Linear(D,D)` for all queries, another for keys and another for values. Each projected feature still depends on all `D` input features; only the resulting output columns are split into heads. Those groups of columns have separate learned parameters, so the packed version represents the same family of linear maps.

For head `h`, calculate

$$A_h=\operatorname{softmax}_{\text{source}}\!\left(\frac{Q_hK_h^T}{\sqrt d}\right),\qquad C_h=A_hV_h.$$

If `D=8,H=2`, the denominator is `sqrt(4)=2`. Using `sqrt(8)` would reduce scores too far. Concatenate each token's contexts:

$$C_{\mathrm{joined}}=[C_0\;\Vert\;C_1]\in\mathbb R^{B\times T\times D}.$$

Finally apply `out_proj`, a `Linear(D,D)`. The heads perform separate retrievals, but that output map can combine information from their feature groups. Concatenation alone puts their features side by side; the projection learns how to mix them.

### The class state and the path through it

The `Attention` constructor in `translation_attention.py` owns four projections:

```text
self.q_proj = nn.Linear(embedding_dimension, embedding_dimension)
self.k_proj = nn.Linear(embedding_dimension, embedding_dimension)
self.v_proj = nn.Linear(embedding_dimension, embedding_dimension)
self.out_proj = nn.Linear(embedding_dimension, embedding_dimension)
```

They are `nn.Module` members, so PyTorch registers their parameters, moves them with the model and includes them in the state dictionary. Creating new `Linear` layers inside `forward` would create new parameters on every call and destroy the intended learning state.

The computational spine of `forward` is:

```text
query_input = src if tgt is None else tgt
q = split_heads(self.q_proj(query_input), self.num_heads)
k = split_heads(self.k_proj(src), self.num_heads)
v = split_heads(self.v_proj(src), self.num_heads)
# Build the allowed-key mask as explained in Lesson 11.
context = F.scaled_dot_product_attention(
    q, k, v, attn_mask=allowed, dropout_p=drop, is_causal=False
)
output = self.out_proj(merge_heads(context))
```

The complete callable class is in the linked file, including dimension checks, the explicit inspection path, mask validation and dropout handling. The constructor accepts `fused_attn=False` to use `attention_math`; `need_weights=True` selects that inspection path even when efficient execution is enabled.

### Efficient SDPA preserves the operation

`torch.nn.functional.scaled_dot_product_attention` accepts the packed `q`, `k`, `v` and performs the same scaled, normalized weighted sum. PyTorch chooses an eligible backend according to device, tensor type, dimensions and other constraints. CPU execution or an unsupported GPU shape can use a mathematical backend. Calling SDPA does **not** prove that a FlashAttention kernel ran, and speed is a separate measurement.

The fused implementations can avoid explicitly retaining a full probability matrix. That can improve memory use, but attention still has interactions between every relevant query and key. With self-attention length `T`, the conceptual score tensor contains `B×H×T²` entries. For `B=2,H=4,T=128`, that is `131072` entries, roughly `0.5 MiB` in float32 for one such tensor. Activations, gradients, other layers and backend workspaces add more memory. Doubling `T` quadruples this count.

Evaluation requires a deliberate dropout choice. The SDPA function follows its `dropout_p` argument; it does not inspect the surrounding module's `.eval()` flag. The class therefore passes `self.attn_dropout if self.training else 0.0`. In the explicit path, dropout is applied to probabilities during training. Dropped weights are rescaled, so the retained probability row need not sum to one in an individual training call.

### Verify both execution paths with the same parameters

Run this from `examples/`:

```python
import torch
from translation_attention import Attention

torch.manual_seed(7)
manual = Attention(8, 2, fused_attn=False).double().eval()
efficient = Attention(8, 2, fused_attn=True).double().eval()
efficient.load_state_dict(manual.state_dict())
x = torch.randn(2, 3, 8, dtype=torch.double)
a = manual(x)
b = efficient(x)
torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
print(a.shape, "paths agree")
```

Expected: `torch.Size([2,3,8]) paths agree`. Matching parameter state and disabled dropout are necessary: two separately initialized layers should not have matching outputs.

The dedicated test includes unequal query/source lengths and a padding mask:

```bash
python translation_checks.py TranslationChecks.test_manual_and_sdpa_match_for_rectangular_mask
python translation_checks.py TranslationChecks.test_head_layout_round_trip_and_identity
```

If layout round-trip fails, inspect the reshape and swap. If parity fails only with masks, inspect the mask convention and key axis before changing tolerances.

### Separate parameter cost from activation cost

At fixed D, four biased D→D projections own 4D²+4D parameters regardless of H. Increasing H changes head width and the conceptual probability count BHTS; it does not multiply projection parameters by H. Query-key and probability-value products each require roughly BTSD multiply-accumulates across all heads. Projection work instead scales with sequence lengths and D². These estimates describe operations, not measured wall time or backend memory allocation.

For D=8 and H=2, there are 288 projection parameters. With B=2,T=4,S=7, scores contain 112 entries. Replacing two heads by four keeps 288 parameters but produces 224 conceptual score entries with width 2 per head. A fused kernel may avoid materializing that full probability array; requesting weights selects the explicit path and changes the inspection cost.

`need_weights=True` returns **pre-dropout** probabilities. With training dropout enabled, those displayed weights are not exactly the weights used to form context. Check both the module mode and what the returned tensor represents before interpreting a plot. Probability plots alone do not establish a linguistic explanation or causal importance.

### Extend parity to gradients and rectangular inputs

An implementation that matches one forward pass can still detach or reuse the wrong parameter during backward. Compare identical parameter states on unequal source/query lengths, valid Boolean masks and deterministic mode. Backpropagate the same scalar function of the output, then compare input and projection gradients. Floating-point kernels can differ slightly; use tolerances appropriate to dtype and backend rather than insisting on bit equality across devices.

For a manual head loop, slice **output rows of the stored projection weights** for each head. All heads read the full input feature vector. Slicing input columns would create a different map in which each head sees only part of the input. Concatenate each token's contexts, then apply the common output projection.

[[practice:heads-practice-1]]
[[practice:heads-transfer]]

### Notes and implementation

Draw the two reshape/transposition directions in my notes using numbered features. Count parameters and rectangular score entries for a changed H, then explain the stored-weight slices of a loop implementation. Run the two named parity/layout checks and the applied gradient probes. Continue to [attention masks](lessons/09-masks.md).
