## Follow the reusable Attention class from call to output

The earlier classes expose the progression from a single head to separate and vectorized heads. The final `Attention` class is the interface intended for the later encoder/decoder. We can now read its methods in execution order and explain every tensor transformation without treating a library call as a black box.

### State the contract before the call

The constructor takes `embedding_dimension`, `num_heads`, `attn_dropout`, and the keyword-only `fused_attn`. It owns four biased projections named `q_proj`, `k_proj`, `v_proj` and `out_proj`. Those names are deliberately retained for the later model code.

`forward(src, tgt=None, attention_mask=None, causal=False, *, need_weights=False)` accepts floating `(B,S,D)` source vectors and optional `(B,T,D)` target/query vectors. The mask is `(B,S)` and True means an allowed key. Without target vectors it performs self-attention. With them it performs cross-attention and rejects `causal=True`.

### Follow the method in six steps

| Step | Actual operation | What to explain |
| --- | --- | --- |
| 1 | `_validate_inputs` | Shape, dtype/device/batch agreement and source-mask length |
| 2 | `query_input = src if tgt is None else tgt` | Which sequence determines output positions |
| 3 | Q/K/V projection, then `_heads` | Full-width affine maps and head feature grouping |
| 4 | `_context` | Visibility, scaling, softmax/value mix and attention dropout |
| 5 | `context.transpose(1,2).flatten(2)` | Restore query-major layout and concatenate head features |
| 6 | `out_proj` | Mix head information at each query position |

The complete frozen module is available as [attention.py](code/src/core/attention.py). The demo is a separate [top-level file](code/src/attention.py), and [the tests](code/tests/test_attention.py) show supported/rejected behavior. These are captured source copies for reading/download; the application remains owned by its project directory.

### Run the three modes with one layer

From the project environment, this is a complete small component example:

```python
import torch
from core.attention import Attention

torch.manual_seed(7)
src = torch.randn(2, 5, 12)
tgt = torch.randn(2, 3, 12)
mask = torch.tensor([[1,1,1,0,0], [1,1,1,1,0]], dtype=torch.bool)
layer = Attention(12, 3).eval()
with torch.inference_mode():
    encoded = layer(src, attention_mask=mask)
    prefix = layer(src, attention_mask=mask, causal=True)
    retrieved = layer(src, tgt=tgt, attention_mask=mask)
    inspected, weights = layer(src, attention_mask=mask, need_weights=True)
print(encoded.shape, prefix.shape)  # both torch.Size([2, 5, 12])
print(retrieved.shape)              # torch.Size([2, 3, 12])
print(weights.shape)                # torch.Size([2, 3, 5, 5])
torch.testing.assert_close(encoded, inspected, atol=1e-6, rtol=1e-5)
```

This intentionally uses one layer only to inspect its API. A complete translation architecture will use separately owned modules in its encoder and decoder; these four calls do not assemble a Transformer.

### Explain what “efficient path” means here

With `fused_attn=True` and no request for weights, `_context` calls `torch.nn.functional.scaled_dot_product_attention`. With `fused_attn=False`, or when `need_weights=True`, it performs the score, mask, softmax and value operations explicitly. Asking for the weights therefore changes which implementation path is used, while preserving the intended numerical function when dropout is disabled.

![The demo compares explicit and SDPA outputs with matched parameters.](images/15-explicit-fused-parity.png)

To compare paths, initialize one layer, copy its state into the other and use identical inputs, masks and dtype. The tests compare float64 outputs and gradients with tight tolerances. The demo uses float32 and allows floating-point rounding differences. A maximum difference near `1e-7` is a numerical observation; it is not a speed measurement.

The SDPA backend selects a suitable kernel for the actual device and inputs. The current CPU check does not prove that GPU FlashAttention ran, that it would be faster, or that memory use improved on another device. Keep the implementation-choice claim separate from a measured hardware result. The [PyTorch API reference](source:sdpa) records the underlying call's semantics.

### Dropout requires an explicit training-state decision

Attention dropout randomly removes/rescales selected weights used to form context during training. In the explicit helper, the returned inspection weights are the normalized values before dropout. Thus a row of returned weights can sum to one even when the weights actually used for context have a different sum in that sampled call.

The functional SDPA API uses the supplied dropout probability directly. Our wrapper passes `self.attn_dropout` in training mode and `0.0` in evaluation mode. Simply calling `.eval()` would not automatically rewrite an unrelated raw functional call; the wrapper's argument choice is essential.

The tests use dropout probability one to make a diagnostic extreme: training context becomes zero and final output equals the output bias. In evaluation mode the same layer returns deterministic attention results. This is a test fixture, not a recommended training setting.

### Choose the right observation tool

Use the explicit path or `need_weights=True` when explaining scores and distributions. Use the default path when exercising the intended production-facing component interface. Preserve exact source/parameters and record what changed before comparing outputs. The [attention comparisons](project/comparisons/01-attention.md) collect these distinctions.

[[practice:interface-check]]

The forward path is clear. We next ask whether a loss can send meaningful gradients through that retrieval calculation, without pretending that one derivative check is a trained translator.
