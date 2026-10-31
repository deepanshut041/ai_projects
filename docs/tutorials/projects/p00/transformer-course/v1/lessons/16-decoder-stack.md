## Lesson 16 — Decoder stack, source retrieval and information flow

**Goal:** implement each decoder layer's three sublayers and verify which source/prefix information can reach a prediction. Prerequisites: [encoder memory](lessons/15-encoder-stack.md), shifted bilingual batches and post-norm residual blocks. Use your scratch copy's `examples/` with the compatible interpreter from Lesson 2. The authoritative reference methods are `TransformerDecoderLayer.forward` and `Transformer.decode` in [translation_model.py](examples/translation_model.py).

### The residual stream always belongs to the target queries

Let encoder memory be $M[B,S,D]$ and the current decoder representation $Z[B,T,D]$. With source-valid mask $K_s[B,S]$ and decoder-key validity $K_t[B,T]$, one layer performs

$$\begin{aligned}
U&=\operatorname{LN}_{d}\!\left(Z+\operatorname{Dropout}(\operatorname{MHA}_{d}(Z,K_t,\mathrm{causal}))\right),\\
V&=\operatorname{LN}_{c}\!\left(U+\operatorname{Dropout}(\operatorname{MHA}_{c}(M,U,K_s))\right),\\
Z'&=\operatorname{LN}_{f}\!\left(V+\operatorname{FFN}(V)\right).
\end{aligned}$$

All residual sums preserve `[B,T,D]`. Cross-attention transports information from S source positions into T query positions; it does not change the target sequence into an S-length array. Adding memory M directly to U is therefore wrong. Equal toy lengths can conceal that error.

The source-style block is post-norm with GELU inside its FFN. The FFN owns its output dropout, while self/cross-attention have their own residual dropout modules. Three distinct LayerNorms have independently learned scale/offset. Sharing one accidentally changes parameters and coupling. The original paper uses the same ordering with a ReLU FFN; swapping to pre-norm changes the function even when every shape stays legal.

[[diagram:decoder-retrieval]]

### Read the custom Attention arguments carefully

The reference call is `cross_attention(src, tgt, attention_mask=src_mask)`. In [translation_attention.py](examples/translation_attention.py), `src` supplies K and V; `tgt`, when present, supplies Q. Thus `cross_attention(memory, target_states, ...)` is correct. This argument order is not the three-tensor signature of PyTorch's built-in MHA. Copying another API's positional argument order can reverse retrieval without an obvious error when S=T.

For B=2,S=5,T=3,D=8,H=2, cross Q has shape `[2,2,3,4]`, K/V `[2,2,5,4]`, weights `[2,2,3,5]`, and the projected result `[2,3,8]`. Source-valid mask `[2,5]` expands over heads and queries. Cross-attention can read any real source key; no target-style triangle belongs on that rectangle. Each decoder layer applies its own K/V projections to the same final encoder memory, not a matching intermediate encoder layer.

### Causality follows the shifted prefix, including the current input

For target content `[a,b,c]`, the teacher-forced decoder input is `[BOS,a,b,c]` and labels are `[a,b,c,EOS]`. Query index 1 sees BOS and a, and predicts b. Allowing the diagonal is legal because the current input is an earlier target token. An unshifted input would reveal its label through the embedding/residual path even with a triangular mask.

The first two predictions cannot depend on input IDs at positions 2 or 3. This restriction must survive every decoder layer. Causal self-attention cannot introduce future prefix information; cross-attention reads target-independent source memory; per-token FFN and feature LayerNorm do not mix positions. Induction then preserves causality through depth. Encoder memory may depend on every source token because the complete source is available.

Padded target queries can still produce logits. Exclude their labels with `-100`; key masking and loss exclusion are different operations. An all-masked source is rejected at the model's encode boundary. Do not infer zero padded outputs from a padding embedding or a mask that only describes keys.

### Retrieval sends gradients into both streams

For one cross-attention head, $R=QK^T/\sqrt d$, $A=\operatorname{softmax}_{\mathrm{keys}}R$, and $O=AV$. Given upstream $G=\partial L/\partial O$,

$$\begin{aligned}
\nabla_V L&=A^TG,\qquad U_A=GV^T,\\
(\nabla_R L)_{ij}&=A_{ij}\left((U_A)_{ij}-\sum_k A_{ik}(U_A)_{ik}\right),\\
\nabla_K L&=(\nabla_R L)^TQ/\sqrt d.
\end{aligned}$$

The memory receives both key and value projection paths; the target stream receives the Q path and its residual path. Contributions from all decoder layers add into the shared memory graph. Detaching M allows decoder projection/head gradients while disconnecting the encoder, so a head-gradient check is insufficient.

For d=1,q=1, keys `[0,log3]`, values `[[2,0],[0,4]]`, weights are `[.25,.75]` and O=`[.5,3]`. With G=`[2,−1]`, value gradients are `[[.5,−.25],[1.5,−.75]]`; score gradients are `[1.5,−1.5]`. Even though the first key is zero, its score derivative need not be zero. The gradient depends on retrieval and on what is transported.

### Verify depth, rectangles and the encoder gradient path

Run this complete probe from `examples/`:

```python
import torch
from translation_model import Transformer, TransformerConfig

torch.set_num_threads(1)
torch.manual_seed(7)
model = Transformer(TransformerConfig(embedding_dimension=8,
    num_attention_heads=2, decoder_depth=2, fused_attn=False)).eval()
src = torch.tensor([[2,4,3,0,0]])
mask = src.ne(0)
prefix = torch.tensor([[2,5,6,7]])
changed = prefix.clone(); changed[:,2:] = 8
with torch.no_grad():
    original = model(src,prefix,mask)
    altered = model(src,changed,mask)
torch.testing.assert_close(original[:,:2],altered[:,:2],atol=1e-6,rtol=0)
memory = model.encode(src,mask)
memory.retain_grad()
logits = model.decode(memory,prefix,mask)
labels = torch.tensor([[5,6,7,3]])
torch.nn.functional.cross_entropy(logits.flatten(0,1),labels.flatten()).backward()
assert logits.shape == (1,4,12)
assert torch.isfinite(memory.grad).all() and memory.grad[mask].abs().sum() > 0
assert model.encodings.src_embeddings.weight.grad.abs().sum() > 0
print('rectangular decoding, future isolation and encoder gradients: passed')
```

The test establishes computation paths under this fixture, not useful source-conditioned translation. After training, a model can still learn to ignore source information; Lesson 20 adds behavioral source controls. Also run `python translation_checks.py TranslationChecks.test_full_model_future_target_isolation` and the encoder/decoder/head gradient test named in Lesson 17.

### Practice and personal notes

[[practice:decoder-stack-transfer]]
[[practice:decoder-stack-gradient-transfer]]

In my notes, draw one layer with the three target residual sums, label the actual `Attention(src,tgt)` roles, and calculate the two-stream gradient example. Mark where a future token could leak if a single later layer omitted its triangle. Explain why cached inference needs absolute prefix positions whereas the current reference simply recomputes the prefix.

Source structure: [pinned translation model](source:translation-code); [original decoder roles](source:attention-paper). Continue with [the complete model](lessons/12-translation-model.md).
