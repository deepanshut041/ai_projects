## Lesson 17 — Assemble the complete translation model

**Goal:** reconstruct the actual reference configuration, route bilingual IDs to target logits, count its state and verify gradients throughout the complete model. Prerequisites: the [encoder](lessons/15-encoder-stack.md) and [decoder](lessons/16-decoder-stack.md) stacks. Work in your disposable copy's `examples/` using Lesson 2's compatible interpreter. [translation_model.py](examples/translation_model.py) owns model classes; [translation_data.py](examples/translation_data.py) owns vocabulary/collation; training and persistence remain separate modules.

### Configuration describes a specific function

`TransformerConfig` is a frozen dataclass. Frozen fields prevent accidental configuration assignment; model Parameters remain trainable. Its miniature defaults are:

| Configuration | Reference choice and consequence |
|---|---|
| Source/target vocabulary sizes | 10/12 in the fitted six-pair fixture; sizes and row meanings come from separate artifacts |
| `embedding_dimension`, `num_attention_heads` | 32/4; each head has width 8 |
| `encoder_depth`, `decoder_depth` | One independently parameterized layer each |
| `mlp_ratio` | Four; FFN expands 32→128→32 with GELU |
| `max_src_len`, `max_tgt_len` | 32 each; two fixed sine/cosine buffers |
| `src_pad_id`, `tgt_pad_id` | 0/1, respectively; never infer one from the other |
| Attention/hidden dropout | Both 0 in the controlled reference fit |
| `fused_attn` | True by default; false gives the explicit inspection path |

Validation requires positive even width, divisibility by positive head count, positive depths/expansion/capacities and padding IDs inside the correct vocabularies. This does not replace checking ID ranges, device/dtype or artifact identity at data boundaries. Matching vocabulary sizes cannot detect permuted token meanings.

The inspected large source uses width 512 and six layers per side; the reference reduces capacity while preserving its post-norm/GELU structure. Its separate embeddings feed an untied target head. Lookup and position are added without sqrt(D) scaling or an extra input-dropout module. Those differ from the original paper's shared vocabulary weights, input scaling/dropout and ReLU FFN. [Pinned structure](source:translation-code), [paper specification](source:attention-paper).

### Follow all axes through the actual fixture

The six English sequences contain BOS, three content IDs and EOS, so source shape is `[6,5]`. Each full French target contains BOS, four words and EOS; shifting gives decoder inputs and labels `[6,5]`. The accidental equality of these lengths is not an architectural requirement.

| Stage | Output | Meaning |
|---|---|---|
| Source embedding plus positions | `[6,5,32]` | Initial continuous source states |
| Encoder stack | `[6,5,32]` | Contextual memory; source position axis retained |
| Decoder embedding plus positions | `[6,5,32]` | Teacher-forced prefix states |
| Causal self/cross/FFN stack | `[6,5,32]` | Prefix-and-source-conditioned representations |
| `head = Linear(32,12)` | `[6,5,12]` | Unnormalized target-vocabulary scores |
| Flatten batch/position for loss | `[30,12]` plus labels`[30]` | Thirty valid categorical decisions, including EOS |

[[diagram:encoder-decoder]]

`Transformer.forward` derives the source-valid mask if absent, calls `encode`, then passes memory and the **same source mask** to `decode`. `decode` derives target-key validity if absent, applies every decoder layer to the same memory, and returns `head(hidden)`. During inference, `encode` runs once for the fixed request and `decode` receives a growing prefix. Returning only the last memory position or mixing target PAD into source masking changes this contract.

### Derive the complete parameter count

Let D=32,F=128, source vocabulary $V_s=10$ and target vocabulary $V_t=12$. One four-projection MHA has $4D^2+4D=4224$ parameters with biases. One FFN has $2DF+F+D=8352$. Each affine LayerNorm has 2 D=64 parameters. Therefore

$$\begin{aligned}
P_{\mathrm{encoder}}&=4224+8352+2(64)=12704,\\
P_{\mathrm{decoder}}&=2(4224)+8352+3(64)=16992,\\
P_{\mathrm{interface}}&=V_sD+V_tD+DV_t+V_t=1100,\\
P_{\mathrm{total}}&=12704+16992+1100=30796.
\end{aligned}$$

Two 32×32 position buffers add 2048 stored scalars but no learned parameters. FP32 parameter values alone occupy 123184 bytes; gradients, optimizer moments, buffers, saved activations and the runtime consume more. A full checkpoint is not a 30796×4-byte file. Adding one encoder layer adds 12704 parameters; increasing target capacity changes buffers/configuration but not these learned counts.

### Register and initialize the same ownership structure

`Transformer` owns `encodings`, an encoder `ModuleList`, a decoder `ModuleList`, `head`, and `config`. The lists create new modules per layer. These names determine state-dict keys such as `decoder.0.cross_attention.v_proj.weight`; rebuilding with unrelated key names requires an explicit migration.

`self.apply(_init_weights)` initializes all registered linear/embedding weights from Normal(0,.02²), linear biases to 0, LayerNorm scales to 1 and offsets to 0. It resets each embedding's padding row to zero. This is a source convention, not an assertion that every deep architecture should use this scale. A zero lookup row still acquires positions and contextual values; masks continue to matter.

In a scratch implementation, assemble the earlier `Embeddings`, `TransformerEncoderLayer` and `TransformerDecoderLayer` classes before implementing `Transformer.__init__/encode/decode/forward`. Preserve the established method arguments; then compare state names, parameter count and controlled outputs with the downloaded reference. Do not place model definitions inside a training notebook or replace your existing learner child with this completed source.

### A full forward/backward probe

Run the complete block from `examples/`:

```python
import torch
from translation_data import make_fixture
from translation_model import Transformer, TransformerConfig

torch.set_num_threads(1)
source, target, batch = make_fixture()
torch.manual_seed(7)
model = Transformer(TransformerConfig(src_vocab_size=source.vocab_size,
    tgt_vocab_size=target.vocab_size,src_pad_id=source.pad_token_id,
    tgt_pad_id=target.pad_token_id))
logits = model(batch['src_input_ids'],batch['tgt_input_ids'],
               batch['src_pad_mask'],batch['tgt_pad_mask'])
labels = batch['tgt_outputs']
loss = torch.nn.functional.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=-100)
loss.backward()
assert logits.shape == (6,5,12) and int(labels.ne(-100).sum()) == 30
assert sum(p.numel() for p in model.parameters()) == 30796
assert sum(b.numel() for b in model.buffers()) == 2048
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
for p in (model.encodings.src_embeddings.weight,
          model.encoder[0].enc_attention.v_proj.weight,
          model.decoder[0].cross_attention.v_proj.weight,model.head.weight):
    assert p.grad.abs().sum() > 0
print('logits:',tuple(logits.shape),'parameters:30796','targets:30')
```

A connected parameter can have zero gradient in a particular fixture; do not require every scalar derivative to be nonzero. For example, a common key-projection bias adds the same score offset across allowed keys and cancels in row softmax. We require finite gradients throughout and nonzero representative encoder, retrieval and head paths. Backward computes derivatives; it does not apply an optimizer update or prove useful translation.

Also run `python translation_model.py` and `python translation_checks.py TranslationChecks.test_gradients_reach_encoder_decoder_and_head`. Preserve complete-model future-target and source-padding isolation checks, not just component parity. Decode an unequal S/T case before accepting a shape contract that only exercised equal fixture lengths.

### Practice and notes

[[practice:translation-model-practice-1]]
[[practice:translation-model-state-transfer]]

In my notes, derive 30796 from the exact module owners, list buffers separately, and draw IDs→memory→prefix states→logits→labels with mask domains. Calculate how adding three French vocabulary entries changes only the target table/head, then explain why a position-buffer change can reject a full state load even when learned parameter count is unchanged. Continue with [translation updates](lessons/13-translation-training.md).
