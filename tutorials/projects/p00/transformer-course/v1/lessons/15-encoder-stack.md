## Lesson 15 — Encoder stack and contextual source memory

**Goal:** implement the whole source-side computation, explain what its memory contains, and verify padding isolation through multiple layers. Prerequisites: learned attention, positions and the post-norm blocks in [Lesson 14](lessons/11-transformer-blocks.md). Work in your disposable source copy's `examples/`; use the compatible PyTorch interpreter established in Lesson 2. The reference implementation is [translation_model.py](examples/translation_model.py), not a replacement for your existing learner repository.

### Memory contains contextual vectors, not predicted English tokens

Let source IDs be $I_s[B,S]$, with batch size B, padded source length S and feature width D. `encodings.forward_src` looks up a separate source table and adds fixed positional features, producing $X^{(0)}[B,S,D]$. This inspected-source design does **not** multiply its embeddings by sqrt(D). The original paper's scaling is a different declared choice, rather than a correction to apply silently.

An encoder layer reads every permitted source position, then transforms features at each position. For layer l, write

$$\begin{aligned}
U^{(l)}&=\operatorname{LN}_{l,a}\!\left(X^{(l-1)}+\operatorname{Dropout}(\operatorname{MHA}_l(X^{(l-1)},K_s))\right),\\
X^{(l)}&=\operatorname{LN}_{l,f}\!\left(U^{(l)}+\operatorname{FFN}_l(U^{(l)})\right),\\
M&=X^{(L_e)}\quad[B,S,D].
\end{aligned}$$

Here $K_s[B,S]$ is a Boolean **valid-key** mask. Both normalizations reduce D independently for each token; no pooling removes the S axis. `FeedForward` already includes its hidden/output dropout and GELU. Adding another output dropout outside it would change the inspected computation. The decoder later projects M into keys and values; the encoder does not choose English vocabulary IDs or produce one vector per sentence.

[[diagram:encoder-memory]]

### Full-source attention is appropriate to this task

At training and inference, the complete English source is available. An encoder query can use a later English token: `bank` can be disambiguated by a later phrase. This is legal conditioning, not target leakage. Applying a target-style triangle would change the source representation and restrict an operation that is meant to be bidirectional.

For valid keys `[True,True,False]`, the allowed matrix for three source queries is

$$A_{\mathrm{allowed}}=\begin{bmatrix}1&1&0\\1&1&0\\1&1&0\end{bmatrix}.$$

Take one query's scores `[0,log(3),large]` and values `[[2,0],[0,4],[100,100]]`. Masking the third score **before** softmax gives weights `[.25,.75,0]` and output `[.5,3]`; the arbitrary padded value is irrelevant. If the second key instead becomes invalid, only the first remains and the output becomes `[2,0]`. A mask change is a changed conditioning set, not a harmless storage rearrangement.

The mask describes keys, not query outputs. A padded query can read real keys, and its positional features/residual/LayerNorm bias can produce a nonzero vector. The reference does not promise zero padded memory rows. It promises that later operations exclude them as keys. Preserve the original validity mask when memory goes to cross-attention.

### Padding isolation must survive the entire stack

Suppose padded input IDs change while the original mask stays fixed. At layer one, every real query receives the same real Q/K/V and excludes the changed padded keys. Its attention output is therefore unchanged. Residual addition, feature normalization and the position-wise FFN do not mix positions, so its final real vector is unchanged. Apply the same reasoning inductively to every layer. Changed padded rows remain excluded at the next layer and at decoder cross-attention.

This argument assumes matching parameters, evaluation mode or matched randomness, the same real positions, and an unchanged supplied mask. Recomputing a mask from the deliberately changed PAD IDs would declare those positions real and invalidate the test. Dropout differences can also hide an otherwise correct invariance.

### Construct distinct layers and preserve the source API

The constructor uses `nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.encoder_depth)])`. Each call creates new attention, FFN and normalization parameters. `[one_layer] * depth` registers repeated references to the same layer and ties parameters across depth. Repeated architecture does not mean shared weights.

`Transformer.encode(src_ids, src_attention_mask=None)` derives a mask if absent, rejects a source with no usable key, creates embeddings/positions, then calls each layer with the same source mask. Its return remains `[B,S,D]`. The complete registered methods live in `translation_model.py`; implement the source route there in your scratch copy after the earlier blocks, rather than adding notebook-global memory.

Run this complete two-layer probe from `examples/`:

```python
import torch
from translation_model import Transformer, TransformerConfig

torch.set_num_threads(1)
torch.manual_seed(7)
model = Transformer(TransformerConfig(embedding_dimension=8,
    num_attention_heads=2, encoder_depth=2, fused_attn=False)).eval()
ids = torch.tensor([[2,4,3,0],[2,5,6,3]])
valid = ids.ne(0)
changed = ids.clone(); changed[0,-1] = 7
with torch.no_grad():
    memory = model.encode(ids, valid)
    altered = model.encode(changed, valid)
assert memory.shape == (2,4,8)
torch.testing.assert_close(memory[valid], altered[valid], atol=1e-6, rtol=0)
assert model.encoder[0].enc_attention.q_proj.weight is not model.encoder[1].enc_attention.q_proj.weight
assert 'encoder.1.enc_attention.q_proj.weight' in model.state_dict()
print('memory:', tuple(memory.shape), 'real-token padding isolation: passed')
```

Expected shape is `(2,4,8)` and all assertions pass. Also test all-masked input rejection, a source beyond `max_src_len`, and the full-model isolation test: `python translation_checks.py TranslationChecks.test_full_model_source_padding_isolation`. Encoder-only parity does not by itself verify the decoder uses its mask.

### Review practice and notes

[[practice:encoder-stack-transfer]]
[[practice:encoder-stack-sharing-transfer]]

In my notes, draw two source layers with separate parameter boxes and one persistent validity mask. Work the changed-key calculation above, then write the induction argument for real-token invariance. Record which padded outputs are allowed to remain nonzero and where the decoder must exclude them. Explain why detaching or caching memory is safe for some inference calls but would cut the encoder's training gradient path.

Source structure: [pinned translation implementation](source:translation-code); [original encoder architecture](source:attention-paper). The custom layer is not implicitly interchangeable with `nn.TransformerEncoderLayer` defaults. Continue with [decoder stacks and retrieval](lessons/16-decoder-stack.md).
