## Lesson 11 — Padding, causal and cross-attention masks

**Goal:** specify legal information for encoder self-attention, decoder self-attention and source cross-attention, then test isolation. Prerequisites: attention axes and teacher-forced likelihood. A valid loss denominator cannot repair information leakage in the forward graph.

[translation_attention.py](examples/translation_attention.py) uses **True = allowed source key**, matching functional SDPA. Other APIs can reverse Boolean meaning; establish semantics at each boundary. [Official contract](source:torch-sdpa).

### A padding mask describes source keys

Suppose two English samples become this batch after padding:

```text
sample0: [BOS] i see cats [EOS]
sample1: [BOS] cats [EOS] [PAD] [PAD]

key_mask:
[1,1,1,1,1]
[1,1,1,0,0]
```

A pad token makes the rectangular batch possible; it is not extra sentence information. Each query must give a padded **key column** zero weight. For sample one, the unmasked probability for the last two columns would read representations that arose only from padding. An embedding with a zero pad row is insufficient because positional encodings, projection biases and later operations can make its representation nonzero.

If keys have length `S`, begin with `[B,S]`. Insert dummy head and query axes to produce `[B,1,1,S]`; broadcast over heads and query rows. This creates no need to physically copy the mask with `repeat`. For encoder attention, queries also have length `S`. For cross-attention, the query length may be different.

### The causal mask protects a prediction from its answer

Imagine French decoder inputs `[BOS,je,vois,les]`. At the position holding `je`, the model is predicting `vois`. It may read `[BOS,je]`, but not the later input `vois`, which already contains the answer. A four-position causal table is

$$M_{\mathrm{causal}}=\begin{bmatrix}
1&0&0&0\\1&1&0&0\\1&1&1&0\\1&1&1&1
\end{bmatrix}.$$

The diagonal is allowed: the current decoder input predicts the *next* label. Position `i` may read key `j` when `j≤i`. That rule makes it possible to calculate all teacher-forced predictions in parallel while preserving the dependency on only a prefix.

Padding and causality answer separate questions and must be combined. If the last input is padding, its key column is blocked even when the triangular rule permits it. The allowed table is the logical intersection:

$$M_{b,i,j}=M_{\mathrm{key},b,j}\land(j\le i).$$

[[diagram:translation-masks]]

### Build the entire allowed table

The complete function in `translation_attention.py` is:

```python
def allowed_keys(key_mask, query_length, *, causal=False):
    if key_mask.ndim != 2 or key_mask.dtype != torch.bool:
        raise ValueError("key_mask must be a Boolean [batch, source_length] tensor")
    source_length = key_mask.shape[-1]
    allowed = key_mask[:, None, None, :].expand(-1, 1, query_length, -1)
    if causal:
        if query_length != source_length:
            raise ValueError("causal self-attention requires equal query/key lengths")
        history = torch.ones(query_length, source_length, dtype=torch.bool,
                             device=key_mask.device).tril()
        allowed = allowed & history[None, None]
    return allowed
```

`None` in tensor indexing inserts an axis; `expand(-1,1,T,-1)` keeps batch/source lengths and presents the same key mask to all `T` queries. `tril()` keeps the lower triangle. The self-attention contract requires matching query/key lengths when causality is requested; this function does not implement cached decoding with different query/key offsets.

Run a direct numeric check:

```python
import torch
from translation_attention import allowed_keys

mask = torch.tensor([[True, True, False]])
print(allowed_keys(mask, 3, causal=True)[0, 0].int().tolist())
```

Expected: `[[1,0,0],[1,1,0],[1,1,0]]`. The final row does not gain access to the pad column.

In the explicit path, illegal scores are replaced with negative infinity **before** softmax. Since `exp(-∞)=0`, they receive no mass and the allowed scores are renormalized. Masking probabilities *after* an ordinary softmax without renormalizing would leave the retained weights summing to less than one and reduce the context magnitude incorrectly.

A wholly blocked row has no valid probability distribution. The reference's explicit operation returns zero weights and zero context for that row, matching the tested efficient operation. Once `out_proj` is applied, its bias can still make the final output nonzero. We reject an entire empty source sequence in the complete translation model because it has no translation context.

### Cross-attention is a rectangular retrieval

The encoder turns English into contextual source vectors, called **memory**. The decoder's current French vectors ask questions of that memory:

$$Q=Y_{\mathrm{French}}W_Q,\quad
K=Z_{\mathrm{English}}W_K,\quad
V=Z_{\mathrm{English}}W_V.$$

With three French query positions and five English source positions, each head's score matrix is `[3,5]`. Rows are French positions; columns are English positions. After multiplying by English values, output length is three. The output updates French, so it follows the query length.

Pass the English padding mask to this computation. A French mask has the wrong key meaning and often the wrong shape. Cross-attention has no French-future table: all English words are already supplied before translating. In this task a triangle across English/French indices would incorrectly equate positions in different languages. Other tasks can define their own cross-attention visibility rules; this translation class deliberately rejects `causal=True` with a separate target.

The final interface handles all three cases:

```text
encoder_update = layer(english, attention_mask=english_mask)
decoder_update = layer(french, attention_mask=french_mask, causal=True)
cross_update = layer(english_memory, tgt=french, attention_mask=english_mask)
```

Each call uses the same *interface*, but the complete model creates separate learned attention modules for these jobs. Reusing one layer's weights for all three would be an additional architectural choice we are not making.

### Change illegal inputs and demand identical legal outputs

A mask test should check behavior, not only print a triangle. For causality, change a future input and compare the earlier outputs:

```python
import torch
from translation_attention import Attention

torch.manual_seed(7)
layer = Attention(8, 2).eval()
x = torch.randn(1, 4, 8)
changed = x.clone()
changed[:, 2:] += 500
before = layer(x, causal=True)
after = layer(changed, causal=True)
torch.testing.assert_close(before[:, :2], after[:, :2])
print("future positions cannot change earlier outputs")
```

Without the causal table this fails. To test padding, perturb only padded key vectors while preserving the mask and compare all valid receiving positions. To test cross-attention, use different French/English lengths and ensure its output uses the French length.

Run the independent cases:

```bash
python translation_checks.py TranslationChecks.test_causal_attention_future_isolation
python translation_checks.py TranslationChecks.test_padding_key_perturbation_does_not_affect_valid_queries
python translation_checks.py TranslationChecks.test_manual_and_sdpa_match_for_rectangular_mask
```

These checks use evaluation mode so dropout cannot introduce unrelated differences. A decoder can pass a triangle-shape check while still leaking answers through another layer; Lesson 17 repeats the future-change test against the whole model.

### Distinguish padded queries from padded keys

The mask above prevents *reading padded keys*. It does not necessarily zero a padded query's output. For a right-padded training target, a query at a padded position can read real prefix tokens and produce a vector. That is acceptable when its corresponding label is excluded from loss. There are therefore two obligations: the attention mask excludes illegal context, and the loss mask excludes targets we are not supervising. Neither replaces the other.

For an ordinary right-padded target, the last `EOS` input may itself have an ignored next label. It is still a valid input key. Ignore flags belong to labels, not automatically to input tokens. This detail becomes visible when we shift targets next.

### Cached causality requires absolute positions

The square table uses query/key indices measured from the same start. For a cache containing keys at absolute positions 0…4 and a new query at position 4, the legal row is `[1,1,1,1,1]`. Applying a fresh 1×5 lower triangle would allow only key 0. The general condition is key_position≤query_position, intersected with key validity. This source rejects rectangular causal calls rather than pretending that it implements a cache.

When adding a cache, maintain the position offset, key order, per-row finished state and source-mask identity together. A cached implementation must match full-prefix logits under evaluation with identical parameters; a faster call with a shifted position table is not equivalent.

### Design controls that can actually falsify a claim

Change future **target inputs**, preserving the earlier prefix and source, to test decoder causality. Change padded source payloads while preserving their mask to test key isolation. Change a real source token to verify that the decoder can use source information; invariance there would suggest an ignored source path or an insensitive fixture. Keep mode, RNG and positional addresses fixed so a comparison isolates the intended condition.

An all-blocked row has no legal categorical distribution. The explicit zero-context convention is useful for a component test, but complete translation rejects an entirely empty source. Do not infer that zero context makes a whole residual block or projected output zero.

[[practice:masks-practice-1]]
[[practice:masks-transfer]]
[[scenario:visibility]]

### Notes and implementation

Draw the three tables and identify receiving rows, source columns, allowed meaning and ignored labels in my notes. Give the cached one-query counterexample. Run component isolation checks now, then whole-model controls after assembly. Continue with [bilingual batches](lessons/10-bilingual-batches.md).
