## Lesson 23 — RoBERTa byte tokens, pair layout and learned positions

**Goal:** trace text into frozen byte-BPE IDs, distinguish character offsets from bytes, and derive padding-aware learned position addresses and capacities. Prerequisites: Tutorial 0's encoding/embedding concepts and Lesson 22's encoder contract. Vocabulary and position identity are part of the model function, not interchangeable preprocessing details.

### Own-format byte-BPE preserves a byte alphabet

[ByteBPETokenizer](examples/roberta_data.py) starts with 256 one-byte symbols and five reserved IDs. An ordinary byte value $b$ initially receives ID $b+5$. Training counts adjacent symbol pairs inside regex pretokenization chunks, weighted by their corpus frequencies. It chooses the most frequent pair, breaks ties lexicographically by byte tuples, merges non-overlapping occurrences left to right and repeats until its merge budget or exhaustion of pairs. Encoding replays the **saved merge order**; it never recomputes frequencies from the input being encoded.

For `café`, UTF-8 bytes are `[99,97,102,195,169]`; initial IDs are `[104,102,107,200,174]`. There are four Unicode code points and five bytes. With no merges, token offsets are `[(0,1),(1,2),(2,3),(3,4),(3,4)]`: both byte tokens for `é` belong to its original half-open character interval. Merging them changes token boundaries while preserving their character ownership.

The regex groups Unicode word, punctuation and whitespace chunks, including possible leading whitespace. Merges do not cross those chunk boundaries. The decoded byte stream retains whitespace; offset construction trims leading whitespace from each token's answer-coordinate interval. Consequently whitespace-only tokens may have empty offsets, and token offsets need not be a partition into disjoint intervals. Those distinctions matter when extracting an answer from the original string.

The artifact stores `kind='course-byte-bpe-v1'` and ordered byte-tuple merges; its SHA-256 fingerprint fixes that mapping. The trainer's 80 merges yield 341 vocabulary rows in this fixture. This pretokenizer/artifact is deliberately simpler than official RoBERTa byte-BPE, and model names/embedding details also differ. A byte alphabet supports unseen valid Unicode input; it does not make arbitrary generated single-byte tokens independently valid UTF-8. `decode` uses replacement characters for invalid byte sequences.

### Reserve IDs and specify every sequence boundary

| Meaning | Course ID | Required operation |
| --- | --- | --- |
| BOS / `<s>` | 0 | Begin single and paired encoder inputs |
| PAD | 1 | Fill short rows; exclude keys; keep reserved position row |
| EOS / `</s>` | 2 | End a text or separate paired segments |
| UNK | 3 | Reserved; complete byte coverage avoids ordinary unseen-character lookup failure |
| MASK | 4 | Explicit corruption replacement, not an ignored target |

Single text is `<s> text </s>`. A question/context pair is `<s> question </s> </s> context </s>`: four special tokens, with two adjacent separators at the boundary. For capacity 64 and question length 10, the context budget is $64-10-4=50$ tokens. Those are token counts after encoding, not word counts. Literal special-token spellings in ordinary text are encoded as text unless a specific helper, such as `fill_mask`, supplies structural handling.

The pinned official `roberta-base` artifact has 50,265 vocabulary entries and MASK ID 50,264; its first four special IDs match the small convention. Its configuration has 514 position rows and a token-type vocabulary of size one. Our training fixture has 341 vocabulary entries, MASK ID 4, 66 position rows and no token-type table. Those are inspected artifact values, not assumptions from moving library defaults. Equal special-number prefixes or equal table sizes cannot establish checkpoint compatibility. Train vocabulary on declared training material before packing and record its exact identity; changing merge ranks under old weights changes the function without changing tensor shapes.

### Derive padding-aware positions and table size

Let $r_{b,t}=\mathbf1[i_{b,t}\ne p]$ with PAD ID $p=1$. The actual function computes

$$P_{b,t}=p+r_{b,t}\sum_{u=0}^{t}r_{b,u}.$$

For input `[0,7,2,1,1]`, real flags are `[1,1,1,0,0]`, cumulative counts `[1,2,3,3,3]`, and position IDs `[2,3,4,1,1]`. Leading padding `[1,1,0,7,2]` gives `[1,1,2,3,4]`: the real tokens keep their addresses. All nonpad IDs, including BOS/EOS/MASK, increment the count.

For $C$ real positions, the maximum address is $C+p$. An embedding table therefore needs $C+p+1$ rows indexed through that maximum: at $C=64,p=1$, 66 rows. Row 0 is unused by this convention. The model also rejects a physical sequence width greater than `context_length`, even if many entries are PAD; available real positions alone do not bypass that shape guard.

[[diagram:roberta-positions]]

Input vectors are $X_{b,t}=\operatorname{Dropout}(\operatorname{LayerNorm}(E[i_{b,t}]+R[P_{b,t}]))$, with token/position tables of width $D$. This model has learned positions rather than the translation model's fixed sinusoids. Its position function derives validity from IDs, **not** an independently supplied attention mask. Changing an interior PAD ID to an ordinary ID while keeping a false key mask changes subsequent position addresses. That perturbation is not a clean key-padding invariance test. Keep real-token positions unchanged when isolating mask behavior.

`padding_idx=1` suppresses ordinary input-lookup gradients for that row. It does not make every padded output vector zero after normalization/projection, nor globally freeze the row when tied vocabulary scoring supplies another gradient path. Key masks and ignored supervised labels remain separate contracts.

### Check bytes, artifact round trip and positions

From `examples/` with the frozen interpreter:

```python
import torch
from roberta_data import ByteBPETokenizer
from roberta_model import position_ids, RobertaConfig, RobertaEmbeddings

plain = ByteBPETokenizer()
ids, offsets = plain.encode_with_offsets('café')
assert ids == [104, 102, 107, 200, 174]
assert offsets == [(0,1), (1,2), (2,3), (3,4), (3,4)]
assert plain.decode(ids) == 'café'
trained = ByteBPETokenizer.train(['café café', 'two  spaces'], merge_count=8)
restored = ByteBPETokenizer.from_dict(trained.to_dict())
assert restored.fingerprint() == trained.fingerprint()
assert restored.decode(restored.encode('🙂 café')) == '🙂 café'
x = torch.tensor([[0,7,2,1,1], [1,1,0,7,2]])
assert position_ids(x).tolist() == [[2,3,4,1,1], [1,1,2,3,4]]
c = RobertaConfig(vocab_size=30, embedding_dimension=8,
    num_transformer_blocks=1, num_attention_heads=2, context_length=5)
assert RobertaEmbeddings(c).position_embeddings.num_embeddings == 7
print(ids, offsets, position_ids(x).tolist())
```

The checks preserve exact original text and the same frozen artifact after reload. Inventing consecutive byte offsets for `é` would corrupt later QA supervision. Using `arange(T)` instead of this cumulative rule changes learned position rows and compatibility.

### Keep official-tokenizer experiments separate

The retained source profile inspected `FacebookAI/roberta-base` at revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`. In a separate optional online environment with `transformers`, acquire **tokenizer files only**, retain their digests/revision, and inspect `all_special_ids`, prefix-space policy and actual offsets. This optional recipe was not executed by offline source checks:

```text
from pathlib import Path
from transformers import RobertaTokenizerFast
revision = 'e2da8e2f811d1448a5b465c236feacd80ffbac7b'
tokenizer = RobertaTokenizerFast.from_pretrained(
    'FacebookAI/roberta-base', revision=revision)
tokenizer.save_pretrained(Path('/tmp/course-official-roberta-tokenizer'))
print(tokenizer.pad_token_id, tokenizer.mask_token_id, tokenizer.all_special_ids)
print(tokenizer('café', return_offsets_mapping=True))
```

This creates a separately declared source profile; it must not replace our tokenizer under existing weights. The online instructional pipeline uses libraries/datasets beyond the self-contained reference. [Official RoBERTa](source:hf-roberta), [tokenizer mechanics](source:tokenizers), [source differences](project/source-map.md).

### Changed-condition practice and notes

[[practice:roberta-tokens-and-positions-practice-1]]

[[practice:roberta-tokens-and-positions-transfer]]

In [my study notes](project/study-notes.md), trace one multibyte character through bytes, IDs and repeated character offsets; derive table capacity from PAD identity; then predict an interior-padding mutation. Name which artifact details must match before reusing learned weights. Next, [dynamic corruption](lessons/18-dynamic-mlm-corruption.md) separates input changes from supervised labels.
