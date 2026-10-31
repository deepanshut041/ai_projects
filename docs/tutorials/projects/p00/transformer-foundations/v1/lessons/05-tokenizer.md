## Lesson 12 — Text encoding, tokenization and vocabulary identity

**Goal:** distinguish visible text, Unicode coordinates, bytes and model token IDs, then understand a reproducible subword tokenizer. Prerequisites: integer tensor contracts and split-safe preprocessing. Encoding is representation; embedding is the next learned map.

### Unicode is not a one-byte alphabet

A Python string is a sequence of Unicode code points, not UTF-8 bytes and not necessarily user-perceived grapheme clusters. The composed string `café` has 4 code points and UTF-8 bytes `[99,97,102,195,169]`, length5. The visible é also has a decomposed spelling `e` followed by a combining accent: visually similar text can have different code-point sequences and offsets.

```python
import unicodedata
s = 'café'
assert len(s) == 4 and list(s.encode('utf-8')) == [99,97,102,195,169]
decomposed = 'cafe\u0301'
assert len(decomposed) == 5
assert unicodedata.normalize('NFC', decomposed) == s
```

NFC performs canonical composition; NFKC additionally applies compatibility mappings that may remove meaningful distinctions. Case folding and whitespace changes are other transformations. Choose and record a policy rather than assuming normalization is always lossless. If an application later extracts spans from original text, preserve original text and an alignment map across any transformations.

### IDs refer to a saved vocabulary

A character tokenizer could assign one ID per observed training character, plus special IDs. A byte tokenizer can start with 256 ordinary byte values and round-trip valid UTF-8 by concatenating bytes before decoding. A word tokenizer produces shorter sequences but needs a policy for unseen words. Subword methods balance vocabulary size, sequence length and coverage.

Token ID37 has no inherent numeric closeness to ID38. Relabeling IDs together with the matching embedding/output rows preserves the represented model; changing IDs alone corrupts it. An unknown-token category may discard distinctions. Byte-level coverage avoids ordinary unknown byte values but does not automatically handle arbitrary special-token syntax or pretrained tokenizer compatibility.

The reference [CharacterVocabulary](examples/prerequisite_text.py) sorts training characters deterministically and persists its mapping. Special PAD/UNK/BOS/EOS values must be defined by the tokenizer contract and not inferred from an unrelated model.

[[diagram:prereq-text]]

### BPE learns repeated adjacent pieces

Byte-pair encoding starts with basic pieces, counts adjacent pairs within allowed boundaries, replaces a chosen frequent pair with a new ID, and repeats to the chosen vocabulary budget. In this educational byte tokenizer, ordinary bytes are 0–255, reserved special IDs are256–259, and the first learned merge is260.

On the document `abab`, adjacent byte pairs include `(97,98)` twice and `(98,97)` once. Merging `(97,98)` gives `[260,260]`; a second merge gives261. Decoding261 recursively yields bytes `97,98,97,98`, then the string `abab`. Keep documents separate during counts/merges: flattening `['a','b']` would invent a pair across a document boundary.

Training picks merges from frequencies with deterministic tie rules. Encoding new text follows the saved merge ranks; it does not retrain the tokenizer or greedily invent a new pair based on that input's frequency. Persist ordered merges, vocabulary/special mappings, normalization rules and training identity. Reloading an artifact must preserve the same ID sequence. The educational implementation departs from word-boundary/pattern handling in other BPE systems; it is not a drop-in GPT tokenizer.

```bash
python3 examples/byte_bpe.py
python3 examples/prerequisite_text.py
python3 -m unittest discover -s examples -p test_prerequisite_text.py
```

### Coordinates must stay explicit

In `The café is open.`, original code-point interval `[4,8)` extracts the word. Its UTF-8 byte interval is `[4,9)`. Reusing `[4,8)` as byte coordinates cuts the two-byte é in half and strict decoding fails. A token may span several characters; a character may occupy several bytes. Token count, byte count and character count are different.

Use half-open spans `[start,end)` with named coordinate systems. For QA, retain the source-character span used to extract an answer; a tokenizer offset map is an alignment tool, not permission to substitute token indices for characters. Round-trip and span extraction are separate tests.

[[scenario:coordinate-system]]
[[practice:encoding-transfer]]

### Notes and sources

Draw `text→normalization→bytes/pieces→IDs` and label every coordinate. Hand-run both BPE merges, state the document-boundary rule and explain why the frozen artifact must travel with a checkpoint. Compute the correct byte and code-point intervals for a changed sentence.

Sources: [Unicode normalization](source:unicode-normalization), [Sennrich et al. subword units](source:bpe-paper), [pinned minbpe source](source:minbpe). Original probes are in [prerequisite_text.py](examples/prerequisite_text.py) and [byte_bpe.py](examples/byte_bpe.py). Continue with [embeddings](lessons/13-embeddings.md).
