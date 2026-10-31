## Lesson 5 — Source and target tokenizers

**Goal:** build a reproducible text-to-ID interface for two languages and understand how subword choices change model contracts. Prerequisites: Tutorial 0 encoding/BPE and this course's source/target vocabulary axes. Tokenization chooses rows; embeddings learn their continuous values.

### Separate vocabularies can have separate special IDs

The offline translation fixture uses [Vocabulary](examples/translation_data.py), an inspectable word/punctuation tokenizer. `words()` normalizes NFC, lowercases and splits with a word-or-punctuation regex. `Vocabulary.fit` sorts observed training pieces and prepends a special-token tuple. Source order is `[PAD],[UNK],[BOS],[EOS]`, while target order is `[UNK],[PAD],[BOS],[EOS]`. Hence source PAD=0 and target PAD=1; BOS/EOS are2/3 in these artifacts.

On training text `I like cats.`, normalized pieces are `['i','like','cats','.']`; sorted ordinary tokens are `['.','cats','i','like']`. Source IDs are BOS2, i6, like7, cats5, period4, EOS3. The target would have the same ordinary indices in this particular single-text example but a different meaning for IDs0/1. Do not infer identical mappings from an accidental matching suffix.

```python
from translation_data import Vocabulary, words
src = Vocabulary.fit(['I like cats.'],source=True)
tgt = Vocabulary.fit(['I like cats.'])
assert words('I like cats.') == ['i','like','cats','.']
assert src.encode('I like cats.') == [2,6,7,5,4,3]
assert src.pad_token_id == 0 and tgt.pad_token_id == 1
assert src.decode(src.encode('I like cats.')) == 'i like cats .'
```

This decode is normalized token text, not exact original typography/case. New words use the artifact's UNK. Length bounds include BOS/EOS; the encoder rejects overlength sequences instead of silently truncating a paired record. The full batch constructor later uses each tokenizer's actual PAD ID.

[[diagram:token-pipeline]]

### Bytes, code points and subword pieces remain distinct

`café` has4 composed code points and UTF-8 bytes `[99,97,102,195,169]`. A multibyte character can span byte-level pieces; decoding a whole valid sequence can round-trip even if a single piece cannot decode alone. NFC and lowercasing can change original coordinates, so translation's normalized word decoder is not an offset-aware QA tokenizer.

The original word-BPE arithmetic in [foundation_math.py](examples/foundation_math.py) remains available. For training frequencies `low:5, lower:2, lowest:1`, adjacent `(l,o)` and `(o,w)` each have count8; lexical tie-breaking selects `(l,o)`. After that merge, `(lo,w)` has count8 and wins. Then `(low,</w>)` count5 wins over `(low,e)` count3. The saved merges are `(l,o)`, `(lo,w)`, `(low,</w>)`; encoding `lower` yields `('low','e','r','</w>')`, while `low` becomes one full-word piece.

```python
from foundation_math import fit_word_bpe, encode_word_bpe, merge_pair
merges = fit_word_bpe({'low':5,'lower':2,'lowest':1},3)
assert merges == [('l','o'),('lo','w'),('low','</w>')]
assert encode_word_bpe('lower',merges) == ('low','e','r','</w>')
assert merge_pair(('a','a','a'),('a','a')) == ('aa','a')
```

A merge pass consumes nonoverlapping occurrences, even when pair counts see overlapping neighbors. Encoding applies saved merge rank, not new-input frequency training. This word-boundary format is pedagogical; it is not RoBERTa's byte-BPE artifact or WordPiece.

### WordPiece is a separate source-style exercise

The inspected translation route uses an English BERT tokenizer and a French-trained WordPiece tokenizer. `train_french_wordpiece(training_texts,output_path,vocab_size=...)` constructs a Tokenizers WordPiece model with UNK, whitespace/punctuation preprocessing, `##` continuation decoding and a postprocessor adding BOS/EOS. It trains on permitted training text, persists JSON and reloads the exact artifact.

Use that adapter only with the optional pinned Tokenizers dependency installed and recorded; the default miniature run is offline and does not download an English pretrained model. A pretrained English artifact must have its model/revision, special IDs and preprocessing identity documented. Its vocabulary width and normalization can differ from our word fixture. Replacing one tokenizer requires a new compatible embedding/model contract, not only a changed filename.

WordPiece and BPE both provide subword pieces, but their training/rank/scoring procedures and segmentation conventions differ. A `##` continuation marker is an artifact convention, not a real character inserted in the original source sentence. The native tokenizers perform their configured postprocessing; do not append BOS/EOS twice in a separate data function.

### Persist meaning, not only size

Record ordered token-to-ID mapping, special IDs, normalization/pretokenization, saved merges/model JSON, training-text identity and version/revision. A remapped artifact can have the same V while swapping embedding/head row meanings. A strict run identity should detect it before loading optimizer/model state. Keep source and target identities separate unless a declared joint-vocabulary architecture really shares them.

A larger subword vocabulary grows embedding/output parameters but can shorten sequences; a smaller one can increase quadratic attention work. The budget must use measured token-length distributions under the actual artifact, not the number of whitespace words in raw data.

[[practice:tokenization-practice-1]]
[[practice:tokenization-transfer]]

### Notes and implementation check

In my notes: hand-encode the sentence, work all three BPE merges and label code-point/byte/token coordinates. Explain which transformations prevent exact-text decode. Draw the tokenizer→embedding/output row compatibility boundary and predict swapping PAD/UNK or ordinary rows. Continue with [corpus ownership and visibility](lessons/06-objectives.md). Sources: [Tokenizers API](source:tokenizers), [pinned translation code](source:translation-code).
