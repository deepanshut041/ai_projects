## Lesson 28 — QA character offsets and overlapping context windows

**Goal:** map annotated answers in original text to inclusive token endpoints, construct question/context windows, and distinguish unavailable-window labels from a learned no-answer decision. Prerequisites: byte/character coordinates, pair layout and backbone transfer. Extractive QA predicts a substring of the supplied context; annotation alignment is part of the supervised task.

### Name every coordinate system

An answer annotation supplies original context string $c$, start character $a$ and nonempty answer text $r$. Its half-open interval is $[a,b)$ with $b=a+|r|$, and the consistency check is `c[a:b] == r`. Here “character” means Python Unicode code-point indexing, not UTF-8 byte indexing or displayed grapheme clusters. Preserve the original context rather than stripping, normalizing or rebuilding it from decoded tokens.

For `a café b`, `café` occupies character interval `[2,6)` and UTF-8 byte interval `[2,7)`. Its final `é` occupies character interval `[5,6)` but two bytes. The unmerged byte tokenizer therefore produces two token offsets `(5,6)`. A token offset maps to original characters; it is not necessarily one byte or a disjoint piece of the text.

The course trims leading whitespace from token answer offsets while preserving whitespace in encoded bytes. Empty intervals may arise for whitespace-only symbols. They are not valid answer endpoints. Question, special and PAD positions carry `None` offsets, explicitly excluding them from context-span ownership.

### Derive inclusive token endpoints from half-open characters

Let token $i$ have nonempty context interval $[l_i,r_i)$. It contains answer start $a$ when $l_i\leq a<r_i$. It contains the exclusive answer end $b$ when $l_i<b\leq r_i$. The asymmetric inequalities keep an answer beginning exactly at a boundary out of the preceding token.

For offsets `[(0,4),(4,8),(8,12)]` and answer `[4,8)`, start and end are both token 1. Token 0 ends where the answer begins, so it is not a valid start. Inclusive token span `[s,e]` then maps back to `context[offsets[s][0]:offsets[e][1]]`.

`character_span_to_tokens` selects the **first** matching start and **last** matching end. Repeated character offsets from multibyte symbols therefore keep the full matching token range when both byte symbols are present. It returns no span if the annotated interval is outside the retained context coverage or cannot be aligned to available endpoints. Record the answer's coordinate convention before calling it; shape checks cannot identify a byte start mistakenly supplied as a character start.

### Allocate question and context space explicitly

`qa_features` independently encodes a stripped question and the unchanged original context. Its pair is `<s> question </s> </s> context-window </s>`. With capacity $C$, question token count $Q$ and four specials, context budget is $W=C-Q-4$. It rejects $W\leq0$ rather than silently removing the question or truncating both streams.

For $C=16,Q=1$, $W=11$. Overlap $O=5$ gives window-start stride $W-O=6$, so context token starts are 0,6,12,… . Each feature preserves the original context offsets; its window start is a global context-token index, while head endpoints are local packed indices. The packed context prefix length is $Q+3$; the final separator occupies another position after the context.

[[diagram:qa-windows]]

Overlap is counted in **tokens**, not characters. A span crossing a boundary can appear completely in a later overlapping window. For fixed token windows, a length-$L$ answer can be guaranteed to fit some window for all start positions when $O\geq L-1$, subject to ordinary coverage at the sequence ends. Derive it: the worst start lies $W-O-1$ positions after a window start, requiring $(W-O-1)+(L-1)\leq W-1$. Answers longer than $W$ cannot fit under any overlap. Inspect actual answer token lengths after the frozen tokenizer; character length alone is insufficient.

### Label unavailable windows without inventing negative QA examples

If the answer maps into the current window, store inclusive start/end indices. Otherwise both labels are `-100`. An ignored feature is not labeled BOS, zero or another fake endpoint. It is unavailable supervision for this positive-answer task; it does not teach an abstention classifier.

The tiny QA builder retains answer-containing training features and requires at least one such feature per training question. Inference still evaluates all context windows. This filtering policy and its positive-only data differ from a full SQuAD 2.0 system with unanswerable examples and a null-score decision. [QA preprocessing](source:hf-qa), [source departures](project/source-map.md).

Byte-level windows introduce another explicit limitation: a cut can split symbols representing one Unicode character. Duplicate offsets can still identify original character coverage even though decoding that window's token bytes independently yields a replacement character. Offset extraction from the **original** context remains the authoritative answer rendering. The current mapper checks character-boundary coverage, not independent UTF-8 decodability of every window. A byte/character-aligned window policy would be a separate tested extension; do not claim this reference implements it.

### Probe an accented answer and a later-window answer

From `examples/` using the frozen interpreter:

```python
from roberta_data import ByteBPETokenizer, qa_features

tok = ByteBPETokenizer()
context = 'a café b'
features = qa_features('?', context, tok, context_length=24, overlap=2,
    answer_start=2, answer_text='café')
f = next(f for f in features if f.start_position != -100)
s, e = f.start_position, f.end_position
assert context[f.offsets[s][0]:f.offsets[e][1]] == 'café'
assert sum(span == (5,6) for span in f.offsets) == 2
assert all(span is None for span in f.offsets[:4])
long = 'alpha beta gamma delta'
windows = qa_features('?', long, tok, context_length=16, overlap=5,
    answer_start=long.index('gamma'), answer_text='gamma')
assert windows[0].start_position == windows[0].end_position == -100
valid = [w for w in windows if w.start_position != -100]
assert valid
for w in valid:
    assert long[w.offsets[w.start_position][0]:w.offsets[w.end_position][1]] == 'gamma'
assert all(w.input_ids[:4] == [0,68,2,2] for w in windows)
print('accented endpoints:', (s,e), 'answer-containing window starts:', [w.window_start for w in valid])
```

The first window of the long case cannot contain `gamma`; at least one later feature can. Labeling its first feature with a fabricated span would train the wrong task. Reconstructing `café` from a byte-index annotation would fail the coordinate contract.

### Recognize the corresponding official fast-tokenizer fields

A separate official-tokenizer profile uses `return_offsets_mapping=True`, `return_overflowing_tokens=True`, a declared `stride`, and `truncation='only_second'`. `sequence_ids(feature_index)` distinguishes question from context positions, while `overflow_to_sample_mapping` links features back to original record IDs. Keep only context-owned offsets as answer candidates and preserve every feature's record identity. These APIs expose the same alignment problem; they do not make the own-format course tokenizer compatible with official weights automatically. [Official QA guide](source:hf-qa).

### Changed-condition practice and notes

[[practice:qa-offsets-and-windows-practice-1]]

[[practice:qa-offsets-and-windows-transfer]]

In [my study notes](project/study-notes.md), trace `café` through character, byte, token and packed-window coordinates. Derive an overlap requirement from the actual longest answer, and show one ignored window. Explain why missing coverage is different from a no-answer label. Next, [QA head and fine-tuning](lessons/22-qa-head-and-fine-tuning.md) trains the endpoints that this alignment produces.
