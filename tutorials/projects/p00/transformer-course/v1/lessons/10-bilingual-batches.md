## Lesson 12 — Bilingual batches and teacher-forced alignment

**Goal:** build the exact source, decoder-input, label and mask tensors for unequal-length examples. Prerequisites: separate tokenizers, conditional likelihood and the three attention masks. A batch can have plausible shapes while its labels are shifted incorrectly.

### Derive prediction slots from a complete target

A full target `[BOS,a,b,EOS]` creates decoder inputs `[BOS,a,b]` and labels `[a,b,EOS]`. Position 0 predicts a from BOS and source memory; position 1 predicts b from BOS/a; position 2 predicts EOS from the complete content prefix. EOS is supervised. BOS is supplied as an input rather than predicted in this objective.

The reference [collate_translation](examples/translation_data.py) first right-pads complete targets and then slices. This order leaves the shorter example's EOS in the decoder input when another row is longer. Its corresponding next label is padding and is ignored. EOS remains a valid input key: validity of an input and validity of its next label are different facts.

[[diagram:target-shift]]

### Work a ragged batch completely

Use source PAD=0, target PAD=1, BOS=2, EOS=3, and arbitrary ordinary IDs 4/5/6/7. Two records are:

```text
source A [2,4,5,3]     full target A [2,6,7,3]
source B [2,5,3]       full target B [2,7,3]

padded source:        source key validity:
[2,4,5,3]             [1,1,1,1]
[2,5,3,0]             [1,1,1,0]

decoder input:        target key validity:       labels:
[2,6,7]               [1,1,1]                    [6,7,3]
[2,7,3]               [1,1,1]                    [7,3,-100]
```

There are five supervised token slots, not six. Both decoder rows have three valid input keys, but causality still prevents early queries from reading later keys. The shorter row's last EOS query can produce a hidden vector without contributing loss. For target B at query 1, legal self-attention keys are BOS and token 7; cross-attention can read all three real source tokens.

```python
import torch
from translation_data import collate_translation
records = [dict(src_ids=[2,4,5,3],tgt_ids=[2,6,7,3]),
           dict(src_ids=[2,5,3],tgt_ids=[2,7,3])]
batch = collate_translation(records,src_pad_id=0,tgt_pad_id=1)
assert batch['src_input_ids'].tolist() == [[2,4,5,3],[2,5,3,0]]
assert batch['tgt_input_ids'].tolist() == [[2,6,7],[2,7,3]]
assert batch['tgt_outputs'].tolist() == [[6,7,3],[7,3,-100]]
assert batch['tgt_pad_mask'].all()
assert (batch['tgt_outputs'] != -100).sum().item() == 5
assert batch['tgt_outputs'].dtype == torch.long
```

### Keep storage ownership explicit

The source clones input and label slices before replacing target PAD labels with −100. A view that shares storage with the padded target can make an in-place label replacement corrupt input IDs. Negative ignore values belong to supervision only: sending −100 into an embedding lookup is invalid.

The five returned fields are `src_input_ids`/`src_pad_mask` [B,S], `tgt_input_ids`/`tgt_pad_mask` [B,T], and `tgt_outputs` [B,T]. IDs and labels are integer tensors; masks are Boolean. Move the complete batch to the model device while retaining those dtypes. Padding both languages to one common length wastes work and has no mathematical requirement: S and T are independent.

### Treat lengths and denominators as data contracts

Capacity includes special tokens. With target capacity 6, four ordinary tokens plus BOS/EOS fit; five ordinary tokens do not. The reference rejects overlong pairs through `encode_pairs`. Silently truncating one language can remove a meaning or EOS, changing the objective. A deliberate chunking policy must preserve alignment and declare its evaluation meaning.

Length bucketing can reduce padding but changes batch/sampler state that recovery must preserve. Valid counts can differ across micro-batches. Suppose one contains losses summing to 6 over three tokens and another sums to 2 over one token. Backpropagating each sum divided by the shared count 4 gives the true token-mean update. If counts instead differ as 3 and 2, the denominator is 5; averaging two batch means gives another weighting. A pad-heavy batch must not acquire extra weight solely from its rectangular size.

### Check behavior beyond printed tensors

Run `python translation_checks.py TranslationChecks.test_collator_shift_ignore_and_no_alias` and the loss/unequal-accumulation checks. Deliberately change an ignored logit row and verify unchanged loss; change a gold EOS row and expect a change. Keep source and target vocabulary identity with the batch artifact. The larger WordPiece adapter from Lesson 5 must return the same five tensor contracts while using its saved special IDs.

[[practice:bilingual-batches-practice-1]]
[[practice:bilingual-batches-transfer]]

### Notes and implementation

Draw a different unequal-length batch in my notes, including each query's next label, self/cross key domains and valid-token count. Explain why a valid EOS input can have an ignored label and why `.clone()` matters. Continue with [position-wise FFNs](lessons/13-translation-ffn.md).
