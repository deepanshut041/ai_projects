## Lesson 6 — Corpus splits and task visibility

**Goal:** define the information each task may use and the corpus identities that make evaluation meaningful. Prerequisites: tokenizer artifacts and supervised objective. A correct network can still train on leaked or misaligned data; architecture correctness and experimental validity need separate checks.

### Preserve paired records as one unit

A bilingual corpus is a sequence of aligned `(source,target)` records. The reference's `aligned_lines` uses `zip_longest`, detects unequal file lengths and filters an empty pair together. Plain `zip` silently drops the extra side. Independently removing empty lines or overlength sentences can pair a source with a different translation while both files still look valid.

If sources are `['one','', 'three']` and targets `['un','deux','trois']`, filtering source first pairs `three` with `deux`. Paired filtering instead retains `(one,un)` and `(three,trois)`. Record original pair IDs and preprocessing decisions so a later shuffle or rebuild cannot change alignment unnoticed.

Fit learned tokenization/normalization using the permitted training corpus. Split by document/group and check duplicates when closely related sentences could share information. Development data chooses settings; a final test set estimates quality after choices are frozen. A randomly selected holdout is not automatically a named WMT test set.

The offline trainer has six training pairs and two held-out combinations under one small vocabulary. Training fit and held-out compositional behavior are useful diagnostics; neither estimates large-corpus translation quality. The source route has different data/tokenization/scale, recorded in [the source map](project/source-map.md).

### Visibility and target ownership differ by task

| Task | Visible model input | Supervised output |
|---|---|---|
| Translation | full real source and shifted causal target prefix | next target ID, including EOS |
| MLM | corrupted sequence, with real tokens visible bidirectionally | original ID only at selected positions |
| QA | packed question and context, bidirectionally visible | start/end positions for available answer windows |

[[diagram:objective-masks]]


For translation, an encoder source position can see later source tokens because the complete source is supplied. A decoder's prediction may not see later target tokens. Cross-attention queries come from the decoder but keys/values cover real source positions; applying a target triangle to those source columns changes the task.

For MLM, the encoder sees both neighbors around a masked input. Selected unchanged tokens still contribute supervision under the corruption scheme. Dynamic corruption changes the observed input/selected labels; it must not replace an original label with the random replacement ID. MLM accuracy over all unselected tokens would mostly reward positions that were never targets.

For QA, the question can influence context representations, but an answer is extracted from original context. A context-only decoding constraint is different from the endpoint training CE's actual normalization domain. A missing answer in a truncated window is not proof the original question is unanswerable; our reference ignores that window's unavailable target rather than using it as a SQuAD2 negative.

### Data boundaries also determine loss weighting

Longer translation targets contribute more terms under a token mean. MLM uses selected-target counts, which vary with stochastic corruption. QA uses endpoint labels/windows; overlapping windows can repeat supervision for one original record. All are legitimate choices only when named. Do not call every returned value a comparable “average loss.”

Persist corpus/split IDs and hashes, tokenizer fitting corpus, preprocessing version and sampling/bucketing rules. A seed does not identify data whose contents later change. The recovery identity includes such artifacts; exact continuation also needs actual next-record/random state.

### A tiny paired-record check

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from translation_data import aligned_lines
with TemporaryDirectory() as folder:
    src, tgt = Path(folder)/'src.txt', Path(folder)/'tgt.txt'
    src.write_text('one\n\nthree\n')
    tgt.write_text('un\ndeux\ntrois\n')
    assert list(aligned_lines(src,tgt)) == [('one','un'),('three','trois')]
    tgt.write_text('un\ndeux\n')
    try:
        list(aligned_lines(src,tgt))
    except ValueError:
        pass
    else:
        raise AssertionError('Misaligned parallel files were accepted')
```

Run from `examples/`; the temporary files do not alter a real dataset. A passing check establishes this collation/preprocessing behavior, not the quality of unknown corpus annotations. Later checks perturb future targets, padded source content and question/context ownership to test information flow directly.

[[practice:objectives-practice-1]]
[[practice:objectives-transfer]]

### Notes and checks

In my notes: draw the three visibility/label domains and work the empty-line alignment counterexample. Define the split unit, tokenizer fitting data, sampled denominator and held-out question for each build. State which perturbation would reveal a leaked target, mixed-up pair or invalid QA window. Continue with [translation embeddings](lessons/07-translation-embeddings.md).
