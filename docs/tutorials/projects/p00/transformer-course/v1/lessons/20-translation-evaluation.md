## Lesson 20 — Translation evaluation, source controls and error analysis

**Goal:** evaluate conditional likelihood and free-running translations under a declared protocol, calculate BLEU's intermediate counts, and test useful source dependence separately from a successful fit. Prerequisites: [training](lessons/13-translation-training.md) and [generation](lessons/14-translation-generation.md). Use the source copy's `examples/` with Lesson 2's interpreter. [translation_train.py](examples/translation_train.py) owns evaluation/reporting and [translation_metrics.py](examples/translation_metrics.py) owns the metric arithmetic and optional SacreBLEU adapter.

### Write down which population the evidence concerns

The reference fits six sentence pairs. Its two held-out pairs recombine familiar words and relations: `i see dogs` and `you see cats`. Their exact strings differ from training records, but the vocabulary, domain and template are highly constrained. These are valuable debugging/composition probes; two successful sentences cannot estimate general English–French translation quality. There is no independent benchmark test split in this miniature experiment.

For a larger experiment, fit tokenizers on the permitted training material, deduplicate related records before assigning groups/splits, select settings on validation, and evaluate the frozen model once on the declared final test. Save corpus identity, language direction, preprocessing, tokenizer artifacts, checkpoint and decoding policy with every score. A random holdout of a combined corpus is not automatically the official WMT test. [Task protocol](source:wmt14).

### Token accuracy, NLL and generation measure different behavior

For valid teacher-forced targets, sum NLLs and counts over all batches, then divide once. If two batches have two/five labels and mean losses 2/1, the corpus mean is 9/7, not 1.5. Token accuracy similarly uses total correct divided by total labels. `evaluate` runs without graphs in evaluation mode and restores the caller's previous mode.

Perplexity is exp(mean **ordinary hard-target** NLL). For target probabilities `[.5,.25,.8]`, sequence NLL is log 10 ≈ 2.302585, token mean ≈ .767528 and perplexity ≈ 2.154435. The conditioning, token units and EOS/padding rules must match before comparing it. Exponentiating a smoothed objective is not that likelihood measure. Raw probabilities are also not calibrated merely because argmax accuracy is high.

Free generation uses its own preceding mistakes. `translation_report` decodes ID rows and reports exact string match plus the explicit fixture BLEU. Inspect EOS/budget completion from IDs when needed; decoded strings alone can hide truncation. Exact match is appropriate for these controlled targets but is too strict as a complete metric when several phrasings are correct. The fixture references already match the vocabulary's NFC/lowercase policy; broader reference preparation must be declared explicitly.

### Derive the metric before trusting a scalar

For each n-gram g in a sentence, clip proposed count by the reference count: $m_g=\min(c_g,r_g)$. Sum matched and proposed counts **across the corpus**, then calculate $p_n=M_n/C_n$. With one reference stream, let c be total candidate length and r total reference length:

$$\begin{aligned}
BP&=\begin{cases}0,&c=0,\\\min(1,\exp(1-r/c)),&c>0,\end{cases}\\
\mathrm{BLEU}_N&=100\,BP\,\exp\left(\frac1N\sum_{n=1}^{N}\log p_n\right).
\end{aligned}$$

For candidate `a a b c` and reference `a b b c`, unigram matches are 3/4: the two a occurrences receive only one a match. Bigrams match `a b` and `b c`, giving 2/3. No trigram matches, so unsmoothed four-order BLEU is 0 despite useful lower-order overlap. Two-order BLEU is $100\sqrt{(3/4)(2/3)}≈70.710678$, with BP=1. Counting every candidate word that appears anywhere in the reference would falsely give perfect unigram precision.

For candidate `a b c` against `a b c d`, one/two-gram precisions are 1, but c=3,r=4 gives BP=exp(−1/3)≈ .716531. Two-order BLEU is ≈ 71.653131. A perfect two-word candidate still has no third/four-order n-grams and gets 0 under this reference's fixed-four-order/no-smoothing rule. Effective-order and smoothing choices change that behavior and must be reported.

Run this standalone metric probe from `examples/`:

```python
import math
from translation_metrics import corpus_bleu_tokens

candidates = [['a','a','b','c']]
references = [['a','b','b','c']]
two = corpus_bleu_tokens(candidates,references,max_order=2)
four = corpus_bleu_tokens(candidates,references)
assert two['matches']==[3,2] and two['totals']==[4,3]
assert math.isclose(two['score'],100*math.sqrt(.5))
assert four['score']==0
short = corpus_bleu_tokens([['a','b','c']],[['a','b','c','d']],max_order=2)
assert math.isclose(short['score'],100*math.exp(-1/3))
print('two-order:',two['score'],'four-order:',four['score'],'short:',short['score'])
```

`corpus_bleu_tokens` returns counts, lengths, precisions, BP and `one-reference|tokenized|order:N|smooth:none`. It is an educational arithmetic metric, not percentage semantic correctness and not an average of sentence BLEUs. [Original metric](source:bleu-paper).

### A reproducible library score needs its signature

The optional `sacrebleu_corpus(predictions,references)` uses raw decoded strings, `BLEU(tokenize='13a',lowercase=False,smooth_method='exp')`, one nested reference stream, and returns the library's signature. This differs from the hand-tokenized unsmoothed fixture metric and from the pinned source's Hugging Face `evaluate` BLEU path. An optional dependency must be present in the recorded environment before using that adapter; its absence does not change the meaning of the core fixture result.

Save version/signature, references, casing/normalization and decoding settings. Beam width, length penalty and greedy restrictions alter candidates before scoring. Do not compare a normalized tiny-corpus number with an official case-sensitive published test result without their protocol differences. [SacreBLEU source/documentation](source:sacrebleu).

### Check behavior with correct and deliberately mismatched sources

A finite encoder gradient proves a graph path, not that the trained model uses it well. Keep target prefixes/labels fixed and replace the source with another valid sentence, moving its source mask too. Compare NLL and free-running outputs. Swapping source rows can reveal reliance on prefix/template regularities; an unchanged output may indicate ignored source, but ambiguous or equivalent sources also need consideration.

This complete controlled fixture probe trains the actual reference, then measures a mismatch:

```python
import torch
from translation_train import make_state,update,evaluate

torch.set_num_threads(1)
state = make_state(seed=7,steps=400)
for _ in range(400):
    update(state)
right = evaluate(state.model,[state.train_batch])
wrong = dict(state.train_batch)
wrong['src_input_ids'] = wrong['src_input_ids'].roll(1,0)
wrong['src_pad_mask'] = wrong['src_pad_mask'].roll(1,0)
mismatch = evaluate(state.model,[wrong])
assert mismatch['loss'] > right['loss']
print('correct source:',right,'mismatched source:',mismatch)
```

This establishes a behavior difference on the controlled pairs, not broad source grounding. Also inspect paired lexical changes, such as cats→dogs, and source-only perturbations with the same target prefix. Keep the future-target/padded-source invariance tests: permitted semantic dependence and forbidden information flow are complementary checks.

### Read errors and resource measurements together

| Observation | Falsifiable next check |
|---|---|
| Immediate EOS | Inspect EOS labels, target lengths and valid-token counts |
| Repetition until capacity | Check last-position selection, prefix update and learned EOS |
| Fluent-looking output with wrong meaning | Compare correct/mismatched source and verify cross-attention arguments/encoder gradients |
| Good teacher forcing, bad generation | Trace the first wrong generated token and its later prefix |
| Excellent tiny holdout, poor new examples | Inspect domain/templates, duplicates and held-out coverage |

Measure update time, generation time and peak process/device memory under fixed lengths, batch size, modes, thread count and warmup. Parameter bytes, allocator peak and whole-process RAM are different quantities. For the one-layer fixture,400 full-batch updates consume 2400 pair presentations,24000 nonpadding input-token presentations and 12000 supervised target labels. They are not unique data counts or GPU throughput evidence. An optimized attention kernel may reduce score materialization; measure the actual backend rather than equating theoretical quadratic storage with observed peak allocation.

### Practice and notes

[[practice:translation-evaluation-transfer]]
[[practice:translation-evaluation-source-transfer]]

In my notes, reproduce all n-gram counts/BP before reporting a score, state the fixture and final-test populations, and record one correctly conditioned/mismatched-source comparison with generated examples. Keep a failed or budget-truncated case instead of reporting only successes. Continue with [complete-state recovery and Colab](lessons/15-checkpoints-and-colab.md).
