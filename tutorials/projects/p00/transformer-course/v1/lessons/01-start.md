## Lesson 1 — Model specifications and task contracts

**Goal:** define what the two model builds compute, what trains them and what counts as a complete implementation. Prerequisite: Tutorial 0’s mathematical and model foundations. Advanced Python is assumed. This course develops the model-specific mathematics and implementation; the [32-topic outline](project/course-outline.md) is the reading order, and [my study notes](project/study-notes.md) will follow the same topics.

### Three tasks require three different prediction contracts

Translation predicts target tokens given a source sequence and target prefix. A masked language model predicts selected original tokens from a corrupted, bidirectionally visible sequence. Extractive QA predicts two positions in a packed question/context sequence, then returns a span of original context text. These outputs are respectively vocabulary classes, vocabulary classes, and sequence positions. Similar tensor dimensions do not make their objectives interchangeable.

Let x be source text, y₁…yₜ target IDs including EOS, and θ learned parameters. Translation models

$$p_\theta(y\mid x)=\prod_{t=1}^{T}p_\theta(y_t\mid x,y_{<t}).$$

Teacher forcing supplies the observed earlier target tokens during training. If the correct-token probabilities for a three-token target are `.5,.25,.8`, its conditional probability is `.1`, total NLL is log10≈2.302585 and token mean NLL≈.767528. At inference, earlier target tokens are the model's own generated outputs. That distinction explains why a good teacher-forced loss alone does not establish correct generation.

For MLM, let C(x) be corrupted IDs and S the selected target positions. The sampled objective is

$$L_{MLM}=-\frac1{|S|}\sum_{i\in S}\log p_\theta(x_i\mid C(x)).$$

The encoder can use tokens on both sides. Selection S decides supervision; replacement decides the observed input. An unchanged selected token still has a label. A batch with |S|=0 needs an explicit skip/reduction policy. It is not an autoregressive sequence-likelihood model merely because its output head uses a vocabulary softmax.

For QA, hidden states produce start and end logits of width T, the packed sequence length. The reference trains two endpoint cross-entropies and combines them; decoding adds context-only, order and length constraints. Its source training denominator includes all packed positions, while the allowed decoded answer domain is narrower. Missing-window labels are ignored; this does not teach an unanswerable-question classifier.

### A complete build includes more than attention

| Build path | Required state and behavior |
|---|---|
| Translation | exact source/target vocabularies, collation/shift/masks, embeddings/positions, encoder and decoder stacks, vocabulary logits/loss/backward, optimizer/schedule, generation, evaluation and recovery |
| RoBERTa MLM | exact byte-token artifact and packing/position rules, dynamic corruption/RNG, bidirectional encoder, tied MLM head, selected-only loss/update/evaluation and compatible backbone export |
| QA fine-tuning | checked backbone/tokenizer identity, context offsets/windows/labels, new endpoint head/optimizer, endpoint losses, constrained span decoding, original-text extraction, EM/F1 and recovery |

A component forward smoke test and an end-to-end learning run answer different questions. A code file called `model.py` is not evidence that all these paths work. Each later topic gives an independent invariant or changed-input check in addition to the reference implementation.

[[diagram:task-contracts]]

### Paper, inspected source and miniature reference

The original Transformer paper specifies post-norm, ReLU FFNs, scaled embeddings and an Adam/inverse-square-root training recipe. The inspected video source uses some different choices: GELU, unscaled separate embeddings, AdamW/cosine and plain CE. This course keeps those source-shaped choices visible instead of silently calling them exact paper reproduction. RoBERTa has its own recipe and artifact details; a small byte-BPE encoder with similar blocks is not automatically compatible with official pretrained weights.

[The source map](project/source-map.md) records the five video/code routes, corrections and deliberate departures. We retain its pinned source/license/caption metadata while authoring original explanations and probes. The full original model source is available separately from prose; essential reasoning belongs here.

### Verify the existing forward contracts

Run this standalone probe from the course's `examples/` folder with the reference interpreter selected in Lesson 2:

```python
import torch
from translation_data import make_fixture
from translation_model import Transformer, TransformerConfig
source, target, batch = make_fixture()
torch.manual_seed(7)
model = Transformer(TransformerConfig(src_vocab_size=source.vocab_size,
                                      tgt_vocab_size=target.vocab_size))
logits = model(batch['src_input_ids'], batch['tgt_input_ids'],
               batch['src_pad_mask'], batch['tgt_pad_mask'])
assert logits.shape == (*batch['tgt_input_ids'].shape, target.vocab_size)
assert sum(p.numel() for p in model.parameters()) == 30796
assert torch.isfinite(logits).all()
```

This is a full untrained forward reference, not a request to overwrite your learner project. The existing foundations child still resumes `linear()`. Later bounded reference runs exercise learning, source-use controls and recovery separately.

[[practice:start-practice-1]]
[[practice:start-transfer]]

### Notes and stopping check

In my notes: draw the translation, MLM and QA input–visibility–target paths. Calculate the three-token likelihood, then change one conditional probability and predict its loss effect. State which source/paper choices I will implement and which observation would demonstrate each complete path. Continue with [source and workspace identity](lessons/02-workspace.md).
