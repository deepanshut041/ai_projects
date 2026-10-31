## Lesson 11 — Datasets, batching and supervision contracts

**Goal:** construct batches without changing the problem or leaking held-out information. Prerequisites: tensor axes, loss reduction and update order. Data contracts deserve the same precision as model equations.

### Define the question and split before fitting preprocessing

Training data supplies parameter gradients. Validation data supports model/hyperparameter selection. A final test split estimates performance after those choices are frozen. Repeatedly inspecting a test result to choose the next change turns it into a development signal.

Split by the unit that prevents leakage: a document, conversation, person or source group when related records would share information. Near duplicates across splits can make held-out performance misleading. Fit learned normalization statistics, vocabularies and tokenizer merges using the permitted training corpus; record any deliberately broader unsupervised pretraining corpus. A tokenizer fitted on all held-out text changes the experimental contract even if no labels were used.

A split manifest records record/group IDs and hashes, acquisition/preprocessing rules and seed. A seed alone cannot recover which data was used if the source corpus changes.

### Sampling defines what a mini-batch estimates

Uniform record sampling with a record mean estimates equal weight per record. Token-normalized sequence loss weights longer records more heavily. Both can be valid, but answer different questions. Shuffling is useful for order effects; it does not repair leakage. Length bucketing reduces padded work but changes batch composition and needs a reproducible order.

For deterministic continuation, preserve the actual shuffle order, cursor, epoch and generator state. Reconstructing only “epoch 3” may consume a different next batch. Worker/prefetch randomness adds state beyond a simple sequential sampler; the reference keeps that controlled rather than claiming exact arbitrary DataLoader recovery.

### Collation turns ragged records into tensors

For two ID sequences `[4,5,6]` and `[7]` with PAD=0, the batch is

$$I=\begin{bmatrix}4&5&6\\7&0&0\end{bmatrix},\quad
K=\begin{bmatrix}1&1&1\\1&0&0\end{bmatrix},\quad \ell=[3,1].$$

I is integer `[B,T]`; K marks valid positions, and ℓ preserves original lengths. An embedding turns I into floating `[B,T,D]`. A decoder can have different source and target lengths S,T; do not force equality just because the sample sentences are short.

[[diagram:data-roles]]

Keep three roles separate:

| Object | Meaning | Operation it controls |
|---|---|---|
| PAD vocabulary ID | a filler input symbol | lookup/collation; possibly an embedding padding row |
| ignored label, here −100 | no supervised answer at this output position | exclusion from the loss/reduction |
| attention mask | permitted query–key connections | score normalization/information flow |

The ignored-label sentinel is not a vocabulary ID and must never enter an embedding lookup. A padding embedding that begins at zero does not alone exclude that token from attention; projections/biases and positions can make its representation nonzero.

### Aggregate totals, not averages of averages

Suppose valid losses for the first record are `[1,2,3]` and the second `[4]`. Sum is10, valid count4, token mean2.5. Averaging record means gives `(2+4)/2=3`. A large padded array containing zeros cannot use its full element count as a denominator. Keep `(loss_sum,valid_count)` across batches and divide once.

```python
import torch
labels = torch.tensor([[4,5,6],[7,-100,-100]])
losses = torch.tensor([[1.,2.,3.],[4.,0.,0.]])
valid = labels.ne(-100)
total = losses[valid].sum()
count = valid.sum()
assert count.item() == 4
assert (total/count).item() == 2.5
```

A real-token count is not always identical to a supervised-target count: a BOS input may predict the first target, some tasks supervise only selected positions, and EOS may be included while padding is ignored. Derive the target shift before counting.

### Validate at the batch boundary

Assert dtype, device, vocabulary range, target range, lengths and mask shape before expensive model work. Check a short last batch and uneven sequences. A `.reshape` that flattens labels in a different order from logits creates a valid shape with incorrect supervision.

[[practice:data-transfer]]

In my notes: draw two ragged records, their padded IDs, valid-key mask and ignored labels. Calculate both record and token means and say which objective the training code chooses. List the state needed to reproduce the next batch.

The collator and totals are in [prerequisite_tensors.py](examples/prerequisite_tensors.py); independent tests include unequal lengths and all-ignored rejection. Source contract: [CrossEntropyLoss](source:cross-entropy). Continue with [encoding and tokenization](lessons/05-tokenizer.md).
