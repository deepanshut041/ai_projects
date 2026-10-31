## Lesson 22 — BERT-to-RoBERTa design and bidirectional objectives

**Goal:** distinguish a contextual encoder's architecture from its pretraining recipe, derive the masked-token objective, and specify what this miniature RoBERTa-style build actually implements. Prerequisites: Tutorial 0's attention/loss foundations and this course's visibility and data contracts. The second build reuses encoder mechanisms, but its targets and permitted information differ from translation.

### Contextual vectors differ from vocabulary rows

A token embedding $E[i]\in\mathbb R^D$ is a learned row associated with vocabulary ID $i$. The encoder output $h_t=f_\theta(\widetilde x)_t$ is a contextual vector that also depends on other permitted tokens, positions and all encoder layers. The same token ID for “bank” can enter with the same embedding in two sentences while its final contextual representation differs. A downstream head receives those contextual vectors, not a permanently meaningful dictionary of word senses.

With batch size $B$, input length $T$, width $D$ and vocabulary size $V$, the backbone outputs $H[B,T,D]$. The MLM head outputs logits $Z[B,T,V]$, classifying vocabulary identity separately at each position. Extractive QA later produces start/end logits $[B,T]$, classifying **positions**. At $V=341,T=64$, the two classification spaces have 341 and 64 classes respectively; they are not interchangeable merely because both use cross-entropy.

### Derive the objective and its visibility

Let $x$ be original token IDs, $M$ a sampled set of selected ordinary positions, and $\widetilde x$ the corrupted input. The selected-target loss is

$$
L_{\mathrm{MLM}}=-\frac{1}{|M|}\sum_{t\in M}\log p_\theta(x_t\mid\widetilde x),\qquad
p_\theta(j\mid\widetilde x,t)=\frac{e^{Z_{t,j}}}{\sum_{k=0}^{V-1}e^{Z_{t,k}}}.
$$

For selected target probabilities 0.7 and 0.2, the loss sum is $-\log0.7-\log0.2≈1.966113$ and the selected mean is $≈0.983056$. Unselected positions supply context but do not add independent vocabulary-classification terms. They can still receive gradients through attention when they influence a selected position.

This objective is not the causal likelihood $\prod_t p(x_t\mid x_{<t})$. Selected conditionals use both left and right context and depend on a sampled corruption process; multiplying them does not automatically define an ordinary left-to-right sequence likelihood. A fixed-corruption loss can measure this reconstruction experiment without being directly comparable to GPT perplexity.

For three real tokens, an encoder permits the all-ones query/key matrix; a causal decoder permits only the lower triangle. Padding still excludes invalid keys in both. A real token to the right of an MLM target is permitted context. The same access would leak future information for next-token generation. Selected tokens follow explicit corruption branches, including an intentional unchanged-input branch; that branch is part of the declared denoising recipe rather than an accidental replacement of every target with its answer.

[[diagram:roberta-objectives]]

### Separate the original designs from recipe changes

BERT pretraining combines a deeply bidirectional encoder, masked-token prediction and next-sentence prediction (NSP). Its input uses token, position and segment embeddings; the NSP classifier distinguishes adjacent versus sampled unrelated segment pairs. RoBERTa studies a revised training recipe including removal of NSP, dynamic corruption, different sequence construction, byte-level BPE, larger batches/more text and longer training. These are empirical choices about data/objective/optimization; they do not follow mathematically from the attention equation. [BERT](source:bert-paper), [RoBERTa](source:roberta-paper).

This course implements MLM, then extractive QA. It has no NSP classifier and no conversational generation objective. Pair separators later mark question/context boundaries without creating an NSP task. Our token and position embeddings are summed, normalized and dropped out; the tiny model omits a token-type embedding table. An official implementation may retain a token-type lookup with IDs fixed to zero, which still affects checkpoint mapping.

The inspected instructional source and the authored reference also differ. Our reference uses padding-aware cumulative positions, consistent artifact PAD IDs, corrected masks and a tied MLM decoder; the pinned demonstration uses simpler position/head choices. Read [the source map](project/source-map.md) with those corrections rather than treating either as an exact official pretrained-library checkpoint.

### Specify the miniature build before allocating it

`RobertaConfig` in [roberta_model.py](examples/roberta_model.py) owns vocabulary size, width, depth, heads, FFN ratio, context capacity, PAD identity, two dropout rates, LayerNorm epsilon and initialization scale. The default trainer resolves $V=341,D=32,L=2,H=4,F=4D=128,T\leq64$. It initializes linear/embedding weights with normal standard deviation 0.02, linear biases with zero, and LayerNorm scale/shift with one/zero. Head width is $D/H=8$.

`vocab_size=301` in the standalone configuration is a constructor default, not the trainer's actual artifact size. Always derive it from the tokenizer used for that run. Context length counts all input tokens including specials; it is not a word count. Zero default dropout is a deliberate CPU inspection profile rather than a claim about the original RoBERTa training recipe.

The tiny corpus is eight authored training documents plus QA training questions for vocabulary fitting; two validation documents are excluded. That is a small combined training fixture. It does not reproduce the source's Wikipedia/BookCorpus pipeline or the paper's broader training corpora and budget. Architecture execution, a local loss decrease and broad language understanding are separate evidence.

### Run an objective/interface probe

Work in the course `examples/` directory using the frozen interpreter from Lesson 2. This complete probe imports the unchanged model rather than writing into your implementation child.

```python
import torch
from roberta_model import RobertaConfig, RobertaForMaskedLM, RobertaForQuestionAnswering

torch.set_num_threads(1)
torch.manual_seed(10)
c = RobertaConfig(vocab_size=30, embedding_dimension=8,
    num_transformer_blocks=1, num_attention_heads=2, context_length=8)
mlm = RobertaForMaskedLM(c).eval()
qa = RobertaForQuestionAnswering(c).eval()
ids = torch.tensor([[0, 4, 8, 2]])
changed = ids.clone()
changed[0, 2] = 20
with torch.no_grad():
    original = mlm(ids)
    alternate = mlm(changed)
    spans = qa(ids)
assert original['hidden_states'].shape == (1, 4, 8)
assert original['logits'].shape == (1, 4, 30)
assert spans['start_logits'].shape == spans['end_logits'].shape == (1, 4)
assert (original['logits'][0, 1] - alternate['logits'][0, 1]).abs().max() > 1e-7
print('head width:', c.embedding_dimension // c.num_attention_heads)
print('MLM vocabulary axis:', original['logits'].shape[-1])
```

Expect head width 4 and vocabulary axis 30. The changed right-context token can alter the earlier masked-position scores under this seed. This verifies permitted information flow and interfaces, not trained semantic understanding. An accidental causal triangle would change the intended objective.

### Changed-condition practice and notes

[[practice:bert-to-roberta-practice-1]]

[[practice:bert-to-roberta-transfer]]

In [my study notes](project/study-notes.md), derive the two-target loss above, draw encoder versus causal visibility, and name the output class axis for MLM and QA. List three recipe changes separately from architecture changes, then explain why a same-width miniature model cannot consume arbitrary official token IDs or weights. Next, [byte tokens and learned positions](lessons/17-roberta-tokens-and-positions.md) make those identities concrete.
