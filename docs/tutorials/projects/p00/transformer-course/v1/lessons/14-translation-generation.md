## Lesson 19 — Autoregressive generation, completion and decoding scores

**Goal:** implement the actual batched greedy path, explain its bounds and compare it with beam scoring without attributing a beam implementation to this source. Prerequisites: the [trained update protocol](lessons/13-translation-training.md), target shifting and rectangular retrieval. Use your scratch `examples/` folder and Lesson 2's compatible interpreter. `Transformer.inference` lives in [translation_model.py](examples/translation_model.py); it calls the same `encode/decode` methods used in training.

### At inference the next prefix must be constructed

The learned conditional distribution is $p(y_t\mid x,y_{<t})$. Teacher forcing supplied recorded previous tokens; generation supplies the tokens actually selected so far. Encode x once to memory M, initialize the decoder prefix with target BOS, run the decoder, take its **last position's logits**, choose a token, append it and repeat.

For target `je vois les chats`, the intended prefix trace is `BOS`→`BOS je`→`BOS je vois`→`BOS je vois les`→`BOS je vois les chats`→EOS. At prefix length 3, decoder row 2 predicts the next token; row 0 still predicts the first token. Selecting row 0 every time repeatedly answers the wrong question.

[[diagram:translation-generation]]

EOS is a learned output factor, not an oracle for the reference sentence length. Including it in training supervision permits learned stopping, but a hard capacity/budget remains necessary. Greedy argmax can operate on logits directly: softmax preserves their ordering. The reference masks the target PAD and BOS output logits to−∞ for active rows, so its greedy selection is over an explicitly restricted domain. EOS and UNK remain possible.

### Keep batch completion separate from batch termination

For B rows, maintain `finished[B]`. Suppose row 0 predicts EOS immediately while row 1 predicts token 5 then EOS. With target BOS 2,EOS 3,PAD 1, output is `[[2,3,1],[2,5,3]]`. The first EOS remains in its row; subsequent shared steps append PAD to that completed row. Stop when `finished.all()` or the budget ends. `any()` would truncate still-active rows.

`max_len` counts the **whole returned sequence**, including BOS and any EOS/PAD positions. A budget 4 permits at most three appended tokens. It must satisfy 2≤max_len≤max_tgt_len. The prefix stays `[B,length]`, even for B=1; unrestricted squeezing would destroy a predictable interface.

The reference's target PAD is 1, unlike source PAD 0. Generation derives prefix-key validity using target PAD; encoding and retrieval use the source mask. Finished rows do not attend to other batch rows. Masking PAD outputs for active rows prevents filler tokens from becoming meaningful prefix content before completion.

### Follow the actual control flow

Inside `Transformer.inference`, the source code performs:

1. Validate budget, target vocabulary ranges and distinct BOS/EOS/PAD IDs.
2. Remember `was_training`, switch to `eval()`, derive or accept the source-valid mask and call `encode` once.
3. Allocate BOS prefixes and `finished` on the source device.
4. Call `decode(memory,generated,source_mask)`, clone the final logits and forbid PAD/BOS.
5. Select argmax, replace selections for already finished rows with PAD, append, update finished flags and test all-completed.
6. Return the ID tensor; `finally` restores caller mode even if an exception occurs.

The method is decorated with `no_grad()`. That disables graph recording; eval independently disables dropout. Both matter for deterministic inference comparisons. Memory reuse here is per request, not a global object whose contents survive into a new sentence. The reference recomputes the entire target prefix each step and has **no decoder K/V cache**.

### A deterministic test isolates the loop from model quality

Run this complete probe from `examples/`. It changes only the local object's decoder method, so the fixture tests completion logic independently of learned weights.

```python
import types
import torch
from translation_model import Transformer,TransformerConfig

torch.set_num_threads(1)
model = Transformer(TransformerConfig(embedding_dimension=8,num_attention_heads=2))
model.train()
def controlled_decode(self,memory,ids,source_mask,target_mask=None):
    logits = torch.zeros(ids.shape[0],ids.shape[1],self.config.tgt_vocab_size,device=ids.device)
    logits[0,-1,3] = 10
    logits[1,-1,5 if ids.shape[1]==1 else 3] = 10
    return logits
model.decode = types.MethodType(controlled_decode,model)
result = model.inference(torch.tensor([[2,4,3],[2,5,3]]),max_len=8)
assert result.tolist() == [[2,3,1],[2,5,3]]
assert model.training  # caller mode restored
budgeted = Transformer(TransformerConfig(embedding_dimension=8,num_attention_heads=2))
with torch.no_grad():
    budgeted.head.weight.zero_(); budgeted.head.bias.zero_(); budgeted.head.bias[5]=10
assert budgeted.inference(torch.tensor([[2,4,3]]),max_len=4).tolist() == [[2,5,5,5]]
print('independent completion and BOS-inclusive budget: passed')
```

Also run `python translation_checks.py TranslationChecks.test_batched_generation_finishes_rows_independently` and `TranslationChecks.test_generation_budget_includes_bos`. These cases can pass for an untrained model and establish no linguistic quality. For a trained path, `python translation_train.py --steps 400 --seed 7` decodes training and held-out fixtures using each actual tokenizer's BOS/EOS IDs.

`Vocabulary.decode` skips BOS/PAD and stops at the first EOS. It can return a plausible string when the ID tensor exhausted its budget without EOS. Inspect IDs/finished status and label such a row truncated; removing structural symbols is not proof of successful completion.

### Greedy is a local choice, beam is a search policy

Suppose the first-token probabilities are P(A)=.55,P(B)=.45, with P(EOS|A)=.6 and P(EOS|B)=.95. Greedy chooses `[A,EOS]`, probability .33. `[B,EOS]` has probability .4275. A width-two beam can retain both prefixes and select the latter; width one reproduces the greedy preference here.

Keep prefix scores as summed log probabilities: `score(A,EOS)=log(.55)+log(.6)≈−1.108663`; the B path scores ≈ −.849801. Products of many small probabilities underflow, while log scores remain comparable. Beam width limits the candidates explored and does not guarantee the globally best sequence. Handle completed candidates separately; stopping on the first EOS found can discard a better completion.

A length penalty explicitly changes ranking. For a declared rule $s(y)=\log p(y)/((5+L)/6)^\alpha$, say whether L counts EOS/BOS. For L=2 and α=.6, the denominator is about 1.096903; for L=4 it is about 1.275425. Negative log scores become less negative when divided by a larger denominator, altering the short-sequence preference. This is a comparison exercise, not code implemented by `Transformer.inference`. The original paper reports beam 4/α=.6; the inspected source and miniature reference use greedy generation. [Original inference settings](source:attention-paper).

### Caching is a separate implementation experiment

The current prefix decoder repeatedly projects earlier target tokens and repeats their attention work. With growing prefix length, recomputation adds substantial work even though encoder memory is reused. A future K/V cache could preserve previous per-layer self-attention projections and source K/V, but it must record absolute positions, layer ownership, key validity, per-row finished state and request identity. The current `allowed_keys(...,causal=True)` requires equal query/key lengths; a one-query cached rectangle needs a different explicit alignment contract.

Do not claim speed from asymptotic reasoning alone. Compare cached and uncached final logits with dropout off, including unequal source lengths and independently finished rows, before measuring actual time/memory. Lesson 20 separates those runtime measurements from translation quality.

### Practice and notes

[[practice:translation-generation-practice-1]]
[[practice:translation-generation-search-transfer]]

In my notes, trace two rows with different EOS times and a budget shorter than one desired output. Compute the greedy/beam scores, state the ranking length convention, and explain which computations are reused by this exact method versus a proposed cache. Record a source/reference/generated-ID example, including whether EOS was observed. Continue with [translation evaluation](lessons/20-translation-evaluation.md).
