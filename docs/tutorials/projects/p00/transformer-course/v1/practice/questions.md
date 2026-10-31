# Changed-case questions for both builds

Attempt a calculation, prediction or drawing before reading the corresponding worked answer. These questions target the actual reference contracts, not implementation-name recall. They do not record learner results or supply a mastery score. Return to the linked lesson when the changed condition exposes a gap.

## 1. Multiplication contracts before attention

**Question.** A query matrix has shape `[3,2]` and a key matrix has shape `[4,2]`. Why must the score product use the transpose of the key matrix? What shape is the result? Calculate the score for query `[1,2]` and key `[3,-1]`.

**Worked answer.** `Q Kᵀ` contracts the shared feature width 2, leaving three query rows and four key columns: `[3,4]`. The dot product is `1×3+2×(-1)=1`. `Q K` is undefined because its inner dimensions are 2 and 4. Review [Lesson 3 — Model tensor interfaces and resource budgets](lessons/03-tensors.md) and [Lesson 9 — Learned Q/K/V and scaled attention](lessons/07-qkv.md).

## 2. Softmax without changing its evidence

**Question.** Scores `[0,log(3)]` normalize to what probabilities? What happens if you add 1,000 to both scores?

**Worked answer.** Exponentials are `[1,3]`, so probabilities are `[0.25,0.75]`. Adding the same finite constant leaves the mathematical probabilities unchanged because the common factor cancels. A stable implementation subtracts the row maximum before exponentiating to avoid overflow. A large absolute logit is not a calibrated confidence by itself. Review [Lesson 4 — Translation likelihood and supervised losses](lessons/04-learning.md).

## 3. Heads change the contracted width

**Question.** With embedding width 24 and three heads, what are the head width and score scale denominator? Can you switch to five heads without changing width?

**Worked answer.** Head width is `24/3=8`; scores divide by `sqrt(8)`. Using `sqrt(24)` would change the probabilities. Five equal heads do not divide width 24, so the reshape contract is invalid. Review [Lesson 10 — Multi-head attention implementation](lessons/08-heads.md).

## 4. The rectangular cross-attention board

**Question.** A translation batch has source length 7 and target-prefix length 4. Which side supplies cross-attention Q, K and V? Is the score matrix square?

**Worked answer.** Target hidden states supply four query rows; encoded source memory supplies seven key/value rows. Per-head scores have shape `[4,7]`, weights multiply source values `[7,d]`, and context has shape `[4,d]`. The score matrix is rectangular. Source pad columns are blocked; the target causal triangle belongs to decoder self-attention, not this source-memory lookup. Review [Lesson 11 — Padding, causal and cross-attention masks](lessons/09-masks.md).

## 5. A low loss can still leak the target

**Question.** You used the full target `[BOS,a,b,EOS]` as both decoder input and labels. Decoder self-attention is causal. Is the setup correct? Write the intended input and labels.

**Worked answer.** It is incorrect because causal attention still permits the current position to read itself. The visible current token is also its label. Intended input is `[BOS,a,b]`; labels are `[a,b,EOS]`. Mask and shift are separate contracts. Review [Lesson 12 — Bilingual batches and teacher-forced alignment](lessons/10-bilingual-batches.md).

## 6. Padding in two vocabularies

**Question.** Source pad ID is 0 and target pad ID is 1. A collator uses `ids != 0` for both masks. What can go wrong even if shapes are correct?

**Worked answer.** Target pad ID 1 is incorrectly allowed as a key, and a real target token with ID 0 can be blocked. Target pad labels may also enter loss if the same wrong constant is reused. Each tokenizer owns its own special IDs; derive each mask and ignored label policy from the corresponding vocabulary. Review [Lesson 12 — Bilingual batches and teacher-forced alignment](lessons/10-bilingual-batches.md).

## 7. Sinusoids and learned positions are different state

**Question.** The translation position table is a registered sinusoidal buffer. RoBERTa positions are an embedding Parameter. Which receives optimizer gradients? What belongs in a checkpoint?

**Worked answer.** The learned position table receives gradients; the fixed sinusoidal buffer does not. A model state dictionary carries relevant parameters and registered buffers for either architecture. Exact continuation also needs optimizer/scheduler/RNG state. “Not trained” does not mean “not part of the model's state contract.” Review [Lesson 8 — Sinusoidal position implementation](lessons/08-translation-positions.md) and [Lesson 23 — RoBERTa byte tokens, pair layout and learned positions](lessons/17-roberta-tokens-and-positions.md).

## 8. Information access changes with the objective

**Question.** Changing a real token to the right changes an earlier RoBERTa masked-position prediction. Is this a bug? What if it changes an earlier translation decoder prediction?

**Worked answer.** Right-context access is intended for the bidirectional MLM encoder. For a causal decoder at an earlier target position, changing a future target token must not change that prediction in evaluation with the same prefix and source. A gold shifted label fed into the current decoder input can create leakage independently of the attention mask. Review [Lesson 22 — BERT-to-RoBERTa design and bidirectional objectives](lessons/16-bert-to-roberta.md) and [Lesson 11 — Padding, causal and cross-attention masks](lessons/09-masks.md).

## 9. Count the selected labels

**Question.** Out of 100 eligible MLM tokens, 15 were selected: 12 masked, 1 randomly replaced and 2 unchanged. What is the loss denominator? What label belongs at the random replacement?

**Worked answer.** The denominator is 15. The random replacement retains the original uncorrupted token as its label; the two unchanged selected inputs retain their original labels too. `-100` marks unselected and special/padding positions. The branch counts are one possible draw, not exact percentages guaranteed for every batch. Review [Lesson 24 — dynamic MLM corruption and selected-target supervision](lessons/18-dynamic-mlm-corruption.md).

## 10. Unequal microbatches need unequal mean weights

**Question.** Microbatch A has one target with loss 2. B has three targets with summed loss 3. Compute the correct group loss and the result of averaging the two per-microbatch means.

**Worked answer.** The target-weighted mean is `(2+3)/(1+3)=1.25`. The mean of means is `(2+3/3)/2=1.5`; it overweights A. Backpropagate each loss sum over the common group count, then perform one optimizer/scheduler update. Review [Lesson 26 — MLM pretraining, counted updates and measurement](lessons/20-mlm-training-and-counting.md).

## 11. Dynamic corruption can select nothing

**Question.** A short MLM batch selects zero ordinary targets. Should you select a separator token to avoid an empty mean? Should its scheduler advance?

**Worked answer.** Do not fabricate a target on an excluded special token. Record count zero and skip the update. The scheduler's completed-update count must not advance for a nonexistent update. A separate declared sampling policy could redraw a batch, but that policy changes the protocol and generator sequence. Review [Lesson 24 — dynamic MLM corruption and selected-target supervision](lessons/18-dynamic-mlm-corruption.md) and [Lesson 26 — MLM pretraining, counted updates and measurement](lessons/20-mlm-training-and-counting.md).

## 12. Tying is object identity

**Question.** You cloned the input embedding weights into the output decoder once. Is the table tied? What must be true for gradients to update one shared table?

**Worked answer.** Equal initial values are not tying. The decoder and embedding must reference the same Parameter object. Input-lookup and output-classification gradients then accumulate into that object, which the optimizer must include once. A padding row can receive a decoder gradient despite input embedding `padding_idx` blocking its lookup gradient. Review [Lesson 25 — bidirectional encoder and tied MLM head](lessons/19-roberta-encoder-and-mlm-head.md).

## 13. A character is not necessarily one byte

**Question.** In `café`, `é` occupies Python character interval `[3,4)` but two UTF-8 bytes. What offsets should the two unmerged byte tokens carry? Could byte indices be used directly to slice a Python string?

**Worked answer.** Both byte tokens carry character interval `(3,4)`. Byte positions 3 and 4 are not separate Python character positions. Directly using byte indices would misalign later characters or extend the answer. The tokenizer must preserve original character ownership through byte merging. Review [Lesson 23 — RoBERTa byte tokens, pair layout and learned positions](lessons/17-roberta-tokens-and-positions.md).

## 14. A half-open boundary belongs to one side

**Question.** Token offsets are `[(0,4),(4,8),(8,12)]`. An answer spans characters `[4,8)`. Which token endpoints should be supervised?

**Worked answer.** Start and end are both token 1. The start uses `left≤start<right`, so token 0 does not contain character 4. The exclusive end uses `left<end≤right`, so end 8 belongs to token 1 for endpoint mapping. Testing both sides inclusively can select the wrong adjacent token. Review [Lesson 28 — QA character offsets and overlapping context windows](lessons/21-qa-offsets-and-windows.md).

## 15. A missing window is not a missing answer

**Question.** A 64-token limit and 10-token question leave how many context tokens? The answer appears after that first context window. Why is `(0,0)` not an ignored target?

**Worked answer.** Four pair-special tokens leave `64−10−4=50` context tokens. Token 0 is a valid position class, so `(0,0)` trains the beginning token. An absent-window span uses explicit ignored endpoints; overflow windows must be checked for full-answer coverage. An answerable-only model does not learn to abstain from those ignored windows. Review [Lesson 28 — QA character offsets and overlapping context windows](lessons/21-qa-offsets-and-windows.md).

## 16. Two maxima need not form an answer

**Question.** Valid context positions 1, 2 and 3 have start scores `[5,1,9]` and end scores `[8,7,1]`. Independent maxima select what pair? With a two-token maximum span, which valid pair has the largest score sum?

**Worked answer.** Independent maxima produce start 3/end 1, a reversed span. Valid sums are `(1,1):13`, `(1,2):12`, `(2,2):8`, `(2,3):2`, `(3,3):10`; span 1–1 wins. Question/separator/padding positions must also be excluded. Review [Lesson 30 — constrained span decoding and QA metrics](lessons/23-span-decoding-and-metrics.md).

## 17. Repeated answer words still count

**Question.** Prediction is `red red blue`; gold is `red blue blue`. What are EM, precision, recall and F1 under multiset-token overlap?

**Worked answer.** EM is 0. Overlap is two tokens—one `red`, one `blue`—rather than three or the size of a unique-word set. Precision and recall are each `2/3`, so F1 is `2/3`. Report fractions or percentages consistently. Review [Lesson 30 — constrained span decoding and QA metrics](lessons/23-span-decoding-and-metrics.md).

## 18. The same checkpoint shape can hide a different task

**Question.** A new tokenizer has the same vocabulary size and a different byte-merge order. The embedding weight shapes match. Can an MLM backbone safely transfer to QA with that tokenizer? What else can silently change a resumed update?

**Worked answer.** No; token-row meanings can differ while shapes match. Transfer requires compatible tokenizer and architecture identity. Exact resume also needs the original corpus, optimizer/scheduler protocol, source semantics, RNG and runtime mapping. A different LambdaLR closure can change future rates even after scheduler state loads. Review [Lesson 29 — QA endpoint head, loss and fine-tuning](lessons/22-qa-head-and-fine-tuning.md) and [Lesson 21 — Full-state continuation, verified generations and Colab](lessons/15-checkpoints-and-colab.md).

## 19. A completed save is more than a file

**Question.** A new checkpoint generation contains half-written weights, while an older generation has valid persisted hashes and a completion record. Which generation should recovery use? When may older verified generations be removed?

**Worked answer.** Ignore the incomplete generation and recover the newest compatible verified one. Publish a completion marker only after verifying the persisted bytes/manifest. Retention runs after a new valid generation exists and keeps at least two verified generations for fallback. A directory timestamp or filename alone is not proof that its state is complete. Review [Lesson 21 — Full-state continuation, verified generations and Colab](lessons/15-checkpoints-and-colab.md).

## 20. State the next unproved claim

**Question.** The tiny QA model returns all eight training answers correctly, and a CPU fresh-process replay matches. What claims remain unproved before calling the learner's RoBERTa build ready? What transfers to GPT?

**Worked answer.** Held-out QA quality, broad MLM quality, learner implementation/explanation and hosted persistent fresh-runtime recovery remain separate gates. Perfect seen-fixture answers establish an overfit sanity check. GPT can reuse tested token/tensor/attention/optimizer/recovery ideas, but needs a causal next-token objective, target shift, decoder architecture and generation contracts. A miniature GPT is not a complete ChatGPT training or deployment system. Review [readiness](project/readiness.md) and [the model explanation](practice/explain-your-model.md).
