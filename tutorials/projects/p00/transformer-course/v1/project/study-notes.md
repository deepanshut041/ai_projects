# My Tutorial 1 study notes

Read the complete lesson, derive the result and predict a changed condition before running code. These fields are blank personal study space; they do not contain completed learner evidence. Save a personal copy before editing.

## Lesson 1 — Model specifications and task contracts

Study prompt: In my notes: draw the translation, MLM and QA input–visibility–target paths. Calculate the three-token likelihood, then change one conditional probability and predict its loss effect. State which source/paper choices I will implement and which observation would demonstrate each complete path. Continue with source and workspace identity.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 2 — Reproducible source and implementation workspace

Study prompt: In my notes: draw the model/helper/trainer/artifact boundaries and record the interpreter/source identity. Explain why changing tokenizer row meanings with identical tensor shapes breaks compatibility. Diagnose a local import failure, wrong loaded Torch version and a weights-only restart as separate problems. Continue with tensor interfaces and budgets.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 3 — Model tensor interfaces and resource budgets

Study prompt: In my notes: label every tensor in the unequal-length path, derive30796 from component terms and calculate parameter/score bytes separately. Predict changing H at fixed D and changing T at fixed model parameters. Identify a same-shape wrong-axis operation that a naive shape assertion would miss. Continue with translation likelihood.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 4 — Translation likelihood and supervised losses

Study prompt: In my notes: derive the conditional factorization, p−one_hot and the valid-token denominator. Work the two different record/token means, then change one target length and predict weighting. Draw which tensor rows are ignored and which learned modules must receive gradients. Continue with source/target tokenizers. API references: CE and the original paper.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 5 — Source and target tokenizers

Study prompt: In my notes: hand-encode the sentence, work all three BPE merges and label code-point/byte/token coordinates. Explain which transformations prevent exact-text decode. Draw the tokenizer→embedding/output row compatibility boundary and predict swapping PAD/UNK or ordinary rows. Continue with corpus ownership and visibility. Sources: Tokenizers API, pinned translation code.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 6 — Corpus splits and task visibility

Study prompt: In my notes: draw the three visibility/label domains and work the empty-line alignment counterexample. Define the split unit, tokenizer fitting data, sampled denominator and held-out question for each build. State which perturbation would reveal a leaked target, mixed-up pair or invalid QA window. Continue with translation embeddings.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 7 — Translation embeddings and vocabulary projection

Study prompt: Derive repeated-ID and output-table gradients in my notes. Draw separate source/target row meanings, mark the PAD lookup rule, and count tied versus untied unique parameters. Inspect `translation_model._init_weights` and the head declaration; verify the table and head identities without changing the learner repository. Continue to sinusoidal positions. References: Embedding API, original architecture.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 8 — Sinusoidal position implementation

Study prompt: Write the D=4 table and offset rotation in my notes. Mark buffer versus Parameter state, source versus target indices, capacity and the missing cache-offset interface. Predict the error from using only the newest token, then inspect the actual full-prefix generation path in Lesson 19. Continue to learned Q/K/V. Source: position representation.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 9 — Learned Q/K/V and scaled attention

Study prompt: In my notes, reproduce Q/K/V, scores, probabilities and context from the worked matrices. Define the row/column meaning for unequal lengths, then derive the two key/value gradient routes. Change a payload and a key separately before running a probe. Continue with multi-head implementation.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 10 — Multi-head attention implementation

Study prompt: Draw the two reshape/transposition directions in my notes using numbered features. Count parameters and rectangular score entries for a changed H, then explain the stored-weight slices of a loop implementation. Run the two named parity/layout checks and the applied gradient probes. Continue to attention masks.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 11 — Padding, causal and cross-attention masks

Study prompt: Draw the three tables and identify receiving rows, source columns, allowed meaning and ignored labels in my notes. Give the cached one-query counterexample. Run component isolation checks now, then whole-model controls after assembly. Continue with bilingual batches.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 12 — Bilingual batches and teacher-forced alignment

Study prompt: Draw a different unequal-length batch in my notes, including each query's next label, self/cross key domains and valid-token count. Explain why a valid EOS input can have an ignored label and why `.clone()` matters. Continue with position-wise FFNs.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 13 — Position-wise FFN and activation choice

Study prompt: Derive the output and backward rows in my notes, count a changed expansion ratio and mark both dropout locations. Recreate the independent-token perturbation on `FeedForward`, then explain where the model mixes positions. Continue with residuals and LayerNorm.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 14 — Residual and normalization blocks

Study prompt: Draw post-norm versus pre-norm graphs in my notes, derive the three-feature gradient and mark all dropout/norm sites and unique parameter owners. Predict a zero-sublayer output before calculating it. Continue with the encoder stack.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 15 — Encoder stack and contextual source memory

Study prompt: In my notes, draw two source layers with separate parameter boxes and one persistent validity mask. Work the changed-key calculation above, then write the induction argument for real-token invariance. Record which padded outputs are allowed to remain nonzero and where the decoder must exclude them. Explain why detaching or caching memory is safe for some inference calls but would cut the encoder's training gradient path.

**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 16 — Decoder stack, source retrieval and information flow

Study prompt: In my notes, draw one layer with the three target residual sums, label the actual `Attention(src,tgt)` roles, and calculate the two-stream gradient example. Mark where a future token could leak if a single later layer omitted its triangle. Explain why cached inference needs absolute prefix positions whereas the current reference simply recomputes the prefix.

**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 17 — Assemble the complete translation model

Study prompt: In my notes, derive 30796 from the exact module owners, list buffers separately, and draw IDs→memory→prefix states→logits→labels with mask domains. Calculate how adding three French vocabulary entries changes only the target table/head, then explain why a position-buffer change can reject a full state load even when learned parameter count is unchanged. Continue with translation updates.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 18 — Translation optimization and complete training updates

Study prompt: In my notes, derive the 2-versus 5 token weighting, calculate the first used/next rates, annotate where gradients remain after commit, and list every optimizer-group/schedule field recovery must preserve. Record actual loss, token accuracy, free-running outputs and protocol identity; do not paste an author's threshold as my result. Continue with autoregressive decoding.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 19 — Autoregressive generation, completion and decoding scores

Study prompt: In my notes, trace two rows with different EOS times and a budget shorter than one desired output. Compute the greedy/beam scores, state the ranking length convention, and explain which computations are reused by this exact method versus a proposed cache. Record a source/reference/generated-ID example, including whether EOS was observed. Continue with translation evaluation.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 20 — Translation evaluation, source controls and error analysis

Study prompt: In my notes, reproduce all n-gram counts/BP before reporting a score, state the fixture and final-test populations, and record one correctly conditioned/mismatched-source comparison with generated examples. Keep a failed or budget-truncated case instead of reporting only successes. Continue with complete-state recovery and Colab.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 21 — Full-state continuation, verified generations and Colab

Study prompt: In my notes, draw complete-update→write→verify→complete→rotate, calculate the rollback/presentation counters, and compare weight-only reload, exact resume and a new task's warm start. Record which local process/storage and hosted deletion checks I actually performed. Continue with BERT-to-RoBERTa design.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 22 — BERT-to-RoBERTa design and bidirectional objectives

Study prompt: In my study notes, derive the two-target loss above, draw encoder versus causal visibility, and name the output class axis for MLM and QA. List three recipe changes separately from architecture changes, then explain why a same-width miniature model cannot consume arbitrary official token IDs or weights. Next, byte tokens and learned positions make those identities concrete.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 23 — RoBERTa byte tokens, pair layout and learned positions

Study prompt: In my study notes, trace one multibyte character through bytes, IDs and repeated character offsets; derive table capacity from PAD identity; then predict an interior-padding mutation. Name which artifact details must match before reusing learned weights. Next, dynamic corruption separates input changes from supervised labels.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 24 — dynamic MLM corruption and selected-target supervision

Study prompt: In my study notes, draw original IDs, eligibility, selection, branch action, labels and attention visibility as separate rows. Calculate expected counts and zero-target probability for a short sentence; predict what restoring only the seed loses. Explain the course's .35 rate and ordinary-only replacement policy without attributing them to every official RoBERTa run. Next, the encoder and tied head implement the vocabulary distribution used by those labels.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 25 — bidirectional encoder and tied MLM head

Study prompt: In my study notes, trace one selected-position gradient through vocabulary bias, tied table, head transform and encoder. Derive the count when the decoder is untied, and explain how that differs from the inspected demonstration and official checkpoint mapping. Next, MLM pretraining and measurement connects this complete model to sampled presentations and counted updates.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 26 — MLM pretraining, counted updates and measurement

Study prompt: In my study notes, work unequal selected counts, draw the scheduler index before/after one update, and distinguish document sampling, input-token budget and supervised-token budget. Record the exact corruption/data protocol beside every result and compare it with the original RoBERTa recipe without claiming an official pretrained artifact. Next, backbone transfer starts a new supervised stage with compatible contextual weights.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 27 — backbone transfer and stage identity

Study prompt: In my study notes, draw the tensors copied and the state reset, derive the 66-parameter QA head, and explain why equal vocabulary size is insufficient. Write the separate acceptance evidence for transfer equality, downstream improvement and exact QA continuation. Next, offsets and windows construct valid supervised spans for that new task.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 28 — QA character offsets and overlapping context windows

Study prompt: In my study notes, trace `café` through character, byte, token and packed-window coordinates. Derive an overlap requirement from the actual longest answer, and show one ignored window. Explain why missing coverage is different from a no-answer label. Next, QA head and fine-tuning trains the endpoints that this alignment produces.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 29 — QA endpoint head, loss and fine-tuning

Study prompt: In my study notes, derive the uniform endpoint gradients, explain what the current denominator contains, and draw gradient flow from both heads into one contextual vector. Predict what freezing the backbone changes and identify the separate artifact checks needed for an official-tokenizer experiment. Next, constrained decoding and metrics turns those scores into a legal original-text answer.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 30 — constrained span decoding and QA metrics

Study prompt: In my study notes, list all valid pairs for the toy case, derive EM/F1 with duplicate tokens, and show a window-wide constant shift. Explain which guarantees come from offsets/constraints and which need learned evidence. Compare the own-format positive-only task with an official dataset/answerability profile. Next, integrated debugging and profiling tests these contracts across all three task paths.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 31 — integrated debugging and profiling

Study prompt: In my study notes, retain one concrete observation, competing hypotheses and the probe that separated them. Calculate a tensor budget independently from process memory, and identify the source/official-profile differences relevant to your failed case. Next, readiness and the GPT handoff assemble the complete task traces and remaining evidence.


**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**



## Lesson 32 — explain the complete models and prepare the Small GPT handoff

Study prompt: In my study notes, draw all three full paths, compare their loss denominators, and identify one unresolved implementation with its independent fixture. State what changes for Small GPT and what your present evidence does not establish. Use that concrete next action to continue rather than a generic completion label.

**My derivation and worked numbers:**


**My diagram, tensor axes and state owners:**


**My prediction, observed check and explanation:**


**Questions and next check:**


