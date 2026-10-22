# P01 completion and GPT-2 readiness

Use this as the exit record for required P01 M1–M3 and M5. Historical GPT-1/classification M4 and early KV-cache M6 remain optional. Actual learner state is P00-M1 not-started; the following rows are requirements, not passed learner gates.

Create `docs/p01-exit.md` in gpt-lab. For each row record status, source/config/data/tokenizer identity, exact command/explanation, actual result, evidence location and next action. Retain earlier failed results. A reference script passing is author verification; your implementation, explanation and hosted evidence are separate.

| Required gate | Evidence to provide | Lesson |
| --- | --- | --- |
| M1: entry and scope | Reviewed P00 exit evidence; independent child/hand-off snapshot; source and runtime pins; tiny config and explicit architecture departures | 1–2 |
| M1: data identity | Licensed/provenanced text, document split before windows/tokenizer training, raw/split/artifact hashes, no exact document overlap, frozen tokenizer compatibility/Unicode checks | 3–4 |
| M2: complete model | Causal self-attention, learned positions, pre-norm and final norm, GELU, tied head; forward/backward; independent attention value/gradient and full-model causal/PAD/bounds checks | 5–7 |
| M3: actual text learning | Fixed-batch diagnostic separate from held-out NLL/perplexity and unigram comparison; declared update/token budgets; finite gradients and inspected LR; final selected protocol before untouched-test evaluation | 7–10 |
| M3: generation | Seed/prompt/checkpoint/temperature/top-k/context/UTF-8 policies; several unselected samples; EOS and max-token bounds; no cross-request state | 8 |
| M3: local recovery | Fresh process parity for full model/Adam/RNG/sampler/schedule/progress; interrupted copy/training and corrupt-newest fallback; new/resume/identity guards; two recent plus best retained under budget | 9–10 |
| M3: Colab Free | Thin common-trainer notebook, exact source ZIP and dependencies, actual Free complete-model training, verified Drive saves and deleted-runtime restore/advance with measured rollback/resources | 11 |
| M5: explain and profile | Exact unique parameter count, storage lifetime ledger, scoped measured CPU profile, one diagnosed failure/change, architecture/source/limitations record and an explanation without reading code | 12 and practice |

## Mathematical exit checks

| Explain by doing | Required demonstration |
| --- | --- |
| Conditional probability, log loss and perplexity | Factor a three-token probability; compute mean NLL and explain its denominator and tokenizer dependence |
| Tensor/head geometry and causal masking | Map [B,T,D]→[B,H,T,d], compute one zero-score average and predict a future-input perturbation |
| LayerNorm/pre-norm/residuals | Normalize a two-feature vector; trace direct/sub-layer derivatives; distinguish post-norm and final norm |
| Learned positions and GELU | Trace repeated token/position gradients; explain Gaussian-CDF gating and exact versus tanh approximation |
| Softmax-cross-entropy gradients | Recover p−one-hot and valid-token mean scaling; check a selected gradient independently |
| AdamW, clipping and warmup/cosine | Calculate the first scalar moments/update; distinguish global gradient clipping; compute resumed global LR |
| Weighted evaluation/accumulation | Combine unequal token counts correctly; distinguish data RNG from dropout RNG |
| Sampling and finite context | Calculate a temperature change, specify top-k and EOS policies, explain context cropping/position reset |
| Parameters and memory | Derive 120832 default parameters; distinguish 4P weights from approximate 16P training-state bytes and BHT² intermediates |

Return to the first explanation/check you cannot reproduce. A full symbolic GPT Jacobian, exact GPT-1 reproduction and custom CUDA kernels are not prerequisites for the required P01 exit.

## Recorded author limits

The reference's local twenty-one tests, fresh-process dropout parity, two-string fitting, 300-update one-book run and notebook/source-export checks are in [reference runs](project/reference-runs.md). Generated text remains visibly weak/repetitive. The author has not evaluated the held-out test chapters or performed a deleted Colab runtime recovery. No learner or hosted milestone advances from this material.

## P02 handoff

After all required rows pass, create `docs/p02-handoff.md` with versioned source/tokenizer/config/data contracts, test commands/tolerances, recovery evidence, sample/error records and measured limits. P02 then covers GPT-2-specific tokenizer/preprocessing/config/weight-layout claims, model fidelity and required KV-cache parity/storage comparisons. Do not repeat pre-norm as if it were new; do not claim GPT-2 compatibility from a matching model name.

Optional cache work must establish absolute-position/mask correctness, full-prefix logits parity, multi-token append, bounds and fresh-request reset before profiling. Optional historical GPT-1/classification is a separate branch with its own normalization/pooling/objective evidence. Neither substitutes for required Free recovery or a learner explanation.

[Return to implementation](project/implementation.md) · [Begin Lesson 1](lessons/01-next-token.md)
