# Local reference results

Checked 2026-10-12: Python 3.12, PyTorch 2.14.1 CPU, NumPy 2.5.4, one training thread. Authored reference evidence is separate from your implementation or understanding. The selected verification record identifies actual source hashes.

From examples/, with the Lesson 2 interpreter:

```bash
python -m unittest discover -s . -p test_foundation_math.py -v
python translation_checks.py
python roberta_checks.py
python course_recovery_checks.py
python course_session_checks.py
python translation_train.py --steps 400
python roberta_train.py --steps 120
```

| Checks | Scope |
| --- | --- |
| 15 foundation checks plus C++ known answers | Matrix orientation, stable NLL, central differences, two Adam steps, Unicode, toy BPE, masks and split guards |
| 22 translation checks | Exact attention, head packing, rectangular efficient parity, behavioral masks, shifts, gradients, weighted accumulation, EOS and BLEU |
| 25 RoBERTa checks | Byte/character offsets, corruption branches, padding/positions, tied head, counting, transfer, QA windows/spans and CPU device transfer |
| Nine session/export checks | Explicit new/resume, all three tasks, stable source/schedule identity, budget refusal, failed-save stop, accounting and cleared notebook export |
| Three fresh-process recovery comparisons | Model/optimizer/schedule/progress/RNG continuation; corrupt/partial fallback, incompatible identity and retention |
| Optional WordPiece and SacreBLEU | Actual train/save/reload, normalization/special-token adapter and reproducible metric signature; no WMT run |

## Translation fixture

30,796 parameters, width 32, four heads, one encoder and one decoder layer. Seed 7, 400 updates, six training pairs with 30 supervised labels per update. NLL: 2.518052 → 0.00013816. Two held-out combinations, ten labels, NLL 0.01370543. All eight fixture strings decoded exactly.

The vocabulary and grammar are deliberately small. Those two combinations test this limited task; they do not estimate real translation quality. Inspect source-dependence and causal checks before interpreting falling loss. Fixture BLEU is a worked calculation, separate from an official WMT evaluation protocol.

## RoBERTa fixture

Own-format byte-BPE vocabulary 341, MLM parameters 39,957. Seed 7, 120 MLM updates: fixed held-out corruption NLL 5.850855 → 3.835181; selected accuracy 0/8 → 2/8. Only eight evaluation targets are selected. Dynamic training and fixed evaluation corruption serve different purposes.

After compatible backbone transfer and 120 QA updates: training loss 4.186882 → 0.014509; eight training answers decoded exactly. This establishes overfit and decoding mechanics on seen examples, not held-out SQuAD quality. The tokenizer and parameter names do not match official pretrained checkpoint conventions.

## Continuation and lesson assembly

Recovery compares eight uninterrupted updates with three, save, and five in a fresh process. Weights, Adam, schedule, step and subsequent Python/NumPy/Torch draws agree. MLM also restores dedicated mask/batch generators. Failed-generation checks preserve compatible older saves and report rollback.

The lesson code was assembled in order: foundational snippets, attention through translation training, MLM forward/backward and QA fine-tuning. Checks use known answers, central differences and behavioral changes alongside shapes.

## Open evidence

Colab/Drive execution and deleted-runtime recovery, CUDA parity, full WMT/SQuAD evaluation, distributed training and large-scale performance remain unverified. Your implementation, explanations, investigations and measured resource profile need their own [readiness evidence](project/readiness.md).

[Selected final verification and source hashes](reference-verification.json).

## Expanded topic verification — 2026-10-13

The original model/trainer/data/recovery sources and recorded bounded results are preserved. The topic review passes the original 71 checks plus 29 independent applied numerical/source controls, and all 39 explicitly runnable lesson Python blocks pass in fresh source-copy processes. New probes compare analytic gradients/finite differences/autograd, parameter budgets, real corruption roles and actual full-model visibility; they do not train another model or establish benchmark quality.

The study/export and root/prefix reader results are recorded in [the topic verification record](topic-verification.json). Hosted Colab execution and learner mastery remain pending. The external workspace migration relocated the learner folders into transformer-labs and removed their nested Git controls; their source bytes match the pre-rewrite preservation baseline. This tutorial task does not recreate those repositories.
