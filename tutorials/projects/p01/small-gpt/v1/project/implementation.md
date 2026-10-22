# Project 1 implementation checklist

[Start with Lesson 1](lessons/01-next-token.md). The twelve lessons are the teaching route. Use this page to resume work, not as a separate course to finish first.

Current workspace learning remains **P00-M1 not-started**. P01 tutorial/reference preparation is available; the independent learner gpt-lab and its evidence are not created by this tutorial. Previewing P01 does not complete P00's exit record.

## One ordered build

All destination paths below are relative to your independent learner project root. Follow the lesson for the create/append operation and the reason behind each function. Complete reference downloads are available beside the code.

| Lesson | Save or implement | Check before continuing |
| --- | --- | --- |
| 1: next token | Scratch math_trace.py / optional C++ trace | Shift ok; compute probability, NLL and perplexity |
| 2: workspace | gpt_config.py; docs/runtime.md; docs/p00-handoff.md | D/H invariant, pinned CPU runtime, actual P00 prerequisite review |
| 3: text/tokenizer | gpt_tokenizer.py snapshot; first gpt_data.py functions; docs/data.md | Pinned acquisition, chapter splits, frozen artifact and Unicode round trip |
| 4: windows | Append windows/Batches/CLI; test_gpt_data.py | Four data tests and exact target coverage |
| 5: attention | gpt_attention.py | Causal hand average, changed-future check |
| 6: model | gpt_model.py; download test_gpt_model.py | [B,T,V] logits, 120832 default parameters; run full tests after loss exists |
| 7: learning | First gpt_objective.py functions; fit_tiny.py | Twelve data/model tests, loss gradient and fixed-batch fitting |
| 8: evaluate/generate | Append evaluation/baseline functions; gpt_sample.py | Valid-token denominator, random-model generation bound; trained CLI later |
| 9: persistence | gpt_checkpoint.py | Read completion/fallback/identity/retention paths; state tests in Lesson 10 |
| 10: trainer | gpt_trainer.py, gpt_train.py, cpu.json, colab-free.json; test_gpt_training.py | Twenty-one tests, actual 150+150 run/resume, fresh-process parity and failure evidence |
| 11: Colab | export_source.py; notebooks/colab.ipynb; docs/colab-recovery.md | ZIP/manifest/notebook checks, then actual Free training and deleted-runtime recovery |
| 12: exit | gpt_profile.py; memory/error/departure/P02 handoff records | Measured profile, one explained diagnosis, required exit rows and learner explanation |

## Additional learner fixtures that must not disappear

The reference's twenty-one tests cover a checked starting point, not every possible environment/failure. Add these in the learner project with precise contracts:

| Destination | Fixture and expected result |
| --- | --- |
| tests/test_fresh_process_resume.py | Spawn 8 uninterrupted versus 4 + process exit + 4; dropout 0.1; compare model, Adam, RNG, sampler, progress and monitoring state at zero tolerance on the declared CPU runtime |
| tests/test_training_interrupt.py | Start a disposable long-enough run, terminate after a confirmed saved update and during a later accumulation window; restore exactly the prior verified state, print rollback and continue |
| tests/test_persistent_interruption.py | Terminate a writer during a new generation's copy or before its completion marker; scanner rejects that generation and older compatible state remains usable |
| tests/test_run_identity.py | new on an existing run fails; resume without valid state fails; changed tokenizer/config/source/runtime fails; no files in the prior run are replaced |
| tests/test_storage_retention.py | Mark an earlier state best, create newer worse states, verify two recent plus best survive; insufficient budget never deletes the only good recovery copy |
| docs/colab-recovery.md | Delete the runtime, reacquire exactly identified source/data, restore/advance the same Drive run, capture resources/rollback/identity; a CPU directory round trip does not pass this row |

Use failure injection only against disposable runs. If you introduce workers, streaming data, AMP, a different device or an epoch sampler, extend the state/checks before calling that path recoverable.

## Record the implementation state

In your child checkpoint record one next unfinished action and its lesson. For each milestone link actual files, tests, runtime/config/source identity, observed results and the explanation you can give. Keep failed attempts with the targets declared before running. The author's [reference runs](project/reference-runs.md) are a comparison, not your evidence.

[Required readiness and P02 boundary](project/readiness.md) · [Source/departure record](research.md) · [Explain the model](practice/explain-your-model.md)
