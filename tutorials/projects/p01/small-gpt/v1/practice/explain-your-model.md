# Explain the model without reading the code

Use this after implementing the relevant part. Do not turn it into another prerequisite reading route. Draw first, then compare with the lesson and tests. Retain personal rehearsal answers only if you choose to record them.

## A two-minute board

Draw token IDs → learned token+position vectors → two pre-norm causal blocks → final LayerNorm → tied vocabulary logits. Label each tensor shape and the shifted labels. Draw the direct residual arrow around normalization+sublayer. Explain which arrays mix positions and which act independently per position.

Trace one string, `hi`: BOS predicts h, h predicts i, i predicts EOS. Explain why the diagonal is permitted but a future key is not. Then name the exact functions in gpt_data.py, gpt_attention.py, gpt_model.py and gpt_objective.py that enforce that story.

## Defend a consequential choice

| Choice | Evidence to show | Tradeoff to explain |
| --- | --- | --- |
| Byte vocabulary first | Frozen artifact/Unicode tests and V260 config | Simple lossless encoding versus longer sequences |
| Reset context per window | Exact target-coverage tests and protocol | Bounded implementation, limited context near boundaries |
| Pre-norm/learned positions/GELU | Formula-to-code trace and variant tests | Deliberate P01 design, not identical to P00 or historical GPT-1 |
| Tied output matrix | 120832 parameter count and non-input-row gradient | Shared parameters with two gradient roles |
| Explicit attention | Float64 SDPA/gradient comparison | Readable reference with quadratic intermediates |
| Strict resume | All-state fresh-process comparison and mismatch rejection | Clear reproducibility boundary, explicit work for device/environment changes |

## Diagnose, then transfer

Choose one real error or disappointing sample. State what you expected, the test/run that contradicted it, the inspected evidence, one change and the next result. Avoid inventing a historical reason for the source's choices: an observed implementation and your own tradeoff analysis are different claims.

Now change one assumption: unequal microbatch lengths, a prompt longer than context, a corrupt newest checkpoint, or a device change after runtime deletion. Predict what should happen and identify which contract/test needs to change. Use the [question bank](practice/questions.md) for prompts and [readiness](project/readiness.md) for completion evidence.
