# Return to the Attention project

The current application has a checked attention component. This course explains setup and that component; later data/model/training work remains pending. Start the reading route at [the first weighted retrieval](lessons/01-weighted-information.md), then continue in lesson order.

## Current evidence

| Area | Implemented state | Review boundary |
| --- | --- | --- |
| Locked local Python package | Prepared; PyTorch/NumPy and optional notebook tools | An installed environment is not a trained model |
| Attention forwards and masks | Implemented in the package module | Checked on the exercised local CPU shapes/dtypes |
| Gradients and reference parity | Existing tests compare independent values and backward paths | No full translation loss/training experiment |
| Tutorial figures and Colab runner | Original figures in the tutorial; six operational notebook cells | Local runner verification; actual hosted execution pending |
| Translation tokenizer/data | Placeholder files | No acquired corpus or frozen tokenization/split contract |
| Full model/training/inference/BLEU | Placeholder files | No trained translator or measured BLEU |
| Persistent hosted recovery | Requirements recorded | Not implemented or executed |

## Resume without recreating setup

Open the existing project under `projects/transformer-labs/attention-is-all-you-need/`, read `README.md`, and run `uv run --frozen --extra notebook pytest -q` if the notebook dependencies are selected. The demonstration is `uv run --frozen python src/attention.py`. From workspace home, `./ai run attention-is-all-you-need` launches that same demo through the workspace controller.

For understanding, select one current question: a query/key matrix cell, one value mixture, a head-layout change, a forbidden-information perturbation or a query gradient. Use the corresponding lesson and predict one changed case before reviewing the answer. Routine model code is already written for this component; no transcription exercise is required.

## Files to continue later

Part 2 starts at `src/data/tokenizer.py` and `src/data/dataset.py`, with commands in `src/tokenizer.py` and `src/prepare_data.py`. Part 3 later fills model/configuration under `src/core/`, trainer/checkpoints under `src/training/`, and the train/inference/BLEU commands under `src/`. Preserve the final Attention argument/projection names and verify the new call sites against their actual masks and tensor roles.

The [readiness record](project/readiness.md) separates source checks, learner explanation, task quality and hosted recovery. The [reference runs](project/reference-runs.md) record current commands and limits. The [source research](research.md) explains the evidence identities and departures from the video reference.
