# My Tutorial 0 implementation and study record

Follow the [25-topic outline](project/course-outline.md) and one ordered lesson route. The chapters are organized by concept; this record helps me resume checks and notes without creating a second theory/build course.

The current learner repository is still `ai_projects/projects/transformer-labs/transformer-foundations-lab`. Its local setup and first prediction are complete; its next saved exercise is `linear()` in `src/transformer_foundations_lab/numerics.py`. The contract is now in [Lesson 1](lessons/01-prediction.md), and the old [Lesson 3 address](lessons/03-numbers.md) retains an explicit continuation link. Inspect my existing file and preserve edits. Do not recreate the repository or replace learner work with the author's completed reference.

## How I will study a topic

I will write the equation and axis meanings, work the numerical fixture, predict a changed condition, then run the check. I will keep a personal copy of [my notes workbook](project/study-notes.md); the optional cleared study notebook imports the same reference helpers. The complete written explanations remain in the lessons. The training/Colab notebook is a separate thin runner using the same Python trainer as local runs.

| Lessons | Result I should be able to produce | Reference source/check |
|---|---|---|
| 1–2: linear algebra/tensors | Rectangular affine outputs; explicit shape/dtype/device/storage/registration contracts | `prerequisite_math.py`, `prerequisite_tensors.py`; known-answer and unequal-axis checks |
| 3–4: calculus/probability | Partial/Jacobian/directional derivatives; likelihood, expectation, entropy and reduction | Scalar central differences, probability identities and original hand fixtures |
| 5–7: networks/init/backprop | Nonlinear forward pass; variance assumptions; every coordinate of a complete backward pass | `deep_learning_math.py` and independent forward-perturbation/FP64 autograd comparisons |
| 8–10: losses/optimization/steps | Stable NLL/BCE signal; Adam history; correct accumulation/schedule/clipping | Math/tensor checks and `prerequisite_mlp_checks.py` |
| 11–12: data/text | Split-safe collation; separate PAD/ignore/mask roles; original spans and saved BPE identity | `prerequisite_text.py`, `byte_bpe.py`, collator/valid-target checks |
| 13–14: embedding/position | Lookup-gradient sums, tied table paths/counts, sinusoidal values and shift/permutation reasoning | Embedding/position probes and independent worked numbers |
| 15–17: attention | Rectangular QKᵀ/softmax/AV, mask invariance, safe empty rows and matched head implementations | `transformer_foundation_probes.py`, independent manual backward and gradient checks |
| 18–20: FFN/residual/norm | Position independence, feature capacity/cost, identity derivative and per-token normalization | Hand fixtures, FFN perturbation and LayerNorm gradient checks |
| 21–22: regularization/seq2seq | Explicit dropout/smoothing contracts and source/prefix/label/inference alignment | Distribution and teacher-forcing/mask probes |
| 23: training/evaluation | Complete bounded nonlinear learning, baselines and failure diagnosis | Shared `prerequisite_mlp.py` trainer on its declared grid and distinct held-out points |
| 24: recovery | Same next batch/update after a fresh process; verified fallback and identity rejection | `prerequisite_checkpoint.py`, fresh-process/retention/corruption checks |
| 25: paper/Colab/readiness | Paper equation/architecture map and exact source delivery; honest next-build and hosted record | Source export, cleared notebooks, [readiness](project/p01-readiness.md) |

## Runnable references and learner evidence

Run standard-library maths/text checks with `python3`. Torch blocks use a recorded compatible environment; the workspace already has a frozen CPU reference interpreter in `ai_projects/projects/transformer-labs/attention-is-all-you-need/.venv/bin/python`. Use it only for references here, not to overwrite or silently reconfigure another child. This course adds no implementation to either child or later model course.

I will distinguish my own implemented function, a passing reference, a worked explanation, measured task behavior and fresh-runtime recovery. The [reference-run record](project/reference-runs.md) documents the bounded author experiment. Its 42-parameter classifier is not a translation model; the 12-parameter backward fixture is not that trainer. Keep datasets, weights, environment folders, generated runs and notebook outputs outside Git.
