# Tutorial 0 reference checks

These are local author results, separate from my learner implementation/explanation, translation-model completion and hosted recovery. Standard-library maths/text use `python3`; Torch examples use the selected reference interpreter from Lesson 2. Run from `examples/` unless a lesson says otherwise.

## Topic-rewrite source checks — 2026-10-12

The completed suite has114 checks:80 existing maths/text/tensor/full-network/classifier/recovery checks and34 new Transformer-foundation probes. Six declared Python/C++ publisher examples pass. The independent review also executes every standalone lesson code block and verifies changed-condition practice arithmetic. The current notes export has25 topic sections and a cleared35-cell study notebook; all9 study code cells execute from an independently extracted source copy.

```bash
python -m unittest test_prerequisite_math test_prerequisite_tensors \
  test_prerequisite_text test_deep_learning_math prerequisite_mlp_checks \
  test_transformer_foundation_probes
python transformer_foundation_probes.py
```

| Area | Meaningful checks |
|---|---|
| Matrices/calculus/loss | Rectangular affine outputs, shared bias/weight paths, stable shifted/extreme logits, independent central differences and full-network FP64 autograd |
| Text/data | UTF-8/code-point/original offsets, invalid/truncated decoding, artifact identity, document-safe BPE, unequal-length collation and correct valid counts |
| Embedding/position | Repeated-ID gradient sums, padding lookup versus tied-output paths, genuinely shared Parameter identity, odd-width/offset sinusoidal values and permutation/position distinction |
| Attention/masks | Independent hand fixtures and analytic backward, every-coordinate perturbations/gradcheck, causal/padded-key invariance, finite zero empty rows and SDPA dropout0 parity |
| Heads/core sublayers | Projected loop/vectorized forward and gradient parity with rectangular lengths, FFN position independence, LayerNorm trailing-axis hand/gradient checks and residual derivative/cancellation |
| Training/regularization | Exact dropout moments, smoothing gradients, schedule/counts, unequal microbatch equivalence, sampler/modes/nonfinite guards and bounded nonlinear fit |
| Recovery/export | Same fresh-process batch/update/optimizer/schedule/RNG, corruption/incomplete/weight-only rejection, retention/failure policy, member digests and cleared notebook/source execution |

The reusable attention module's summary deliberately uses an additional one-feature fixture with values `[[0,0],[6,0],[4,2]]`, producing `[4,1]`. Lesson15 uses a two-feature query and different values, producing `[2,2]`. Passing that lesson's inputs to the same reusable function reproduces its answer; the summary's different fixture is not a discrepancy.

## The classifier's declared domain

The actual trainer has2 numeric inputs,8 Tanh hidden features and2 logits:42 parameters,168 FP32 parameter bytes. Label1 means opposite feature signs. Its64 training and36 distinct held-out grid points are away from ambiguous axes. At seed7 and1,000 updates, held-out NLL changes from.7687182426 with18/36 correct to about.0001299955 with36/36 correct. This measures only the declared grid/domain, not text or translation. Whole-process RAM and timing include much more than parameter bytes and vary by run.

Choose a fresh disposable directory:

```bash
python prerequisite_mlp.py --mode new --run-dir /tmp/tutorial0-classifier-run \
  --total-steps 1000 --max-updates 1000
```

Keep the fixed total goal when slicing sessions. Lesson24 explains verified generation acceptance and fresh-process continuation; Lesson25 uses the same trainer in the thin Colab notebook. The checkpoint code does not promise fsync/atomic-rename storage durability. Local file verification is distinct from remote-drive survival.

The controlled2→2→2 forward/backward fixture has12 parameters and a supplied arithmetic target; it is not the42-parameter task trainer. Its complete parameter/input derivatives match independent forward perturbations and FP64 autograd. Historical74/80-check records belong to earlier preparations; the current source manifest and rebuild package bind this rewrite.

[Current source/export verification](reference-verification.json) records local proof types. Learner `linear()` remains pending in the existing child; hosted installation, Drive survival/deleted-runtime recovery and GPU/mixed-precision behavior remain unverified.
