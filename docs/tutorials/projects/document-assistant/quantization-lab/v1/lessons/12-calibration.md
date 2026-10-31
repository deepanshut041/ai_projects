## 12. Learn which weight errors the inputs amplify

Two weights can have equal reconstruction error while one changes a layer output much more. Calibration measures the inputs a layer actually receives. It does not train the model to answer labels, and it must not read the final evaluation answers.

### Start with two equally sized errors

Take one output row with two input channels. Candidate A has weight error $e_A=[0.25,0]$; candidate B has $e_B=[0,0.25]$. Both have mean squared weight error $(0.25^2+0)/2=0.03125$. Calibration inputs are $x^{(1)}=[4,1]$ and $x^{(2)}=[4,-1]$. For A, both output errors are 1, so output MSE is 1. For B they are 0.25 and -0.25, so output MSE is 0.0625. Equal weight MSE concealed a sixteenfold difference in this output-loss fixture.

### Derive the Gram matrix from a sample average

Define $n$ calibration input vectors of dimension $d$. For one row error $e\in\mathbb R^d$, average squared output error is

$$L(e)=\frac1n\sum_{t=1}^n(e^Tx^{(t)})^2.$$

Expanding the dot product gives $\sum_i\sum_j e_i e_j x_i^{(t)}x_j^{(t)}$. Exchange finite sums and define

$$G_{ij}=\frac1n\sum_t x_i^{(t)}x_j^{(t)},\qquad L(e)=e^TGe.$$

$G$ is the uncentered second-moment/Gram matrix, not automatically a centered covariance matrix. Diagonal entries measure mean squared channel magnitudes; off-diagonals measure cross-products. In our fixture $G=[[16,0],[0,1]]$: the second channel changes sign and its cross-products cancel. This is why the first-channel error costs more. If channels correlate, off-diagonal terms allow errors to cancel or reinforce; Lesson 14 uses that fact.

![Calibration weighting](figures/calibration.svg)

The matrix is positive semidefinite because $e^TGe$ is an average of squares. It can be singular when samples do not span all input directions. A second-order method that requires an inverse must address this; more calibration samples do not guarantee all directions are observed.

### Calculate the same loss in two different ways

Create `12_calibration.py`. `gram()` builds $G$ from samples. `quadratic()` evaluates the expanded matrix form. `reconstruction()` directly evaluates each output error and squares it. Agreement between the latter two checks our axis/reduction choices; the visible known answer provides a check beyond comparing two copies of a formula. The collector rejects empty and ragged sample matrices before accessing their channel dimensions.

[Download the complete reference](examples/12_calibration.py)

```python
"""Calibration statistics in input-channel coordinates; no gradient training."""
import json


def gram(samples):
    if not samples or not samples[0]:
        raise ValueError("Need a rectangular nonempty sample matrix")
    n, d = len(samples), len(samples[0])
    if any(len(x) != d for x in samples):
        raise ValueError("Need a rectangular nonempty sample matrix")
    return [[sum(x[i] * x[j] for x in samples) / n for j in range(d)] for i in range(d)]


def quadratic(error, matrix):
    return sum(error[i] * matrix[i][j] * error[j]
               for i in range(len(error)) for j in range(len(error)))


def reconstruction(error, samples):
    return sum(sum(e*x for e, x in zip(error, sample))**2 for sample in samples) / len(samples)


def main():
    samples = [[4.0, 1.0], [4.0, -1.0]]
    matrix = gram(samples)
    a, b = [0.25, 0.0], [0.0, 0.25]
    assert matrix == [[16.0, 0.0], [0.0, 1.0]]
    assert quadratic(a, matrix) == reconstruction(a, samples) == 1.0
    assert quadratic(b, matrix) == reconstruction(b, samples) == 0.0625
    print(json.dumps({"gram": matrix, "same_weight_mse": True,
                      "output_mse": [quadratic(a, matrix), quadratic(b, matrix)]}))


if __name__ == "__main__":
    main()
```

From the scratch directory containing that file, run:

```bash
python3 12_calibration.py
```

Expected checked output (JSON formatting may differ):

```json
{
  "gram": [
    [
      16.0,
      0.0
    ],
    [
      0.0,
      1.0
    ]
  ],
  "same_weight_mse": true,
  "output_mse": [
    1.0,
    0.0625
  ]
}
```

Expect Gram diagonals 16 and 1, and losses 1 and 0.0625. If both candidate losses are equal, you may be calculating weight MSE instead of activation-weighted output MSE. If a mean changes when you duplicate every sample, check the denominator. If padding dominates statistics, verify which real token positions were included.

### Design one calibration comparison at a time

For phase 04, first hold method, sample token count, seed, group size, precision and runtime fixed. Compare general-text and document-domain calibration with declared document-disjoint membership. Preserve selected text hashes, tokenization, sequence lengths, masks, sampling order and collected statistics. Labels are unnecessary for numerical calibration; tuning recipe choice on development answers is a separate step.

A collector hook must capture the actual input to the targeted linear operation with its channel axis documented. Flattening batch/token axes into samples can be valid, but include only intended real token positions. Retain per-layer counts and bounded summary statistics; do not keep giant activation dumps merely because a hook can produce them. Measure calibration memory/time separately from inference.

Overfitting can occur even without gradient training: searching many scales or recipes against one small development set adapts the choice to that set. Keep a held-out reconstruction probe and task-level development report, then freeze before final evaluation. A lower layer reconstruction loss is a diagnostic, not a guarantee of better grounded answers.

### A candidate importance-matrix route has its own file contract

The pinned [llama.cpp collector documentation](source:llama-imatrix) describes a runtime-native importance file. The following is a **future conditional protocol** after the model/runtime compatibility gate passes, from that runtime checkout, with the verified BF16 container and a frozen permitted calibration text file already present:

```bash
./build/bin/llama-imatrix -m artifacts/gemma4-reference-bf16.gguf -f data/calibration.txt -o artifacts/domain-imatrix.gguf
./build/bin/llama-quantize --imatrix artifacts/domain-imatrix.gguf artifacts/gemma4-reference-bf16.gguf artifacts/gemma4-q4-domain.gguf Q4_K_M
```

The collector file is not the toy Python Gram JSON. Retain its digest, completed chunks/tokens, tensor coverage and warnings; inspect actual supported operations. Generate the general-text alternative with the same declared token/chunk budget, then derive both candidates from the same floating source. Merely requesting an importance file does not prove every tensor used it, and this route is not automatically AWQ or GPTQ. No collector/model command was executed for tutorial preparation.

[[practice:calibration-transfer]]

We now know what “important channel” can mean operationally. Next we rescale channels without changing the original floating operation, then observe why the quantized operation can nevertheless improve or worsen.
