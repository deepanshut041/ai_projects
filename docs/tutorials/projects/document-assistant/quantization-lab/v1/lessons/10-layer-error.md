## 10. Follow a weight error into the layer output

Packed bytes make compression concrete, but the assistant operates on layer outputs. Two artifacts can have the same average weight error and very different behavior on a document. We first trace one linear layer with all axes and numbers visible.

### Rows produce outputs, columns consume inputs

Let a column input $x\in\mathbb R^2$ enter weights $W\in\mathbb R^{2\times2}$. Row $r$ contains weights contributing to output channel $r$: $y_r=\sum_j W_{rj}x_j$. A bias is omitted because we hold it unchanged in this worked example. Our values are

$$W=\begin{bmatrix}0.24&1.02\\-0.76&0.49\end{bmatrix},\quad x=\begin{bmatrix}4\\1\end{bmatrix}.$$

Then $y_1=0.24(4)+1.02(1)=1.98$ and $y_2=-0.76(4)+0.49(1)=-2.55$. An illustrative reconstructed matrix is

$$\hat W=\begin{bmatrix}0&1\\-1&0.5\end{bmatrix},\quad\hat y=\begin{bmatrix}1\\-3.5\end{bmatrix}.$$

Define weight error $E=\hat W-W$, so $E=[[-0.24,-0.02],[-0.24,0.01]]$. Output error is not another arbitrary quantity: subtraction distributes through the linear operation,

$$\Delta y=\hat Wx-Wx=(\hat W-W)x=Ex.$$

For the first row it is $-0.24(4)-0.02(1)=-0.98$; for the second $-0.24(4)+0.01(1)=-0.95$. The larger first input amplifies errors in the first weight column. Reversing input magnitudes would change which column matters most even though the weight matrix and its reconstruction errors are unchanged.

![Layer error matrix calculation](figures/layer-error.svg)

### A norm bound gives a warning, not an answer score

The Euclidean norm of a vector is $\|x\|_2=\sqrt{\sum_jx_j^2}$. The Frobenius norm of a matrix is $\|E\|_F=\sqrt{\sum_{r,j}E_{rj}^2}$. Apply the Cauchy–Schwarz inequality to each output row, then sum squared row bounds to obtain $\|Ex\|_2\le\|E\|_F\|x\|_2$. This explains why a small matrix error can still produce a large output error when input magnitude is large. The bound is generally loose, and it says nothing directly about an answer being semantically supported.

Create `10_layer_error.py`. `matvec()` checks row lengths before computing dot products; otherwise `zip()` could silently ignore a missing channel. The program checks output subtraction against independently applying the error matrix and verifies the norm bound. The matrix is an authored numerical example, not a tensor extracted from Gemma.

[Download the complete reference](examples/10_layer_error.py)

```python
"""Rows are output channels; columns are input channels."""
import json
import math


def matvec(matrix, vector):
    if any(len(row) != len(vector) for row in matrix):
        raise ValueError("Input dimension mismatch")
    return [sum(w * x for w, x in zip(row, vector)) for row in matrix]


def main():
    weights = [[0.24, 1.02], [-0.76, 0.49]]
    restored = [[0.0, 1.0], [-1.0, 0.5]]
    x = [4.0, 1.0]
    error = [[q - w for q, w in zip(qrow, wrow)] for qrow, wrow in zip(restored, weights)]
    y, qy = matvec(weights, x), matvec(restored, x)
    delta = [a - b for a, b in zip(qy, y)]
    assert all(abs(a - b) < 1e-12 for a, b in zip(delta, matvec(error, x)))
    error_norm = math.sqrt(sum(v*v for row in error for v in row))
    x_norm = math.sqrt(sum(v*v for v in x))
    delta_norm = math.sqrt(sum(v*v for v in delta))
    assert delta_norm <= error_norm * x_norm + 1e-12
    print(json.dumps({"float_output": [round(v, 4) for v in y],
                      "quantized_output": qy, "delta": [round(v, 4) for v in delta]}))


if __name__ == "__main__":
    main()
```

From the scratch directory containing that file, run:

```bash
python3 10_layer_error.py
```

Expected checked output (JSON formatting may differ):

```json
{
  "float_output": [
    1.98,
    -2.55
  ],
  "quantized_output": [
    1.0,
    -3.5
  ],
  "delta": [
    -0.98,
    -0.95
  ]
}
```

Expect outputs `[1.98,-2.55]` and `[1.0,-3.5]`, with deltas `[-0.98,-0.95]`. If dimensions still look correct but values differ, inspect row/column orientation. A tensor library often represents a batch as rows and computes `X @ W.T`; that is the same per-example operation under a different layout. Write the axes before translating formulas.

### Error flows through a network, not just a matrix

A Transformer adds residual paths, normalization, nonlinear feed-forward operations and attention. A residual addition passes a branch's error into a later state; normalization can redistribute its effect across channels; an attention score perturbation can alter how information is mixed. The final token decision depends on the relative logit margin. If the largest two logits are 2.00 and 1.99, a 0.02 differential shift can flip greedy decoding. A different token changes future history, as Lesson 7 showed.

This is a reason to measure task outputs, not a claim that every layer error compounds monotonically. Errors may cancel, be attenuated or grow. Weight MSE is a useful diagnostic but not a replacement for question-level evidence review. Preserve layer reconstruction diagnostics and final answer changes as separate evidence.

[[practice:layer-transfer]]

We now have a mathematical reason to control inputs when comparing compressed weights. Before adding advanced quantization, we define how real converted artifacts and their runtime factors are kept matched.
