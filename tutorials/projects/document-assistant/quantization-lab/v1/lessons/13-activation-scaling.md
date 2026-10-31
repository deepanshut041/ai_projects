## 13. Protect a sensitive channel through an equivalent scaling

Calibration showed that large input channels amplify weight error. One response is to store selected weights at higher precision. Another is to change the numerical coordinates before quantization while preserving the original floating computation. This lesson derives that transformation and its limits.

### An identity before an approximation

Let $S=\operatorname{diag}(s_1,\ldots,s_d)$ with every $s_j>0$. Multiplying by a diagonal matrix scales channels independently. Its inverse has diagonal entries $1/s_j$, so

$$Wx=(WS)(S^{-1}x).$$

For output row $w=[0.24,1.02]$, input $x=[4,1]$ and scales $[4,1]$, transformed weights are $[0.96,1.02]$ and transformed input is $[1,1]$. Both give $1.98$. This identity does not reduce precision or introduce error in exact arithmetic. It merely changes coordinates. It is also not permission to scale only the weights: forgetting the inverse input scaling changes the model.

After quantizing the transformed weights, the operation is $Q(WS)S^{-1}x$. Quantization is not a linear operation, so this generally differs from $Q(W)x$. That difference gives us room to reduce reconstruction loss. It can also make it worse.

### Work a small grid all the way through

Use a deliberately coarse symmetric ternary quantizer: each block reconstructs $[-a,0,a]$ with $a=\max|w|$. It has three levels; it is not an actual four-bit AWQ format. With no scaling, `[0.24,1.02]` becomes `[0,1.02]`. For $x=[4,1]$, output falls from 1.98 to 1.02, error -0.96.

With scales `[4,1]`, `[0.96,1.02]` rounds to `[1.02,1.02]`, and transformed input `[1,1]` gives 2.04. Error is +0.06. Across inputs `[4,1]` and `[4,-1]`, mean squared error drops from $0.96^2=0.9216$ to $0.06^2=0.0036$ in this fixture.

Larger is not always better. Scale 8 produces weights `[1.92,1.02]`, rounded to `[1.92,1.92]`, with input `[0.5,1]`; output becomes 2.88 instead of 1.98. The group maximum changed, coarsening the other channel's grid. Across the two samples, MSE is 0.81. We therefore evaluate a bounded scale grid rather than assuming monotonically increasing protection.

![Worked scaling](figures/scaling.svg)

### Execute the complete equivalent transform and search

Create `13_activation_scaling.py`. `transformed()` returns both sides of the identity. `loss()` checks float equivalence for every sample before applying quantization; an incorrect inverse would fail before any score comparison. It then evaluates the same output-error objective as Lesson 12. `main()` compares four predetermined scales.

[Download the complete reference](examples/13_activation_scaling.py)

```python
"""Diagonal equivalence and a tiny scale search, not the full AWQ algorithm."""
import json


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def transformed(weights, sample, scales):
    return [w*s for w, s in zip(weights, scales)], [x/s for x, s in zip(sample, scales)]


def ternary(values):
    scale = max(map(abs, values))
    return [round(v / scale) * scale for v in values]


def loss(weights, samples, scales):
    errors = []
    for x in samples:
        ws, xs = transformed(weights, x, scales)
        assert abs(dot(weights, x) - dot(ws, xs)) < 1e-12
        errors.append((dot(ternary(ws), xs) - dot(weights, x))**2)
    return sum(errors) / len(errors)


def main():
    weights, samples = [0.24, 1.02], [[4.0, 1.0], [4.0, -1.0]]
    scores = {str(s): round(loss(weights, samples, [s, 1.0]), 6) for s in [1, 2, 4, 8]}
    assert scores == {"1": 0.9216, "2": 0.9216, "4": 0.0036, "8": 0.81}
    print(json.dumps({"scale_loss": scores, "best_first_channel_scale": 4,
                      "scope": "ternary teaching grid"}))


if __name__ == "__main__":
    main()
```

From the scratch directory containing that file, run:

```bash
python3 13_activation_scaling.py
```

Expected checked output (JSON formatting may differ):

```json
{
  "scale_loss": {
    "1": 0.9216,
    "2": 0.9216,
    "4": 0.0036,
    "8": 0.81
  },
  "best_first_channel_scale": 4,
  "scope": "ternary teaching grid"
}
```

The expected losses are 0.9216, 0.9216, 0.0036 and 0.81 for scales 1, 2, 4 and 8. Those values describe this tiny ternary fixture only. Trying a different input distribution can change the winner even when float equivalence remains exact.

### Connect the mechanism to AWQ without overstating the reference

[AWQ](source:awq-paper) uses activation information to identify salient weight channels and equivalent scaling to protect them under weight-only quantization. The [pinned implementation](source:awq-code) includes layer-specific scaling transformations and a scale search. Our two-channel program isolates the mathematical idea; it omits architecture-specific folding, clipping search, real quantizer groups, packing and inference kernels. It does not implement or validate AWQ for Gemma 4.

For an eventual supported backend, check where the inverse scale is absorbed. A preceding normalization/linear operation can sometimes be reparameterized to avoid a separate runtime multiply, but graph branches and shared consumers constrain that rewrite. Verify full-precision equivalence before quantization and inspect whether the exporter applied the transformation to each intended layer. A flag named AWQ that is skipped for unsupported patterns is not a completed AWQ experiment.

[[practice:scaling-transfer]]

The phase-04 comparison holds data budget, group size, bits, runtime and placement fixed while changing one supported recipe. Record scale search cost and the actual transformed layers. We have protected channels individually; next we exploit correlations between channels to explain a second-order compensation step.
