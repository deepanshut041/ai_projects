## 14. Derive why later weights can compensate for an earlier rounding error

Channel rescaling changes the quantization grid. A different idea keeps track of how errors interact under correlated inputs. If one weight is forced onto a grid, a still-unquantized weight can shift to reduce the resulting output error. We derive one constrained step rather than hiding it behind an algorithm name.

### Expand a two-channel quadratic

From Lesson 12, a row's reconstruction loss is $L(e)=e^TGe$. Use the authored positive-definite matrix

$$G=\begin{bmatrix}4&2\\2&2\end{bmatrix}.$$

Positive definiteness means every nonzero error direction has positive loss. Here the leading diagonal is positive and determinant $4(2)-2(2)=4>0$. Expanding gives $L=4e_1^2+4e_1e_2+2e_2^2$. Suppose quantization fixes the first error at $e_1=0.25$. We can still choose $e_2$. Without compensation, $e_2=0$ gives $L=0.25$.

A derivative measures how a small change in $e_2$ changes loss while $e_1$ is held fixed. For the square term, the slope over a small step $h$ is $((e_2+h)^2-e_2^2)/h=2e_2+h$, which approaches $2e_2$ as $h$ approaches zero. A constant has slope zero; a linear term $c e_2$ has slope $c$. Apply these rules to each term while treating $e_1$ as constant:

$$\frac{\partial L}{\partial e_2}=4e_1+4e_2.$$

Set it to zero: $e_2=-e_1=-0.25$. The second derivative is 4, positive, so this stationary point is the unique minimum in this one-dimensional slice. Substituting gives $0.25-0.25+0.125=0.125$, half the uncompensated loss. The negative compensation cancels some effect because the input channels are positively correlated.

For general $G$, with $G_{22}>0$, the same derivative gives $e_2^*=-G_{21}e_1/G_{22}$. If the cross term is zero, the optimum is zero: independent sensitivity alone creates no compensation. A negative cross term reverses its sign.

### Connect the inverse form

For a positive-definite quadratic matrix $H$ and a single constrained coordinate error $e_j=\delta$, minimizing $e^THe$ yields

$$e=\frac{\delta}{(H^{-1})_{jj}}H^{-1}_{:,j}.$$

One way to see this is to introduce a constraint multiplier: $2He+\lambda u_j=0$, where $u_j$ is the coordinate basis vector. Multiply by $H^{-1}$ and choose $\lambda$ so coordinate $j$ equals $\delta$. In our example, $G^{-1}=[[0.5,-0.5],[-0.5,1]]$; its first column divided by 0.5 is `[1,-1]`, recovering `[0.25,-0.25]`. A constant factor such as the Hessian $H=2G$ cancels in this ratio.

This derivation assumes a stable inverse and unconstrained remaining coordinates. Subsequent quantization restricts those coordinates too; it can lose some of the compensation. It is not a proof of globally optimal discrete model quantization.

![Worked compensation](figures/compensation.svg)

### Verify with a different computation

Create `14_compensation.py`. It calculates the analytic optimum, searches 2001 candidate values independently and checks the finite-difference slope near the optimum. The grid includes -0.25 exactly. Agreement on this fixture tests the sign, cross-term factor and reduction; a test that only repeats the derivative formula would be weaker.

[Download the complete reference](examples/14_compensation.py)

```python
"""One constrained quadratic step; no claim to implement production GPTQ."""
import json


def cost(e1, e2, h):
    return h[0][0]*e1*e1 + 2*h[0][1]*e1*e2 + h[1][1]*e2*e2


def main():
    h, e1 = [[4.0, 2.0], [2.0, 2.0]], 0.25
    e2 = -h[1][0] * e1 / h[1][1]
    # Independent exhaustive scalar search includes the analytic minimum.
    candidates = [i / 1000 for i in range(-1000, 1001)]
    best = min(candidates, key=lambda e: cost(e1, e, h))
    assert abs(best - e2) < 1e-12 and e2 == -0.25
    epsilon = 1e-6
    slope = (cost(e1, e2 + epsilon, h) - cost(e1, e2 - epsilon, h)) / (2*epsilon)
    assert abs(slope) < 1e-9
    assert cost(e1, e2, h) < cost(e1, 0.0, h)
    print(json.dumps({"fixed_error": e1, "compensation": e2,
                      "before": cost(e1, 0, h), "after": cost(e1, e2, h)}))


if __name__ == "__main__":
    main()
```

From the scratch directory containing that file, run:

```bash
python3 14_compensation.py
```

Expected checked output (JSON formatting may differ):

```json
{
  "fixed_error": 0.25,
  "compensation": -0.25,
  "before": 0.25,
  "after": 0.125
}
```

If the best error is +0.25, inspect the definition `quantized minus original` and the sign in the compensation. If the loss becomes negative for this matrix, inspect the quadratic expansion. Positive semidefinite Gram losses cannot be negative except tiny roundoff noise.

### Where GPTQ adds the practical machinery

[GPTQ](source:gptq-paper) uses layerwise reconstruction and approximate second-order information for post-training quantization. The [pinned source](source:gptq-code) adds damping, inverse/Cholesky machinery, ordered/block updates and quantizer integration. Damping adds a positive diagonal term, often tied to a diagonal statistic, to stabilize poorly observed directions. It changes the objective, so record its value and rule rather than calling it merely a numerical implementation detail.

Our program omits full sequential quantization, grouping, activation ordering, production tensor collection and kernels. It is a verified derivation of one compensation step, not a GPTQ export. Before a real advanced-method comparison, verify support for the exact architecture, real completed layer coverage, calibration identities and fresh-process loading.

[[practice:compensation-transfer]]

We can now explain two different calibration-aware mechanisms: changing coordinates and redistributing constrained error. The next lesson chooses among precision/group options under an explicit storage cap instead of assuming one recipe must suit every tensor.
