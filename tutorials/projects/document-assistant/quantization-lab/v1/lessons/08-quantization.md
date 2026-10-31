## 8. Derive the integer grid instead of memorizing “INT4”

A floating weight can take many representable values. A $b$-bit code has only $2^b$ possible bit patterns. Quantization selects one of a finite set of reconstructed values. The code, the scale and sometimes a zero point together define the approximation. Casting a weight to an integer without those rules is a different operation.

### Construct an affine grid

Choose integer codes $q\in[q_{\min},q_{\max}]$ and a positive step $s$. A zero point $z$ is the integer code that reconstructs real zero. Reconstruction is $\hat x=s(q-z)$. To encode, divide by the step to express a value in code units, round to a nearby code and clip to the representable interval:

$$q=\operatorname{clip}\bigl(\operatorname{round}(x/s)+z,q_{\min},q_{\max}\bigr).$$

Here `round` uses ties to even. The [ONNX QuantizeLinear definition](source:onnx-quantize) specifies this rounding/saturation convention for its operator; our teaching implementation makes the convention explicit rather than assuming every runtime uses it. A fixed $s,z$ affine scheme is the topic here, not a claim about the precise internals of GGUF presets.

For an unsigned three-bit interval $[0,7]$ and desired real range $[-2,4]$, the endpoint-derived step is $s=(4-(-2))/7=6/7$. Zero point is $z=\operatorname{round}(2/(6/7))=2$. Thus codes 0 and 7 reconstruct $-12/7$ and $30/7$, approximately $-1.7143$ and $4.2857$. Integer zero-point rounding means the original endpoints are not both reproduced exactly. The zero itself is exact: code 2 reconstructs 0.

Encoding $x=1$ gives `round(7/6)+2=3`, reconstructing $6/7\approx0.8571$. Encoding 7 would exceed the grid and clips to code 7, reconstructing about 4.2857. The error outside the represented range is clipping error, not bounded by half a step. For unclipped nearest rounding, $|x-\hat x|\le s/2$; this bound says nothing about model answer quality.

![Quantizer grid and clipping](figures/quantizer-grid.svg)

### Symmetric grids make another deliberate choice

A common symmetric teaching scheme uses $q\in[-Q,Q]$, $Q=2^{b-1}-1$, zero point 0 and $s=\max|x|/Q$. At four bits this uses levels -7 through 7, leaving one of the sixteen signed patterns unused. Some formats instead use -8 through 7 or more complex block schemes. State which one you mean. For an all-zero block, choose a harmless positive scale and zero codes; dividing by `max(abs(x))=0` is undefined.

Ties affect reproducibility: ties-to-even maps 0.5→0, 1.5→2, 2.5→2 and -0.5→0. A hand-written “add 0.5 then truncate” rule is different for negatives and ties. In real conversion, the format/kernel's exact rule is authoritative.

### Weight precision, activation precision and accumulation differ

“W4A16” describes four-bit stored weights and sixteen-bit activations under a declared scheme. It does not mean the whole network computes four-bit arithmetic. A weight-only kernel may unpack/dequantize blocks while multiplying floating activations and accumulate at another precision. An integer-weight-and-activation operation instead needs scales for both sides. For symmetric codes, $w=s_wq_w$, $x=s_xq_x$, so a dot product is $s_ws_x\sum_jq_{w,j}q_{x,j}$. With codes `[2,-1]` and `[3,4]`, scales 0.5 and 0.25, the integer sum is $6-4=2$ and reconstructed output is $0.125(2)=0.25$. The corresponding float vectors `[1,-0.5]` and `[0.75,1]` give the same result here. Affine zero points require subtracting each side's zero point inside the products.

The accumulator needs enough range for summed products; storage width alone does not specify it. Record weight/activation/cache/storage/compute/accumulation types separately where the backend exposes them. Post-training quantization calibrates/converts an existing model; quantization-aware training changes training to account for simulated quantization effects. Our project begins with post-training weight studies, without gradient updates or a newly trained model.

### Implement both rules and inspect reconstructed values

Create `08_scalar_quant.py`. `affine()` requires a nonzero range containing zero. Constant nonzero ranges need a separately declared widening/range policy; the function rejects invalid bounds rather than silently inventing one. `symmetric()` handles the all-zero block. Both return reconstruction so error can be inspected in original units.

[Download the complete reference](examples/08_scalar_quant.py)

```python
"""Explicit affine/symmetric quantizers, with round-to-nearest-even."""
import json


def affine(values, low, high, bits):
    if not low < high or bits < 1:
        raise ValueError("Require a nonzero range and positive bit count")
    if not low <= 0 <= high:
        raise ValueError("This zero-preserving fixture includes zero in the range")
    qmax = 2 ** bits - 1
    scale = (high - low) / qmax
    zero = min(qmax, max(0, round(-low / scale)))
    codes = [min(qmax, max(0, round(x / scale) + zero)) for x in values]
    restored = [scale * (q - zero) for q in codes]
    return scale, zero, codes, restored


def symmetric(values, bits):
    if bits < 2 or not values:
        raise ValueError("Need at least two bits and a nonempty block")
    qmax = 2 ** (bits - 1) - 1
    scale = max(map(abs, values)) / qmax
    if scale == 0:
        return [0] * len(values), [0.0] * len(values), 1.0
    codes = [min(qmax, max(-qmax, round(x / scale))) for x in values]
    return codes, [q * scale for q in codes], scale


def main():
    scale, zero, codes, restored = affine([-2, -1, 0, 1, 4, 7], -2, 4, 3)
    assert zero == 2 and codes == [0, 1, 2, 3, 7, 7]
    assert symmetric([0, 0], 4)[1] == [0.0, 0.0]
    assert [round(x) for x in [0.5, 1.5, 2.5, -0.5]] == [0, 2, 2, 0]
    print(json.dumps({"scale": round(scale, 6), "zero": zero, "codes": codes,
                      "restored": [round(x, 6) for x in restored]}))


if __name__ == "__main__":
    main()
```

From the scratch directory containing that file, run:

```bash
python3 08_scalar_quant.py
```

Expected checked output (JSON formatting may differ):

```json
{
  "scale": 0.857143,
  "zero": 2,
  "codes": [
    0,
    1,
    2,
    3,
    7,
    7
  ],
  "restored": [
    -1.714286,
    -0.857143,
    0.0,
    0.857143,
    4.285714,
    4.285714
  ]
}
```

Check the last two reconstructions: 4 and 7 both map to the largest code in this fixture. The clipped outlier does not retain its magnitude. If reconstructed zero is nonzero, inspect the order of subtracting the zero point. If a scale is negative, range validation is wrong. Keep finite-value validation in the eventual tensor implementation; NaNs and infinities should not flow unnoticed into an integer artifact.

[[practice:quantization-transfer]]

### Quantization error needs a distribution

For a narrow block containing mostly small weights and one outlier, the outlier can set a large step that erases small values. Clipping can reduce typical error while damaging the outlier. Which is acceptable depends on activations and task behavior, not a universal rule that all outliers should be retained or removed. Later calibration lessons quantify this sensitivity.

We can now encode/reconstruct values mathematically. Next we store the codes as actual packed bytes and count the metadata that the label “four bits” leaves out.
