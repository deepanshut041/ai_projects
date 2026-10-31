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
