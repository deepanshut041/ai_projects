"""P00 first-session arithmetic; standard library only, no training or files.

Run with --walk for an annotated explanation. Default JSON is checked by the
tutorial example runner. Predict before running; change inputs in your own copy.
"""
import argparse
import json
import math


def linear(rows, weight, bias):
    """Explicit X @ W + bias; W has input-by-output layout."""
    if not rows or not weight or not bias:
        raise ValueError("Use nonempty matrices and bias")
    input_width, output_width = len(weight), len(bias)
    if any(len(row) != input_width for row in rows):
        raise ValueError("Input row must match the weight input width")
    if any(len(row) != output_width for row in weight):
        raise ValueError("Weight row must match the output width")
    return [[sum(row[d] * weight[d][f] for d in range(input_width)) + bias[f]
             for f in range(output_width)] for row in rows]


def learning_step(weight, x, target, rate):
    """One analytic SGD step for squared error, not an autograd engine."""
    if not all(math.isfinite(v) for v in (weight, x, target, rate)) or rate <= 0:
        raise ValueError("Use finite values and a positive learning rate")
    prediction = weight * x
    residual = prediction - target
    gradient = 2 * residual * x
    updated = weight - rate * gradient
    return {"prediction": prediction, "loss_before": residual ** 2,
            "gradient": gradient, "new_weight": updated,
            "loss_after": (updated * x - target) ** 2}


def checked_results():
    result = linear([[1, 2], [3, 4]], [[1, 0, 2], [0, 1, 3]], [0, 0, 1])
    assert result == [[1, 2, 9], [3, 4, 19]]
    baseline = learning_step(2.0, 3.0, 1.0, 0.01)
    assert baseline["gradient"] == 30.0
    assert math.isclose(baseline["loss_after"], 16.81, abs_tol=1e-12)
    changed = learning_step(1.0, 2.0, 0.0, 0.1)
    assert changed["gradient"] == 8.0
    assert math.isclose(changed["new_weight"], 0.2, abs_tol=1e-12)
    assert math.isclose(changed["loss_after"], 0.16, abs_tol=1e-12)
    too_large = learning_step(2.0, 3.0, 1.0, 0.2)
    assert too_large["loss_after"] == 169.0 > too_large["loss_before"]
    rejected = False
    try:
        linear([[1]], [[1, 2], [3, 4]], [0, 0])
    except ValueError:
        rejected = True
    assert rejected
    return {"linear": result, "prediction": 6.0, "loss_before": 25.0,
            "gradient": 30.0, "new_weight": 1.7,
            "loss_after": round(baseline["loss_after"], 8),
            "changed_loss_after": round(changed["loss_after"], 8),
            "overshoot_loss": too_large["loss_after"], "shape_rejection": rejected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walk", action="store_true", help="Print annotated arithmetic")
    args = parser.parse_args()
    result = checked_results()
    if args.walk:
        print("P00 FIRST SESSION — checked arithmetic, no PyTorch needed\n")
        print("1. A layer combines input numbers with adjustable weights.")
        print("   X=[[1,2],[3,4]], W=[[1,0,2],[0,1,3]], b=[0,0,1]")
        print("   XW+b =", result["linear"])
        print("   First row, third output: 1*2 + 2*3 + 1 = 9.\n")
        print("2. One adjustable number can learn from an error.")
        print("   w=2, x=3, target=1: prediction=6; loss=(6-1)^2=25.")
        print("   dL/dw=2*(6-1)*3=30; rate=0.01 gives w=2-0.01*30=1.7.")
        print("   New prediction=5.1; new loss=16.81.\n")
        print("3. Direction is not enough: step size matters.")
        print("   With rate=0.2: w=-4, prediction=-12, loss=169 (worse).")
        print("   Try next: x=2, target=0, w=1, rate=0.1.")
        print("   Predict gradient, new weight and loss; remove --walk to check JSON.")
        print("\nContinue with chapters 2 and 3. This is a starting exercise,")
        print("not a completed model or a completed P00 milestone.")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
