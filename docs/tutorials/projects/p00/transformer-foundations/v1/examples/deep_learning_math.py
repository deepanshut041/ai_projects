"""Inspect a shared derivative, binary loss and a complete Tanh network backward pass."""

from __future__ import annotations

import argparse
import json
import math

from prerequisite_math import linear, logit_gradient, matmul, nll, softmax, transpose


def shared_parameter(weight):
    errors = [2 * weight - 1, 3 * weight - 1]
    contributions = [2 * errors[0] * 2, 2 * errors[1] * 3]
    return {"loss": sum(error * error for error in errors),
            "contributions": contributions, "gradient": sum(contributions)}


def binary_loss(logit, target):
    if not math.isfinite(logit) or type(target) is not int or target not in (0, 1):
        raise ValueError("A finite logit and binary integer target are required")
    if logit >= 0:
        probability = 1 / (1 + math.exp(-logit))
    else:
        term = math.exp(logit)
        probability = term / (1 + term)
    # Keep the loss finite without taking log of a rounded zero probability.
    loss = max(logit, 0) - logit * target + math.log1p(math.exp(-abs(logit)))
    error = probability - target
    return {"probability": probability, "cross_entropy": loss,
            "cross_entropy_gradient": error,
            "half_squared_loss": 0.5 * error * error,
            "half_squared_gradient": error * probability * (1 - probability)}


def mlp_trace(inputs, hidden_weights, hidden_bias, output_weights, output_bias, target):
    """One labelled row; matrices use input-by-output orientation, with no reduction."""
    z = linear([inputs], hidden_weights, hidden_bias)
    hidden = [[math.tanh(value) for value in z[0]]]
    logits = linear(hidden, output_weights, output_bias)[0]
    loss = nll(logits, target)
    delta_output = [logit_gradient(logits, target)]
    grad_hidden = matmul(delta_output, transpose(output_weights))
    delta_hidden = [[gradient * (1 - activation * activation)
                     for gradient, activation in zip(grad_hidden[0], hidden[0])]]
    return {"preactivation": z[0], "hidden": hidden[0], "logits": logits,
            "probabilities": softmax(logits), "loss": loss,
            "grad_logits": delta_output[0], "grad_hidden": grad_hidden[0],
            "grad_preactivation": delta_hidden[0],
            "grad_output_weights": matmul(transpose(hidden), delta_output),
            "grad_output_bias": delta_output[0],
            "grad_hidden_weights": matmul(transpose([inputs]), delta_hidden),
            "grad_hidden_bias": delta_hidden[0],
            "grad_inputs": matmul(delta_hidden, transpose(hidden_weights))[0]}


def worked_fixture():
    return {"inputs": [1.0, 2.0], "hidden_weights": [[1.0, -1.0], [-0.5, 0.5]],
            "hidden_bias": [0.0, math.log(3) / 2],
            "output_weights": [[1.0, -1.0], [0.0, 2 * math.log(3)]],
            "output_bias": [0.0, 0.0], "target": 1}


def torch_comparison(fixture=None):
    import torch
    from torch.nn import functional as F

    values = fixture or worked_fixture()
    names = ("inputs", "hidden_weights", "hidden_bias", "output_weights", "output_bias")
    tensors = {name: torch.tensor(values[name], dtype=torch.float64, requires_grad=True)
               for name in names}
    hidden = torch.tanh(tensors["inputs"].unsqueeze(0) @ tensors["hidden_weights"]
                        + tensors["hidden_bias"])
    logits = hidden @ tensors["output_weights"] + tensors["output_bias"]
    loss = F.cross_entropy(logits, torch.tensor([values["target"]]))
    loss.backward()
    expected = mlp_trace(**values)
    maximum = abs(float(loss.detach()) - expected["loss"])
    for name, tensor in tensors.items():
        known = torch.tensor(expected["grad_" + name], dtype=torch.float64)
        maximum = max(maximum, float((tensor.grad - known).abs().max()))
    return {"maximum_absolute_difference": maximum,
            "matches": maximum < 1e-12, "dtype": "float64", "device": "cpu"}


def report():
    trace = mlp_trace(**worked_fixture())
    binary = binary_loss(-math.log(99), 1)
    return {"shared_gradient": shared_parameter(1.0)["gradient"],
            "binary_probability": round(binary["probability"], 9),
            "binary_cross_entropy_gradient": round(binary["cross_entropy_gradient"], 9),
            "binary_half_squared_gradient": round(binary["half_squared_gradient"], 9),
            "hidden": [round(value, 9) for value in trace["hidden"]],
            "probabilities": [round(value, 9) for value in trace["probabilities"]],
            "loss": round(trace["loss"], 9),
            "grad_hidden_weights": [[round(value, 9) for value in row]
                                    for row in trace["grad_hidden_weights"]],
            "grad_output_weights": [[round(value, 9) for value in row]
                                    for row in trace["grad_output_weights"]]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch", action="store_true")
    args = parser.parse_args()
    result = report()
    if args.torch:
        result["torch"] = torch_comparison()
        if not result["torch"]["matches"]:
            raise AssertionError("Manual network derivatives disagree with autograd")
    print(json.dumps(result, indent=2))
