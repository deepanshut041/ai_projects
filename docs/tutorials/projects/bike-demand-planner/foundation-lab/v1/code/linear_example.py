"""Compare single-observation updates from the same initial parameters."""

import json

from bike_demand_planner.linear import (
    half_squared_error, predict, sgd_step, squared_error_gradients,
)


if __name__ == "__main__":
    working_day = 1.0
    hour = 8.0
    hour_scale = 4.0
    features = [working_day, hour / hour_scale]
    weights = [10.0, 5.0]
    bias = 20.0
    targets = [45.0, 35.0]
    learning_rate = 0.01

    cases = []
    for target in targets:
        # Each target starts from the same weights and bias.
        prediction = predict(features, weights, bias)
        loss = half_squared_error(prediction, target)
        weight_gradients, bias_gradient = squared_error_gradients(features, prediction, target)
        next_weights, next_bias = sgd_step(
            weights, bias, weight_gradients, bias_gradient, learning_rate,
        )
        next_prediction = predict(features, next_weights, next_bias)
        next_loss = half_squared_error(next_prediction, target)
        cases.append({
            "target": target,
            "before": {
                "weights": weights, "bias": bias, "prediction": prediction,
                "residual": prediction - target, "loss": loss,
            },
            "gradients": {"weights": weight_gradients, "bias": bias_gradient},
            "after": {
                "weights": next_weights, "bias": next_bias,
                "prediction": next_prediction,
                "residual": next_prediction - target, "loss": next_loss,
            },
        })

    print(json.dumps({
        "illustrative_example": True,
        "feature_order": ["working_day", "hour_divided_by_four"],
        "features": features,
        "learning_rate": learning_rate,
        "cases": cases,
    }, indent=2))
