"""Independent answers and failure cases for single-observation arithmetic."""

import math
import unittest

from bike_demand_planner.linear import (
    half_squared_error, predict, sgd_step, squared_error_gradients,
)


class LinearTests(unittest.TestCase):
    def test_known_prediction_and_objective(self):
        self.assertEqual(predict([1.0, 2.0], [10.0, 5.0], 20.0), 40.0)
        self.assertEqual(half_squared_error(40.0, 45.0), 12.5)

    def test_known_update_and_unchanged_inputs(self):
        weights = [10.0, 5.0]
        gradients, bias_gradient = squared_error_gradients([1.0, 2.0], 40.0, 45.0)
        self.assertEqual(gradients, [-5.0, -10.0])
        self.assertEqual(bias_gradient, -5.0)
        next_weights, next_bias = sgd_step(weights, 20.0, gradients, bias_gradient, 0.01)
        self.assertEqual(weights, [10.0, 5.0])
        self.assertAlmostEqual(next_weights[0], 10.05)
        self.assertAlmostEqual(next_weights[1], 5.10)
        self.assertAlmostEqual(next_bias, 20.05)
        next_prediction = predict([1.0, 2.0], next_weights, next_bias)
        self.assertAlmostEqual(next_prediction, 40.30)
        self.assertAlmostEqual(half_squared_error(next_prediction, 45.0), 11.045)

    def test_gradients_match_finite_differences_on_a_different_case(self):
        features, weights, bias, target = [2.0, -3.0], [0.7, -1.2], 0.4, -2.0
        prediction = predict(features, weights, bias)
        gradients, bias_gradient = squared_error_gradients(features, prediction, target)
        epsilon = 1e-6

        def objective(parameters):
            value = predict(features, parameters[:2], parameters[2])
            return half_squared_error(value, target)

        parameters = [*weights, bias]
        for index, analytic in enumerate([*gradients, bias_gradient]):
            with self.subTest(parameter=index):
                plus, minus = parameters.copy(), parameters.copy()
                plus[index] += epsilon
                minus[index] -= epsilon
                numerical = (objective(plus) - objective(minus)) / (2 * epsilon)
                self.assertTrue(math.isclose(analytic, numerical, rel_tol=1e-6, abs_tol=1e-6))

    def test_lower_target_reverses_the_update(self):
        features, weights, bias, target = [1.0, 2.0], [10.0, 5.0], 20.0, 35.0
        prediction = predict(features, weights, bias)
        self.assertEqual(prediction, 40.0)
        self.assertEqual(half_squared_error(prediction, target), 12.5)
        gradients, bias_gradient = squared_error_gradients(features, prediction, target)
        self.assertEqual(gradients, [5.0, 10.0])
        self.assertEqual(bias_gradient, 5.0)
        next_weights, next_bias = sgd_step(weights, bias, gradients, bias_gradient, 0.01)
        self.assertEqual(weights, [10.0, 5.0])
        self.assertEqual(bias, 20.0)
        self.assertAlmostEqual(next_weights[0], 9.95)
        self.assertAlmostEqual(next_weights[1], 4.90)
        self.assertAlmostEqual(next_bias, 19.95)
        next_prediction = predict(features, next_weights, next_bias)
        self.assertAlmostEqual(next_prediction, 39.70)
        self.assertAlmostEqual(next_prediction - target, 4.70)
        self.assertAlmostEqual(half_squared_error(next_prediction, target), 11.045)

    def test_zero_residual_and_zero_rate_leave_parameters_unchanged(self):
        gradients, bias_gradient = squared_error_gradients([1.0, 2.0], 40.0, 40.0)
        self.assertEqual(sgd_step([10.0, 5.0], 20.0, gradients, bias_gradient, 0.01),
                         ([10.0, 5.0], 20.0))
        self.assertEqual(sgd_step([10.0, 5.0], 20.0, [-5.0, -10.0], -5.0, 0.0),
                         ([10.0, 5.0], 20.0))

    def test_mismatched_or_empty_vectors_are_rejected(self):
        for features, weights in [([], []), ([1.0, 2.0], [3.0])]:
            with self.subTest(features=features), self.assertRaises(ValueError):
                predict(features, weights, 0.0)
        with self.assertRaises(ValueError):
            squared_error_gradients([], 0.0, 1.0)
        with self.assertRaises(ValueError):
            sgd_step([1.0], 0.0, [1.0, 2.0], 0.0, 0.01)

    def test_nonfinite_values_and_negative_rate_are_rejected(self):
        for value in [math.nan, math.inf, -math.inf]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    predict([value], [1.0], 0.0)
                with self.assertRaises(ValueError):
                    half_squared_error(1.0, value)
                with self.assertRaises(ValueError):
                    squared_error_gradients([1.0], value, 1.0)
                with self.assertRaises(ValueError):
                    sgd_step([1.0], 0.0, [1.0], 0.0, value)
        with self.assertRaises(ValueError):
            sgd_step([1.0], 0.0, [1.0], 0.0, -0.01)


if __name__ == "__main__":
    unittest.main()
