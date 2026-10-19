"""Analytic answers, finite differences, fit isolation and stable solver checks."""

import unittest

import numpy as np

from bike_demand_planner.regression import (
    batch_step, fit_linear_regression, fit_standardizer, forward,
    least_squares_reference, loss_and_gradients,
)


X = np.array([[1, 0], [0, 1], [1, 1], [2, -1]], dtype=np.float64)
Y = np.array([5, 2, 4, 8], dtype=np.float64)


class RegressionTests(unittest.TestCase):
    def test_unequal_dimensions_forward_loss_and_known_gradients(self):
        prediction = forward(X, [1, 1], 0)
        np.testing.assert_array_equal(prediction, [1, 1, 2, 1])
        self.assertEqual(prediction.dtype, np.float64)
        loss, dw, db = loss_and_gradients(X, Y, [1, 1], 0)
        self.assertEqual(loss, 8.75)
        np.testing.assert_array_equal(dw, [-5, 1])
        self.assertEqual(db, -3.5)
        loss_twice, dw_twice, db_twice = loss_and_gradients(np.tile(X, (2, 1)), np.tile(Y, 2), [1, 1], 0)
        self.assertEqual(loss_twice, loss)
        np.testing.assert_array_equal(dw_twice, dw)
        self.assertEqual(db_twice, db)

    def test_two_updates_and_an_overshooting_step_have_independent_answers(self):
        weights = np.array([1.0, 1.0])
        w1, b1 = batch_step(X, Y, weights, 0, .1)
        np.testing.assert_array_equal(weights, [1, 1])
        np.testing.assert_allclose(w1, [1.5, .9], atol=1e-12)
        self.assertAlmostEqual(b1, .35)
        np.testing.assert_allclose(forward(X, w1, b1), [1.85, 1.25, 2.75, 2.45], atol=1e-12)
        self.assertAlmostEqual(loss_and_gradients(X, Y, w1, b1)[0], 5.35625)
        w2, b2 = batch_step(X, Y, w1, b1, .1)
        np.testing.assert_allclose(w2, [1.8875, .81125], atol=1e-12)
        self.assertAlmostEqual(b2, .6175)
        wrong_w, wrong_b = batch_step(X, Y, weights, 0, 1)
        self.assertAlmostEqual(loss_and_gradients(X, Y, wrong_w, wrong_b)[0], 13.625)

    def test_batch_gradient_matches_central_finite_differences(self):
        parameters = np.array([.7, -1.2, .4])
        _, dw, db = loss_and_gradients(X, Y, parameters[:2], parameters[2])
        analytical = np.append(dw, db)
        numerical = []
        for index in range(3):
            plus, minus = parameters.copy(), parameters.copy()
            plus[index] += 1e-6
            minus[index] -= 1e-6
            numerical.append((loss_and_gradients(X, Y, plus[:2], plus[2])[0]
                              - loss_and_gradients(X, Y, minus[:2], minus[2])[0]) / 2e-6)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-6, atol=1e-7)

    def test_shapes_empty_values_nonfinite_and_bad_fit_settings_are_rejected(self):
        for features, targets, weights, bias in (
            ([], [], [1, 1], 0), (X.T, Y, [1, 1], 0), (X, Y[:, None], [1, 1], 0),
            (X, Y, [[1, 1]], 0), (X, Y, [1], 0), (X, Y, [1, 1], [0]),
            ([[np.nan, 0]], [1], [1, 1], 0), (X, [5, 2, 4, np.inf], [1, 1], 0),
        ):
            with self.subTest(features=features), self.assertRaises(ValueError):
                loss_and_gradients(features, targets, weights, bias)
        for settings in ({'learning_rate': 0}, {'learning_rate': 10}, {'learning_rate': True},
                         {'max_steps': 0}, {'max_steps': True}, {'gradient_tolerance': 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                fit_linear_regression(X, Y, **settings)

    def test_standardizer_uses_training_population_statistics_and_handles_constants(self):
        train = np.array([[1, 0], [1, 2]], dtype=np.float64)
        scaler = fit_standardizer(train)
        self.assertEqual(scaler.means, (1, 1))
        self.assertEqual(scaler.scales, (1, 1))
        self.assertEqual(scaler.constant_columns, (0,))
        np.testing.assert_array_equal(scaler.transform([[1, 100]]), [[0, 99]])
        train[:] = 999
        self.assertEqual(scaler.means, (1, 1))

    def test_fit_agrees_with_stable_reference_and_raw_coordinate_predictions(self):
        fit = fit_linear_regression(X, Y, gradient_tolerance=1e-10)
        self.assertTrue(fit.converged)
        reference, rank = least_squares_reference(X, Y, fit.model.standardizer)
        self.assertEqual(rank, 3)
        np.testing.assert_allclose(fit.model.predict(X), Y, atol=1e-8)
        np.testing.assert_allclose(fit.model.predict(X), reference.predict(X), atol=1e-8)
        raw_weights, raw_bias = fit.model.raw_parameters()
        np.testing.assert_allclose(raw_weights, [2, -1], atol=1e-8)
        self.assertAlmostEqual(raw_bias, 3, places=8)
        np.testing.assert_allclose(forward(X, raw_weights, raw_bias), fit.model.predict(X), atol=1e-10)
        self.assertTrue(all(b.half_mse <= a.half_mse + 1e-12 for a, b in zip(fit.history, fit.history[1:])))

    def test_rank_deficiency_is_not_solved_by_inverting_a_normal_matrix(self):
        duplicated = np.column_stack((X[:, 0], X[:, 0], np.ones(len(X))))
        fit = fit_linear_regression(duplicated, Y)
        reference, rank = least_squares_reference(duplicated, Y, fit.model.standardizer)
        self.assertLess(rank, duplicated.shape[1] + 1)
        self.assertTrue(fit.converged)
        np.testing.assert_allclose(fit.model.predict(duplicated), reference.predict(duplicated), atol=1e-6)

    def test_update_limit_is_explicit_and_prediction_does_not_fit_to_new_inputs(self):
        fit = fit_linear_regression(X, Y, max_steps=1, gradient_tolerance=1e-14)
        self.assertFalse(fit.converged)
        self.assertEqual(fit.stop_reason, 'maximum_updates')
        self.assertEqual([point.step for point in fit.history], [0, 1])
        old_state = (fit.model.weights, fit.model.bias, fit.model.standardizer)
        fit.model.predict([[100, -100]])
        self.assertEqual((fit.model.weights, fit.model.bias, fit.model.standardizer), old_state)


if __name__ == '__main__':
    unittest.main()
