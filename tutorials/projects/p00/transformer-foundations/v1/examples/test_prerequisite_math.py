"""Independent known answers, perturbations and shape-boundary checks."""

from copy import deepcopy
import json
import math
import unittest

from prerequisite_math import (adam, affine_derivatives, central_difference, linear,
    logit_gradient, matmul, mean, nll, norm, population_variance,
    scalar_derivatives, sgd, softmax, standardize, transpose)


class PrerequisiteTests(unittest.TestCase):
    def test_preserved_linear_and_shared_bias(self):
        rows, weights = [[1, 2], [3, 4]], [[1, 0, 2], [0, 1, 3]]
        self.assertEqual(linear(rows, weights, [0, 0, 1]), [[1, 2, 9], [3, 4, 19]])
        self.assertEqual(linear(rows, weights, [0, 0, 0]), [[1, 2, 8], [3, 4, 18]])

    def test_rectangular_product_and_transpose(self):
        self.assertEqual(matmul([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]]), [[58, 64], [139, 154]])
        self.assertEqual(transpose([[1, 2, 3], [4, 5, 6]]), [[1, 4], [2, 5], [3, 6]])

    def test_shapes_rejected(self):
        for rows, weights, bias in [([], [[1]], [0]), ([[1, 2]], [[1]], [0]), ([[1], [2, 3]], [[1]], [0]), ([[1]], [[2, 3]], [0])]:
            with self.assertRaises(ValueError):
                linear(rows, weights, bias)

    def test_mean_population_variance_and_norm_are_distinct(self):
        self.assertEqual(mean([1, 3]), 2)
        self.assertEqual(population_variance([1, 3]), 1)
        self.assertEqual(norm([3, 4]), 5)
        self.assertEqual(population_variance([101, 103]), 1)
        self.assertNotEqual(norm([1, 3]), norm([101, 103]))

    def test_constant_standardization_and_empty_failure(self):
        self.assertEqual(standardize([5, 5]), [0, 0])
        for function in (mean, population_variance, norm, softmax):
            with self.assertRaises(ValueError):
                function([])

    def test_softmax_known_answer_and_common_shift(self):
        for shift in (0, 1000, -1000):
            actual = softmax([shift, shift + math.log(3)])
            self.assertAlmostEqual(actual[0], 0.25)
            self.assertAlmostEqual(actual[1], 0.75)
        with self.assertRaises(ValueError):
            softmax([float('inf'), 0])

    def test_nll_underflow_and_target_rejection(self):
        self.assertAlmostEqual(nll([0, math.log(3)], 1), -math.log(0.75))
        self.assertEqual(softmax([0, -1000])[1], 0)
        self.assertEqual(nll([0, -1000], 1), 1000)
        with self.assertRaises(ValueError):
            nll([0, 1], 2)

    def test_nll_common_shift_invariance_and_huge_equal_logits(self):
        base = [0.0, math.log(3)]
        for target in range(2):
            expected = nll(base, target)
            for offset in (-1000.0, 1000.0):
                self.assertAlmostEqual(nll([value + offset for value in base], target), expected, places=11)
            for offset in (-1e16, 1e16):
                loss = nll([offset, offset], target)
                self.assertGreater(loss, 0)
                self.assertAlmostEqual(loss, math.log(2), places=12)

    def test_scalar_partial_derivatives_and_sgd(self):
        result = scalar_derivatives(2, 3, 1)
        self.assertEqual(result['grad_weight'], 30)
        self.assertEqual(result['grad_bias'], 10)
        self.assertAlmostEqual(central_difference(lambda w: scalar_derivatives(w, 3, 1)['loss'], 2), 30)
        self.assertAlmostEqual(central_difference(lambda b: scalar_derivatives(2, 3, 1, b)['loss'], 0), 10)
        changed = sgd([2, 0], [30, 10], 0.01)
        self.assertEqual(changed, [1.7, -0.1])
        self.assertEqual(scalar_derivatives(changed[0], 3, 1, changed[1])['loss'], 16)

    def test_large_sgd_step_overshoots(self):
        weight = sgd([2], [30], 0.2)[0]
        self.assertEqual(scalar_derivatives(weight, 3, 1)['loss'], 169)

    def test_every_affine_input_weight_bias_gradient_by_central_difference(self):
        rows, weights, bias = [[1, -2], [3, 0.5]], [[0.2, 1, -1], [2, -3, 0.4]], [0.1, -0.2, 0.3]
        targets = [[0.2, -0.4, 1], [2, 0, -1]]
        analytic = affine_derivatives(rows, weights, bias, targets)
        for argument, data, key in [(0, rows, 'grad_inputs'), (1, weights, 'grad_weights')]:
            for r, row in enumerate(data):
                for c, value in enumerate(row):
                    def changed(replacement):
                        arguments = [rows, weights, bias]
                        arguments[argument] = deepcopy(data)
                        arguments[argument][r][c] = replacement
                        return affine_derivatives(*arguments, targets)['loss']
                    self.assertAlmostEqual(analytic[key][r][c], central_difference(changed, value), places=6)
        for c, value in enumerate(bias):
            def changed(replacement):
                copied = bias[:]
                copied[c] = replacement
                return affine_derivatives(rows, weights, copied, targets)['loss']
            self.assertAlmostEqual(analytic['grad_bias'][c], central_difference(changed, value), places=6)

    def test_affine_known_gradient_and_reduction_denominator(self):
        result = affine_derivatives([[1, 2], [3, 4]], [[1, 0], [0, 1]], [0, 0], [[0, 0], [0, 0]])
        self.assertEqual(result['loss'], 7.5)
        self.assertEqual(result['grad_weights'], [[5, 7], [7, 10]])
        self.assertEqual(result['grad_bias'], [2, 3])

    def test_logit_derivative_finite_difference(self):
        logits = [0.2, -0.4, 0.8]
        for target in range(3):
            gradient = logit_gradient(logits, target)
            self.assertAlmostEqual(sum(gradient), 0)
            for index, value in enumerate(logits):
                def changed(replacement):
                    copied = logits[:]
                    copied[index] = replacement
                    return nll(copied, target)
                self.assertAlmostEqual(gradient[index], central_difference(changed, value), places=7)

    def test_adam_history_and_zero_new_gradient(self):
        first, state = adam([2], [30])
        self.assertAlmostEqual(first[0], 1.99)
        second, state = adam(first, [0], state)
        self.assertAlmostEqual(state.first[0], 2.7)
        self.assertAlmostEqual(state.second[0], 0.8991)
        self.assertLess(second[0], first[0])
        reset, _ = adam(first, [0])
        self.assertEqual(reset, first)

    def test_adam_opposing_moments_independent_two_step_answer(self):
        first, state = adam([1], [2], beta1=0.5, beta2=0.5, learning_rate=0.1, epsilon=1e-6)
        second, state = adam(first, [-1], state, beta1=0.5, beta2=0.5, learning_rate=0.1, epsilon=1e-6)
        self.assertEqual((state.first, state.second), ([0], [1.5]))
        self.assertEqual(first, second)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(PrerequisiteTests))
    print(json.dumps({'tests': result.testsRun, 'passed': result.wasSuccessful()}, sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
