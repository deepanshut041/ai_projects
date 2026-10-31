"""Known answers, central differences and boundary checks for foundation probes."""

import math
import unittest

from foundation_math import (
    AdamState, adam_step, assert_split_disjoint, categorical_loss_and_gradient,
    decode_word_bpe, dot, encode_word_bpe, finite_difference, fit_word_bpe, linear,
    linear_backward, matmul, merge_pair, scalar_mse_and_gradients,
    stable_softmax, transpose, utf8_ids, utf8_text, visibility_mask,
)


class FoundationChecks(unittest.TestCase):
    def assert_vector_close(self, actual, expected, places=7):
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right, places=places)

    def test_rectangular_matrix_known_answer(self):
        self.assertEqual(matmul([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]]), [[58, 64], [139, 154]])
        self.assertEqual(transpose([[1, 2, 3], [4, 5, 6]]), [[1, 4], [2, 5], [3, 6]])
        self.assertEqual(dot([2, -1], [3, 4]), 2)

    def test_linear_bias_broadcast(self):
        self.assertEqual(linear([[1, 0], [0, 1], [1, 1]], [[1, 0], [1, 1]], [0.5, -0.5]), [[1.5, -0.5], [1.5, 0.5], [2.5, 0.5]])

    def test_shape_errors(self):
        for left, right in [([[1, 2]], [[1, 2]]), ([[1], [2, 3]], [[1]]), ([], [[1]])]:
            with self.assertRaises(ValueError):
                matmul(left, right)
        with self.assertRaises(ValueError):
            linear([[1, 2]], [[3], [4]], [0, 0])

    def test_linear_backward_finite_difference_all_entries(self):
        inputs, weights, bias = [[1, -2], [0.5, 3]], [[2, -1, 0.5], [4, 3, -2]], [0.1, 0.2, -0.3]
        grad_output = [[0.2, -0.5, 0.8], [-1, 2, 0.3]]
        grad_inputs, grad_weights, grad_bias = linear_backward(inputs, weights, grad_output)

        def objective(x, w, b):
            output = linear(x, w, b)
            return sum(value * grad_output[r][c] for r, row in enumerate(output) for c, value in enumerate(row))

        for original, gradient, argument in [(inputs, grad_inputs, 0), (weights, grad_weights, 1)]:
            for row, values in enumerate(original):
                for column, value in enumerate(values):
                    def changed(replacement):
                        copied = [part[:] for part in original]
                        copied[row][column] = replacement
                        arguments = [inputs, weights, bias]
                        arguments[argument] = copied
                        return objective(*arguments)
                    self.assertAlmostEqual(gradient[row][column], finite_difference(changed, value), places=7)
        for column, value in enumerate(bias):
            def changed(replacement):
                copied = bias[:]
                copied[column] = replacement
                return objective(inputs, weights, copied)
            self.assertAlmostEqual(grad_bias[column], finite_difference(changed, value), places=7)

    def test_softmax_known_probability_and_shift_invariance(self):
        self.assert_vector_close(stable_softmax([0, math.log(3)]), [0.25, 0.75])
        self.assert_vector_close(stable_softmax([1000, 1000 + math.log(3)]), [0.25, 0.75])
        self.assertAlmostEqual(sum(stable_softmax([-500, 0, 500])), 1)

    def test_categorical_gradient_finite_difference(self):
        logits = [-0.7, 0.5, 1.2]
        loss, probabilities, gradient = categorical_loss_and_gradient(logits, 1)
        self.assertAlmostEqual(loss, -math.log(probabilities[1]))
        self.assertAlmostEqual(sum(gradient), 0)
        for index, value in enumerate(logits):
            def changed(replacement):
                copied = logits[:]
                copied[index] = replacement
                return categorical_loss_and_gradient(copied, 1)[0]
            self.assertAlmostEqual(gradient[index], finite_difference(changed, value), places=7)

    def test_loss_stays_finite_when_target_probability_underflows(self):
        loss, probabilities, gradient = categorical_loss_and_gradient([0, -1000], 1)
        self.assertEqual(probabilities[1], 0)
        self.assertEqual(loss, 1000)
        self.assertEqual(gradient, [1, -1])

    def test_softmax_invalid_inputs(self):
        for logits in [[], [float("inf"), 0], [float("nan")]]:
            with self.assertRaises(ValueError):
                stable_softmax(logits)
        with self.assertRaises(ValueError):
            categorical_loss_and_gradient([1, 2], 2)

    def test_scalar_gradient_and_step_reduce_loss(self):
        inputs, weights, bias, target = [2, -1], [0.5, -1], 0, 1
        prediction, loss, grad_weights, grad_bias = scalar_mse_and_gradients(inputs, weights, bias, target)
        self.assertEqual((prediction, loss, grad_weights, grad_bias), (2, 0.5, [2, -1], 1))
        changed_weights = [value - 0.1 * gradient for value, gradient in zip(weights, grad_weights)]
        changed_bias = bias - 0.1 * grad_bias
        new_prediction, new_loss, *_ = scalar_mse_and_gradients(inputs, changed_weights, changed_bias, target)
        self.assertAlmostEqual(new_prediction, 1.4)
        self.assertAlmostEqual(new_loss, 0.08)
        for index, weight in enumerate(weights):
            def changed(replacement):
                copied = weights[:]
                copied[index] = replacement
                return scalar_mse_and_gradients(inputs, copied, bias, target)[1]
            self.assertAlmostEqual(grad_weights[index], finite_difference(changed, weight), places=7)

    def test_adam_two_steps_independent_unrolled_answer(self):
        updated, first = adam_step([1], [2], learning_rate=0.1, beta1=0.5, beta2=0.5, epsilon=1e-6)
        self.assertAlmostEqual(updated[0], 1 - 0.1 * 2 / (2 + 1e-6))
        self.assertEqual(first, AdamState(1, [1], [2]))
        updated_again, second = adam_step(updated, [-1], first, learning_rate=0.1, beta1=0.5, beta2=0.5, epsilon=1e-6)
        self.assertEqual(second, AdamState(2, [0], [1.5]))
        self.assertEqual(updated_again, updated)
        # Replacing the state by zeros would instead make the parameter increase.
        reset, _ = adam_step(updated, [-1], learning_rate=0.1, beta1=0.5, beta2=0.5, epsilon=1e-6)
        self.assertGreater(reset[0], updated[0])

    def test_utf8_multibyte_roundtrip(self):
        self.assertEqual(utf8_ids("café"), [99, 97, 102, 195, 169])
        for text in ["", "café", "bonjour", "🙂"]:
            self.assertEqual(utf8_text(utf8_ids(text)), text)
        with self.assertRaises(UnicodeDecodeError):
            utf8_text([195])

    def test_bpe_weighted_counts_and_roundtrip(self):
        merges = fit_word_bpe({"low": 5, "lower": 2, "lowest": 1}, 3)
        self.assertEqual(merges, [("l", "o"), ("lo", "w"), ("low", "</w>")])
        self.assertEqual(encode_word_bpe("lower", merges), ("low", "e", "r", "</w>"))
        for word in ["low", "lower", "lowest", "new", ""]:
            self.assertEqual(decode_word_bpe(encode_word_bpe(word, merges)), word)

    def test_bpe_nonoverlapping_merge(self):
        self.assertEqual(merge_pair(("a", "a", "a"), ("a", "a")), ("aa", "a"))
        with self.assertRaises(ValueError):
            fit_word_bpe({"low": 0}, 1)
        with self.assertRaises(ValueError):
            decode_word_bpe(("l", "o", "w"))

    def test_padding_and_causal_visibility(self):
        self.assertEqual(visibility_mask(3, [True, True, False]), [[True, True, False]] * 3)
        self.assertEqual(visibility_mask(3, [True, True, False], causal=True), [[True, False, False], [True, True, False], [True, True, False]])
        self.assertEqual(visibility_mask(2, [True, False, True]), [[True, False, True], [True, False, True]])
        with self.assertRaises(ValueError):
            visibility_mask(2, [True, True, True], causal=True)

    def test_group_leakage_is_rejected(self):
        assert_split_disjoint({"train": ["doc-a", "doc-a"], "validation": ["doc-b"]})
        with self.assertRaisesRegex(ValueError, "doc-a"):
            assert_split_disjoint({"train": ["doc-a"], "validation": ["doc-a"]})


if __name__ == "__main__":
    unittest.main(verbosity=2)
