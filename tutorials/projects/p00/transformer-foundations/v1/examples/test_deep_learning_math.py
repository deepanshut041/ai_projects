"""Compare new teaching derivatives with forward perturbations and known answers."""

from copy import deepcopy
import math
import unittest

from deep_learning_math import binary_loss, mlp_trace, shared_parameter, worked_fixture
from prerequisite_math import central_difference


class DeepLearningMathChecks(unittest.TestCase):
    def test_shared_weight_adds_both_paths(self):
        result = shared_parameter(1.0)
        self.assertEqual(result["contributions"], [4.0, 12.0])
        self.assertEqual(result["loss"], 5.0)
        self.assertEqual(result["gradient"], 16.0)
        measured = central_difference(lambda w: (2 * w - 1)**2 + (3 * w - 1)**2, 1.0)
        self.assertAlmostEqual(measured, 16.0, places=7)

    def test_binary_cross_entropy_gradient_from_forward_loss(self):
        for target in (0, 1):
            for logit in (-8.0, -math.log(99), 0.0, math.log(3), 8.0):
                result = binary_loss(logit, target)
                for loss, gradient in [("cross_entropy", "cross_entropy_gradient"),
                                       ("half_squared_loss", "half_squared_gradient")]:
                    measured = central_difference(lambda a: binary_loss(a, target)[loss], logit)
                    self.assertAlmostEqual(measured, result[gradient], places=7)
        wrong = binary_loss(-math.log(99), 1)
        self.assertAlmostEqual(wrong["probability"], 0.01)
        self.assertAlmostEqual(wrong["cross_entropy_gradient"], -0.99)
        self.assertAlmostEqual(wrong["half_squared_gradient"], -0.009801)

    def test_extreme_binary_logits_and_invalid_contract(self):
        for logit, target, expected in [(-1000, 1, 1000), (1000, 0, 1000),
                                        (1000, 1, 0), (-1000, 0, 0)]:
            self.assertEqual(binary_loss(logit, target)["cross_entropy"], expected)
        for logit, target in [(float('nan'), 1), (float('inf'), 0), (0, 2), (0, True)]:
            with self.assertRaises(ValueError):
                binary_loss(logit, target)

    def test_controlled_two_layer_hand_answers(self):
        result = mlp_trace(**worked_fixture())
        self.assertAlmostEqual(result["hidden"][0], 0.0)
        self.assertAlmostEqual(result["hidden"][1], 0.5)
        self.assertAlmostEqual(result["probabilities"][0], 0.25)
        self.assertAlmostEqual(result["probabilities"][1], 0.75)
        self.assertAlmostEqual(result["loss"], math.log(4 / 3))
        expected = [[0.5, -3 * math.log(3) / 8], [1.0, -3 * math.log(3) / 4]]
        for actual_row, expected_row in zip(result["grad_hidden_weights"], expected):
            for actual, known in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, known)
        self.assertEqual(result["grad_output_weights"][0], [0.0, 0.0])
        self.assertAlmostEqual(result["grad_output_weights"][1][0], 0.125)

    def test_every_coordinate_against_forward_perturbations(self):
        fixtures = [worked_fixture(),
                    {"inputs": [0.7, -1.2],
                     "hidden_weights": [[0.2, -0.4, 0.8], [0.9, 0.3, -0.5]],
                     "hidden_bias": [0.1, -0.2, 0.3],
                     "output_weights": [[0.2, 0.4, -0.6], [-0.5, 0.8, 0.1], [0.7, -0.3, 0.2]],
                     "output_bias": [0.1, -0.2, 0.05], "target": 2}]
        for fixture in fixtures:
            trace = mlp_trace(**fixture)
            for name in ("inputs", "hidden_weights", "hidden_bias", "output_weights", "output_bias"):
                value = fixture[name]
                matrix = isinstance(value[0], list)
                positions = [(r, c) for r, row in enumerate(value) for c in range(len(row))] if matrix else [(i,) for i in range(len(value))]
                for indices in positions:
                    def forward_at(changed):
                        candidate = deepcopy(fixture)
                        if matrix:
                            candidate[name][indices[0]][indices[1]] = changed
                        else:
                            candidate[name][indices[0]] = changed
                        return mlp_trace(**candidate)["loss"]
                    old = value[indices[0]][indices[1]] if matrix else value[indices[0]]
                    known = trace["grad_" + name]
                    analytic = known[indices[0]][indices[1]] if matrix else known[indices[0]]
                    measured = central_difference(forward_at, old)
                    with self.subTest(shape=len(fixture["hidden_bias"]), name=name, indices=indices):
                        self.assertTrue(math.isclose(measured, analytic, rel_tol=2e-6, abs_tol=2e-8), (measured, analytic))

    def test_target_change_reverses_class_signal_without_changing_forward(self):
        fixture = worked_fixture()
        original = mlp_trace(**fixture)
        fixture['target'] = 0
        changed = mlp_trace(**fixture)
        self.assertEqual(original['hidden'], changed['hidden'])
        self.assertEqual(original['logits'], changed['logits'])
        self.assertAlmostEqual(changed['grad_logits'][0], -0.75)
        self.assertAlmostEqual(changed['grad_logits'][1], 0.75)


if __name__ == "__main__":
    unittest.main()
