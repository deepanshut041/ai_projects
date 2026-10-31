"""Known-answer and independent library checks for tensor prerequisites."""

import copy
import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from prerequisite_tensors import (
    FeatureMLP, categorical_nll, collate_token_records, feature_statistics,
    manual_layer_norm, stable_logsumexp, stable_softmax, standard_normal_cdf,
    token_mean, valid_loss_totals, weighted_information,
)


class PrerequisiteTensorChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_padding_ids_visibility_and_labels_known_answer(self):
        batch = collate_token_records([
            {"input_ids": [4, 9, 2], "labels": [1, 0, 1]},
            {"input_ids": [8], "labels": [0]},
        ])
        self.assertEqual(batch["input_ids"].tolist(), [[4, 9, 2], [8, 0, 0]])
        self.assertEqual(batch["valid"].tolist(), [[True, True, True], [True, False, False]])
        self.assertEqual(batch["labels"].tolist(), [[1, 0, 1], [0, -100, -100]])
        self.assertEqual(batch["input_ids"].dtype, torch.long)
        self.assertEqual(batch["valid"].dtype, torch.bool)

    def test_visible_input_can_have_no_supervised_label(self):
        batch = collate_token_records([{"input_ids": [4, 9], "labels": [1, -100]}])
        self.assertTrue(batch["valid"][0, 1])
        self.assertEqual(int((batch["labels"] != -100).sum()), 1)

    def test_collation_does_not_modify_records(self):
        records = [{"input_ids": [4, 9], "labels": [1, 0]}]
        original = copy.deepcopy(records)
        batch = collate_token_records(records)
        batch["input_ids"][0, 0] = 15
        self.assertEqual(records, original)

    def test_bad_ids_and_misaligned_labels_are_rejected(self):
        for record in ({"input_ids": [4., 9], "labels": [1, 0]},
                       {"input_ids": [4, 0], "labels": [1, 0]},
                       {"input_ids": [4, 9], "labels": [1]}):
            with self.assertRaises(ValueError):
                collate_token_records([record])

    def test_true_token_mean_does_not_count_padded_slots(self):
        losses = torch.tensor([[1., 1., 1.], [3., 100., 100.]], requires_grad=True)
        labels = torch.tensor([[1, 0, 1], [0, -100, -100]])
        loss = token_mean(losses, labels)
        self.assertEqual(float(loss.detach()), 1.5)
        loss.backward()
        torch.testing.assert_close(losses.grad, torch.tensor([[.25, .25, .25], [.25, 0., 0.]]))

    def test_unequal_batch_aggregation_uses_sums_and_counts(self):
        first = valid_loss_totals(torch.tensor([1., 1., 1.]), torch.tensor([0, 0, 0]))
        second = valid_loss_totals(torch.tensor([3., 3.]), torch.tensor([0, 0]))
        mean = (first[0] + second[0]) / (first[1] + second[1])
        self.assertAlmostEqual(float(mean), 1.8, places=6)
        self.assertNotAlmostEqual(float(mean), 2.)

    def test_ignored_nonfinite_slot_cannot_pollute_loss(self):
        losses = torch.tensor([1., float("nan")])
        labels = torch.tensor([0, -100])
        self.assertEqual(float(token_mean(losses, labels)), 1.)
        with self.assertRaises(ValueError):
            token_mean(losses, torch.tensor([-100, -100]))

    def test_softmax_small_known_answer_and_large_shift(self):
        row = torch.tensor([0., math.log(3)], dtype=torch.float64)
        expected = torch.tensor([.25, .75], dtype=torch.float64)
        torch.testing.assert_close(stable_softmax(row), expected)
        torch.testing.assert_close(stable_softmax(row + 1000), expected)

    def test_logsumexp_known_answer_and_library(self):
        row = torch.tensor([1000., 1000. + math.log(3)], dtype=torch.float64)
        self.assertAlmostEqual(float(stable_logsumexp(row)), 1000 + math.log(4), places=10)
        x = torch.randn(2, 3, 4, dtype=torch.float64)
        torch.testing.assert_close(stable_logsumexp(x), torch.logsumexp(x, dim=-1))

    def test_blocked_rows_are_zero_and_illegal_scores_do_not_matter(self):
        scores = torch.tensor([[0., math.log(3)], [999., -999.]], dtype=torch.float64)
        allowed = torch.tensor([[True, False], [False, False]])
        expected = torch.tensor([[1., 0.], [0., 0.]], dtype=torch.float64)
        torch.testing.assert_close(stable_softmax(scores, allowed), expected)
        changed = scores.clone()
        changed[~allowed] = float("nan")
        torch.testing.assert_close(stable_softmax(changed, allowed), expected)

    def test_probability_axis_is_independent_per_row(self):
        x = torch.tensor([[0., 0.], [0., math.log(3)]], dtype=torch.float64)
        torch.testing.assert_close(stable_softmax(x), torch.tensor([[.5, .5], [.25, .75]], dtype=torch.float64))

    def test_nll_one_hot_and_direct_probability_agree(self):
        logits = torch.tensor([[0., math.log(3)]], dtype=torch.float64)
        loss = categorical_nll(logits, torch.tensor([1]))
        self.assertAlmostEqual(float(loss), -math.log(.75), places=12)
        one_hot = torch.tensor([[0., 1.]], dtype=torch.float64)
        torch.testing.assert_close(loss, -(one_hot * stable_softmax(logits).log()).sum())

    def test_nll_outputs_and_gradients_match_library(self):
        x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([[1, 3, -100], [0, 2, 1]])
        ours = categorical_nll(x, labels)
        expected = F.cross_entropy(x.flatten(0, 1), labels.flatten(), ignore_index=-100)
        torch.testing.assert_close(ours, expected)
        ours_gradient, = torch.autograd.grad(ours, x, retain_graph=True)
        library_gradient, = torch.autograd.grad(expected, x)
        torch.testing.assert_close(ours_gradient, library_gradient)

    def test_nll_huge_common_shifts_preserve_loss_and_gradient(self):
        labels = torch.tensor([0, 1])
        known_gradient = torch.tensor([[-.25, .25], [.25, -.25]])
        for common_shift in (-1e20, -1e8, 0., 1e8, 1e20):
            with self.subTest(common_shift=common_shift):
                logits = torch.full((2, 2), common_shift, dtype=torch.float32, requires_grad=True)
                ours = categorical_nll(logits, labels)
                library = F.cross_entropy(logits, labels)
                torch.testing.assert_close(ours, torch.tensor(math.log(2), dtype=torch.float32))
                torch.testing.assert_close(ours, library)
                ours_gradient, = torch.autograd.grad(ours, logits, retain_graph=True)
                library_gradient, = torch.autograd.grad(library, logits)
                torch.testing.assert_close(ours_gradient, known_gradient)
                torch.testing.assert_close(ours_gradient, library_gradient)

    def test_nll_huge_shift_preserves_ignored_rows_and_reductions(self):
        logits = torch.full((3, 3), 1e20, dtype=torch.float32, requires_grad=True)
        labels = torch.tensor([2, -100, 0])
        expected_losses = torch.tensor([math.log(3), 0., math.log(3)], dtype=torch.float32)
        for reduction in ("none", "sum", "mean"):
            with self.subTest(reduction=reduction):
                ours = categorical_nll(logits, labels, reduction=reduction)
                library = F.cross_entropy(logits, labels, ignore_index=-100, reduction=reduction)
                expected = expected_losses if reduction == "none" else (
                    expected_losses.sum() if reduction == "sum" else expected_losses.sum() / 2)
                torch.testing.assert_close(ours, expected)
                torch.testing.assert_close(ours, library)
        mean = categorical_nll(logits, labels)
        library = F.cross_entropy(logits, labels, ignore_index=-100)
        ours_gradient, = torch.autograd.grad(mean, logits, retain_graph=True)
        library_gradient, = torch.autograd.grad(library, logits)
        known_gradient = torch.tensor([[1 / 6, 1 / 6, -1 / 3], [0., 0., 0.], [-1 / 3, 1 / 6, 1 / 6]])
        torch.testing.assert_close(ours_gradient, known_gradient)
        torch.testing.assert_close(ours_gradient, library_gradient)

    def test_weighted_information_actual_matrix_result(self):
        weights = torch.tensor([[.25, .75], [.6, .4], [0., 0.]], dtype=torch.float64)
        values = torch.tensor([[2., 0.], [0., 4.]], dtype=torch.float64)
        expected = torch.tensor([[.5, 3.], [1.2, 1.6], [0., 0.]], dtype=torch.float64)
        torch.testing.assert_close(weighted_information(weights, values), expected)

    def test_value_change_preserves_selection(self):
        weights = torch.tensor([[.25, .75]], dtype=torch.float64)
        before = weights.clone()
        first = weighted_information(weights, torch.tensor([[2., 0.], [0., 4.]], dtype=torch.float64))
        second = weighted_information(weights, torch.tensor([[2., 0.], [0., 40.]], dtype=torch.float64))
        self.assertEqual(float(first[0, 0]), float(second[0, 0]))
        self.assertEqual(float(second[0, 1]), 30.)
        torch.testing.assert_close(weights, before)

    def test_population_feature_statistics_and_norm(self):
        x = torch.tensor([[1., 3.]], dtype=torch.float64)
        mean, variance = feature_statistics(x)
        self.assertEqual(float(mean), 2.)
        self.assertEqual(float(variance), 1.)
        expected = torch.tensor([[-1., 1.]], dtype=torch.float64) / math.sqrt(1 + 1e-5)
        torch.testing.assert_close(manual_layer_norm(x), expected)

    def test_layer_norm_matches_library_and_gradient_check(self):
        x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
        scale = torch.randn(4, dtype=torch.float64, requires_grad=True)
        bias = torch.randn(4, dtype=torch.float64, requires_grad=True)
        torch.testing.assert_close(manual_layer_norm(x, scale, bias), F.layer_norm(x, (4,), scale, bias))
        self.assertTrue(torch.autograd.gradcheck(lambda a, b, c: manual_layer_norm(a, b, c), (x, scale, bias)))

    def test_module_registration_buffer_and_shape(self):
        model = FeatureMLP()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 42)
        self.assertIn("input_scale", dict(model.named_buffers()))
        self.assertNotIn("input_scale", dict(model.named_parameters()))
        self.assertIn("input_scale", model.state_dict())
        self.assertEqual(tuple(model(torch.ones(3, 2)).shape), (3, 2))

    def test_linear_known_weight_orientation(self):
        layer = nn.Linear(2, 2).double()
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[1., 2.], [-1., .5]], dtype=torch.float64))
            layer.bias.copy_(torch.tensor([.5, -.5], dtype=torch.float64))
        torch.testing.assert_close(layer(torch.tensor([[3., 4.]], dtype=torch.float64)),
                                   torch.tensor([[11.5, -1.5]], dtype=torch.float64))

    def test_embedding_lookup_and_gradient_row_ownership(self):
        table = nn.Embedding(4, 3, padding_idx=0).double()
        with torch.no_grad():
            table.weight.copy_(torch.tensor([[0., 0., 0.], [1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=torch.float64))
        ids = torch.tensor([[2, 1, 0]])
        output = table(ids)
        torch.testing.assert_close(output, torch.tensor([[[4., 5., 6.], [1., 2., 3.], [0., 0., 0.]]], dtype=torch.float64))
        output.sum().backward()
        self.assertEqual(table.weight.grad[0].abs().sum().item(), 0.)
        self.assertEqual(table.weight.grad[3].abs().sum().item(), 0.)
        self.assertEqual(table.weight.grad[2].tolist(), [1., 1., 1.])

    def test_relu_creates_nonlinearity(self):
        inputs = torch.tensor([-2., 0., 3.])
        torch.testing.assert_close(F.relu(inputs), torch.tensor([0., 0., 3.]))
        self.assertNotEqual(float(F.relu(torch.tensor(-1.)) + F.relu(torch.tensor(1.))), float(F.relu(torch.tensor(0.))))

    def test_tanh_known_answers_and_saturation(self):
        x = torch.tensor([-1., 0., 1., 10.], dtype=torch.float64, requires_grad=True)
        y = torch.tanh(x)
        torch.testing.assert_close(y[:3], torch.tensor([-0.7615941559557649, 0., 0.7615941559557649], dtype=torch.float64))
        y.sum().backward()
        self.assertEqual(float(x.grad[1]), 1.)
        self.assertLess(float(x.grad[3]), 1e-7)
        self.assertTrue(bool((y.abs() <= 1).all()))

    def test_dropout_eval_and_no_grad_are_independent(self):
        dropout = nn.Dropout(.5)
        x = torch.ones(64, requires_grad=True)
        dropout.eval()
        output = dropout(x)
        torch.testing.assert_close(output, x)
        self.assertTrue(output.requires_grad)
        dropout.train()
        with torch.no_grad():
            changed = dropout(x)
        self.assertFalse(changed.requires_grad)
        self.assertTrue(bool((changed == 0).any()))
        self.assertTrue(bool((changed == 2).any()))

    def test_gaussian_cdf_known_center_and_symmetry(self):
        x = torch.tensor([-1., 0., 1.], dtype=torch.float64)
        probabilities = standard_normal_cdf(x)
        self.assertAlmostEqual(float(probabilities[1]), .5, places=12)
        self.assertAlmostEqual(float(probabilities[0]), .15865525393145707, places=12)
        self.assertAlmostEqual(float(probabilities[2]), 1 - float(probabilities[0]), places=12)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
