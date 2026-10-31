"""Independent arithmetic, gradient and information-flow checks for the probes."""

import itertools
import math
import unittest

import torch
from torch.nn import functional as F

import transformer_foundation_probes as probes


DTYPE = torch.float64


def tensor(values, *, grad=False):
    return torch.tensor(values, dtype=DTYPE, requires_grad=grad)


def random_tensor(shape, generator, *, grad=False):
    return torch.randn(shape, generator=generator, dtype=DTYPE).requires_grad_(grad)


class FoundationProbeChecks(unittest.TestCase):
    def close(self, actual, expected, *, atol=1e-12):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=1e-12)

    def attention_fixture(self, *, grad=False):
        return (tensor([[1.]], grad=grad),
                tensor([[0.], [math.log(2)], [math.log(3)]], grad=grad),
                tensor([[0., 0.], [6., 0.], [4., 2.]], grad=grad))

    def test_position_known_even_and_odd_coordinates(self):
        self.close(probes.sinusoidal_positions(1, 4), tensor([[0., 1., 0., 1.]]))
        self.close(probes.sinusoidal_positions(1, 5), tensor([[0., 1., 0., 1., 0.]]))
        expected = tensor([[math.sin(1), math.cos(1), math.sin(.01), math.cos(.01)]])
        self.close(probes.sinusoidal_positions(1, 4, offset=1), expected)
        odd = probes.sinusoidal_positions(2, 5)
        self.assertAlmostEqual(float(odd[1, 4]), math.sin(10000 ** (-4 / 5)), places=14)
        self.assertEqual(probes.sinusoidal_positions(0, 3).shape, (0, 3))

    def test_fixed_position_shift_is_pairwise_rotation(self):
        original = probes.sinusoidal_positions(1, 6, offset=3)[0]
        shifted = probes.sinusoidal_positions(1, 6, offset=5)[0]
        for pair in range(3):
            angle = 2 * 10000 ** (-2 * pair / 6)
            rotation = tensor([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
            self.close(rotation @ original[2 * pair:2 * pair + 2], shifted[2 * pair:2 * pair + 2])

    def test_embedding_lookup_equals_one_hot_multiply(self):
        table = tensor([[0., 0.], [1., 2.], [3., 4.], [-1., 1.]])
        ids = torch.tensor([2, 1, 2])
        one_hot = F.one_hot(ids, 4).to(DTYPE)
        self.close(F.embedding(ids, table), one_hot @ table)
        self.close(F.embedding(ids, table), tensor([[3., 4.], [1., 2.], [3., 4.]]))

    def test_repeated_ids_sum_and_padding_lookup_does_not_contribute(self):
        result = probes.embedding_gradient_probe()
        self.close(result["lookup_grad"], tensor([[0., 0.], [2., 2.], [4., 4.], [0., 0.]]))
        self.assertEqual(result["lookup_grad"].shape, (4, 2))
        self.assertGreater(float(result["tied_output_grad"][0, 0]), 0)
        self.assertEqual(float(result["tied_output_grad"][0, 1]), 0)

    def test_padding_gradient_suppression_does_not_force_assigned_value_to_zero(self):
        table = tensor([[5., 7.], [1., 2.]], grad=True)
        looked_up = F.embedding(torch.tensor([0, 1]), table, padding_idx=0)
        self.close(looked_up[0], tensor([5., 7.]))
        looked_up.sum().backward()
        self.close(table.grad, tensor([[0., 0.], [1., 1.]]))

    def test_tied_table_is_one_parameter_with_two_gradient_paths(self):
        # Restore the global RNG after constructors: the probe suite owns no seed.
        with torch.random.fork_rng():
            model = torch.nn.Module()
            model.embedding = torch.nn.Embedding(4, 2, padding_idx=0, dtype=DTYPE)
            model.output = torch.nn.Linear(2, 4, bias=False, dtype=DTYPE)
        model.output.weight = model.embedding.weight
        self.assertIs(model.output.weight, model.embedding.weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        self.assertEqual(len(optimizer.param_groups[0]["params"]), 1)
        lookup_loss = model.embedding(torch.tensor([0, 2])).sum()
        output_loss = model.output(tensor([[1., 0.]])).sum()
        (lookup_loss + output_loss).backward()
        # Output projection contributes [1,0] to every row; lookup adds [1,1] at 2.
        self.close(model.embedding.weight.grad, tensor([[1., 0.], [1., 0.], [2., 1.], [1., 0.]]))

    def test_attention_hand_weights_values_and_mask_renormalization(self):
        q, k, v = self.attention_fixture()
        output, weights = probes.explicit_attention(q, k, v)
        self.close(weights, tensor([[1 / 6, 1 / 3, 1 / 2]]))
        self.close(output, tensor([[4., 1.]]))
        masked, attention = probes.explicit_attention(q, k, v, torch.tensor([[True, False, True]]))
        self.close(attention, tensor([[.25, 0., .75]]))
        self.close(masked, tensor([[3., 1.5]]))
        zero_values = v.clone()
        zero_values[1] = 0
        self.close(probes.explicit_attention(q, k, zero_values)[0], tensor([[2., 1.]]))

    def test_attention_changes_values_without_changing_weights(self):
        q, k, v = self.attention_fixture()
        first, weights = probes.explicit_attention(q, k, v)
        second, changed_weights = probes.explicit_attention(q, k, v + tensor([2., -3.]))
        self.close(weights, changed_weights)
        self.close(second, first + tensor([[2., -3.]]))

    def test_stable_softmax_extreme_logits_and_translation(self):
        scores = tensor([[1000., 1000. + math.log(2), 1000. + math.log(3)]])
        self.close(probes.masked_softmax(scores), tensor([[1 / 6, 1 / 3, 1 / 2]]), atol=1e-13)
        self.close(probes.masked_softmax(scores), probes.masked_softmax(scores - 1000), atol=1e-13)
        self.assertTrue(torch.isfinite(probes.masked_softmax(tensor([[10000., -10000.]]))).all())

    def test_attention_manual_backward_independent_one_query_numbers(self):
        q, k, v = self.attention_fixture()
        gradients = probes.attention_backward(q, k, v, tensor([[1., 0.]]))
        # Upstream weight-gradient [0,6,4] has probability-weighted mean 4.
        self.close(gradients["scores"], tensor([[-2 / 3, 2 / 3, 0.]]))
        self.close(gradients["q"], tensor([[2 * math.log(2) / 3]]))
        self.close(gradients["k"], tensor([[-2 / 3], [2 / 3], [0.]]))
        self.close(gradients["v"], tensor([[1 / 6, 0.], [1 / 3, 0.], [1 / 2, 0.]]))

    def test_manual_attention_backward_matches_autograd_rectangular_batches(self):
        generator = torch.Generator().manual_seed(23)
        q, k, v = [random_tensor(shape, generator, grad=True) for shape in [(2, 3, 4), (2, 5, 4), (2, 5, 2)]]
        upstream = random_tensor((2, 3, 2), generator)
        allowed = torch.tensor([[[True, False, True, True, False], [False] * 5, [True] * 5]])
        output, _ = probes.explicit_attention(q, k, v, allowed)
        automatic = torch.autograd.grad((output * upstream).sum(), (q, k, v))
        manual = probes.attention_backward(q, k, v, upstream, allowed)
        for name, reference in zip(("q", "k", "v"), automatic):
            self.close(manual[name], reference)
        self.assertTrue(torch.equal(manual["scores"][:, 1], torch.zeros((2, 5), dtype=DTYPE)))

    def test_attention_gradcheck_includes_masked_and_empty_rows(self):
        generator = torch.Generator().manual_seed(11)
        args = [random_tensor(shape, generator, grad=True) for shape in [(2, 2), (3, 2), (3, 2)]]
        allowed = torch.tensor([[True, False, True], [False, False, False]])
        self.assertTrue(torch.autograd.gradcheck(lambda q, k, v: probes.explicit_attention(q, k, v, allowed)[0], tuple(args), eps=1e-6, atol=1e-5))

    def test_manual_attention_gradients_against_central_differences(self):
        q, k, v = self.attention_fixture()
        upstream = tensor([[1., -.5]])
        manual = probes.attention_backward(q, k, v, upstream)
        for name, arg_index, coordinate in [("q", 0, (0, 0)), ("k", 1, (1, 0)), ("v", 2, (2, 1))]:
            plus, minus = [x.clone() for x in (q, k, v)], [x.clone() for x in (q, k, v)]
            plus[arg_index][coordinate] += 1e-6
            minus[arg_index][coordinate] -= 1e-6
            loss = lambda xs: float((probes.explicit_attention(*xs)[0] * upstream).sum())
            difference = (loss(plus) - loss(minus)) / 2e-6
            self.assertAlmostEqual(float(manual[name][coordinate]), difference, places=8)

    def test_empty_attention_rows_have_finite_zero_output_and_gradients(self):
        q, k, v = self.attention_fixture(grad=True)
        output, weights = probes.explicit_attention(q, k, v, torch.zeros((1, 3), dtype=torch.bool))
        self.assertTrue(torch.equal(weights, torch.zeros_like(weights)))
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        gradients = torch.autograd.grad(output.sum(), (q, k, v))
        for grad in gradients:
            self.assertTrue(torch.isfinite(grad).all())
            self.assertTrue(torch.equal(grad, torch.zeros_like(grad)))

    def test_padding_key_value_invariance_and_zero_gradient(self):
        generator = torch.Generator().manual_seed(2)
        q = random_tensor((2, 3, 4), generator, grad=True)
        k = random_tensor((2, 5, 4), generator, grad=True)
        v = random_tensor((2, 5, 2), generator, grad=True)
        allowed = torch.tensor([True, True, True, False, False])
        reference, _ = probes.explicit_attention(q, k, v, allowed)
        changed_k, changed_v = k.detach().clone(), v.detach().clone()
        changed_k[:, 3:] += 10000
        changed_v[:, 3:] -= 10000
        self.close(reference, probes.explicit_attention(q, changed_k, changed_v, allowed)[0])
        dk, dv = torch.autograd.grad(reference.sum(), (k, v))
        self.assertTrue(torch.equal(dk[:, 3:], torch.zeros_like(dk[:, 3:])))
        self.assertTrue(torch.equal(dv[:, 3:], torch.zeros_like(dv[:, 3:])))

    def test_future_token_changes_cannot_change_prefix_outputs(self):
        x = tensor([[1., 0.], [0., 1.], [1., 2.], [-1., 3.]])
        positions = torch.arange(4)
        allowed = probes.causal_allowed(positions, positions)
        output, _ = probes.explicit_attention(x, x, x, allowed)
        changed = x.clone()
        changed[2:] += tensor([100., -30.])
        changed_output, _ = probes.explicit_attention(changed, changed, changed, allowed)
        self.close(output[:2], changed_output[:2])
        self.assertGreater(float((output[2:] - changed_output[2:]).abs().sum()), 1)

    def test_cached_rectangular_causality_uses_absolute_positions(self):
        allowed = probes.causal_allowed(torch.tensor([3, 4]), torch.arange(5))
        self.assertEqual(allowed.tolist(), [[True, True, True, True, False], [True] * 5])
        self.assertNotEqual(allowed.tolist(), torch.ones((2, 5), dtype=torch.bool).tril().tolist())

    def test_explicit_attention_sdpa_evaluation_parity(self):
        generator = torch.Generator().manual_seed(3)
        q, k, v = [random_tensor(shape, generator) for shape in [(2, 2, 3, 4), (2, 2, 5, 4), (2, 2, 5, 2)]]
        allowed = torch.tensor([[[[True, False, True, True, False], [False] * 5, [True] * 5]]])
        explicit, _ = probes.explicit_attention(q, k, v, allowed)
        reference = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
        self.close(explicit, reference)

    def test_content_attention_permutation_equivariance_and_fixed_position_distinction(self):
        x = tensor([[1., 0., 1., 0.], [0., 2., 0., 1.], [2., -1., 1., 3.]])
        permutation = torch.tensor([2, 0, 1])
        plain = probes.explicit_attention(x, x, x)[0]
        reordered = probes.explicit_attention(x[permutation], x[permutation], x[permutation])[0]
        self.close(reordered, plain[permutation])
        positions = probes.sinusoidal_positions(3, 4)
        positioned = probes.explicit_attention(x + positions, x + positions, x + positions)[0]
        reassigned_positions = x[permutation] + positions
        changed = probes.explicit_attention(reassigned_positions, reassigned_positions, reassigned_positions)[0]
        self.assertGreater(float((changed - positioned[permutation]).abs().max()), .1)

    def multihead_fixture(self, *, grad=False):
        generator = torch.Generator().manual_seed(27)
        streams = [random_tensor(shape, generator, grad=grad) for shape in [(2, 3, 8), (2, 5, 8), (2, 5, 8)]]
        weights = [random_tensor((8, 8), generator, grad=grad) / 4 for _ in range(4)]
        biases = [random_tensor((8,), generator, grad=grad) / 5 for _ in range(4)]
        allowed = torch.ones((2, 2, 3, 5), dtype=torch.bool)
        allowed[0, 0, 1] = False
        allowed[1, :, :, 4] = False
        return streams + weights, biases, allowed

    def test_projected_multihead_loop_vectorized_forward_and_gradient_parity(self):
        args, biases, allowed = self.multihead_fixture(grad=True)
        vectorized, weights = probes.multihead_attention(*args, heads=2, allowed=allowed, biases=biases)
        loop, loop_weights = probes.multihead_attention(*args, heads=2, allowed=allowed, biases=biases, implementation="loop")
        self.assertEqual(vectorized.shape, (2, 3, 8))
        self.assertEqual(weights.shape, (2, 2, 3, 5))
        self.close(vectorized, loop)
        self.close(weights, loop_weights)
        all_args = tuple(args + biases)
        vector_grad = torch.autograd.grad(vectorized.square().sum(), all_args, retain_graph=True)
        loop_grad = torch.autograd.grad(loop.square().sum(), all_args)
        for actual, expected in zip(vector_grad, loop_grad):
            self.close(actual, expected, atol=1e-11)

    def test_projected_multihead_matches_independent_sdpa_head_path(self):
        args, biases, allowed = self.multihead_fixture()
        xq, xk, xv, wq, wk, wv, wo = args
        projected = [x @ w + bias for x, w, bias in zip((xq, xk, xv), (wq, wk, wv), biases[:3])]
        heads = [x.reshape(2, x.shape[1], 2, 4).permute(0, 2, 1, 3) for x in projected]
        reference = F.scaled_dot_product_attention(*heads, attn_mask=allowed, dropout_p=0.0)
        reference = reference.permute(0, 2, 1, 3).reshape(2, 3, 8) @ wo + biases[3]
        actual, _ = probes.multihead_attention(*args, heads=2, allowed=allowed, biases=biases)
        self.close(actual, reference)

    def test_output_bias_can_change_wholly_masked_multihead_output(self):
        args, biases, _ = self.multihead_fixture()
        allowed = torch.zeros((2, 2, 3, 5), dtype=torch.bool)
        output, weights = probes.multihead_attention(*args, heads=2, allowed=allowed, biases=biases)
        self.close(output, biases[3].expand(2, 3, 8))
        self.assertTrue(torch.equal(weights, torch.zeros_like(weights)))

    def test_positionwise_ffn_hand_values_and_position_independence(self):
        x = tensor([[2., -1.], [-1., 2.]], grad=True)
        w1, w2 = tensor([[1., -1., 0.], [0., 1., 1.]]), tensor([[1., 0.], [0., 1.], [1., 1.]])
        args = (w1, tensor([0., 0., 0.]), w2, tensor([1., -1.]))
        output = probes.positionwise_ffn(x, *args)
        self.close(output, tensor([[3., -1.], [3., 4.]]))
        changed = x.detach().clone()
        changed[0] += 10
        self.close(probes.positionwise_ffn(changed, *args)[1], output[1])
        # A loss at position 1 has no dependency on position 0.
        grad, = torch.autograd.grad(output[1].sum(), x)
        self.close(grad[0], tensor([0., 0.]))

    def test_layernorm_feature_axis_hand_values_affine_and_constant(self):
        x = tensor([[[1., 3.], [2., 6.]], [[5., 5.], [-1., 1.]]])
        # eps=1 makes the denominators sqrt(2),sqrt(5),1,sqrt(2).
        expected = tensor([[[-1 / math.sqrt(2), 1 / math.sqrt(2)], [-2 / math.sqrt(5), 2 / math.sqrt(5)]], [[0., 0.], [-1 / math.sqrt(2), 1 / math.sqrt(2)]]])
        normalized = probes.layer_norm(x, 2, eps=1)
        self.close(normalized, expected)
        scale, bias = tensor([2., 3.]), tensor([1., -1.])
        self.close(probes.layer_norm(x, 2, scale=scale, bias=bias, eps=1), expected * scale + bias)
        self.close(probes.layer_norm(x, 2, scale=scale, bias=bias, eps=1)[1, 0], bias)

    def test_layernorm_trailing_multiple_axes_and_pytorch_parity(self):
        x = tensor([[[1., 2.], [3., 4.]], [[8., 8.], [8., 8.]]])
        # First sample mean=2.5, population variance=1.25.
        expected = tensor([[[-1.5 / math.sqrt(2.25), -.5 / math.sqrt(2.25)], [.5 / math.sqrt(2.25), 1.5 / math.sqrt(2.25)]], [[0., 0.], [0., 0.]]])
        actual = probes.layer_norm(x, (2, 2), eps=1)
        self.close(actual, expected)
        self.close(actual, F.layer_norm(x, (2, 2), eps=1))
        # Normalize each final-axis pair versus all positions/features: distinct contracts.
        self.assertGreater(float((actual - probes.layer_norm(x, 2, eps=1)).abs().max()), .3)

    def test_layernorm_gradient_check_and_train_eval_invariance(self):
        x = tensor([[1., 2., 4.], [3., -2., .5]], grad=True)
        gamma, beta = tensor([2., 1., 3.], grad=True), tensor([.1, .2, .3], grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda a, g, b: probes.layer_norm(a, 3, scale=g, bias=b), (x, gamma, beta)))
        with torch.random.fork_rng():
            module = torch.nn.LayerNorm(3, dtype=DTYPE)
        module.train()
        training = module(x)
        module.eval()
        self.close(training, module(x))

    def test_residual_norm_zero_branch_and_matching_dimensions(self):
        x = tensor([[1., 3.]], grad=True)
        zero = lambda values: values * 0
        pre = probes.residual_norm(x, zero, placement="pre")
        post = probes.residual_norm(x, zero, placement="post")
        self.close(pre, x)
        self.close(post, probes.layer_norm(x, 2))
        self.assertGreater(float((pre - post).detach().abs().max()), 1)
        pre_grad, = torch.autograd.grad(pre.sum(), x)
        self.close(pre_grad, tensor([[1., 1.]]))
        with self.assertRaises(ValueError):
            probes.residual_norm(x, lambda values: values[..., :1])

    def test_residual_jacobian_is_identity_plus_branch_jacobian(self):
        x = tensor([1., 2.], grad=True)
        weight = tensor([[2., 1.], [-1., 3.]])
        jacobian = torch.autograd.functional.jacobian(lambda values: values + values @ weight, x)
        self.close(jacobian, torch.eye(2, dtype=DTYPE) + weight.T)

    def test_inverted_dropout_exact_expectation_variance_and_eval(self):
        x = tensor([2., -3.])
        outputs = [probes.inverted_dropout(x, torch.tensor(mask), .5) for mask in itertools.product((False, True), repeat=2)]
        stacked = torch.stack(outputs)
        self.close(stacked.mean(0), x)
        # Var = p/(1-p) * x^2 at p=.5.
        self.close(stacked.var(0, correction=0), x.square())
        self.close(probes.inverted_dropout(x, torch.tensor([False, False]), .5, training=False), x)
        self.close(probes.inverted_dropout(x, torch.tensor([False, False]), 0), x)

    def test_label_smoothing_hand_distribution_loss_and_gradient(self):
        targets = torch.tensor([1, -100])
        distribution = probes.smoothed_targets(targets, 4, .1)
        self.close(distribution, tensor([[.025, .925, .025, .025], [0., 0., 0., 0.]]))
        logits = tensor([[0., math.log(2), math.log(3), math.log(4)]], grad=True)
        manual_loss = -(distribution[:1] * logits.log_softmax(-1)).sum()
        reference = F.cross_entropy(logits, targets[:1], label_smoothing=.1)
        self.close(manual_loss, reference)
        actual_grad, = torch.autograd.grad(manual_loss, logits)
        self.close(actual_grad, tensor([[.1, .2, .3, .4]]) - distribution[:1])

    def test_schedule_linear_warmup_peak_and_inverse_sqrt(self):
        expected = {1: .0625, 2: .125, 4: .25, 16: .125}
        for step, value in expected.items():
            self.assertAlmostEqual(probes.transformer_learning_rate(step, width=4, warmup=4), value)
        self.assertGreater(probes.transformer_learning_rate(4, width=4, warmup=4), probes.transformer_learning_rate(5, width=4, warmup=4))
        with self.assertRaises(ValueError):
            probes.transformer_learning_rate(0)

    def test_parameter_counts_match_small_and_paper_dimensions(self):
        small = probes.parameter_counts(vocabulary=4, width=2, hidden=3, heads=1)
        self.assertEqual(small, {"three_untied_vocabulary_matrices": 24, "one_tied_vocabulary_matrix": 8, "multihead": 24, "ffn": 17})
        paper = probes.parameter_counts(vocabulary=37000, width=512, hidden=2048, heads=8)
        self.assertEqual(paper["ffn"], 2099712)
        self.assertEqual(paper["multihead"], 1050624)
        self.assertEqual(probes.parameter_counts(vocabulary=4, width=8, hidden=32, heads=2)["multihead"], probes.parameter_counts(vocabulary=4, width=8, hidden=32, heads=4)["multihead"])

    def test_shape_dtype_and_mask_contract_rejections(self):
        q, k, v = self.attention_fixture()
        bad_calls = [
            lambda: probes.explicit_attention(q, k, v, tensor([[1., 0., 1.]])),
            lambda: probes.explicit_attention(q, k, v, torch.ones((2, 2), dtype=torch.bool)),
            lambda: probes.explicit_attention(q.long(), k, v),
            lambda: probes.explicit_attention(q, k.float(), v),
            lambda: probes.sinusoidal_positions(2, 0),
            lambda: probes.layer_norm(tensor([[1., 2.]]), 3),
            lambda: probes.parameter_counts(vocabulary=4, width=7, hidden=28, heads=2),
            lambda: probes.inverted_dropout(tensor([1.]), torch.tensor([True]), 1),
            lambda: probes.smoothed_targets(torch.tensor([4]), 4),
        ]
        for call in bad_calls:
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()

    def test_report_is_deterministic_without_global_rng_changes(self):
        before = torch.get_rng_state().clone()
        first = probes.report()
        second = probes.report()
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(first["attention"]["empty_output"], [[0., 0.]])


if __name__ == "__main__":
    unittest.main()
