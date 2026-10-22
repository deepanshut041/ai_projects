"""Hand fixtures, actual-module controls and gradient checks for applied probes."""

import copy
import math
from dataclasses import replace
import unittest

import torch
from torch.nn import functional as F

import model_learning_probes as probes
from translation_attention import Attention
from translation_model import Transformer, TransformerConfig, FeedForward, TransformerEncoderLayer
from translation_train import token_statistics, train_update
from translation_metrics import corpus_bleu_tokens
from roberta_model import (RobertaConfig, RobertaForMaskedLM, RobertaForQuestionAnswering,
                           RobertaEncoderLayer, position_ids, masked_token_loss, optimizer_groups)
from roberta_data import (ByteBPETokenizer, corrupt_mlm, character_span_to_tokens, qa_features,
                          best_span, answer_metrics, normalize_answer)
from roberta_train import build_training_bundle, training_step


def tensor(values, *, grad=False):
    return torch.tensor(values, dtype=torch.float64, requires_grad=grad)


class ModelLearningChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.rng = torch.random.fork_rng()
        self.rng.__enter__()
        torch.manual_seed(37)

    def tearDown(self):
        self.rng.__exit__(None, None, None)

    def close(self, actual, expected, *, atol=1e-12):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=atol)

    def test_valid_token_nll_and_gradient_hand_fixture(self):
        logits = tensor([[[0., math.log(3)], [0., math.log(3)], [900., -900.]]], grad=True)
        labels = torch.tensor([[1, 0, -100]])
        result = probes.token_loss_and_gradient(logits, labels)
        self.assertEqual(result["count"], 2)
        self.assertAlmostEqual(float(result["loss"].detach()), math.log(16 / 3) / 2)
        expected = tensor([[[.125, -.125], [-.375, .375], [0., 0.]]])
        self.close(result["gradient"], expected)
        self.close(torch.autograd.grad(result["loss"], logits)[0], expected)
        source_sum, source_count, _ = token_statistics(logits, labels)
        self.close(source_sum, result["loss_sum"])
        self.assertEqual(int(source_count), 2)

    def test_unequal_selected_counts_share_one_denominator(self):
        logits = tensor([[[0., math.log(3)], [0., 0.]], [[math.log(3), 0.], [0., math.log(3)]]], grad=True)
        labels = torch.tensor([[1, -100], [0, 1]])
        gradients, total = probes.accumulated_logit_gradients([logits[:1], logits[1:]], [labels[:1], labels[1:]])
        self.assertEqual(total, 3)
        self.close(torch.cat(gradients), probes.token_loss_and_gradient(logits, labels)["gradient"])
        # Averaging the two microbatch means would give unequal per-token weight.
        wrong = torch.cat([probes.token_loss_and_gradient(logits[:1], labels[:1])["gradient"] / 2,
                           probes.token_loss_and_gradient(logits[1:], labels[1:])["gradient"] / 2])
        self.assertFalse(torch.allclose(torch.cat(gradients), wrong))

    def test_nll_finite_differences_and_all_ignored_gradients(self):
        logits = tensor([[1000., 999., -1000.], [.2, -.1, .6]], grad=True)
        labels = torch.tensor([2, 1])
        result = probes.token_loss_and_gradient(logits, labels)
        self.assertTrue(torch.isfinite(result["loss"]))
        for coordinate in [(0, 2), (1, 0)]:
            plus, minus = logits.detach().clone(), logits.detach().clone()
            plus[coordinate] += 1e-5
            minus[coordinate] -= 1e-5
            central = (float(probes.token_loss_and_gradient(plus, labels)["loss"]) - float(probes.token_loss_and_gradient(minus, labels)["loss"])) / 2e-5
            self.assertAlmostEqual(float(result["gradient"][coordinate].detach()), central, places=7)
        ignored = probes.token_loss_and_gradient(logits, torch.tensor([-100, -100]))
        self.assertEqual(ignored["count"], 0)
        self.close(ignored["gradient"], torch.zeros_like(logits))
        self.close(torch.autograd.grad(ignored["loss"], logits)[0], torch.zeros_like(logits))

    def test_learned_rectangular_attention_two_query_hand_fixture(self):
        result = probes.projected_attention(*probes.learned_attention_fixture())
        self.close(result["weights"], tensor([[[[1 / 7, 3 / 7, 3 / 7], [2 / 5, 1 / 5, 2 / 5]]]]))
        self.close(result["merged"], tensor([[[8 / 7, 18 / 7], [8 / 5, 9 / 5]]]))
        self.close(result["output"], tensor([[[8 / 7, 10 / 7], [8 / 5, 1 / 5]]]))
        masked = probes.projected_attention(*probes.learned_attention_fixture(), allowed=torch.tensor([True, True, False]))
        self.close(masked["output"], tensor([[[.5, 1.75], [4 / 3, -1 / 3]]]))

    def test_learned_attention_manual_all_projection_gradients(self):
        args = [x.detach().clone().requires_grad_() for x in probes.learned_attention_fixture()]
        upstream = tensor([[[1., -.5], [.25, 2.]]])
        result = probes.projected_attention(*args)
        automatic = torch.autograd.grad((result["output"] * upstream).sum(), tuple(args))
        manual = probes.projected_attention_backward(*args, upstream)
        for name, expected in zip(("query", "source", "wq", "wk", "wv", "wo"), automatic):
            self.close(manual[name], expected)
        # Source receives both the matching-key path and transported-value path.
        self.assertGreater(float(manual["source"].abs().sum()), 1)

    def test_learned_attention_matches_actual_layer_forward_and_backward(self):
        query, source, *weights = [x.detach().clone().requires_grad_() for x in probes.learned_attention_fixture()]
        layer = Attention(2, 1, fused_attn=False).double().eval()
        with torch.no_grad():
            for projection, weight in zip((layer.q_proj, layer.k_proj, layer.v_proj, layer.out_proj), weights):
                projection.weight.copy_(weight.T)
                projection.bias.zero_()
        upstream = tensor([[[1., -.5], [.25, 2.]]])
        actual = layer(source, query)
        own = probes.projected_attention(query, source, *weights)["output"]
        self.close(actual, own)
        actual_parameters = [layer.q_proj.weight, layer.k_proj.weight, layer.v_proj.weight, layer.out_proj.weight]
        actual_gradients = torch.autograd.grad((actual * upstream).sum(), [query, source, *actual_parameters])
        manual = probes.projected_attention_backward(query, source, *weights, upstream)
        for index, name in enumerate(("query", "source", "wq", "wk", "wv", "wo")):
            expected = actual_gradients[index] if index < 2 else actual_gradients[index].T
            self.close(manual[name], expected)

    def test_projected_multihead_gradcheck_and_sdpa_equivalence(self):
        generator = torch.Generator().manual_seed(8)
        shapes = [(1, 2, 4), (1, 3, 4)] + [(4, 4)] * 4
        args = [(torch.randn(shape, dtype=torch.float64, generator=generator) / 3).requires_grad_() for shape in shapes]
        allowed = torch.tensor([[[[True, False, True], [False, False, False]]]])
        self.assertTrue(torch.autograd.gradcheck(lambda *xs: probes.projected_attention(*xs, heads=2, allowed=allowed)["output"], tuple(args), fast_mode=True))
        result = probes.projected_attention(*args, heads=2, allowed=allowed)
        sdpa = F.scaled_dot_product_attention(result["q"], result["k"], result["v"], attn_mask=allowed, dropout_p=0.)
        output = sdpa.transpose(1, 2).reshape(1, 2, 4) @ args[-1]
        self.close(result["output"], output)
        upstream = tensor([[[1., 2., 3., 4.], [4., 3., 2., 1.]]])
        automatic = torch.autograd.grad((result["output"] * upstream).sum(), tuple(args))
        manual = probes.projected_attention_backward(*args, upstream, heads=2, allowed=allowed)
        for name, expected in zip(("query", "source", "wq", "wk", "wv", "wo"), automatic):
            self.close(manual[name], expected)

    def test_separate_vocabularies_and_exact_full_model_budget(self):
        config = TransformerConfig(src_vocab_size=7, tgt_vocab_size=9, embedding_dimension=8,
                                   num_attention_heads=2, mlp_ratio=2)
        model = Transformer(config).double()
        self.assertEqual(model.encodings.src_embeddings.weight.shape, (7, 8))
        self.assertEqual(model.encodings.tgt_embeddings.weight.shape, (9, 8))
        self.assertEqual(model.head.weight.shape, (9, 8))
        self.assertIsNot(model.head.weight, model.encodings.tgt_embeddings.weight)
        budget = probes.translation_parameter_budget(source_vocabulary=7, target_vocabulary=9, width=8, heads=2, ratio=2)
        self.assertEqual(budget, {"embeddings": 128, "head": 81, "encoder": 600, "decoder": 904, "total": 1713})
        self.assertEqual(sum(p.numel() for p in model.parameters()), budget["total"])
        self.assertEqual(probes.attention_score_elements(batch=2, heads=2, source_length=5, target_length=3), 196)

    def test_source_positions_and_registered_buffer_are_different_interfaces(self):
        config = TransformerConfig(src_vocab_size=7, tgt_vocab_size=9, embedding_dimension=4, num_attention_heads=2)
        model = Transformer(config).double()
        encoding = model.encodings.src_positional_encodings
        self.assertIn("encodings", dict(encoding.named_buffers()))
        self.assertEqual(list(encoding.parameters()), [])
        self.close(encoding(torch.zeros((1, 1, 4), dtype=torch.float64)), tensor([[[0., 1., 0., 1.]]]))
        self.assertEqual(position_ids(torch.tensor([[0, 6, 1, 7, 2, 1]])).tolist(), [[2, 3, 1, 4, 5, 1]])
        with self.assertRaises(ValueError):
            encoding(torch.zeros((1, config.max_src_len + 1, 4), dtype=torch.float64))

    def test_actual_encoder_postnorm_order_with_zero_branches(self):
        config = TransformerConfig(embedding_dimension=4, num_attention_heads=2, mlp_ratio=2)
        encoder = TransformerEncoderLayer(config).double().eval()
        with torch.no_grad():
            for module in (encoder.enc_attention, encoder.feed_forward):
                for parameter in module.parameters():
                    parameter.zero_()
        x = tensor([[[1., 2., 4., 8.]]])
        expected = F.layer_norm(F.layer_norm(x, (4,)), (4,))
        self.close(encoder(x), expected)
        self.assertFalse(torch.allclose(encoder(x), x))
        roberta_config = RobertaConfig(vocab_size=20, embedding_dimension=4, num_attention_heads=2, num_transformer_blocks=1)
        roberta_encoder = RobertaEncoderLayer(roberta_config).double().eval()
        with torch.no_grad():
            for module in (roberta_encoder.attention, roberta_encoder.feed_forward):
                for parameter in module.parameters():
                    parameter.zero_()
        self.close(roberta_encoder(x, torch.tensor([[True]])), expected)

    def test_inspected_gelu_ffn_known_values_gradient_and_position_sharing(self):
        ffn = FeedForward(TransformerConfig(embedding_dimension=2, num_attention_heads=1, mlp_ratio=1)).double().eval()
        with torch.no_grad():
            for linear in (ffn.intermediate_dense, ffn.output_dense):
                linear.weight.copy_(torch.eye(2, dtype=torch.float64))
                linear.bias.zero_()
        x = tensor([[[0., 1.], [-1., 2.]]], grad=True)
        out = ffn(x)
        self.close(out[0, 0], tensor([0., .5 * (1 + math.erf(1 / math.sqrt(2)))]))
        grad = torch.autograd.grad(out[0, 0].sum(), x)[0]
        expected_derivative = .5 * (1 + math.erf(1 / math.sqrt(2))) + math.exp(-.5) / math.sqrt(2 * math.pi)
        self.close(grad[0, 0], tensor([.5, expected_derivative]))
        self.close(grad[0, 1], tensor([0., 0.]))

    def test_attention_dropout_train_reproducibility_and_eval_determinism(self):
        layer = Attention(4, 2, attn_dropout=.5, fused_attn=False).double()
        x = tensor([[[1., 2., 3., 4.], [3., 1., -2., 4.], [4., 3., 2., 1.]]])
        layer.train()
        state = torch.get_rng_state().clone()
        first = layer(x)
        torch.set_rng_state(state)
        self.close(layer(x), first)
        layer.eval()
        state = torch.get_rng_state().clone()
        self.close(layer(x), layer(x))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        # Independent inverted-dropout expectation at p=.5: values 0 or 2x.
        value = tensor([2., -3.])
        alternatives = torch.stack([torch.zeros_like(value), 2 * value])
        self.close(alternatives.mean(0), value)
        self.close(alternatives.var(0, correction=0), value.square())

    def test_mlm_conditional_branch_counts_targets_and_expectations(self):
        result = probes.controlled_mlm_probe()
        self.assertEqual(result["labels"], [[-100, 6, 7, 8, -100, -100, -100]])
        self.assertEqual([result[key] for key in ["selected", "mask_branch", "random_branch", "unchanged_branch"]], [3, 1, 1, 1])
        self.assertTrue(result["original_ids_unchanged"] and result["random_token_ordinary"] and result["unchanged_selected_label_preserved"])
        expected = probes.corruption_expectations(100, .15)
        self.assertEqual([expected[key] for key in ["selected", "mask", "random", "unchanged"]], [15., 12., 1.5, 1.5])
        self.assertAlmostEqual(probes.corruption_expectations(2, .35)["zero_selection_probability"], .65 ** 2)

    def test_zero_selected_targets_are_permitted_without_forcing_one(self):
        tokenizer = ByteBPETokenizer()
        ids = torch.tensor([[0, 6, 7, 2, 1]])
        output, labels = corrupt_mlm(ids, tokenizer, torch.Generator().manual_seed(3), probability=0)
        self.assertTrue(torch.equal(output, ids))
        self.assertTrue(torch.equal(labels, torch.full_like(ids, -100)))
        bundle = build_training_bundle("mlm", total_steps=4)
        bundle.protocol["masking_probability"] = 0
        parameters = {name: value.clone() for name, value in bundle.model.state_dict().items()}
        optimizer, scheduler = copy.deepcopy(bundle.optimizer.state_dict()), copy.deepcopy(bundle.scheduler.state_dict())
        batch_rng = bundle.batch_generator.get_state().clone()
        result = training_step(bundle)
        self.assertFalse(result["updated"])
        self.assertEqual((result["target_count"], bundle.step), (0, 0))
        self.assertEqual(bundle.optimizer.state_dict(), optimizer)
        self.assertEqual(bundle.scheduler.state_dict(), scheduler)
        for name, value in bundle.model.state_dict().items():
            self.close(value, parameters[name])
        # Sampling/corruption may still consume RNG even when no update occurs.
        self.assertFalse(torch.equal(batch_rng, bundle.batch_generator.get_state()))

    def test_mlm_selected_only_logit_gradients_and_actual_head_loss(self):
        logits = tensor([[[0., math.log(3)], [0., math.log(3)], [0., 0.]]], grad=True)
        labels = torch.tensor([[1, -100, 0]])
        own = probes.token_loss_and_gradient(logits, labels)
        loss_sum, count = masked_token_loss(logits, labels)
        self.assertEqual(count, 2)
        self.close(loss_sum, own["loss_sum"])
        self.close(torch.autograd.grad(loss_sum / count, logits)[0], own["gradient"])
        self.close(own["gradient"][0, 1], tensor([0., 0.]))

    def test_tied_mlm_table_has_joint_lookup_and_output_gradient_paths(self):
        model = RobertaForMaskedLM(RobertaConfig(vocab_size=7, embedding_dimension=2, num_attention_heads=1, num_transformer_blocks=1, context_length=8)).double()
        table = model.roberta.embeddings.word_embeddings
        self.assertIs(model.mlm_head.decoder.weight, table.weight)
        ids = torch.tensor([1, 5, 5])
        lookup_loss = table(ids).sum()
        hidden = tensor([[2., 0.]])
        output_loss = model.mlm_head.decoder(hidden).sum()
        (lookup_loss + output_loss).backward()
        expected = torch.tensor([[2., 0.]] * 7, dtype=torch.float64)
        expected[5] += tensor([2., 2.])
        self.close(table.weight.grad, expected)
        self.close(table.weight.grad[1], tensor([2., 0.]))
        parameters = list(model.parameters())
        self.assertEqual(sum(p is table.weight for p in parameters), 1)

    def test_unique_mlm_and_qa_parameter_budgets(self):
        config = RobertaConfig(vocab_size=20, embedding_dimension=8, num_transformer_blocks=1, num_attention_heads=2, context_length=16, mlp_ratio=2)
        for task, constructor in [("mlm", RobertaForMaskedLM), ("qa", RobertaForQuestionAnswering)]:
            model = constructor(config)
            budget = probes.roberta_parameter_budget(vocabulary=20, width=8, layers=1, heads=2, context_length=16, ratio=2, task=task)
            self.assertEqual(budget["backbone"], 920)
            self.assertEqual(budget["head"], 108 if task == "mlm" else 18)
            self.assertEqual(sum(p.numel() for p in model.parameters()), budget["total"])

    def test_backbone_transfer_copies_weights_and_starts_clean_task_optimizer(self):
        config = RobertaConfig(vocab_size=20, embedding_dimension=8, num_transformer_blocks=1, num_attention_heads=2, context_length=16, mlp_ratio=2)
        mlm = RobertaForMaskedLM(config).double()
        optimizer = torch.optim.AdamW(optimizer_groups(mlm), lr=.001)
        result = mlm(torch.tensor([[0, 6, 7, 2]]), labels=torch.tensor([[-100, 6, -100, -100]]))
        result["loss"].backward()
        optimizer.step()
        self.assertGreater(len(optimizer.state), 0)
        qa = RobertaForQuestionAnswering(config).double()
        qa_head = {name: value.clone() for name, value in qa.qa_head.state_dict().items()}
        qa.load_mlm_backbone(mlm)
        for name, value in qa.roberta.state_dict().items():
            self.close(value, mlm.roberta.state_dict()[name])
        for name, value in qa.qa_head.state_dict().items():
            self.close(value, qa_head[name])
        self.assertIsNot(qa.roberta.embeddings.word_embeddings.weight, mlm.roberta.embeddings.word_embeddings.weight)
        task_optimizer = torch.optim.AdamW(optimizer_groups(qa), lr=.001)
        self.assertEqual(len(task_optimizer.state), 0)
        with self.assertRaises(ValueError):
            RobertaForQuestionAnswering(replace(config, vocab_size=21)).load_mlm_backbone(mlm)

    def test_original_utf8_half_open_coordinates_and_duplicate_byte_offsets(self):
        self.assertEqual(probes.utf8_byte_character_map("café"), [(0, 1), (1, 2), (2, 3), (3, 4), (3, 4)])
        tokenizer = ByteBPETokenizer()
        ids, offsets = tokenizer.encode_with_offsets("café!")
        self.assertEqual(offsets, [(0, 1), (1, 2), (2, 3), (3, 4), (3, 4), (4, 5)])
        self.assertEqual(tokenizer.decode(ids), "café!")
        self.assertEqual(character_span_to_tokens(offsets, 0, 4), (0, 4))
        self.assertEqual(character_span_to_tokens(offsets, 4, 5), (5, 5))
        self.assertEqual(character_span_to_tokens(offsets, 3, 4), (3, 4))
        self.assertIsNone(character_span_to_tokens(offsets, 4, 4))
        restored = ByteBPETokenizer.from_dict(tokenizer.to_dict())
        self.assertEqual(restored.encode_with_offsets("café!"), (ids, offsets))

    def test_overlapping_windows_include_only_available_answer_labels(self):
        tokenizer = ByteBPETokenizer()
        context = "alpha beta café omega"
        answer = "café"
        features = qa_features("q", context, tokenizer, context_length=14, overlap=4,
                               answer_start=context.index(answer), answer_text=answer)
        self.assertGreater(len(features), 1)
        available = [feature for feature in features if feature.start_position != -100]
        self.assertTrue(available)
        self.assertTrue(any(feature.start_position == -100 for feature in features))
        for feature in available:
            start, end = feature.start_position, feature.end_position
            left, right = feature.offsets[start][0], feature.offsets[end][1]
            self.assertEqual(context[left:right], answer)
            self.assertTrue(all(feature.offsets[index] is not None for index in range(start, end + 1)))
        for feature in features:
            self.assertEqual(len(feature.input_ids), 14)
            self.assertEqual(feature.start_position == -100, feature.end_position == -100)

    def test_qa_endpoint_loss_gradients_independent_hand_fixture(self):
        scores = tensor([[math.log(n) for n in [1, 2, 3, 4]]], grad=True)
        ends = scores.detach().clone().requires_grad_()
        result = probes.qa_endpoint_loss_and_gradient(scores, ends, torch.tensor([2]), torch.tensor([3]))
        self.assertAlmostEqual(float(result["loss"].detach()), math.log(25 / 3) / 2)
        self.close(result["start_gradient"], tensor([[.05, .1, -.35, .2]]))
        self.close(result["end_gradient"], tensor([[.05, .1, .15, -.3]]))
        left, right = torch.autograd.grad(result["loss"], (scores, ends))
        self.close(left, result["start_gradient"])
        self.close(right, result["end_gradient"])

    def test_actual_qa_ce_domain_includes_question_special_and_padding_positions(self):
        config = RobertaConfig(vocab_size=20, embedding_dimension=8, num_transformer_blocks=1, num_attention_heads=2, context_length=16, mlp_ratio=2)
        model = RobertaForQuestionAnswering(config).double()
        ids = torch.tensor([[0, 6, 2, 2, 7, 2, 1], [0, 6, 2, 2, 8, 2, 1]])
        starts, ends = torch.tensor([4, -100]), torch.tensor([4, -100])
        result = model(ids, start_positions=starts, end_positions=ends)
        own = probes.qa_endpoint_loss_and_gradient(result["start_logits"], result["end_logits"], starts, ends)
        self.close(result["loss"], own["loss"])
        gradients = torch.autograd.grad(result["loss"], (result["start_logits"], result["end_logits"]), retain_graph=True)
        self.close(gradients[0], own["start_gradient"])
        self.close(gradients[1], own["end_gradient"])
        self.assertGreater(float(gradients[0][0, 1]), 0)  # question competes
        self.assertGreater(float(gradients[0][0, 6]), 0)  # padding competes
        self.close(gradients[0][1], torch.zeros(7, dtype=torch.float64))
        result["loss"].backward()
        self.assertGreater(float(model.roberta.embeddings.word_embeddings.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.qa_head.weight.grad.abs().sum()), 0)

    def test_joint_span_decoding_rejects_independent_argmax_and_question_peaks(self):
        starts, ends = [100., 1., 8., 0.], [100., 9., 2., 7.]
        offsets = [None, (0, 1), (2, 3), (4, 5)]
        candidates = probes.valid_span_candidates(starts, ends, offsets, max_tokens=2)
        best = max(candidates, key=lambda item: item["score"])
        self.assertEqual((best["start"], best["end"], best["score"]), (2, 3, 15.))
        actual = best_span(starts, ends, offsets, "a b c", max_answer_tokens=2)
        self.assertEqual((actual["start_token"], actual["end_token"], actual["text"]), (2, 3, "b c"))
        # Context-only independent argmax would choose reversed endpoints 2,1.
        self.assertEqual((max(range(1, 4), key=starts.__getitem__), max(range(1, 4), key=ends.__getitem__)), (2, 1))

    def test_cross_window_raw_score_is_not_shift_invariant(self):
        offsets = [(0, 1), (2, 3)]
        first = best_span([2., 0.], [2., 0.], offsets, "a b")
        second = best_span([0., 1.], [0., 1.], offsets, "a b")
        shifted = best_span([5., 6.], [5., 6.], offsets, "a b")
        self.assertGreater(first["score"], second["score"])
        self.assertGreater(shifted["score"], first["score"])
        self.close(tensor([0., 1.]).softmax(0), tensor([5., 6.]).softmax(0))

    def test_normalized_em_and_multiset_f1_worked_example(self):
        prediction, gold = "The red apple apple!", "a red red apple"
        predicted, expected = normalize_answer(prediction).split(), normalize_answer(gold).split()
        self.assertEqual(predicted, ["red", "apple", "apple"])
        self.assertAlmostEqual(probes.multiset_f1(predicted, expected), 2 / 3)
        self.assertEqual(answer_metrics(prediction, [gold]), {"exact_match": 0., "f1": 2 / 3})
        self.assertEqual(answer_metrics("The Café!", ["café"])["exact_match"], 1)

    def test_bleu_clipping_brevity_geometric_mean_and_short_zero_policy(self):
        actual = corpus_bleu_tokens([["a", "b", "a", "b"]], [["a", "b", "c", "d", "e", "f"]], max_order=2)
        self.assertEqual(actual["matches"], [2, 1])
        self.assertEqual(actual["totals"], [4, 3])
        own = probes.bleu_from_counts([2, 1], [4, 3], 4, 6)
        self.assertAlmostEqual(own["brevity_penalty"], math.exp(-.5))
        self.assertAlmostEqual(own["score"], 100 * math.exp(-.5) / math.sqrt(6))
        self.assertAlmostEqual(actual["score"], own["score"])
        self.assertEqual(corpus_bleu_tokens([["a", "b"]], [["a", "b"]])["score"], 0.)

    def test_actual_complete_models_visibility_and_source_dependence_controls(self):
        controls = probes.model_control_probe()
        for name, value in controls.items():
            if name != "scope":
                self.assertTrue(value, name)
        # These are untrained architectural checks, not generated quality checks.
        self.assertIn("untrained", controls["scope"])

    def test_teacher_forcing_alignment_gradients_and_unequal_updates_actual_model(self):
        from translation_data import collate_translation
        records = [{"src_ids": [2, 4, 3], "tgt_ids": [2, 5, 6, 3]}, {"src_ids": [2, 5, 6, 3], "tgt_ids": [2, 7, 3]}]
        batch = collate_translation(records, 0, 1)
        self.assertEqual(batch["tgt_input_ids"].tolist(), [[2, 5, 6], [2, 7, 3]])
        self.assertEqual(batch["tgt_outputs"].tolist(), [[5, 6, 3], [7, 3, -100]])
        model = Transformer(TransformerConfig(src_vocab_size=7, tgt_vocab_size=9, embedding_dimension=8, num_attention_heads=2, mlp_ratio=2)).double()
        split_model = copy.deepcopy(model)
        first = torch.optim.SGD(model.parameters(), lr=.01)
        second = torch.optim.SGD(split_model.parameters(), lr=.01)
        whole = train_update(model, first, [batch], max_grad_norm=1e9)
        micro = [{key: value[index:index + 1] for key, value in batch.items()} for index in range(2)]
        divided = train_update(split_model, second, micro, max_grad_norm=1e9)
        self.assertEqual((whole["tokens"], divided["tokens"]), (5, 5))
        for actual, expected in zip(model.parameters(), split_model.parameters()):
            self.close(actual, expected)
        self.assertGreater(float(model.encodings.src_embeddings.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.encodings.tgt_embeddings.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.head.weight.grad.abs().sum()), 0)

    def test_report_repeatability_and_global_rng_restoration(self):
        before = torch.get_rng_state().clone()
        self.assertEqual(probes.report(), probes.report())
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
