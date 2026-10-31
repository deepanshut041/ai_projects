"""Independent arithmetic, corruption, encoder, span and bounded-training checks."""

import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from roberta_data import ByteBPETokenizer, answer_metrics, best_span, character_span_to_tokens, corrupt_mlm, pack_documents, pad_sequences, qa_batch, qa_features
from roberta_model import RobertaAttention, RobertaConfig, RobertaForMaskedLM, RobertaForQuestionAnswering, masked_token_loss, position_ids
from roberta_train import build_training_bundle, export_backbone, fill_mask, fixed_mlm_evaluation, load_backbone_for_qa, next_batch, run_demo, to_model_device, training_step


class DataChecks(unittest.TestCase):
    def test_packing_retains_the_tail_and_boundaries(self):
        tokenizer = ByteBPETokenizer()
        expected = tokenizer.encode("ab") + tokenizer.encode("c")
        chunks = pack_documents(["ab", "c"], tokenizer, context_length=4)
        self.assertEqual([token for chunk in chunks for token in chunk], expected)
        self.assertEqual(list(map(len, chunks)), [4, 3])

    def test_byte_merges_round_trip_unicode_and_whitespace(self):
        tokenizer = ByteBPETokenizer.train(["café café", "the red apple", "你好世界"], merge_count=20)
        for text in ["café", " café", "two  spaces", "你好!", "🙂 unseen text"]:
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(tokenizer.fingerprint(), ByteBPETokenizer.from_dict(tokenizer.to_dict()).fingerprint())

    def test_padding_keys_are_not_created_after_padding(self):
        batch = pad_sequences([[0, 10, 2], [0, 2]])
        self.assertEqual(batch["input_ids"].tolist(), [[0, 10, 2], [0, 2, 1]])
        self.assertEqual(batch["attention_mask"].tolist(), [[True, True, True], [True, True, False]])

    def test_all_three_corruption_branches_keep_original_targets(self):
        tokenizer = ByteBPETokenizer()
        original = torch.tensor([[0, 5, 6, 7, 8, 1, 2, 4]])
        untouched = original.clone()
        draws = torch.tensor([[0.0, 0.1, 0.82, 0.94, 0.3, 0.0, 0.0, 0.0]])
        corrupted, labels = corrupt_mlm(original, tokenizer, torch.Generator().manual_seed(4), probability=1.0, selection_draws=torch.zeros_like(original, dtype=torch.float), replacement_draws=draws)
        self.assertTrue(torch.equal(original, untouched))
        self.assertEqual(labels.tolist(), [[-100, 5, 6, 7, 8, -100, -100, -100]])
        self.assertEqual(corrupted[0, 1].item(), 4)
        self.assertEqual(corrupted[0, 3].item(), 7)
        self.assertEqual(corrupted[0, 4].item(), 4)
        self.assertIn(corrupted[0, 2].item(), range(5, tokenizer.vocab_size))
        self.assertTrue(torch.equal(corrupted[0, [0, 5, 6, 7]], original[0, [0, 5, 6, 7]]))

    def test_random_replacements_can_exceed_the_vocabulary_count(self):
        original = torch.full((8, 128), 6)
        corrupted, labels = corrupt_mlm(original, ByteBPETokenizer(), torch.Generator().manual_seed(4), probability=1, selection_draws=torch.zeros_like(original, dtype=torch.float), replacement_draws=torch.full_like(original, 0.85, dtype=torch.float))
        self.assertEqual(labels.ne(-100).sum().item(), 1024)
        self.assertTrue((corrupted >= 5).all())

    def test_half_open_boundaries_do_not_leak_to_next_token(self):
        offsets = [(0, 1), (1, 2), (2, 3)]
        self.assertEqual(character_span_to_tokens(offsets, 0, 1), (0, 0))
        self.assertEqual(character_span_to_tokens(offsets, 1, 2), (1, 1))
        self.assertIsNone(character_span_to_tokens(offsets, 0, 4))

    def test_multibyte_characters_use_character_not_byte_offsets(self):
        tokenizer = ByteBPETokenizer()
        context = "a café b"
        features = qa_features("?", context, tokenizer, context_length=24, overlap=2, answer_start=2, answer_text="café")
        feature = next(feature for feature in features if feature.start_position != -100)
        start, end = feature.start_position, feature.end_position
        self.assertEqual(context[feature.offsets[start][0] : feature.offsets[end][1]], "café")
        self.assertEqual(tokenizer.decode(feature.input_ids[start : end + 1]).strip(), "café")
        self.assertEqual(sum(span == (5, 6) for span in feature.offsets), 2)

    def test_overlapping_windows_recover_an_answer_after_first_window(self):
        tokenizer = ByteBPETokenizer()
        context = "alpha beta gamma delta"
        features = qa_features("?", context, tokenizer, context_length=16, overlap=5, answer_start=context.index("gamma"), answer_text="gamma")
        self.assertEqual(features[0].start_position, -100)
        valid = [feature for feature in features if feature.start_position != -100]
        self.assertGreaterEqual(len(valid), 1)
        for feature in valid:
            self.assertEqual(context[feature.offsets[feature.start_position][0] : feature.offsets[feature.end_position][1]], "gamma")
        self.assertTrue(all(feature.input_ids[:4] == [0, 68, 2, 2] for feature in features))

    def test_question_overflow_and_bad_answer_annotations_raise(self):
        tokenizer = ByteBPETokenizer()
        with self.assertRaises(ValueError):
            qa_features("question too long", "x", tokenizer, context_length=8, overlap=1)
        with self.assertRaises(ValueError):
            qa_features("?", "alpha", tokenizer, context_length=16, overlap=1, answer_start=0, answer_text="beta")

    def test_decoder_rejects_question_and_reversed_span_peaks(self):
        prediction = best_span([100, 5, 1, 9], [100, 8, 7, 1], [None, (0, 1), (2, 3), (4, 5)], "a b c", max_answer_tokens=2)
        self.assertEqual(prediction["text"], "a")
        self.assertEqual((prediction["start_token"], prediction["end_token"]), (1, 1))

    def test_squad_normalization_and_multiset_overlap(self):
        self.assertEqual(answer_metrics("The RED apple!", ["red apple"]), {"exact_match": 1.0, "f1": 1.0})
        metrics = answer_metrics("red red blue", ["red blue blue"])
        self.assertEqual(metrics["exact_match"], 0)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)
        self.assertEqual(answer_metrics("", [""]), {"exact_match": 1.0, "f1": 1.0})


class ModelChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        self.config = RobertaConfig(vocab_size=30, embedding_dimension=8, num_transformer_blocks=1, num_attention_heads=2, context_length=16)

    def test_padding_aware_position_numbers(self):
        ids = torch.tensor([[0, 7, 2, 1, 1], [1, 0, 7, 2, 1]])
        self.assertEqual(position_ids(ids).tolist(), [[2, 3, 4, 1, 1], [1, 2, 3, 4, 1]])

    def test_attention_matches_independent_explicit_matrix_path(self):
        layer = RobertaAttention(self.config).double().eval()
        x = torch.randn(1, 3, 8, dtype=torch.float64)
        mask = torch.tensor([[True, True, False]])
        q, k, v = [projection(x).view(1, 3, 2, 4).transpose(1, 2) for projection in (layer.q_proj, layer.k_proj, layer.v_proj)]
        scores = q @ k.transpose(-2, -1) / 2.0
        scores[..., 2] = -torch.inf
        weights = scores.softmax(-1)
        expected = layer.out_proj((weights @ v).transpose(1, 2).reshape(1, 3, 8))
        torch.testing.assert_close(layer(x, mask), expected, rtol=1e-12, atol=1e-12)

    def test_padded_values_do_not_affect_real_token_hidden_states(self):
        model = RobertaForMaskedLM(self.config).eval()
        ids = torch.tensor([[0, 6, 2, 1, 1]])
        mask = ids.ne(1)
        changed = ids.clone()
        changed[0, 3:] = torch.tensor([25, 26])
        torch.testing.assert_close(model(ids, mask)["logits"][:, :3], model(changed, mask)["logits"][:, :3])

    def test_encoder_can_use_right_context(self):
        model = RobertaForMaskedLM(self.config).eval()
        ids = torch.tensor([[0, 4, 8, 2]])
        changed = ids.clone()
        changed[0, 2] = 20
        difference = (model(ids)["logits"][0, 1] - model(changed)["logits"][0, 1]).abs().max()
        self.assertGreater(difference.item(), 1e-7)

    def test_tied_parameter_is_one_object_with_decoder_gradient(self):
        model = RobertaForMaskedLM(self.config)
        embedding = model.roberta.embeddings.word_embeddings.weight
        self.assertIs(model.mlm_head.decoder.weight, embedding)
        result = model(torch.tensor([[0, 4, 2]]), labels=torch.tensor([[-100, 6, -100]]))
        result["loss"].backward()
        self.assertGreater(embedding.grad[6].abs().sum().item(), 0)

    def test_loss_denominator_is_number_of_selected_targets(self):
        logits = torch.zeros(1, 4, 5, dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([[1, -100, 2, 3]])
        summed, count = masked_token_loss(logits, labels)
        self.assertEqual(count, 3)
        self.assertAlmostEqual(float(summed.detach()), 3 * math.log(5), places=12)
        summed.backward()
        self.assertEqual(logits.grad[0, 1].abs().sum().item(), 0)

    def test_all_ignored_mlm_and_qa_losses_are_finite_zero(self):
        ids = torch.tensor([[0, 6, 2]])
        mlm = RobertaForMaskedLM(self.config)
        result = mlm(ids, labels=torch.full_like(ids, -100))
        self.assertEqual(result["target_count"], 0)
        self.assertEqual(result["loss"].item(), 0)
        result["loss"].backward()
        qa = RobertaForQuestionAnswering(self.config)
        result = qa(ids, start_positions=torch.tensor([-100]), end_positions=torch.tensor([-100]))
        self.assertEqual(result["target_count"], 0)
        self.assertEqual(result["loss"].item(), 0)

    def test_qa_loss_is_two_position_classifications(self):
        qa = RobertaForQuestionAnswering(self.config)
        ids = torch.tensor([[0, 6, 7, 2], [0, 9, 2, 1]])
        starts, ends = torch.tensor([1, -100]), torch.tensor([2, -100])
        result = qa(ids, start_positions=starts, end_positions=ends)
        expected = (F.cross_entropy(result["start_logits"][:1], starts[:1]) + F.cross_entropy(result["end_logits"][:1], ends[:1])) / 2
        torch.testing.assert_close(result["loss"], expected)
        self.assertEqual(result["target_count"], 1)


class TrainingChecks(unittest.TestCase):
    def test_cpu_sampling_and_model_device_transfer_preserve_targets(self):
        bundle = build_training_bundle("mlm", total_steps=10, device="cpu")
        batch = next_batch(bundle)
        generator_state = bundle.mask_generator.get_state().clone()
        moved = to_model_device(batch, bundle.model)
        for name, value in batch.items():
            self.assertEqual(value.device.type, "cpu")
            torch.testing.assert_close(moved[name], value)
        self.assertTrue(torch.equal(generator_state, bundle.mask_generator.get_state()))
        result = bundle.model(**moved)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(bundle.model.mlm_head.dense.weight.grad).all())

    def test_fill_mask_is_one_token_with_explicit_candidates(self):
        bundle = build_training_bundle("mlm", total_steps=10)
        candidates = fill_mask(bundle.model, bundle.tokenizer, "Paris is <mask>.", top_k=3)
        self.assertEqual(len(candidates), 3)
        self.assertTrue(all(candidate["token_id"] >= 5 for candidate in candidates))
        self.assertLessEqual(sum(candidate["probability"] for candidate in candidates), 1)
        with self.assertRaises(ValueError):
            fill_mask(bundle.model, bundle.tokenizer, "No missing marker.")
        with self.assertRaises(ValueError):
            fill_mask(bundle.model, bundle.tokenizer, "<mask> and <mask>")

    def test_checkpoint_identity_tracks_schedule_seed_corpus_and_protocol(self):
        import roberta_train

        bundle = build_training_bundle("mlm", seed=7, total_steps=10)
        baseline = bundle.identity()
        self.assertNotEqual(baseline, build_training_bundle("mlm", seed=7, total_steps=11).identity())
        self.assertNotEqual(baseline, build_training_bundle("mlm", seed=8, total_steps=10).identity())
        bundle.accumulation_steps = 2
        self.assertNotEqual(baseline, bundle.identity())
        bundle.accumulation_steps = 1
        roberta_train.TRAIN_TEXTS.append("A new document changes corpus identity.")
        try:
            self.assertNotEqual(baseline, bundle.identity())
        finally:
            roberta_train.TRAIN_TEXTS.pop()
        self.assertEqual(baseline, bundle.identity())

    def test_unequal_microbatch_target_counts_match_combined_gradient(self):
        config = RobertaConfig(vocab_size=30, embedding_dimension=8, num_transformer_blocks=1, num_attention_heads=2, context_length=16)
        torch.manual_seed(1)
        model = RobertaForMaskedLM(config)
        copy = RobertaForMaskedLM(config)
        copy.load_state_dict(model.state_dict())
        ids = torch.tensor([[0, 4, 4, 2], [0, 4, 4, 2]])
        labels = torch.tensor([[-100, 7, -100, -100], [-100, 8, 9, -100]])
        model(ids, labels=labels)["loss"].backward()
        for row in range(2):
            (copy(ids[row:row+1], labels=labels[row:row+1])["loss_sum"] / 3).backward()
        for parameter, counterpart in zip(model.parameters(), copy.parameters()):
            torch.testing.assert_close(parameter.grad, counterpart.grad, atol=2e-7, rtol=3e-5)

    def test_strict_backbone_transfer_and_tokenizer_mismatch(self):
        mlm = build_training_bundle("mlm", total_steps=10)
        qa = build_training_bundle("qa", total_steps=10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "backbone.pt"
            export_backbone(mlm, path)
            load_backbone_for_qa(qa, path)
            for name, value in mlm.model.roberta.state_dict().items():
                torch.testing.assert_close(qa.model.roberta.state_dict()[name], value)
            qa.tokenizer = ByteBPETokenizer()
            with self.assertRaises(ValueError):
                load_backbone_for_qa(qa, path)

    def test_tiny_training_reduces_losses_and_decodes_seen_spans(self):
        reports = run_demo(steps=120)
        self.assertLess(reports["mlm"]["after"]["loss"], reports["mlm"]["before"]["loss"])
        self.assertLess(reports["qa"]["training_fixture_loss_after"], 0.1)
        self.assertEqual(sum(prediction["exact_match"] for prediction in reports["qa"]["seen_fixture_predictions"]), 8)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
