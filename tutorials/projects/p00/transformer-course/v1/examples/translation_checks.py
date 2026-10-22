"""Independent checks for attention, bilingual training and generation."""

import copy
import math
from pathlib import Path
import tempfile
import types
import unittest

import torch
from torch.nn import functional as F

from translation_attention import Attention, allowed_keys, attention_math, merge_heads, split_heads
from translation_data import TRAIN_PAIRS, Vocabulary, aligned_lines, collate_translation, make_fixture
from translation_metrics import corpus_bleu_tokens
from translation_model import PositionalEncoding, Transformer, TransformerConfig
from translation_train import adamw_groups, evaluate, logits_for, make_state, token_statistics, train_update, update


class TranslationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)

    def assertTensor(self, actual, expected, *, atol=1e-6):
        torch.testing.assert_close(actual, expected.to(actual), atol=atol, rtol=atol)

    def test_dot_product_and_weighted_value_known_answer(self):
        q = torch.tensor([[[[1., 0.]]]])
        k = torch.tensor([[[[1., 0.], [0., 1.]]]])
        v = torch.tensor([[[[2., 0.], [0., 4.]]]])
        context, weights = attention_math(q, k, v)
        p = math.exp(1 / math.sqrt(2)) / (math.exp(1 / math.sqrt(2)) + 1)
        self.assertTensor(weights, torch.tensor([[[[p, 1 - p]]]]))
        self.assertTensor(context, torch.tensor([[[[2 * p, 4 * (1 - p)]]]]))

    def test_head_layout_round_trip_and_identity(self):
        x = torch.arange(16).reshape(1, 2, 8)
        packed = split_heads(x, 2)
        self.assertEqual(packed[0, 1, 0].tolist(), [4, 5, 6, 7])
        self.assertTensor(merge_heads(packed), x)

    def test_padding_and_causality_intersection(self):
        allowed = allowed_keys(torch.tensor([[True, True, False]]), 3, causal=True)
        self.assertEqual(allowed[0, 0].tolist(), [[True, False, False], [True, True, False], [True, True, False]])

    def test_fully_blocked_attention_row_is_zero(self):
        q, k, v = (torch.randn(1, 2, 3, 4) for _ in range(3))
        allowed = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
        output, weights = attention_math(q, k, v, allowed)
        self.assertEqual(int(torch.count_nonzero(output)), 0)
        self.assertEqual(int(torch.count_nonzero(weights)), 0)
        self.assertTensor(F.scaled_dot_product_attention(q, k, v, attn_mask=allowed), output)

    def test_manual_and_sdpa_match_for_rectangular_mask(self):
        manual = Attention(8, 2, fused_attn=False).double().eval()
        efficient = Attention(8, 2, fused_attn=True).double().eval()
        efficient.load_state_dict(manual.state_dict())
        source, target = torch.randn(2, 5, 8, dtype=torch.double), torch.randn(2, 3, 8, dtype=torch.double)
        mask = torch.tensor([[True, True, True, False, False], [True] * 5])
        self.assertTensor(manual(source, target, mask), efficient(source, target, mask), atol=1e-10)

    def test_padding_key_perturbation_does_not_affect_valid_queries(self):
        layer = Attention(8, 2).eval()
        source = torch.randn(1, 4, 8)
        changed = source.clone()
        changed[:, 2:] += 1000
        mask = torch.tensor([[True, True, False, False]])
        self.assertTensor(layer(source, attention_mask=mask)[:, :2], layer(changed, attention_mask=mask)[:, :2])

    def test_causal_attention_future_isolation(self):
        layer = Attention(8, 2).eval()
        x = torch.randn(1, 4, 8)
        changed = x.clone()
        changed[:, 2:] += 500
        self.assertTensor(layer(x, causal=True)[:, :2], layer(changed, causal=True)[:, :2])

    def test_collator_shift_ignore_and_no_alias(self):
        records = [{"src_ids": [2, 4, 3], "tgt_ids": [2, 8, 9, 3]},
                   {"src_ids": [2, 5, 6, 3], "tgt_ids": [2, 7, 3]}]
        batch = collate_translation(records, 0, 1)
        self.assertEqual(batch["src_input_ids"].tolist(), [[2, 4, 3, 0], [2, 5, 6, 3]])
        self.assertEqual(batch["tgt_input_ids"].tolist(), [[2, 8, 9], [2, 7, 3]])
        self.assertEqual(batch["tgt_outputs"].tolist(), [[8, 9, 3], [7, 3, -100]])
        self.assertTrue(bool(batch["tgt_pad_mask"].all()))
        self.assertEqual(records[1]["tgt_ids"], [2, 7, 3])

    def test_parallel_file_mismatch_and_paired_empty_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            src, tgt = Path(directory) / "x.en", Path(directory) / "x.fr"
            src.write_text("one\n\nthree\n")
            tgt.write_text("un\ndeux\ntrois\n")
            self.assertEqual(list(aligned_lines(src, tgt)), [("one", "un"), ("three", "trois")])
            tgt.write_text("un\n")
            with self.assertRaisesRegex(ValueError, "line 2"):
                list(aligned_lines(src, tgt))

    def test_separate_vocabularies_normalization_and_special_tokens(self):
        source = Vocabulary.fit(["cats"], source=True)
        target = Vocabulary.fit(["École"])
        self.assertNotEqual(source.pad_token_id, target.pad_token_id)
        self.assertEqual(target.encode("e\u0301cole"), target.encode("ÉCOLE"))
        self.assertEqual(target.decode(target.encode("école")), "école")
        self.assertIn(target.tokens.index("[UNK]"), target.encode("unseen"))

    def test_sinusoidal_known_positions_and_buffer(self):
        positions = PositionalEncoding(3, 4)
        self.assertTensor(positions.encodings[0], torch.tensor([0., 1., 0., 1.]))
        self.assertTensor(positions.encodings[1], torch.tensor([math.sin(1), math.cos(1), math.sin(.01), math.cos(.01)]))
        self.assertNotIn("encodings", dict(positions.named_parameters()))
        self.assertIn("encodings", positions.state_dict())
        with self.assertRaises(ValueError):
            positions(torch.zeros(1, 4, 4))

    def test_full_model_future_target_isolation(self):
        model = Transformer(TransformerConfig(embedding_dimension=8, num_attention_heads=2)).eval()
        src, target = torch.tensor([[2, 4, 3]]), torch.tensor([[2, 5, 6, 7]])
        changed = target.clone()
        changed[:, 2:] = 8
        self.assertTensor(model(src, target)[:, :2], model(src, changed)[:, :2])

    def test_full_model_source_padding_isolation(self):
        model = Transformer(TransformerConfig(embedding_dimension=8, num_attention_heads=2)).eval()
        src, target = torch.tensor([[2, 4, 3, 0]]), torch.tensor([[2, 5]])
        changed = src.clone()
        changed[:, -1] = 7
        mask = src != 0
        self.assertTensor(model(src, target, mask), model(changed, target, mask))

    def test_gradients_reach_encoder_decoder_and_head(self):
        state = make_state(steps=20)
        loss, _, _ = token_statistics(logits_for(state.model, state.train_batch), state.train_batch["tgt_outputs"])
        loss.backward()
        for parameter in (state.model.encodings.src_embeddings.weight,
                          state.model.encoder[0].enc_attention.v_proj.weight,
                          state.model.decoder[0].cross_attention.v_proj.weight,
                          state.model.head.weight):
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_loss_ignores_padding_and_uses_token_denominator(self):
        logits = torch.tensor([[[0., 0.], [0., math.log(3)], [100., -100.]]])
        labels = torch.tensor([[0, 1, -100]])
        loss, count, correct = token_statistics(logits, labels)
        self.assertAlmostEqual(float(loss), -math.log(.5) - math.log(.75), places=6)
        self.assertEqual(int(count), 2)
        self.assertEqual(int(correct), 2)

    def test_unequal_microbatch_accumulation_matches_full_batch(self):
        records = [{"src_ids": [2, 4, 3], "tgt_ids": [2, 5, 3]},
                   {"src_ids": [2, 4, 5, 3], "tgt_ids": [2, 5, 6, 7, 3]}]
        full = collate_translation(records, 0, 1)
        chunks = [collate_translation([row], 0, 1) for row in records]
        a = Transformer(TransformerConfig(embedding_dimension=8, num_attention_heads=2))
        b = copy.deepcopy(a)
        opt_a, opt_b = torch.optim.SGD(a.parameters(), lr=.01), torch.optim.SGD(b.parameters(), lr=.01)
        train_update(a, opt_a, [full], max_grad_norm=1e6)
        train_update(b, opt_b, chunks, max_grad_norm=1e6)
        for left, right in zip(a.parameters(), b.parameters()):
            self.assertTensor(left, right, atol=2e-6)

    def test_evaluation_restores_mode(self):
        state = make_state(steps=20)
        state.model.train()
        result = evaluate(state.model, [state.train_batch])
        self.assertTrue(state.model.training)
        self.assertEqual(result["tokens"], 30)

    def test_batched_generation_finishes_rows_independently(self):
        model = Transformer(TransformerConfig(embedding_dimension=8, num_attention_heads=2))
        model.train()

        def fixed_decode(this, memory, ids, source_mask, target_mask=None):
            logits = torch.zeros(ids.shape[0], ids.shape[1], this.config.tgt_vocab_size)
            logits[0, -1, 3] = 10
            logits[1, -1, 5 if ids.shape[1] == 1 else 3] = 10
            return logits

        model.decode = types.MethodType(fixed_decode, model)
        result = model.inference(torch.tensor([[2, 4, 3], [2, 5, 3]]), max_len=8)
        self.assertEqual(result.tolist(), [[2, 3, 1], [2, 5, 3]])
        self.assertTrue(model.training)

    def test_generation_budget_includes_bos(self):
        model = Transformer(TransformerConfig(embedding_dimension=8, num_attention_heads=2))
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.zero_()
            model.head.bias[5] = 10
        result = model.inference(torch.tensor([[2, 4, 3]]), max_len=4)
        self.assertEqual(result.tolist(), [[2, 5, 5, 5]])

    def test_bleu_clipping_brevity_and_identical_reference(self):
        exact = corpus_bleu_tokens([["je", "vois", "les", "chats"]], [["je", "vois", "les", "chats"]])
        self.assertEqual(exact["score"], 100)
        clipped = corpus_bleu_tokens([["chat"] * 3], [["chat", "est", "ici"]], max_order=1)
        self.assertEqual(clipped["matches"], [1])
        self.assertAlmostEqual(clipped["score"], 100 / 3)
        short = corpus_bleu_tokens([["je", "vois"]], [["je", "vois", "les", "chats"]], max_order=1)
        self.assertAlmostEqual(short["score"], 100 * math.exp(-1))

    def test_optimizer_groups_exclude_layernorm_and_bias(self):
        state = make_state(steps=20)
        groups = adamw_groups(state.model)
        plain = {id(p) for p in groups[1]["params"]}
        self.assertIn(id(state.model.encoder[0].layer_norm.weight), plain)
        self.assertIn(id(state.model.head.bias), plain)
        self.assertNotIn(id(state.model.head.weight), plain)

    def test_bounded_fixture_training_and_generation(self):
        state = make_state(steps=400)
        before = evaluate(state.model, [state.train_batch])["loss"]
        for _ in range(400):
            update(state)
        after = evaluate(state.model, [state.train_batch])
        self.assertLess(after["loss"], before / 10)
        self.assertEqual(after["token_accuracy"], 1)
        ids = state.model.inference(state.train_batch["src_input_ids"], max_len=12)
        self.assertEqual([state.tgt_tokenizer.decode(row.tolist()) for row in ids], [t for _, t in TRAIN_PAIRS])
        self.assertEqual(state.step, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
