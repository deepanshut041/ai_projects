"""Check known arithmetic, training behavior, full-state replay and failed saves."""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from prerequisite_checkpoint import generations, load_latest, restore_payload, save_generation, snapshot, verified_generation
from prerequisite_mlp import Config, Sampler, TinyClassifier, evaluate, fixture, make_training_state, nll_statistics, run_session, update


def compare(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            compare(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            compare(a, b)
    else:
        assert left == right, (left, right)


def draws():
    return [random.random(), float(np.random.random()), float(torch.rand(()))]


def resumed_process(directory, destination):
    state = make_training_state()
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    restored = load_latest(directory, state)
    random_draws = draws()
    updates = [update(state) for _ in range(5)]
    torch.save({"snapshot": snapshot(state), "draws": random_draws, "updates": updates, "restored": restored}, destination)


class NumericalChecks(unittest.TestCase):
    def test_known_labels_and_disjoint_held_out_points(self):
        data = fixture()
        train, labels = data["train"]
        validation, validation_labels = data["validation"]
        self.assertEqual((len(train), len(validation)), (64, 36))
        self.assertEqual(labels.sum().item(), 32)
        self.assertEqual(validation_labels.sum().item(), 18)
        self.assertFalse(set(map(tuple, train.tolist())) & set(map(tuple, validation.tolist())))
        corners = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        self.assertEqual((corners[:, 0] * corners[:, 1] < 0).long().tolist(), [0, 1, 1, 0])

    def test_two_layer_forward_matches_hand_calculated_tanh(self):
        model = TinyClassifier(hidden_features=2)
        with torch.no_grad():
            model.hidden.weight.copy_(torch.eye(2))
            model.hidden.bias.zero_()
            model.output.weight.copy_(torch.tensor([[1., -1.], [-1., 1.]]))
            model.output.bias.copy_(torch.tensor([0.5, -0.5]))
        logits = model(torch.tensor([[0., 0.], [1., -1.]]))
        value = 2 * math.tanh(1) + 0.5
        torch.testing.assert_close(logits, torch.tensor([[0.5, -0.5], [value, -value]]))
        self.assertEqual(sum(parameter.numel() for parameter in TinyClassifier().parameters()), 42)

    def test_uniform_nll_is_log_two_per_record(self):
        total, count, correct = nll_statistics(torch.zeros(3, 2), torch.tensor([0, 1, 0]))
        self.assertEqual((count, correct), (3, 2))
        self.assertAlmostEqual(float(total), 3 * math.log(2), places=6)
        with self.assertRaises(ValueError):
            nll_statistics(torch.zeros(0, 2), torch.tensor([], dtype=torch.long))
        with self.assertRaises(ValueError):
            nll_statistics(torch.zeros(3, 2), torch.tensor([0., 1., 0.]))

    def test_hidden_weight_gradient_agrees_with_finite_difference(self):
        state = make_training_state()
        inputs = torch.tensor([[0.7, -0.3], [-1.1, 0.7]])
        labels = torch.tensor([1, 1])
        loss = torch.nn.functional.cross_entropy(state.model(inputs), labels)
        loss.backward()
        actual = float(state.model.hidden.weight.grad[0, 0])
        with torch.no_grad():
            original = state.model.hidden.weight[0, 0].clone()
            state.model.hidden.weight[0, 0] = original + 1e-3
            positive = float(torch.nn.functional.cross_entropy(state.model(inputs), labels))
            state.model.hidden.weight[0, 0] = original - 1e-3
            negative = float(torch.nn.functional.cross_entropy(state.model(inputs), labels))
            state.model.hidden.weight[0, 0] = original
        self.assertAlmostEqual(actual, (positive - negative) / 2e-3, delta=2e-4)

    def test_unequal_microbatches_match_the_combined_gradient_update(self):
        combined = make_training_state()
        split = make_training_state()
        features, labels = combined.data["train"]
        x, y = features[[0, 1, 20, 40]], labels[[0, 1, 20, 40]]
        update(combined, batches=[(x, y)])
        update(split, batches=[(x[:1], y[:1]), (x[1:], y[1:])])
        for parameter, counterpart in zip(combined.model.parameters(), split.model.parameters()):
            torch.testing.assert_close(parameter, counterpart, atol=2e-7, rtol=1e-6)

    def test_update_clears_gradients_and_steps_schedule_once(self):
        state = make_training_state(accumulation_steps=2)
        first = update(state)
        second = update(state)
        self.assertEqual(first["examples"], 32)
        self.assertAlmostEqual(first["learning_rate_used"], 0.03 / 20)
        self.assertAlmostEqual(second["learning_rate_used"], 0.03 * 2 / 20)
        self.assertEqual((state.step, state.examples_seen), (2, 64))
        self.assertTrue(all(parameter.grad is None for parameter in state.model.parameters()))

    def test_evaluation_does_not_consume_sampler_or_change_parameters(self):
        state = make_training_state()
        before = deepcopy(state.model.state_dict())
        sampler = deepcopy(state.sampler.state_dict())
        state.model.train()
        report = evaluate(state)
        self.assertEqual(report["count"], 36)
        compare(state.model.state_dict(), before)
        compare(state.sampler.state_dict(), sampler)
        self.assertTrue(state.model.training)
        self.assertTrue(all(parameter.grad is None for parameter in state.model.parameters()))

    def test_sampler_restore_preserves_next_order_across_epoch_boundary(self):
        sampler = Sampler(4, 9)
        first = sampler.next(3)
        saved = sampler.state_dict()
        expected = sampler.next(5)
        other = Sampler(4, 999)
        other.load_state_dict(saved)
        compare(other.next(5), expected)
        self.assertEqual(len(first), 3)
        self.assertEqual(other.epoch, 1)

    def test_nonfinite_loss_stops_before_optimizer_update(self):
        state = make_training_state()
        before = deepcopy(state.model.state_dict())
        with self.assertRaises(FloatingPointError):
            update(state, batches=[(torch.full((2, 2), torch.nan), torch.tensor([0, 1]))])
        self.assertEqual(state.step, 0)
        compare(state.model.state_dict(), before)

    def test_training_fits_declared_grid_and_held_out_sign_cases(self):
        state = make_training_state()
        before = evaluate(state)
        for _ in range(1000):
            update(state)
        after = evaluate(state)
        self.assertLess(after["loss"], before["loss"])
        self.assertGreaterEqual(after["accuracy"], 0.95)
        self.assertEqual(evaluate(state, "train")["correct"], 64)
        corners = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        self.assertEqual(state.model(corners).argmax(-1).tolist(), [0, 1, 1, 0])


class RecoveryChecks(unittest.TestCase):
    def test_storage_budget_rejects_before_replacing_a_good_generation(self):
        state = make_training_state()
        with tempfile.TemporaryDirectory() as directory:
            update(state)
            older = save_generation(directory, state)
            before = (Path(directory) / older["generation"] / "state.pt").read_bytes()
            update(state)
            with self.assertRaises(OSError):
                save_generation(directory, state, storage_budget_bytes=2 * older["bytes"])
            self.assertEqual(len(generations(directory)), 1)
            self.assertEqual((Path(directory) / older["generation"] / "state.pt").read_bytes(), before)
            fresh = make_training_state()
            self.assertEqual(load_latest(directory, fresh)["step"], 1)

    def test_low_free_space_stops_the_session_and_preserves_recovery(self):
        import prerequisite_checkpoint
        import prerequisite_mlp
        from types import SimpleNamespace

        calls = []
        actual_update = prerequisite_mlp.update

        def recorded_update(state, batches=None):
            calls.append(state.step)
            return actual_update(state, batches)

        with tempfile.TemporaryDirectory() as directory:
            available = prerequisite_checkpoint.shutil.disk_usage(directory)
            with patch.object(prerequisite_checkpoint.shutil, "disk_usage", side_effect=[available, SimpleNamespace(free=0)]), patch.object(prerequisite_mlp, "update", side_effect=recorded_update):
                with self.assertRaises(OSError):
                    run_session(directory, mode="new", max_updates=10, checkpoint_updates=2)
            self.assertEqual(calls, [0, 1])
            self.assertEqual(len(generations(directory)), 1)
            self.assertEqual(load_latest(directory, make_training_state())["step"], 0)

    def test_rotation_preserves_verified_generations_of_another_identity(self):
        state = make_training_state(seed=7)
        foreign = make_training_state(seed=8)
        with tempfile.TemporaryDirectory() as directory:
            foreign_record = save_generation(directory, foreign)
            for _ in range(3):
                update(state)
                save_generation(directory, state)
            folders = generations(directory)
            self.assertEqual(len(folders), 3)
            foreign_folder = Path(directory) / foreign_record["generation"]
            self.assertIsNotNone(verified_generation(foreign_folder))
            self.assertEqual(load_latest(directory, make_training_state(seed=7))["step"], 3)
            self.assertEqual(load_latest(directory, make_training_state(seed=8))["step"], 0)

    def test_session_and_persistence_budget_caps_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            for settings in ({"max_updates": 1001}, {"max_seconds": 3601}, {"checkpoint_updates": 101}, {"checkpoint_seconds": 301}, {"max_seconds": float("nan")}):
                with self.assertRaises(ValueError):
                    run_session(directory, mode="new", **settings)

    def test_fresh_process_resume_matches_full_training_state_and_random_draws(self):
        state = make_training_state()
        for _ in range(3):
            update(state)
        with tempfile.TemporaryDirectory() as directory:
            save_generation(directory, state)
            expected_draws = draws()
            expected_updates = [update(state) for _ in range(5)]
            expected = snapshot(state)
            output = Path(directory) / "resumed.pt"
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--resume-probe", directory, "--output", str(output)], check=True)
            actual = torch.load(output, weights_only=True)
            compare(expected, actual["snapshot"])
            compare(expected_draws, actual["draws"])
            compare(expected_updates, actual["updates"])
            self.assertEqual(actual["restored"]["step"], 3)
            self.assertEqual(actual["snapshot"]["step"], 8)

    def test_corrupt_and_incomplete_generations_fall_back(self):
        state = make_training_state()
        with tempfile.TemporaryDirectory() as directory:
            update(state)
            older = save_generation(directory, state)
            update(state)
            newer = save_generation(directory, state)
            with (Path(directory) / newer["generation"] / "state.pt").open("ab") as file:
                file.write(b"bad trailing bytes")
            partial = Path(directory) / "99999999-interrupted"
            partial.mkdir()
            (partial / "state.pt").write_bytes(b"unfinished")
            restored = make_training_state()
            result = load_latest(directory, restored)
            self.assertEqual(result["generation"], older["generation"])
            self.assertEqual(restored.step, 1)
            self.assertEqual(len(result["skipped"]), 2)

    def test_a_deserializable_weight_only_generation_is_not_complete_state(self):
        import hashlib

        state = make_training_state()
        with tempfile.TemporaryDirectory() as directory:
            update(state)
            older = save_generation(directory, state)
            update(state)
            newer = save_generation(directory, state)
            folder = Path(directory) / newer["generation"]
            payload = snapshot(state)
            del payload["optimizer"]
            torch.save(payload, folder / "state.pt")
            blob = (folder / "state.pt").read_bytes()
            manifest = json.loads((folder / "manifest.json").read_text())
            manifest["bytes"], manifest["sha256"] = len(blob), hashlib.sha256(blob).hexdigest()
            manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
            (folder / "manifest.json").write_bytes(manifest_bytes)
            (folder / "complete.json").write_text(json.dumps({"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}))
            self.assertIsNone(verified_generation(folder))
            fresh = make_training_state()
            self.assertEqual(load_latest(directory, fresh)["generation"], older["generation"])
            self.assertEqual(fresh.step, 1)

    def test_retention_keeps_two_verified_generations(self):
        state = make_training_state()
        with tempfile.TemporaryDirectory() as directory:
            for _ in range(4):
                update(state)
                save_generation(directory, state)
            valid = [folder for folder in generations(directory) if verified_generation(folder) is not None]
            self.assertEqual(len(valid), 2)
            self.assertEqual([verified_generation(folder)["step"] for folder in valid], [4, 3])

    def test_identity_rejection_precedes_parameter_mutation(self):
        saved_state = make_training_state()
        update(saved_state)
        payload = snapshot(saved_state)
        for key in ("source_sha256", "config", "data_sha256", "torch", "parameter_dtypes", "threads"):
            changed = deepcopy(payload)
            changed["identity"][key] = "different"
            fresh = make_training_state()
            before = deepcopy(fresh.model.state_dict())
            with self.assertRaises(ValueError):
                restore_payload(changed, fresh)
            compare(before, fresh.model.state_dict())
            self.assertEqual(fresh.step, 0)

    def test_pending_gradients_and_unsupported_precision_cannot_save(self):
        state = make_training_state()
        features, labels = state.data["train"]
        torch.nn.functional.cross_entropy(state.model(features[:2]), labels[:2]).backward()
        with self.assertRaises(ValueError):
            snapshot(state)
        state.optimizer.zero_grad(set_to_none=True)
        state.model.double()
        with self.assertRaises(ValueError):
            snapshot(state)

    def test_new_resume_and_session_slices_do_not_reset_the_total_goal(self):
        with tempfile.TemporaryDirectory() as parent:
            directory = Path(parent) / "run"
            first = run_session(directory, mode="new", max_updates=3)
            self.assertEqual(first["resources"]["completed_updates"], 3)
            second = run_session(directory, mode="resume", max_updates=2)
            self.assertEqual(second["restored"]["step"], 3)
            self.assertEqual(second["resources"]["completed_updates"], 5)
            self.assertEqual(second["remaining_updates"], 995)
            with self.assertRaises(ValueError):
                run_session(directory, mode="new", max_updates=1)
            with self.assertRaises(ValueError):
                run_session(Path(parent) / "missing", mode="resume", max_updates=1)

    def test_persistence_failure_stops_without_continuing_unprotected_updates(self):
        import prerequisite_checkpoint
        import prerequisite_mlp

        calls = []
        actual_save = prerequisite_checkpoint.save_generation
        actual_update = prerequisite_mlp.update

        def failing_save(directory, state, keep=2):
            if state.step >= 2:
                raise OSError("simulated persistent storage failure")
            return actual_save(directory, state, keep)

        def recorded_update(state, batches=None):
            calls.append(state.step)
            return actual_update(state, batches)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(prerequisite_checkpoint, "save_generation", side_effect=failing_save), patch.object(prerequisite_mlp, "update", side_effect=recorded_update):
                with self.assertRaises(OSError):
                    run_session(directory, mode="new", max_updates=10, checkpoint_updates=2)
            self.assertEqual(calls, [0, 1])
            restored = make_training_state()
            self.assertEqual(load_latest(directory, restored)["step"], 0)


class ExportChecks(unittest.TestCase):
    def test_notebook_is_output_cleared_and_uses_the_common_trainer(self):
        import ast
        from prerequisite_export import SOURCE_FILES, notebook

        data = notebook()
        for cell in data["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                ast.parse("".join(cell["source"]))
        source = "\n".join("".join(cell["source"]) for cell in data["cells"])
        self.assertIn('"prerequisite_mlp.py", "--mode", mode', source)
        self.assertIn('"--total-steps", "1000"', source)
        self.assertIn('"--max-seconds", "3600"', source)
        self.assertNotIn("tiny_transformer.py", SOURCE_FILES)
        self.assertNotIn("fresh_process_resume.py", SOURCE_FILES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-probe")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.resume_probe:
        resumed_process(args.resume_probe, args.output)
    else:
        suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        print(json.dumps({"tests": result.testsRun, "passed": result.wasSuccessful(), "fresh_process_resume": result.wasSuccessful(), "hosted_execution": "unverified"}))
        raise SystemExit(0 if result.wasSuccessful() else 1)
