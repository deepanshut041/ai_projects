"""Check bounded sessions, refusal boundaries, accounting and failed persistence."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import course_session
from course_checkpoint import _verified
from course_source_export import export_sources


class SessionChecks(unittest.TestCase):
    def run_quiet(self, task, directory, **kwargs):
        return course_session.run_session(task, directory, emit=lambda record: None,
                                          total_steps=12, additional_updates=2,
                                          save_updates=1, **kwargs)

    def test_all_three_tasks_new_then_resume_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            for task in ("translation", "mlm", "qa"):
                directory = Path(temporary) / task
                first = self.run_quiet(task, directory, mode="new")
                second = self.run_quiet(task, directory, mode="resume")
                self.assertEqual((first["step"], second["step"]), (2, 4))
                self.assertGreater(second["input_tokens_seen"], first["input_tokens_seen"])
                self.assertGreater(second["supervised_targets_seen"], first["supervised_targets_seen"])
                self.assertEqual((first["attempts"], second["attempts"]), (2, 4))
                valid = [path for path in (directory / "checkpoints").iterdir() if _verified(path)]
                self.assertEqual(len(valid), 2)

    def test_session_slices_match_uninterrupted_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            for task in ("translation", "mlm", "qa"):
                sliced = Path(temporary) / (task + "-sliced")
                whole = Path(temporary) / (task + "-whole")
                self.run_quiet(task, sliced, mode="new")
                self.run_quiet(task, sliced, mode="resume")
                course_session.run_session(task, whole, mode="new", total_steps=12,
                                           additional_updates=4, emit=lambda record: None)
                latest = [sorted((directory / "checkpoints").iterdir())[-1] / "state.pt"
                          for directory in (sliced, whole)]
                left, right = [torch.load(path, weights_only=True) for path in latest]
                self.assertEqual(left["step"], right["step"])
                self.assertEqual(left["data_state"], right["data_state"])
                for name, value in left["model"].items():
                    torch.testing.assert_close(value, right["model"][name], rtol=0, atol=0)

    def test_wall_time_budget_stops_before_an_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = []
            result = course_session.run_session("translation", Path(temporary) / "timed",
                                                mode="new", total_steps=12, max_seconds=1e-9,
                                                emit=records.append)
            self.assertEqual(result["step"], 0)
            self.assertFalse(any(record["event"] == "update" for record in records))

    def test_new_refuses_existing_and_resume_never_starts_new(self):
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                self.run_quiet("translation", existing, mode="new")
            with self.assertRaises(FileNotFoundError):
                self.run_quiet("translation", Path(temporary) / "missing", mode="resume")
            directory = Path(temporary) / "run"
            self.run_quiet("translation", directory, mode="new")
            for folder in (directory / "checkpoints").iterdir():
                (folder / "complete.json").unlink()
            with self.assertRaisesRegex(ValueError, "compatible"):
                self.run_quiet("translation", directory, mode="resume")

    def test_resume_refuses_changed_duration_without_mutating_run_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "run"
            self.run_quiet("translation", directory, mode="new")
            original = (directory / "run.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "schedule"):
                course_session.run_session("translation", directory, mode="resume", total_steps=24,
                                           additional_updates=1, emit=lambda record: None)
            self.assertEqual((directory / "run.json").read_bytes(), original)

    def test_failed_save_stops_before_next_update_and_keeps_older_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "run"
            updates = []
            real_save = course_session.save_generation
            calls = 0

            def injected_failure(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated persistent storage failure")
                return real_save(*args, **kwargs)

            with patch("course_session.save_generation", side_effect=injected_failure):
                with self.assertRaisesRegex(RuntimeError, "training stopped"):
                    course_session.run_session("translation", directory, mode="new", total_steps=12,
                                               additional_updates=5, save_updates=1,
                                               emit=lambda record: updates.append(record))
            self.assertEqual(len([item for item in updates if item["event"] == "update"]), 1)
            valid = [path for path in (directory / "checkpoints").iterdir() if _verified(path)]
            self.assertEqual(len(valid), 1)
            self.assertEqual(_verified(valid[0])["step"], 0)

    def test_retention_budget_rejects_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(OSError, "budget"):
                self.run_quiet("translation", Path(temporary) / "tiny", mode="new", storage_budget_bytes=1)

    def test_accounting_does_not_change_sampler_or_global_rng(self):
        state = course_session.build("mlm", 7, 12, "cpu")
        sampler = state.batch_generator.get_state().clone()
        global_rng = torch.get_rng_state().clone()
        examples, tokens = course_session.input_counts("mlm", state)
        self.assertEqual(examples, state.batch_size)
        self.assertGreater(tokens, 0)
        self.assertTrue(torch.equal(state.batch_generator.get_state(), sampler))
        self.assertTrue(torch.equal(torch.get_rng_state(), global_rng))

    def test_exported_notebook_is_clear_compilable_and_pins_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.zip"
            report = export_sources(path)
            notebook = json.loads(Path(report["notebook"]).read_text())
            code = []
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])
                    source = cell["source"]
                    compile(source, "generated-colab-cell", "exec")
                    code.append(source)
            joined = "\n".join(code)
            self.assertIn(report["sha256"], joined)
            self.assertIn("run_session", joined)
            self.assertIn('MODE = "new"', joined)
            self.assertIn("source-manifest.json", joined)
            self.assertIn("drive.mount", joined)


if __name__ == "__main__":
    unittest.main(verbosity=2)
