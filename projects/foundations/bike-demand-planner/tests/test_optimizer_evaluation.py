"""Pilot isolation, comparable work budgets, provenance and failure preservation."""

import json
import tempfile
import unittest
from pathlib import Path

from bike_demand_planner.data import sha256
from bike_demand_planner.optimizer_evaluation import run_optimizer_study
from test_evaluation import PROJECT_ROOT, prepare_fixture


class OptimizerEvaluationTests(unittest.TestCase):
    def test_validation_changes_leave_pilot_choices_and_all_fits_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); models = []; choices = []
            for name, target in (("original", 16), ("changed", 1000)):
                processed = prepare_fixture(root / name, target)
                (processed / "test.csv").unlink(); (root / name / "hour.csv").unlink()
                directory = root / ("study-" + name)
                report = run_optimizer_study(processed, directory, PROJECT_ROOT, rates=(.01, .1), seeds=(17,),
                                             passes=2, pilot_updates=3)
                self.assertEqual(report["status"], "completed")
                self.assertFalse(report["test_evaluated"])
                protocol = json.loads((directory / "protocol.json").read_text())
                self.assertFalse(protocol["validation_used_for_selection"])
                choices.append(report["selected_rates"])
                models.append({identity: (directory / rel / "model.json").read_bytes()
                               for identity, rel in report["trial_directories"].items()})
                for identity, rel in report["trial_directories"].items():
                    record = json.loads((directory / rel / "run.json").read_text())
                    self.assertEqual(record["optimization"]["examples_processed"], 4)
                    for filename, expected in record["artifacts_sha256"].items():
                        self.assertEqual(sha256((directory / rel / filename).read_bytes()), expected)
                if name == "original": original = report["summaries"]["optimizer"]["sgd"]["mae"]["mean"]
                else: self.assertNotEqual(original, report["summaries"]["optimizer"]["sgd"]["mae"]["mean"])
            self.assertEqual(choices[0], choices[1])
            self.assertEqual(models[0], models[1])

    def test_failed_study_retains_reason_and_existing_outputs_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);directory = root / "study"
            with self.assertRaises(FileNotFoundError):
                run_optimizer_study(root / "missing", directory, PROJECT_ROOT)
            receipt = (directory / "study.json").read_bytes()
            self.assertEqual(json.loads(receipt)["status"], "failed")
            with self.assertRaises(FileExistsError):
                run_optimizer_study(root / "missing", directory, PROJECT_ROOT)
            self.assertEqual(receipt, (directory / "study.json").read_bytes())
            with self.assertRaises(ValueError):
                run_optimizer_study(root / "missing", root / "invalid", PROJECT_ROOT, rates=(float("nan"),))
            self.assertFalse((root / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
