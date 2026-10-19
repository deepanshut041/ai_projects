"""Training-only transforms, aligned evaluation and durable linear run receipts."""

import json
import tempfile
import unittest
from pathlib import Path

from bike_demand_planner.data import sha256
from bike_demand_planner.linear_evaluation import run_linear_regression
from test_evaluation import PROJECT_ROOT, prepare_fixture


class LinearEvaluationTests(unittest.TestCase):
    def test_validation_outcomes_do_not_change_parameters_transform_or_stopping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = []
            for name, target in (("original", 16), ("changed", 1000)):
                directory = prepare_fixture(root / name, target)
                (directory / "test.csv").unlink()
                (root / name / "hour.csv").unlink()
                record = run_linear_regression(directory, root / f"run-{name}", PROJECT_ROOT)
                results.append(record)
                self.assertEqual(record["status"], "completed")
                self.assertTrue(record["optimization"]["converged"])
                self.assertFalse(record["test_evaluated"])
                self.assertFalse(record["configuration"]["validation_used_for_fitting"])
                self.assertEqual(record["model"]["standardizer"]["training_count"], 2)
                self.assertEqual(record["metrics"]["validation"]["count"], 2)
                self.assertLess(record["reference"]["agreement"]["train"]["maximum_absolute_prediction_difference"], 1e-6)
                for filename, identity in record["artifacts_sha256"].items():
                    self.assertEqual(sha256((root / f"run-{name}" / filename).read_bytes()), identity)
            self.assertEqual(results[0]["model"], results[1]["model"])
            self.assertEqual(results[0]["optimization"], results[1]["optimization"])
            self.assertEqual(results[0]["metrics"]["train"], results[1]["metrics"]["train"])
            self.assertNotEqual(results[0]["metrics"]["validation"], results[1]["metrics"]["validation"])

    def test_unconverged_run_is_recorded_and_failures_cannot_overwrite_a_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = prepare_fixture(root / "data")
            record = run_linear_regression(directory, root / "limited", PROJECT_ROOT, max_steps=1)
            self.assertEqual(record["status"], "completed")
            self.assertFalse(record["optimization"]["converged"])
            self.assertEqual(record["optimization"]["stop_reason"], "maximum_updates")
            with self.assertRaises(FileNotFoundError):
                run_linear_regression(root / "missing", root / "failed", PROJECT_ROOT)
            receipt = (root / "failed/run.json").read_bytes()
            self.assertEqual(json.loads(receipt)["status"], "failed")
            with self.assertRaises(FileExistsError):
                run_linear_regression(directory, root / "failed", PROJECT_ROOT)
            self.assertEqual((root / "failed/run.json").read_bytes(), receipt)


if __name__ == "__main__":
    unittest.main()
