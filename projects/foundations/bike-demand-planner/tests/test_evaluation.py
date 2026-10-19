"""Frozen partition identity, row alignment and held-out fit isolation."""

import json
import tempfile
import unittest
from pathlib import Path

from bike_demand_planner.data import sha256
from bike_demand_planner.evaluation import (
    Forecast, HourIdentity, LabelledHour, hour_workingday_key, load_labelled_partition,
    run_hour_workingday_baseline, run_mean_baseline, score_forecasts,
)
from bike_demand_planner.features import encode_calendar
from bike_demand_planner.splits import prepare_calendar_dataset
from test_splits import source_rows, write_source


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def prepare_fixture(root, validation_target=16):
    rows = source_rows()
    for index, target in ((0, 10), (1, 30), (3, validation_target), (4, validation_target)):
        rows[index].update(casual="0", registered=str(target), cnt=str(target))
    root.mkdir(parents=True)
    identity = write_source(root / "hour.csv", rows)
    directory = root / "processed"
    prepare_calendar_dataset(root / "hour.csv", directory, identity)
    return directory


class EvaluationTests(unittest.TestCase):
    def test_group_key_uses_the_encoded_hour_and_working_day_not_outcomes(self):
        features = tuple(encode_calendar(dict(hr=8, weekday=1, mnth=1, holiday=0, workingday=1)))
        identity = HourIdentity(1, "2012-01-09T08:00:00", "2012-01-09T07:00:00")
        for target in (0, 420):
            self.assertEqual(hour_workingday_key(LabelledHour(identity, features, target)), (8, 1))
        for changed in (features[:44], (1.0,) * 45, (0.0,) * 45):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                hour_workingday_key(LabelledHour(identity, changed, 420))

    def test_group_run_uses_training_only_fallback_and_unchanged_validation_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = []
            for label, target in (("first", 16), ("changed", 1000)):
                directory = prepare_fixture(root / label, target)
                (directory / "test.csv").unlink()
                (root / label / "hour.csv").unlink()
                report = run_hour_workingday_baseline(directory, root / f"run-{label}", PROJECT_ROOT)
                results.append(report)
                self.assertEqual(report["model"]["fallback"]["value"], 20)
                self.assertEqual(report["model"]["observed_group_count"], 2)
                self.assertEqual(report["fallback_counts"], {"train": 0, "validation": 1})
                self.assertEqual(report["metrics"]["train"]["mae"], 0)
                self.assertFalse(report["test_evaluated"])
                self.assertEqual(report["status"], "completed")
                self.assertEqual(report["task"]["features_used"][-1], "workingday")
                self.assertEqual(len(report["task"]["features_used"]), 25)
                for name, identity in report["artifacts_sha256"].items():
                    self.assertEqual(sha256((root / f"run-{label}" / name).read_bytes()), identity)
            self.assertEqual(results[0]["model"], results[1]["model"])
            self.assertEqual(results[0]["metrics"]["validation"]["mae"], 9)
            self.assertEqual(results[1]["metrics"]["validation"]["mae"], 975)

    def test_loader_rejects_test_before_any_file_access(self):
        with self.assertRaisesRegex(ValueError, "Only train and validation"):
            load_labelled_partition(Path("does-not-exist"), "test")

    def test_metrics_refuse_reordered_missing_and_duplicate_forecasts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = prepare_fixture(Path(temporary) / "data")
            rows, _ = load_labelled_partition(directory, "train")
            forecasts = [Forecast(row.identity, 20) for row in rows]
            self.assertEqual(score_forecasts(rows, forecasts)["mae"], 10)
            for changed in (forecasts[::-1], forecasts[:1], [forecasts[0], forecasts[0]]):
                with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "identities"):
                    score_forecasts(rows, changed)

    def test_changed_partition_and_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = prepare_fixture(Path(temporary) / "data")
            train = directory / "train.csv"
            train.write_bytes(train.read_bytes() + b"changed\n")
            with self.assertRaisesRegex(ValueError, "train.csv differs"):
                load_labelled_partition(directory, "train")
            manifest = directory / "split-manifest.json"
            content = json.loads(manifest.read_text())
            content["counts"]["train"] = 9000
            manifest.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError, "manifest identity"):
                load_labelled_partition(directory, "train")

    def test_validation_changes_do_not_change_fit_and_test_files_are_unneeded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = []
            for label, target in (("first", 16), ("changed", 1000)):
                directory = prepare_fixture(root / label, target)
                (directory / "test.csv").unlink()
                (root / label / "hour.csv").unlink()
                report = run_mean_baseline(directory, root / f"run-{label}", PROJECT_ROOT)
                results.append(report)
                self.assertEqual(report["model"]["value"], 20)
                self.assertEqual(report["metrics"]["train"]["mae"], 10)
                self.assertFalse(report["test_evaluated"])
                self.assertEqual(report["status"], "completed")
                self.assertIn("content_sha256", report["source"])
                for name, identity in report["artifacts_sha256"].items():
                    self.assertEqual(sha256((root / f"run-{label}" / name).read_bytes()), identity)
            self.assertEqual(results[0]["model"], results[1]["model"])
            self.assertEqual(results[0]["metrics"]["validation"]["mae"], 4)
            self.assertEqual(results[1]["metrics"]["validation"]["mae"], 980)

    def test_failed_run_retains_reason_and_existing_run_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "failed"
            with self.assertRaises(FileNotFoundError):
                run_mean_baseline(root / "missing", run, PROJECT_ROOT)
            record_bytes = (run / "run.json").read_bytes()
            record = json.loads(record_bytes)
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["failure"]["type"], "FileNotFoundError")
            with self.assertRaises(FileExistsError):
                run_mean_baseline(root / "missing", run, PROJECT_ROOT)
            self.assertEqual((run / "run.json").read_bytes(), record_bytes)


if __name__ == "__main__":
    unittest.main()
