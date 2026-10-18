"""Date-boundary, holdout isolation and frozen-export regression checks."""

import csv
import io
import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from bike_demand_planner.data import RAW_FIELDS, sha256
from bike_demand_planner.splits import prepare_calendar_dataset, split_for_target


def source_rows():
    dates = ["2011-01-03T08:00:00", "2011-12-31T23:00:00", "2012-01-01T00:00:00",
             "2012-01-01T01:00:00", "2012-06-30T23:00:00", "2012-07-01T00:00:00",
             "2012-07-01T01:00:00"]
    rows = []
    for index, label in enumerate(dates, start=1):
        stamp = datetime.fromisoformat(label)
        rows.append(dict(zip(RAW_FIELDS, map(str, (
            index, stamp.date().isoformat(), 1, stamp.year - 2011, stamp.month,
            stamp.hour, 0, (stamp.weekday() + 1) % 7, int(stamp.weekday() < 5),
            1, 0.24, 0.28, 0.8, 0.0, 3, 13, 16,
        )), strict=True)))
    return rows


def write_source(path, rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=RAW_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    content = output.getvalue().encode()
    path.write_bytes(content)
    return sha256(content)


class SplitTests(unittest.TestCase):
    def test_boundaries_include_issue_time_gap_and_reject_unsupported_timestamps(self):
        for label, expected in [
            ("2011-12-31T23:00:00", "train"), ("2012-01-01T00:00:00", "embargo"),
            ("2012-01-01T01:00:00", "validation"), ("2012-06-30T23:00:00", "validation"),
            ("2012-07-01T00:00:00", "embargo"), ("2012-07-01T01:00:00", "test"),
            ("2012-12-31T23:00:00", "test"),
        ]:
            with self.subTest(label=label):
                self.assertEqual(split_for_target(datetime.fromisoformat(label)), expected)
        for stamp in [datetime(2010, 12, 31, 23), datetime(2013, 1, 1),
                      datetime(2011, 1, 1, 8, 30), datetime(2011, 1, 1, tzinfo=UTC)]:
            with self.subTest(stamp=stamp), self.assertRaises(ValueError):
                split_for_target(stamp)

    def test_membership_is_disjoint_complete_and_test_export_has_no_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = write_source(root / "hour.csv", source_rows())
            manifest = prepare_calendar_dataset(root / "hour.csv", root / "processed", identity)
            self.assertEqual(manifest["counts"], {"train": 2, "validation": 2, "test": 1, "embargo": 2})
            with (root / "processed/membership.csv").open() as source:
                membership = list(csv.DictReader(source))
            ids = [row["record_id"] for row in membership]
            self.assertEqual(len(ids), 7)
            self.assertEqual(len(set(ids)), 7)
            with (root / "processed/test.csv").open() as source:
                test_reader = csv.DictReader(source)
                self.assertNotIn("target", test_reader.fieldnames)
                self.assertNotIn("cnt", test_reader.fieldnames)
                self.assertNotIn("registered", test_reader.fieldnames)
                self.assertEqual([row["record_id"] for row in test_reader], ["7"])
            with (root / "processed/train.csv").open() as source:
                self.assertEqual([row["target"] for row in csv.DictReader(source)], ["16", "16"])
            self.assertEqual(sum(manifest["counts"].values()), 7)

    def test_previous_stage_labels_exist_before_evaluation_issue_times(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = write_source(root / "hour.csv", source_rows())
            manifest = prepare_calendar_dataset(root / "hour.csv", root / "processed", identity)
            bounds = manifest["observed_bounds"]
            for previous, following in [("train", "validation"), ("validation", "test")]:
                available = datetime.fromisoformat(bounds[previous]["last"]) + timedelta(hours=1)
                issue = datetime.fromisoformat(bounds[following]["first"]) - timedelta(hours=1)
                self.assertLessEqual(available, issue)

    def test_heldout_changes_do_not_fit_categories_or_training_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = source_rows()
            identity = write_source(root / "first.csv", rows)
            original = prepare_calendar_dataset(root / "first.csv", root / "first", identity)
            for row in rows[2:]:
                row.update(holiday="1", workingday="0", casual="9000", registered="1000", cnt="10000")
            changed_identity = write_source(root / "changed.csv", rows)
            changed = prepare_calendar_dataset(root / "changed.csv", root / "changed", changed_identity)
            for key in ("membership_sha256", "feature_schema_sha256", "training_feature_summary"):
                self.assertEqual(original[key], changed[key])
            self.assertEqual(original["artifacts_sha256"]["train.csv"], changed["artifacts_sha256"]["train.csv"])
            self.assertNotEqual(original["source_hour_sha256"], changed["source_hour_sha256"])
            self.assertIn("holiday", original["training_feature_summary"]["constant_features"])
            self.assertEqual(original["feature_count"], 45)

    def test_frozen_outputs_repeat_exactly_and_reject_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = write_source(root / "hour.csv", source_rows())
            first = prepare_calendar_dataset(root / "hour.csv", root / "processed", identity)
            repeated = prepare_calendar_dataset(root / "hour.csv", root / "processed", identity)
            self.assertEqual(first, repeated)
            retained = json.loads((root / "processed/split-manifest.json").read_text())
            self.assertEqual(retained, first)
            train = root / "processed/train.csv"
            train.write_text(train.read_text() + "changed\n")
            with self.assertRaisesRegex(ValueError, "Frozen artifact train.csv differs"):
                prepare_calendar_dataset(root / "hour.csv", root / "processed", identity)

    def test_wrong_source_identity_and_empty_partitions_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = write_source(root / "hour.csv", source_rows()[:1])
            with self.assertRaisesRegex(ValueError, "source identity"):
                prepare_calendar_dataset(root / "hour.csv", root / "processed", "wrong hash")
            with self.assertRaisesRegex(ValueError, "must contain observations"):
                prepare_calendar_dataset(root / "hour.csv", root / "processed", identity)
            self.assertFalse((root / "processed").exists())


if __name__ == "__main__":
    unittest.main()
