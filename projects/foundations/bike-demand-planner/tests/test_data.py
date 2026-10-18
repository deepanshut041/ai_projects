"""Known rows, temporal failures and cached-source integrity checks."""

import csv
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from bike_demand_planner.data import (
    CALENDAR_FIELDS, RAW_FIELDS, audit_hourly_csv, parse_hourly_row, prepare_hourly_data,
)


def hourly_row(**changes):
    row = dict(zip(RAW_FIELDS, (
        "1", "2011-01-01", "1", "0", "1", "0", "0", "6", "0", "1",
        "0.24", "0.2879", "0.81", "0", "3", "13", "16",
    ), strict=True))
    row.update(changes)
    return row


def csv_text(rows, fields=RAW_FIELDS):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


class DataTests(unittest.TestCase):
    def test_known_row_separates_calendar_identity_and_target(self):
        record = parse_hourly_row(hourly_row())
        self.assertEqual(record.target_time.isoformat(), "2011-01-01T00:00:00")
        self.assertIsNone(record.target_time.tzinfo)
        self.assertEqual(record.record_id, 1)
        self.assertEqual(record.target, 16)
        self.assertEqual(list(record.calendar), list(CALENDAR_FIELDS))
        self.assertEqual(record.calendar["weekday"], 6)
        for excluded in ("cnt", "casual", "registered", "instant", "temp", "weathersit"):
            self.assertNotIn(excluded, record.calendar)

    def test_malformed_fields_and_calendar_alignment_are_rejected(self):
        for changes in (
            {"cnt": ""}, {"hr": "24"}, {"temp": "nan"}, {"windspeed": "1.1"},
            {"cnt": "-1"}, {"cnt": "17"}, {"yr": "1"}, {"mnth": "2"},
            {"weekday": "0"}, {"workingday": "1"}, {"holiday": "2"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                parse_hourly_row(hourly_row(**changes))

    def test_unrecorded_hour_is_reported_without_adding_zero_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hour.csv"
            path.write_text(csv_text([hourly_row(), hourly_row(instant="2", hr="2")]))
            report = audit_hourly_csv(path)
            self.assertEqual(report["rows"], 2)
            self.assertEqual(report["expected_naive_hour_slots"], 3)
            self.assertEqual(report["unrecorded_naive_hour_slots"], 1)
            self.assertEqual(report["gap_count"], 1)
            self.assertEqual(report["first_gaps"][0]["unrecorded_hours"], 1)
            self.assertEqual(report["scope"], "raw_structural_audit")
            self.assertNotIn("split_manifest", report)

    def test_duplicate_ids_duplicate_hours_and_reversed_order_are_rejected(self):
        cases = [
            [hourly_row(), hourly_row(hr="1")],
            [hourly_row(), hourly_row(instant="2")],
            [hourly_row(hr="2"), hourly_row(instant="2", hr="1")],
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hour.csv"
            for rows in cases:
                path.write_text(csv_text(rows))
                with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "CSV line 3"):
                    audit_hourly_csv(path)

    def test_empty_data_and_wrong_header_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hour.csv"
            for content in (csv_text([]), csv_text([hourly_row()], tuple(reversed(RAW_FIELDS)))):
                path.write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    audit_hourly_csv(path)

    def test_acquisition_retains_hashes_and_reuses_cache_without_extracting_other_files(self):
        buffer = io.BytesIO()
        with ZipFile(buffer, "w") as archive:
            archive.writestr("hour.csv", csv_text([hourly_row()]))
            archive.writestr("Readme.txt", "Test source documentation")
            archive.writestr("../outside.txt", "Must not be extracted")
        content = buffer.getvalue()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "data"
            with patch("bike_demand_planner.data.urlopen", return_value=io.BytesIO(content)) as download:
                report = prepare_hourly_data("https://example.test/source", "https://example.test/data.zip", directory)
            download.assert_called_once()
            self.assertEqual(report["source"]["archive_sha256"], hashlib.sha256(content).hexdigest())
            self.assertFalse((Path(temporary) / "outside.txt").exists())
            self.assertEqual(sorted(p.name for p in (directory / "raw").iterdir()),
                             ["Readme.txt", "bike-sharing.zip", "hour.csv", "source.json"])
            with patch("bike_demand_planner.data.urlopen", side_effect=AssertionError("Unexpected network access")):
                cached = prepare_hourly_data("https://example.test/source", "https://example.test/data.zip", directory)
            self.assertEqual(cached["source"], report["source"])
            (directory / "raw/hour.csv").write_text("Modified local content")
            with self.assertRaisesRegex(ValueError, "Existing hour.csv differs"):
                prepare_hourly_data("https://example.test/source", "https://example.test/data.zip", directory)
            archive_path = directory / "raw/bike-sharing.zip"
            archive_path.write_bytes(content + b"changed")
            with self.assertRaisesRegex(ValueError, "Cached archive disagrees"):
                prepare_hourly_data("https://example.test/source", "https://example.test/data.zip", directory)


if __name__ == "__main__":
    unittest.main()
