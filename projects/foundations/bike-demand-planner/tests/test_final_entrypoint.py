"""Run discovery and saved-result reuse must survive a fresh source checkout."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import final_evaluation_example as entry
from bike_demand_planner.artifacts import digest, json_bytes


class FinalEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def run_main(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = entry.main(self.root)
        return code, json.loads(output.getvalue())

    def test_missing_prerequisites_return_actionable_status_without_target_access(self):
        with patch.object(entry, "run_final_assessment") as run:
            code, result = self.run_main()
        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "needs_run")
        self.assertIn("error_review_example.py", result["message"])
        run.assert_not_called()

    def test_study_follows_reviewed_identity_instead_of_latest_unrelated_study(self):
        review = self.root/"runs/errors-new"
        expected = self.root/"runs/trees-reviewed"
        review.mkdir(parents=True)
        expected.mkdir(parents=True)
        (review/"review.json").write_text("{}")
        (expected/"study.json").write_text("{}")
        unrelated = self.root/"runs/trees-zzz"
        unrelated.mkdir()
        (unrelated/"study.json").write_text("{}")
        with patch.object(entry, "verified_review", return_value={"protocol":{"source_run":"trees-reviewed"}}):
            self.assertEqual(entry.assessment_inputs(self.root), (review, expected))

    def test_invalid_reviewed_path_is_rejected(self):
        review = self.root/"runs/errors-new"
        review.mkdir(parents=True)
        (review/"review.json").write_text("{}")
        with patch.object(entry, "verified_review", return_value={"protocol":{"source_run":"../outside"}}):
            with self.assertRaisesRegex(ValueError, "plain reviewed"):
                entry.assessment_inputs(self.root)

    def write_assessment(self, name="final-001", status="completed"):
        protocol = self.root/"models/final-assessment/protocol.json"
        protocol.parent.mkdir(parents=True, exist_ok=True)
        protocol.write_text("{}")
        run = self.root/"runs"/name
        run.mkdir(parents=True)
        (run/"target-free-predictions.json").write_text("{}")
        receipt = {"version":"hourly-final-assessment-v1", "status":status, "test_evaluated":True,
                   "selection_changed":False, "protocol_sha256":digest(protocol.read_bytes()),
                   "artifacts_sha256":{"target-free-predictions.json":digest(b"{}")}}
        (run/"assessment.json").write_bytes(json_bytes(receipt))
        return run

    def test_repeat_verifies_saved_result_without_reopening_targets(self):
        self.write_assessment()
        with patch.object(entry, "assessment_inputs") as inputs, patch.object(entry, "run_final_assessment") as run:
            code, result = self.run_main()
        self.assertEqual(code, 0)
        self.assertTrue(result["saved_evidence_verified"])
        inputs.assert_not_called()
        run.assert_not_called()

    def test_latest_failure_does_not_fall_back_to_an_older_success(self):
        self.write_assessment()
        self.write_assessment("final-002", "failed")
        code, result = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn("latest assessment", result["message"])

    def test_changed_saved_evidence_is_rejected(self):
        run = self.write_assessment()
        (run/"target-free-predictions.json").write_text("changed")
        code, result = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn("evidence changed", result["message"])
