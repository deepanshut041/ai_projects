"""Predictions are preserved across process boundaries without outcomes or fitting."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bike_demand_planner.artifacts import digest, json_bytes, read_json
from bike_demand_planner.saved_predictor import SavedPredictor
from replay_fixture import rehash, write_bundle


class SavedPredictorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle = self.root/"bundle"
        write_bundle(self.bundle)

    def tearDown(self):
        self.temp.cleanup()

    def test_known_answers_absent_group_and_input_nonmutation(self):
        model = SavedPredictor(self.bundle)
        for time,expected in [("2012-01-09T08:00:00",30.),("2012-01-08T08:00:00",10.),("2012-01-09T02:00:00",20.)]:
            request=model.request_for_target(time);before=copy.deepcopy(request)
            with patch("bike_demand_planner.regression.fit_standardizer",side_effect=AssertionError("No fitting")),patch("bike_demand_planner.evaluation.load_labelled_partition",side_effect=AssertionError("No targets")):
                result=model.predict(request)
            self.assertEqual(result["prediction"],expected)
            self.assertEqual(result["baseline"],20.)
            self.assertFalse(result["target_available"])
            self.assertNotIn("target",result)
            self.assertEqual(before,request)
        self.assertTrue(result["fallback_used"])

    def test_strict_contract_and_recorded_calendar_authority(self):
        model=SavedPredictor(self.bundle);request=model.request_for_target("2012-01-09T08:00:00")
        changed=copy.deepcopy(request);changed["calendar"].update(holiday=1,workingday=0)
        with self.assertRaisesRegex(ValueError,"authority"):model.predict(changed)
        changed=copy.deepcopy(request);changed["target"]=35
        with self.assertRaises(ValueError):model.predict(changed)
        with self.assertRaises(ValueError):model.request_for_target("2012-07-01T08:00:00")
        with self.assertRaises(ValueError):model.request_for_target("2013-01-09T08:00:00")

    def test_corruption_trusted_hash_and_rehashed_schema_change_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"trusted"):SavedPredictor(self.bundle,expected_manifest_sha256="wrong")
        path=self.bundle/"model.json";original=path.read_bytes();path.write_bytes(original+b" ")
        with self.assertRaisesRegex(ValueError,"identity"):SavedPredictor(self.bundle)
        path.write_bytes(original)
        candidate=read_json(self.bundle/"candidate.json");candidate["feature_names"].reverse()
        (self.bundle/"candidate.json").write_bytes(json_bytes(candidate));rehash(self.bundle)
        with self.assertRaisesRegex(ValueError,"contract"):SavedPredictor(self.bundle)

    def test_nonfinite_json_and_duplicate_groups_fail_even_with_updated_file_hashes(self):
        model=read_json(self.bundle/"model.json");model["groups"].append(model["groups"][0])
        (self.bundle/"model.json").write_bytes(json_bytes(model))
        candidate=read_json(self.bundle/"candidate.json");candidate["model_sha256"]=digest((self.bundle/"model.json").read_bytes())
        (self.bundle/"candidate.json").write_bytes(json_bytes(candidate));rehash(self.bundle)
        with self.assertRaisesRegex(ValueError,"duplicate"):SavedPredictor(self.bundle)
        (self.bundle/"calendar.json").write_text('{"value":NaN}');rehash(self.bundle)
        with self.assertRaisesRegex(ValueError,"Nonfinite"):SavedPredictor(self.bundle)

    def test_fresh_process_uses_only_the_copied_bundle_and_target_free_request(self):
        model=SavedPredictor(self.bundle);request=model.request_for_target("2012-01-09T08:00:00")
        source=self.root/"requests.json";source.write_bytes(json_bytes([request]));output=self.root/"prediction.json"
        subprocess.run([sys.executable,"-I","-m","bike_demand_planner.prediction_worker","--bundle",str(self.bundle),
                        "--expected-sha256",model.identity,"--requests",str(source),"--output",str(output)],cwd=self.root,check=True,capture_output=True)
        self.assertEqual(read_json(output),[model.predict(request)])
