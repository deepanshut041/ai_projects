"""End-to-end HTTP separation, unsupported inputs and artifact compatibility."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request,urlopen

from bike_demand_planner.application import ReplayApplication,make_server
from bike_demand_planner.saved_predictor import SavedPredictor
from replay_fixture import write_app_data,write_bundle


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);write_bundle(self.root/"bundle")
        self.predictor=SavedPredictor(self.root/"bundle");write_app_data(self.root/"data",self.predictor)
        self.app=ReplayApplication(self.root/"bundle",self.root/"data")
        self.server=make_server(self.app,0);self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.origin=f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):self.server.shutdown();self.server.server_close();self.thread.join();self.temp.cleanup()

    def get(self,path):
        with urlopen(self.origin+path) as r:return json.load(r)

    def post(self,value,origin=None):
        headers={"Content-Type":"application/json"}
        if origin:headers["Origin"]=origin
        return urlopen(Request(self.origin+"/api/predict",data=json.dumps(value).encode(),headers=headers))

    def test_prediction_and_observation_are_separate_and_missing_is_not_zero(self):
        request=self.get("/api/request?target_time=2012-01-09T08%3A00%3A00")
        with self.post(request) as r:prediction=json.load(r)
        self.assertEqual(prediction["prediction"],30);self.assertNotIn("target",prediction)
        observed=self.get("/api/observation?target_time=2012-01-09T08%3A00%3A00")
        self.assertEqual(observed["target"],35);self.assertEqual(prediction["prediction"]-observed["target"],-5)
        missing=self.get("/api/observation?target_time=2012-01-09T03%3A00%3A00")
        self.assertFalse(missing["available"]);self.assertNotIn("target",missing)

    def test_bad_dates_unknown_fields_and_foreign_origin_are_explicit_errors(self):
        for time in ["2012-07-01T08:00:00","2013-01-01T08:00:00","2012-01-09T08:30:00"]:
            with self.assertRaises(HTTPError) as error:self.get("/api/request?target_time="+time)
            self.assertEqual(error.exception.code,400)
        request=self.predictor.request_for_target("2012-01-09T08:00:00");request["target"]=35
        with self.assertRaises(HTTPError) as error:self.post(request)
        self.assertEqual(error.exception.code,400)
        del request["target"]
        with self.assertRaises(HTTPError) as error:self.post(request,origin="https://example.com")
        self.assertEqual(error.exception.code,403)

    def test_app_identity_and_evidence_mismatch_fail_before_serving(self):
        self.assertFalse(self.get("/api/info")["final_test_evaluated"])
        with urlopen(self.origin+"/") as r:self.assertIn(b"Bike Demand Planner",r.read())
        with self.assertRaises(HTTPError) as error:self.get("/candidate.json")
        self.assertEqual(error.exception.code,404)
        (self.root/"data/replay.json").write_text("{}");
        with self.assertRaisesRegex(ValueError,"identity"):ReplayApplication(self.root/"bundle",self.root/"data")
