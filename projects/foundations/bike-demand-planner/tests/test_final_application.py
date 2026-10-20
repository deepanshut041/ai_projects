"""Final evidence is identified separately from the historical request domain."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from bike_demand_planner.artifacts import digest, json_bytes
from bike_demand_planner.final_application import FinalApplication, make_final_server
from bike_demand_planner.saved_predictor import SavedPredictor
from replay_fixture import write_app_data, write_bundle


class FinalApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        write_bundle(self.root/'bundle');p=SavedPredictor(self.root/'bundle');write_app_data(self.root/'data',p)
        self.choice={'model_sha256':p.candidate['model_sha256'],'policy':'raw','validation':{},'fit':{},'model_json_bytes':10}
        protocol={'choices':{'groups':self.choice}};(self.root/'protocol.json').write_bytes(json_bytes(protocol))
        (self.root/'predictions.json').write_text('{}')
        self.report={'version':'hourly-final-assessment-v1','status':'completed','test_evaluated':True,'selection_changed':False,
                     'selected':'groups','protocol_sha256':digest((self.root/'protocol.json').read_bytes()),'bundle_sha256':p.identity,
                     'artifacts_sha256':{'predictions.json':digest(b'{}')},'coverage':{'count':1},'examples':[],
                     'source_provenance':{},'cost_scope':'local only','interpretation':'fixed',
                     'results':{'groups':{**self.choice,'test':{'metrics':{'mae':5}},'slices':{'workingday_rush':{'yes':{'metrics':{}}},'month':{}},'predict_batch_seconds':.1}}}
        self.save()
    def save(self): (self.root/'assessment.json').write_bytes(json_bytes(self.report))
    def app(self):return FinalApplication(self.root/'bundle',self.root/'data',self.root/'assessment.json',self.root/'protocol.json')
    def tearDown(self):self.temp.cleanup()

    def test_assessment_http_and_target_free_validation_prediction(self):
        app=self.app();server=make_final_server(app,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            origin=f'http://127.0.0.1:{server.server_port}'
            with urlopen(origin+'/api/info') as r:self.assertTrue(json.load(r)['final_test_evaluated'])
            with urlopen(origin+'/api/assessment') as r:self.assertEqual(json.load(r)['rows'][0]['test']['mae'],5)
            with urlopen(origin+'/') as r:self.assertIn(b'fixed forecast',r.read())
            request=app.predictor.request_for_target('2012-01-09T08:00:00');self.assertNotIn('target',app.predictor.predict(request))
            with self.assertRaises(HTTPError):urlopen(origin+'/api/request?target_time=2012-07-02T08:00:00')
        finally:server.shutdown();server.server_close();thread.join()

    def test_protocol_or_selected_identity_mismatch_rejected(self):
        for field,value in [('selected','forest'),('protocol_sha256','other'),('selection_changed',True)]:
            with self.subTest(field=field):
                prior=self.report[field];self.report[field]=value;self.save()
                with self.assertRaisesRegex(ValueError,'identities'):self.app()
                self.report[field]=prior

    def test_changed_assessment_evidence_rejected_before_serving(self):
        (self.root/'predictions.json').write_text('changed')
        with self.assertRaisesRegex(ValueError,'evidence changed'):self.app()
