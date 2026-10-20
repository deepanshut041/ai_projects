"""Independent arithmetic and failure boundaries for the fixed final assessment."""

import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bike_demand_planner.artifacts import digest, json_bytes
from bike_demand_planner.data import RAW_FIELDS
from bike_demand_planner.evaluation import HourIdentity, load_labelled_partition
from bike_demand_planner.features import encode_calendar
from bike_demand_planner.final_evaluation import CalendarHour, join_final_targets, predict_frozen, verify_protocol


class FinalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.features=tuple(encode_calendar({'hr':8,'weekday':1,'mnth':7,'holiday':0,'workingday':1}))
        self.row=CalendarHour(HourIdentity(1,'2012-07-02T08:00:00','2012-07-02T07:00:00'),self.features)
        raw=dict(zip(RAW_FIELDS,['1','2012-07-02','3','1','7','8','0','1','1','1','.5','.5','.5','.1','10','25','35'],strict=True))
        output=io.StringIO();writer=csv.DictWriter(output,fieldnames=RAW_FIELDS,lineterminator='\n');writer.writeheader();writer.writerow(raw)
        self.raw=output.getvalue().encode();(self.root/'raw.csv').write_bytes(self.raw)
        self.saved={'protocol_sha256':'p','identities':[[1,self.row.identity.target_time,self.row.identity.forecast_issue_time]],'predictions':{'groups':[40]}}
        (self.root/'target-free-predictions.json').write_bytes(json_bytes(self.saved))
        self.access={'version':'hourly-final-label-access-v1','protocol_sha256':'p','source_hour_sha256':digest(self.raw),'count':1,
                     'forecasts_saved_before_labels':True,'predictions_sha256':digest((self.root/'target-free-predictions.json').read_bytes())}
        self.save_access()

    def save_access(self): (self.root/'label-access.json').write_bytes(json_bytes(self.access))
    def tearDown(self): self.temp.cleanup()
    def join(self):return join_final_targets(self.root/'raw.csv',[self.row],self.root/'label-access.json','p',digest(self.raw))

    def test_independent_group_lookup_and_missing_group_fallback(self):
        model={'groups':[{'hour':8,'workingday':1,'value':40,'training_count':2}],
               'fallback':{'value':20,'training_count':3}}
        holiday=encode_calendar({'hr':8,'weekday':3,'mnth':7,'holiday':1,'workingday':0})
        result=predict_frozen('groups',model,[self.features,holiday],'raw')
        np.testing.assert_array_equal(result,[40,20]);self.assertEqual(model['groups'][0]['value'],40)

    def test_verified_target_join_preserves_identity_features_and_observation(self):
        rows=self.join();self.assertEqual(rows[0].identity,self.row.identity);self.assertEqual(rows[0].features,self.features)
        self.assertEqual(rows[0].target,35);self.assertEqual(40-rows[0].target,5)

    def test_bad_receipt_rejected_before_raw_file_access(self):
        for field,value in [('protocol_sha256','other'),('count',2),('source_hour_sha256','other'),('forecasts_saved_before_labels',False)]:
            with self.subTest(field=field):
                prior=self.access[field];self.access[field]=value;self.save_access()
                (self.root/'raw.csv').unlink(missing_ok=True)
                with self.assertRaisesRegex(ValueError,'receipt'):self.join()
                self.access[field]=prior

    def test_changed_predictions_rejected_before_missing_raw_is_opened(self):
        (self.root/'target-free-predictions.json').write_text('{}');(self.root/'raw.csv').unlink()
        with self.assertRaisesRegex(ValueError,'forecasts changed'):self.join()

    def test_reordered_prediction_identity_rejected(self):
        self.saved['identities'][0][0]=2;(self.root/'target-free-predictions.json').write_bytes(json_bytes(self.saved))
        self.access['predictions_sha256']=digest((self.root/'target-free-predictions.json').read_bytes());self.save_access()
        with self.assertRaisesRegex(ValueError,'membership'):self.join()

    def test_raw_tamper_rejected(self):
        (self.root/'raw.csv').write_bytes(self.raw+b'\n')
        with self.assertRaisesRegex(ValueError,'source hash'):self.join()

    def test_calendar_mismatch_is_rejected_even_when_ids_match(self):
        wrong=CalendarHour(self.row.identity,tuple(encode_calendar({'hr':8,'weekday':1,'mnth':7,'holiday':1,'workingday':0})))
        with self.assertRaisesRegex(ValueError,'calendar'):join_final_targets(self.root/'raw.csv',[wrong],self.root/'label-access.json','p',digest(self.raw))

    def test_development_loader_still_rejects_test_before_io(self):
        with self.assertRaisesRegex(ValueError,'Only train and validation'):load_labelled_partition(self.root/'missing','test')

    def test_protocol_hash_rejects_before_reading_project(self):
        file=self.root/'protocol.json';file.write_text('{}')
        with self.assertRaisesRegex(ValueError,'protocol hash'):verify_protocol(self.root/'missing',file,'wrong')

    def test_frozen_global_prediction_and_policy_are_explicit(self):
        self.assertEqual(predict_frozen('global',{'value':12,'training_count':3},[self.features],'raw').tolist(),[12])
        with self.assertRaisesRegex(ValueError,'Unknown frozen model'):predict_frozen('other',{},[self.features],'raw')
        with self.assertRaisesRegex(ValueError,'policy'):predict_frozen('global',{'value':12,'training_count':3},[self.features],'rounded')
