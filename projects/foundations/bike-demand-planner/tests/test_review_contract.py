"""Named calendar inputs, temporal domain and non-mutating model sensitivity."""

import copy
import unittest

from bike_demand_planner.baselines import fit_hour_workingday_baseline
from bike_demand_planner.error_review import _request, contract_probes
from bike_demand_planner.features import FEATURE_NAMES
from bike_demand_planner.review_contract import probe_group_response, review_request_features
from dataclasses import asdict


class ReviewContractTests(unittest.TestCase):
    def setUp(self):
        self.model=asdict(fit_hour_workingday_baseline([(8,1),(8,0),(2,1)],[350,100,7.5]))

    def test_valid_calendar_changes_have_known_responses_without_changing_state(self):
        before=copy.deepcopy(self.model)
        monday=probe_group_response(self.model,_request('2012-01-09T08:00:00'))
        june=probe_group_response(self.model,_request('2012-06-04T08:00:00'))
        sunday=probe_group_response(self.model,_request('2012-01-08T08:00:00'))
        self.assertEqual(monday['prediction'],350);self.assertEqual(june['prediction'],350);self.assertEqual(sunday['prediction'],100)
        self.assertEqual(self.model,before);self.assertFalse(monday['target_available'])

    def test_absent_group_uses_training_mean_and_does_not_create_support(self):
        result=probe_group_response(self.model,_request('2012-01-09T09:00:00'))
        self.assertTrue(result['fallback_used']);self.assertAlmostEqual(result['prediction'],(350+100+7.5)/3)
        self.assertEqual(result['group_training_count'],0)

    def test_boundary_hours_and_declared_rejections_are_enforced(self):
        for time in ['2012-01-01T01:00:00','2012-06-30T23:00:00']:
            self.assertEqual(len(review_request_features(_request(time))),45)
        result=contract_probes(self.model);self.assertEqual(len(result['rejected']),14)
        self.assertEqual(len(result['accepted']),4)
        for names in [None,list(reversed(FEATURE_NAMES))]:
            with self.assertRaises(ValueError):review_request_features(_request('2012-01-09T08:00:00'),feature_names=names)
        bad=_request('2012-01-09T08:00:00');bad['target_time']='2012-02-30T08:00:00'
        with self.assertRaises(ValueError):review_request_features(bad)


if __name__=='__main__':unittest.main()
