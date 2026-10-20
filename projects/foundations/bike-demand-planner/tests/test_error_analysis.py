"""Slice arithmetic, empty populations, time gaps and aligned saved predictions."""

import csv
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from bike_demand_planner.error_analysis import (calendar_level_comparison, error_summary, largest_misses,
                                               read_aligned_predictions, slice_diagnostics, temporal_diagnostics)
from bike_demand_planner.evaluation import Forecast, HourIdentity, LabelledHour, _write_predictions
from bike_demand_planner.features import encode_calendar


def row(time, target, record_id=1):
    d=datetime.fromisoformat(time);day=(d.weekday()+1)%7
    features=tuple(encode_calendar(dict(hr=d.hour,weekday=day,mnth=d.month,holiday=0,workingday=int(day in range(1,6)))))
    return LabelledHour(HourIdentity(record_id,time,(d-timedelta(hours=1)).isoformat()),features,target)


class ErrorAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.rows=[row('2012-01-09T02:00:00',5,1),row('2012-01-09T08:00:00',420,2),row('2012-01-09T13:00:00',20,3)]
        self.predictions=np.asarray([7.5,349.5,20.])

    def test_signed_absolute_and_deficit_denominators_have_independent_answers(self):
        result=error_summary(self.rows,self.predictions);m=result['metrics']
        self.assertAlmostEqual(m['mae'],73/3);self.assertAlmostEqual(m['mean_signed_error'],-68/3)
        self.assertAlmostEqual(m['mean_underprediction'],70.5/3);self.assertEqual(m['mean_underprediction_when_under'],70.5)
        self.assertEqual(m['underprediction_count'],1);self.assertEqual(result['overprediction_count'],1)
        self.assertEqual(result['mean_overprediction_when_over'],2.5)
        self.assertEqual(result['absolute_error_sum'],result['deficit_sum']+result['excess_sum'])

    def test_slice_mass_reconstructs_overall_but_unweighted_means_do_not(self):
        groups=slice_diagnostics(self.rows,self.predictions)['hour_window'];overall=error_summary(self.rows,self.predictions)
        self.assertEqual(groups['rush']['count'],1);self.assertEqual(groups['other']['count'],2)
        weighted=sum(v['count']*v['metrics']['mae'] for v in groups.values())/3
        self.assertEqual(weighted,overall['metrics']['mae'])
        self.assertNotEqual(sum(v['metrics']['mae'] for v in groups.values())/2,weighted)

    def test_empty_slices_are_undefined_rather_than_zero_error(self):
        groups=slice_diagnostics(self.rows,self.predictions)
        self.assertEqual(groups['season']['autumn']['count'],0);self.assertIsNone(groups['season']['autumn']['metrics'])
        self.assertEqual(groups['month']['1']['observed_months'],[1]);self.assertIsNone(error_summary([],[])['mean_target'])

    def test_largest_misses_use_absolute_error_and_deterministic_time_ties(self):
        result=largest_misses(self.rows,self.predictions,2)
        self.assertEqual([r['record_id'] for r in result],[2,1]);self.assertEqual(result[0]['residual'],-70.5)
        tied=largest_misses(self.rows,[7,422,22],3);self.assertEqual([r['record_id'] for r in tied],[1,2,3])

    def test_temporal_pairs_exclude_missing_hours_and_keep_partial_week_counts(self):
        rows=[row('2012-01-01T01:00:00',10,1),row('2012-01-01T02:00:00',10,2),
              row('2012-01-01T04:00:00',10,3),row('2012-01-01T05:00:00',10,4)]
        result=temporal_diagnostics(rows,[11,12,13,14])
        self.assertEqual(result['consecutive_hour_pairs'],2);self.assertAlmostEqual(result['consecutive_hour_residual_correlation'],1)
        self.assertEqual(result['gaps'][0]['unrecorded_slots_between'],1)
        self.assertEqual(result['week'][0]['start_date'],'2011-12-26');self.assertEqual(result['week'][0]['count'],4)
        with self.assertRaises(ValueError):temporal_diagnostics(rows[::-1],[1,2,3,4])

    def test_calendar_level_comparison_reweights_cells_and_counts_unmatched_rows(self):
        train=[row('2011-01-03T08:00:00',10,1),row('2011-01-04T08:00:00',30,2),row('2011-02-01T08:00:00',100,3)]
        valid=[row('2012-01-09T08:00:00',40,4),row('2012-01-10T08:00:00',60,5),row('2012-02-06T08:00:00',150,6),row('2012-03-05T08:00:00',200,7)]
        result=calendar_level_comparison(train,valid);shared=result['groups']['all_shared']
        self.assertEqual(result['shared_validation_count'],3);self.assertEqual(result['unmatched_validation_count'],1)
        self.assertAlmostEqual(shared['historical_cell_mean_reweighted'],140/3)
        self.assertAlmostEqual(shared['validation_target_mean'],250/3)
        self.assertAlmostEqual(shared['observed_level_difference'],110/3)
        self.assertIsNone(result['groups']['3']['observed_level_difference'])

    def test_saved_predictions_reject_reordering_changed_targets_and_error_arithmetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'predictions.csv'
            forecasts=[Forecast(r.identity,float(p)) for r,p in zip(self.rows,self.predictions)]
            _write_predictions(path,self.rows,forecasts);original=path.read_text()
            np.testing.assert_array_equal(read_aligned_predictions(path,self.rows),self.predictions)
            lines=original.splitlines();path.write_text('\n'.join([lines[0],*lines[:0:-1]])+'\n')
            with self.assertRaises(ValueError):read_aligned_predictions(path,self.rows)
            path.write_text(original.replace(',5,7.5,',',6,7.5,',1))
            with self.assertRaises(ValueError):read_aligned_predictions(path,self.rows)
            path.write_text(original.replace(',2.5,2.5,6.25,',',2.5,3.5,6.25,',1))
            with self.assertRaises(ValueError):read_aligned_predictions(path,self.rows)


if __name__=='__main__':unittest.main()
