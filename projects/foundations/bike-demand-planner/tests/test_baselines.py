"""Known constant fits and independent hourly-error calculations."""

import math
import unittest

from bike_demand_planner.baselines import fit_hour_workingday_baseline, fit_mean_baseline
from bike_demand_planner.metrics import regression_metrics


class BaselineTests(unittest.TestCase):
    def test_calendar_fit_keeps_hour_and_working_day_separate(self):
        keys = [(8, 1), (8, 1), (2, 1), (8, 0)]
        targets = [10.0, 30.0, 2.0, 100.0]
        model = fit_hour_workingday_baseline(keys, targets)
        self.assertEqual(model.predict([(8, 1), (2, 1), (8, 0)]), [20, 2, 100])
        self.assertEqual([(group.hour, group.workingday, group.training_count) for group in model.groups],
                         [(2, 1, 1), (8, 0, 1), (8, 1, 2)])
        keys[0], targets[0] = (9, 1), 1000
        self.assertEqual(model.predict([(8, 1)]), [20])

    def test_absent_group_uses_row_weighted_training_mean_without_adding_a_group(self):
        model = fit_hour_workingday_baseline([(8, 1), (8, 1), (2, 1), (8, 0)], [10, 30, 2, 100])
        self.assertEqual(model.fallback.value, 35.5)
        self.assertEqual(model.fallback.training_count, 4)
        self.assertEqual(model.predict([(2, 0), (23, 1)]), [35.5, 35.5])
        self.assertEqual(len(model.groups), 3)
        self.assertEqual(model.predict([]), [])

    def test_group_fit_and_prediction_reject_misalignment_and_invalid_calendar_keys(self):
        with self.assertRaisesRegex(ValueError, "one calendar group"):
            fit_hour_workingday_baseline([(8, 1)], [1, 2])
        for key in ((24, 1), (-1, 0), (8, 2), (8.0, 1), (True, 0), (8, True), (8,), "8,1"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                fit_hour_workingday_baseline([key], [1])
            with self.subTest(predict=key), self.assertRaises(ValueError):
                fit_hour_workingday_baseline([(8, 1)], [1]).predict([key])
        for targets in ([], [-1], [math.nan], [math.inf]):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                fit_hour_workingday_baseline([(8, 1)] * len(targets), targets)

    def test_fit_uses_training_mean_and_prediction_needs_no_targets(self):
        targets = [2.0, 4.0, 12.0]
        model = fit_mean_baseline(targets)
        targets[0] = 1000.0
        self.assertEqual(model.value, 6.0)
        self.assertEqual(model.training_count, 3)
        self.assertEqual(model.predict(2), [6.0, 6.0])
        self.assertEqual(model.predict(0), [])

    def test_invalid_training_targets_and_prediction_sizes_are_rejected(self):
        for targets in ([], [-1], [math.nan], [math.inf]):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                fit_mean_baseline(targets)
        for count in (-1, 1.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                fit_mean_baseline([1]).predict(count)

    def test_metrics_match_hand_calculation_and_explicit_deficit_denominators(self):
        result = regression_metrics([10, 30], [12, 22])
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["mae"], 5)
        self.assertAlmostEqual(result["rmse"], math.sqrt(34))
        self.assertEqual(result["mean_signed_error"], -3)
        self.assertEqual(result["mean_underprediction"], 4)
        self.assertEqual(result["underprediction_count"], 1)
        self.assertEqual(result["mean_underprediction_when_under"], 8)

    def test_signed_cancellation_does_not_hide_absolute_error(self):
        result = regression_metrics([10, 30], [20, 20])
        self.assertEqual(result["mean_signed_error"], 0)
        self.assertEqual(result["mae"], 10)
        self.assertEqual(result["rmse"], 10)

    def test_no_underprediction_is_zero_with_undefined_conditional_mean(self):
        result = regression_metrics([0, 1], [0, 3])
        self.assertEqual(result["mean_underprediction"], 0)
        self.assertEqual(result["underprediction_count"], 0)
        self.assertIsNone(result["mean_underprediction_when_under"])
        self.assertEqual(result["mae"], 1)

    def test_metrics_reject_missing_nonfinite_or_overflowing_values(self):
        for targets, predictions in (([], []), ([1], []), ([math.inf], [1]), ([1], [math.nan]), ([1e308], [-1e308])):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                regression_metrics(targets, predictions)


if __name__ == "__main__":
    unittest.main()
