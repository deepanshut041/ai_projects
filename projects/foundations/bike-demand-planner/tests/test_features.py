"""Independent calendar-column positions and input-boundary checks."""

import unittest

from bike_demand_planner.features import FEATURE_NAMES, calendar_schema, encode_calendar


class FeatureTests(unittest.TestCase):
    def test_monday_at_eight_has_four_known_active_columns(self):
        features = encode_calendar({"hr": 8, "weekday": 1, "mnth": 1, "holiday": 0, "workingday": 1})
        self.assertEqual(len(features), 45)
        self.assertEqual([index for index, value in enumerate(features) if value], [8, 25, 31, 44])
        self.assertEqual([FEATURE_NAMES[index] for index in (8, 25, 31, 44)],
                         ["hr_08", "weekday_01", "mnth_01", "workingday"])
        self.assertEqual(set(features), {0.0, 1.0})

    def test_missing_extra_forbidden_and_invalid_values_are_rejected(self):
        valid = {"hr": 8, "weekday": 1, "mnth": 1, "holiday": 0, "workingday": 1}
        cases = [
            {key: value for key, value in valid.items() if key != "hr"},
            {**valid, "cnt": 154}, {**valid, "temp": 0.5}, {**valid, "yr": 0},
            {**valid, "hr": 24}, {**valid, "weekday": 7}, {**valid, "mnth": 0},
            {**valid, "holiday": 2}, {**valid, "hr": 8.0},
            {**valid, "workingday": True}, {**valid, "workingday": 0},
        ]
        for calendar in cases:
            with self.subTest(calendar=calendar), self.assertRaises(ValueError):
                encode_calendar(calendar)

    def test_predeclared_categories_do_not_need_to_be_observed_or_fitted(self):
        schema = calendar_schema()
        features = encode_calendar({"hr": 23, "weekday": 0, "mnth": 12, "holiday": 0, "workingday": 0})
        self.assertEqual([index for index, value in enumerate(features) if value], [23, 24, 42])
        self.assertEqual(calendar_schema(), schema)
        self.assertFalse(schema["learned_transform"])
        self.assertEqual(schema["feature_count"], 45)
        self.assertNotIn("yr", schema["input_fields"])
        self.assertNotIn("season", schema["input_fields"])


if __name__ == "__main__":
    unittest.main()
