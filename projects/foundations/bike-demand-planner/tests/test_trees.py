"""Independent leaf arithmetic, ensemble rules and fitted-state contracts."""

import copy
import json
import unittest

import numpy as np

from bike_demand_planner.features import encode_calendar
from bike_demand_planner.trees import TreeSpec, fit_tree_model, predict_snapshot, trace_snapshot


def calendar(hour, day=1):
    return encode_calendar({"hr": hour, "weekday": day, "mnth": 1, "holiday": 0, "workingday": 1})


class TreeTests(unittest.TestCase):
    def setUp(self):
        self.X = np.asarray([calendar(2), calendar(8), calendar(2), calendar(8)])
        self.y = np.asarray([1., 179., 2., 195.])

    def test_leaf_means_split_and_changed_hour_match_independent_arithmetic(self):
        model, receipt = fit_tree_model(self.X, self.y, TreeSpec("tree", 1, 1))
        np.testing.assert_array_equal(predict_snapshot(model, [calendar(2), calendar(8)]), [1.5,187.])
        path = trace_snapshot(model, calendar(8))
        self.assertEqual(path["leaf_value"], (179+195)/2)
        self.assertEqual(path["distinct_training_samples"], 2)
        self.assertEqual(len(path["steps"]), 1)
        self.assertEqual(receipt["total_nodes"], 3)
        self.assertEqual(receipt["training_replay_maximum_difference"], 0)

    def test_minimum_leaf_support_prevents_the_four_row_split(self):
        model,_ = fit_tree_model(self.X, self.y, TreeSpec("tree", 3, 3))
        np.testing.assert_array_equal(predict_snapshot(model, self.X), [94.25]*4)

    def test_forest_is_the_mean_of_its_actual_component_leaves(self):
        model,_ = fit_tree_model(self.X, self.y, TreeSpec("forest", 3, 1, 7))
        request = calendar(8)
        components = [trace_snapshot(model, request, tree_index=i)["leaf_value"] for i in range(7)]
        self.assertAlmostEqual(predict_snapshot(model, [request])[0], sum(components)/7)
        self.assertTrue(all(t["weighted_samples"][0] == 4 for t in model["trees"]))
        self.assertTrue(any(t["samples"][0] < 4 for t in model["trees"]))
        self.assertGreaterEqual(min(components), min(self.y)); self.assertLessEqual(max(components),max(self.y))

    def test_boosting_first_residual_tree_and_shrinkage_have_known_answers(self):
        model,_ = fit_tree_model(self.X, self.y, TreeSpec("boosting", 1, 1, 1, .1))
        self.assertEqual(model["initial_prediction"], 94.25)
        self.assertEqual(trace_snapshot(model,calendar(8))["leaf_value"], 92.75)
        self.assertEqual(trace_snapshot(model,calendar(2))["leaf_value"], -92.75)
        np.testing.assert_allclose(predict_snapshot(model,[calendar(2),calendar(8)]), [84.975,103.525],rtol=0,atol=1e-12)

    def test_multiple_boosting_stages_use_the_running_prediction(self):
        model,_ = fit_tree_model(self.X, self.y, TreeSpec("boosting", 1, 1, 2, .1))
        first = 94.25 + .1*92.75
        second_residual_mean = ((179-first)+(195-first))/2
        self.assertAlmostEqual(trace_snapshot(model,calendar(8),tree_index=1)["leaf_value"], second_residual_mean)
        self.assertAlmostEqual(predict_snapshot(model,[calendar(8)])[0],first+.1*second_residual_mean)

    def test_json_roundtrip_and_prediction_batch_leave_fitted_state_unchanged(self):
        model,_ = fit_tree_model(self.X,self.y,TreeSpec("forest",3,1,3))
        before=json.dumps(model,sort_keys=True)
        restored=json.loads(before)
        alone=predict_snapshot(restored,[calendar(8)])[0]
        together=predict_snapshot(restored,[calendar(2),calendar(8),calendar(22)])[1]
        self.assertEqual(alone,together); self.assertEqual(before,json.dumps(restored,sort_keys=True))

    def test_same_seed_replays_identical_fitted_state(self):
        spec=TreeSpec("forest",3,1,5)
        first,_=fit_tree_model(self.X,self.y,spec,seed=29)
        second,_=fit_tree_model(self.X,self.y,spec,seed=29)
        self.assertEqual(first,second)

    def test_invalid_calendar_targets_settings_and_cycles_are_rejected(self):
        for kwargs in [dict(family="unknown"),dict(family="tree",n_estimators=2),dict(family="forest",n_estimators=65),
                       dict(family="boosting",max_depth=4),dict(family="tree",min_samples_leaf=0),dict(family="tree",learning_rate=float('nan'))]:
            with self.assertRaises(ValueError): TreeSpec(**kwargs)
        for bad in [self.y[:3],np.asarray([1,2,3,float('nan')]),np.asarray([1,2,3,-1])]:
            with self.assertRaises(ValueError):fit_tree_model(self.X,bad,TreeSpec("tree"))
        broken=self.X.copy();broken[0,0]=1
        with self.assertRaises(ValueError):fit_tree_model(broken,self.y,TreeSpec("tree"))
        model,_=fit_tree_model(self.X,self.y,TreeSpec("tree",1,1))
        cyclic=copy.deepcopy(model);cyclic['trees'][0]['children_left'][0]=0
        with self.assertRaises(ValueError):predict_snapshot(cyclic,self.X)
        reordered=copy.deepcopy(model);reordered['feature_names'].reverse()
        with self.assertRaises(ValueError):predict_snapshot(reordered,self.X)


if __name__ == "__main__": unittest.main()
