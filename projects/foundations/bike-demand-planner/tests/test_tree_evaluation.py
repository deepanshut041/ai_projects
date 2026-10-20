"""Chronological fit isolation, seed aggregation and retained result integrity."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bike_demand_planner.comparison import summarize_tree_study
from bike_demand_planner.tree_evaluation import run_tree_study
from bike_demand_planner.trees import TreeSpec
from test_evaluation import PROJECT_ROOT, prepare_fixture


SMALL = {"tree": [TreeSpec("tree",2,1)], "forest": [TreeSpec("forest",2,1,2)],
         "boosting": [TreeSpec("boosting",2,1,2)]}


class TreeEvaluationTests(unittest.TestCase):
    def test_validation_changes_leave_every_fitted_model_and_protocol_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); snapshots=[]; protocols=[]
            for name,target in [("first",16),("changed",1000)]:
                processed=prepare_fixture(root/name,target)
                (processed/'test.csv').unlink();(root/name/'hour.csv').unlink()
                directory=root/(name+'-study')
                report=run_tree_study(processed,directory,PROJECT_ROOT,candidates=SMALL,seeds=(17,29),pilot_rows=3)
                self.assertEqual(report['status'],'completed');self.assertFalse(report['validation_used_for_fitting'])
                self.assertFalse(report['test_evaluated']);self.assertTrue(report['all_models_fitted_before_validation'])
                snapshots.append({n:(directory/rel/'model.json').read_bytes() for n,rel in report['trial_directories'].items()})
                protocols.append((directory/'protocol.json').read_bytes())
                self.assertEqual(summarize_tree_study(directory)['trial_count'],10)
                ranking=report['selection']['ranking']
                for row in ranking:
                    scores=[report['results'][n]['metrics']['validation'][row['policy']]['mae'] for n in row['trials']]
                    self.assertEqual(row['mean_validation_mae'],float(np.mean(scores)))
                    self.assertEqual(row['representative_trial'],row['trials'][0])
                    self.assertIn('seed17',row['representative_trial']) if row['configuration'].endswith('-c0') else None
                self.assertEqual(ranking[0]['configuration'],report['selection']['configuration'])
            self.assertEqual(snapshots[0],snapshots[1]);self.assertEqual(protocols[0],protocols[1])

    def test_saved_comparison_refuses_changed_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);processed=prepare_fixture(root/'data',16);directory=root/'study'
            report=run_tree_study(processed,directory,PROJECT_ROOT,candidates=SMALL,seeds=(17,))
            summary=summarize_tree_study(directory);self.assertEqual(summary['trial_count'],7)
            path=directory/'trials/groups/validation-raw-predictions.csv'
            path.write_text(path.read_text()+'altered\n')
            with self.assertRaises(ValueError):summarize_tree_study(directory)

    def test_failures_no_overwrite_and_preflight_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'failed'
            with self.assertRaises(FileNotFoundError):run_tree_study(root/'missing',directory,PROJECT_ROOT,candidates=SMALL)
            original=(directory/'study.json').read_bytes();self.assertEqual(json.loads(original)['status'],'failed')
            with self.assertRaises(FileExistsError):run_tree_study(root/'missing',directory,PROJECT_ROOT,candidates=SMALL)
            self.assertEqual(original,(directory/'study.json').read_bytes())
            with self.assertRaises(ValueError):run_tree_study(root/'missing',root/'invalid',PROJECT_ROOT,seeds=(17,17))
            self.assertFalse((root/'invalid').exists())
            processed=prepare_fixture(root/'data',16)
            with patch('bike_demand_planner.tree_evaluation.fit_tree_model',side_effect=ValueError('retained pilot failure')):
                with self.assertRaises(ValueError):run_tree_study(processed,root/'pilot-failed',PROJECT_ROOT,candidates=SMALL)
            pilot=json.loads((root/'pilot-failed/pilot/tree/pilot.json').read_text())
            self.assertEqual(pilot['status'],'failed');self.assertIn('retained pilot failure',pilot['failure']['message'])


if __name__ == '__main__': unittest.main()
