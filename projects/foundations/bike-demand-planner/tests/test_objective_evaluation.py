"""Validation isolation, versioned evidence and declared model/policy selection."""
import json
import tempfile
import unittest
from pathlib import Path

from bike_demand_planner.data import sha256
from bike_demand_planner.objective_evaluation import run_objective_study
from test_evaluation import PROJECT_ROOT,prepare_fixture


class ObjectiveEvaluationTests(unittest.TestCase):
    def test_changed_validation_keeps_all_models_and_pilot_choices_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);models=[];rates=[]
            for name,target in [('first',16),('changed',1000)]:
                processed=prepare_fixture(root/name,target)
                (processed/'test.csv').unlink();(root/name/'hour.csv').unlink()
                directory=root/(name+'-study')
                report=run_objective_study(processed,directory,PROJECT_ROOT,rates=(.1,),pilot_steps=2,main_steps=3,scaling_steps=2)
                self.assertEqual(report['status'],'completed');self.assertFalse(report['test_evaluated'])
                self.assertTrue(report['selection']['validation_used_for_selection'])
                self.assertFalse(report['validation_used_for_fitting'])
                rates.append(report['selected_rates'])
                models.append({key:(directory/rel/'model.json').read_bytes() for key,rel in report['trial_directories'].items()})
                for key,rel in report['trial_directories'].items():
                    receipt=json.loads((directory/rel/'run.json').read_text())
                    for filename,digest in receipt['artifacts_sha256'].items():
                        self.assertEqual(sha256((directory/rel/filename).read_bytes()),digest)
                    raw=receipt['metrics']['validation']['raw']['mae'];clip=receipt['metrics']['validation']['nonnegative']['mae']
                    self.assertLessEqual(clip,raw)
                best=report['selection']['ranking'][0]
                self.assertEqual(best['model'],report['selection']['model'])
                self.assertEqual(best['policy'],report['selection']['policy'])
            self.assertEqual(models[0],models[1]);self.assertEqual(rates[0],rates[1])

    def test_failures_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'failed'
            with self.assertRaises(FileNotFoundError):run_objective_study(root/'missing',directory,PROJECT_ROOT)
            original=(directory/'study.json').read_bytes()
            self.assertEqual(json.loads(original)['status'],'failed')
            with self.assertRaises(FileExistsError):run_objective_study(root/'missing',directory,PROJECT_ROOT)
            self.assertEqual(original,(directory/'study.json').read_bytes())
            with self.assertRaises(ValueError):run_objective_study(root/'missing',root/'invalid',PROJECT_ROOT,rates=(float('nan'),))
            self.assertFalse((root/'invalid').exists())


if __name__=="__main__":unittest.main()
