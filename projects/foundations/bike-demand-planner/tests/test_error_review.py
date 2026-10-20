"""A complete saved-result review without refitting or opening reserved data."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bike_demand_planner.data import sha256
from bike_demand_planner.error_review import run_error_review
from bike_demand_planner.splits import prepare_calendar_dataset
from bike_demand_planner.tree_evaluation import run_tree_study
from bike_demand_planner.trees import TreeSpec
from test_evaluation import PROJECT_ROOT
from test_splits import source_rows,write_source


def source_fixture(root):
    root.mkdir();rows=source_rows()
    for index,target in [(0,10),(1,30),(3,20),(4,30)]:
        rows[index].update(casual='0',registered=str(target),cnt=str(target))
    identity=write_source(root/'hour.csv',rows)
    processed=root/'processed';prepare_calendar_dataset(root/'hour.csv',processed,identity)
    candidates={'tree':[TreeSpec('tree',2,100)],'forest':[TreeSpec('forest',2,100,2)],'boosting':[TreeSpec('boosting',2,100,2)]}
    source=root/'source-study';report=run_tree_study(processed,source,PROJECT_ROOT,candidates=candidates,seeds=(17,))
    assert report['selection']['configuration']=='groups'
    (processed/'test.csv').unlink();(root/'hour.csv').unlink()
    return processed,source


class ErrorReviewTests(unittest.TestCase):
    def test_review_preserves_source_models_and_freezes_their_exact_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);processed,source=source_fixture(root/'data')
            original={str(f.relative_to(source)):sha256(f.read_bytes()) for f in source.rglob('*') if f.is_file()}
            with patch('sklearn.tree.DecisionTreeRegressor.fit',side_effect=AssertionError('No review refits')):
                report=run_error_review(processed,source,root/'review',PROJECT_ROOT,make_plots=False)
            self.assertEqual(report['status'],'completed');self.assertFalse(report['models_refitted']);self.assertFalse(report['test_evaluated'])
            self.assertEqual(report['selected_candidate'],'groups/raw');self.assertEqual(len(report['results']),7)
            self.assertEqual(report['results']['groups']['validation']['summary']['metrics']['mae'],0)
            self.assertEqual(original,{str(f.relative_to(source)):sha256(f.read_bytes()) for f in source.rglob('*') if f.is_file()})
            candidate=json.loads((root/'review/candidate.json').read_text())
            self.assertFalse(candidate['serving_implemented']);self.assertFalse(candidate['recovery_verified'])
            self.assertEqual((root/'review/candidate-model.json').read_bytes(),(source/'trials/groups/model.json').read_bytes())
            for name,digest in report['artifacts_sha256'].items():self.assertEqual(sha256((root/'review'/name).read_bytes()),digest)
            self.assertEqual(report['candidate_sha256'],sha256((root/'review/candidate.json').read_bytes()))

    def test_reordered_predictions_are_rejected_even_after_their_checksum_is_updated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);processed,source=source_fixture(root/'data');directory=source/'trials/groups'
            path=directory/'validation-raw-predictions.csv';lines=path.read_text().splitlines()
            path.write_text('\n'.join([lines[0],*reversed(lines[1:])])+'\n')
            record=json.loads((directory/'run.json').read_text());record['artifacts_sha256'][path.name]=sha256(path.read_bytes())
            (directory/'run.json').write_text(json.dumps(record,indent=2)+'\n')
            with self.assertRaisesRegex(ValueError,'identities, order or targets'):
                run_error_review(processed,source,root/'review',PROJECT_ROOT,make_plots=False)
            failed=json.loads((root/'review/review.json').read_text());self.assertEqual(failed['status'],'failed')

    def test_failures_are_retained_and_existing_review_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'review'
            with self.assertRaises(FileNotFoundError):run_error_review(root/'missing',root/'missing-source',directory,PROJECT_ROOT,make_plots=False)
            original=(directory/'review.json').read_bytes();self.assertEqual(json.loads(original)['status'],'failed')
            with self.assertRaises(FileExistsError):run_error_review(root/'missing',root/'missing-source',directory,PROJECT_ROOT,make_plots=False)
            self.assertEqual((directory/'review.json').read_bytes(),original)


if __name__=='__main__':unittest.main()
