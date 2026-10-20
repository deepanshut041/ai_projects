"""State, publication, fallback and context checks for complete training saves."""

import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bike_demand_planner.artifacts import publish_directory
from bike_demand_planner.checkpoints import load_generation,save_generation
from bike_demand_planner.features import FEATURE_NAMES
from bike_demand_planner.optimization import TrialConfig,fit_trial
from bike_demand_planner.optimizers import OptimizerConfig
from bike_demand_planner.recovery import payload_for
from bike_demand_planner.regression import fit_standardizer


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        X=np.zeros((65,45));X[:,8]=np.arange(65)%2;y=np.arange(65,dtype=float)
        self.scaler=fit_standardizer(X);self.Z=self.scaler.transform(X);self.y=y
        self.config=TrialConfig(optimizer=OptimizerConfig("adam"),learning_rate=.01,batch_size=32,max_examples=195,decay_examples=65)
        self.context=json.loads(json.dumps({"version":"hourly-training-context-v1","feature_names":list(FEATURE_NAMES),"configuration":asdict(self.config),"standardizer":asdict(self.scaler),"training_count":65,"source":"fixture","training_sha256":"fixture"}))
        self.first=payload_for(fit_trial(self.Z,y,self.config,pause_after_updates=1))

    def tearDown(self):self.temp.cleanup()

    def test_round_trip_retains_optimizer_order_schedule_transform_and_monitoring(self):
        save_generation(self.root/"saves",self.first,self.context)
        restored,receipt=load_generation(self.root/"saves",self.context)
        self.assertEqual(restored,self.first);self.assertFalse(receipt["fallback_used"])
        full=fit_trial(self.Z,self.y,self.config)
        resumed=fit_trial(self.Z,self.y,self.config,continuation=restored["trainer_state"])
        np.testing.assert_array_equal(full.parameters,resumed.parameters)
        for key in ["optimizer","batch_stream","schedule"]:self.assertEqual(full.state[key],resumed.state[key])

    def test_newest_corrupt_and_unpublished_save_fall_back_without_overwrite(self):
        saves=self.root/"saves";first=save_generation(saves,self.first,self.context)
        before=(first/"state.json").read_bytes()
        second=save_generation(saves,self.first,self.context);(second/"state.json").write_text("truncated")
        pending=saves/".generation-000003.pending";pending.mkdir();(pending/"state.json").write_text("partial")
        restored,receipt=load_generation(saves,self.context)
        self.assertEqual(restored,self.first);self.assertEqual(receipt["loaded_generation"],"generation-000001")
        self.assertTrue(receipt["fallback_used"]);self.assertEqual(len(receipt["rejected_generations"]),1)
        self.assertEqual(receipt["ignored_incomplete"],[pending.name]);self.assertEqual((first/"state.json").read_bytes(),before)

    def test_source_data_or_schema_context_changes_are_not_silently_resumed(self):
        saves=self.root/"saves";save_generation(saves,self.first,self.context)
        for key in ["source","training_sha256","feature_names"]:
            changed=copy.deepcopy(self.context);changed[key]="changed"
            with self.assertRaisesRegex(ValueError,"No compatible"):load_generation(saves,changed)

    def test_inconsistent_step_cursor_schedule_and_failed_state_are_rejected(self):
        for edit in [lambda p:p["trainer_state"]["optimizer"].update(step=8),lambda p:p["trainer_state"]["schedule"].update(next_learning_rate=99),lambda p:p["trainer_state"].update(resumable=False)]:
            changed=copy.deepcopy(self.first);edit(changed)
            with self.assertRaises(ValueError):save_generation(self.root/"saves",changed,self.context)

    def test_failed_publication_never_becomes_a_loadable_generation(self):
        with patch("bike_demand_planner.artifacts.os.rename",side_effect=OSError("simulated interrupted publication")):
            with self.assertRaises(OSError):save_generation(self.root/"saves",self.first,self.context)
        self.assertFalse((self.root/"saves/generation-000001").exists())
        with self.assertRaises(ValueError):load_generation(self.root/"saves",self.context)
        destination=self.root/"published";publish_directory(destination,{"record.json":b"{}"})
        with self.assertRaises(FileExistsError):publish_directory(destination,{"record.json":b"changed"})
