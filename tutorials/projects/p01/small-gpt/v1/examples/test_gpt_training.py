import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from gpt_config import Config
from gpt_checkpoint import publish, recover, verified, sha
from gpt_data import windows
from gpt_objective import lr_at, evaluate
from gpt_sample import choose, generate
from gpt_tokenizer import ByteBPE, digest, EOS
from gpt_trainer import Trainer, TrainConfig


def fixture(dropout=0.1):
    splits={"train":["red cat", "blue dog", "red dog", "blue cat"],
            "validation":["small cat"], "test":["small dog"]}
    tok=ByteBPE.train(splits["train"],260)
    manifest={"split_hashes":{k:digest(v) for k,v in splits.items()}}
    return Trainer(Config(width=16,heads=4,hidden=32,context=8,dropout=dropout),
                   TrainConfig(total=20,warmup=2,batch=2),splits,tok,manifest),tok


class TrainingTests(unittest.TestCase):
    def test_schedule_is_global(self):
        self.assertEqual(lr_at(1,20,2,.001,.0001),.0005)
        self.assertEqual(lr_at(2,20,2,.001,.0001),.001)
        self.assertEqual(lr_at(20,20,2,.001,.0001),.0001)

    def test_resume_dropout_optimizer_rng_and_next_batch(self):
        a,_=fixture(); [a.update() for _ in range(4)]; state=copy.deepcopy(a.state())
        expected=[a.update() for _ in range(3)]
        b,_=fixture(); b.restore(state); actual=[b.update() for _ in range(3)]
        self.assertEqual(expected,actual)
        for name,value in a.model.state_dict().items():
            torch.testing.assert_close(value,b.model.state_dict()[name],rtol=0,atol=0)
        self.assertEqual(a.batches.draws,b.batches.draws)

    def test_validation_does_not_consume_training_randomness(self):
        a,_=fixture(); before=torch.get_rng_state().clone(); batch=a.batches.state()
        a.validation()
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertTrue(torch.equal(batch["rng"],a.batches.state()["rng"]))
        self.assertTrue(a.model.training)

    def test_checkpoint_roundtrip_corrupt_fallback_and_retention(self):
        a,_=fixture()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for _ in range(4): a.update(); a.validation(); publish(root,a.state())
            good,_=verified(root); self.assertGreaterEqual(len(good),2); self.assertLessEqual(len(good),3)
            newest=good[-1][1]; (newest/"state.pt").write_bytes(b"corrupt")
            state,report=recover(root,a.identity)
            self.assertLess(state["step"],4); self.assertTrue(report["rejected"])
            with self.assertRaises(ValueError): recover(root,{"different":"identity"})

    def test_unloadable_and_wrong_schema_generations_fall_back(self):
        a,_=fixture()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); a.validation(); publish(root,a.state()); a.update()
            publish(root,a.state()); good,_=verified(root); newest=good[-1][1]
            (newest/"state.pt").write_bytes(b"not a torch state")
            manifest=json.loads((newest/"manifest.json").read_text())
            manifest.update(bytes=(newest/"state.pt").stat().st_size,sha256=sha(newest/"state.pt"))
            (newest/"manifest.json").write_text(json.dumps(manifest))
            (newest/"COMPLETE").write_text(sha(newest/"manifest.json"))
            self.assertEqual(recover(root,a.identity)[0]["step"],0)
            publish(root,a.state()); good,_=verified(root); newest=good[-1][1]
            manifest=json.loads((newest/"manifest.json").read_text()); manifest["schema"]=9
            (newest/"manifest.json").write_text(json.dumps(manifest))
            (newest/"COMPLETE").write_text(sha(newest/"manifest.json"))
            self.assertEqual(recover(root,a.identity)[0]["step"],0)

    def test_interrupted_copy_and_storage_failure_preserve_previous(self):
        a,_=fixture()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); a.validation(); publish(root,a.state()); a.update()
            def partial(source,destination):
                Path(destination).write_bytes(b"partial"); raise OSError("simulated copy failure")
            with patch("gpt_checkpoint.shutil.copyfile",side_effect=partial):
                with self.assertRaises(OSError): publish(root,a.state())
            with self.assertRaises(OSError): publish(root,a.state(),budget_bytes=1)
            state,report=recover(root,a.identity)
            self.assertEqual(state["step"],0); self.assertTrue(report["rejected"])

    def test_no_checkpoint_and_changed_identity_fail(self):
        a,_=fixture(); state=copy.deepcopy(a.state()); state["identity"]["tokenizer"]="wrong"
        with self.assertRaises(ValueError): a.restore(state)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError): recover(temp,a.identity)

    def test_sampling_seed_bound_and_forced_eos(self):
        a,tok=fixture(); m=a.model
        first=generate(m,tok,"red",10,0.8,5,17)
        self.assertEqual(first,generate(m,tok,"red",10,0.8,5,17)); self.assertTrue(m.training)
        self.assertLessEqual(len(first["generated_ids"]),10)
        scores=torch.zeros(tok.size); scores[EOS]=100
        self.assertEqual(choose(scores,0,None,torch.Generator()),EOS)
        class AlwaysEOS(torch.nn.Module):
            config=m.config
            def __init__(self): super().__init__(); self.weight=torch.nn.Parameter(torch.zeros(1))
            def forward(self,ids): return scores.expand(1,ids.shape[1],-1)
        self.assertEqual(generate(AlwaysEOS(),tok,"",10,0,None)["generated_ids"],[])
        with self.assertRaises(ValueError): choose(scores,-1,4,torch.Generator())
        with self.assertRaises(ValueError): choose(scores,1,0,torch.Generator())

    def test_evaluation_token_weighting(self):
        a,_=fixture(dropout=0); x,y=a.valid
        first=evaluate(a.model,x,y,batch=1)
        second=evaluate(a.model,x,y,batch=50)
        self.assertAlmostEqual(first["nll"],second["nll"],places=5)
        self.assertEqual(first["tokens"],second["tokens"])


if __name__ == "__main__":
    unittest.main()
