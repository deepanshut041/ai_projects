import tempfile
from pathlib import Path
import unittest
import torch
from gpt_data import windows, Batches
from gpt_tokenizer import ByteBPE, BOS, EOS, PAD


class DataTests(unittest.TestCase):
    def test_shift_and_tail(self):
        x, y = windows(["hi"], ByteBPE(), 4)
        self.assertEqual(x.tolist(), [[BOS, 104, 105, PAD]])
        self.assertEqual(y.tolist(), [[104, 105, EOS, PAD]])

    def test_targets_once_and_document_boundaries(self):
        x, y = windows(["abcde", "xy"], ByteBPE(), 3)
        self.assertEqual(y[y != PAD].tolist(), [97,98,99,100,101,EOS,120,121,EOS])
        self.assertEqual(x[-1,0].item(), BOS)

    def test_unicode_frozen_artifact(self):
        tok = ByteBPE.train(["abab", "abab"], 262)
        self.assertEqual(tok.merges, [(97,98),(260,260)])
        for value in ["", "你好🙂", "<BOS>", "e\u0301"]:
            self.assertEqual(tok.decode(tok.encode(value)), value)
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/"tokenizer.json"; tok.save(p)
            self.assertEqual(tok.identity, ByteBPE.load(p).identity)

    def test_sampler_restores_next_draw(self):
        x, y = windows(["abcdef"], ByteBPE(), 2)
        stream = Batches(x, y, 3); stream.next(3); state = stream.state()
        expected = stream.next(9)
        stream.restore(state)
        actual = stream.next(9)
        self.assertTrue(torch.equal(expected[0], actual[0]))
        self.assertEqual(stream.draws, 12)


if __name__ == "__main__":
    unittest.main()
