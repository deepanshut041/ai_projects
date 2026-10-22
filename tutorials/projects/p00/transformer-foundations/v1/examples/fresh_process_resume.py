"""Verify same-environment full-state CPU parity across a fresh process."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from tiny_transformer import Config, Trainer, publish, verify_generation


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        full = Trainer(Config(dropout=0.1))
        full_losses = [full.update() for _ in range(8)]
        expected = verify_generation(publish(full, root / "full"))
        split = Trainer(Config(dropout=0.1))
        prefix_losses = [split.update() for _ in range(3)]
        publish(split, root / "split")
        # Trusted fixed verification program; dynamic inputs are argv entries.
        code = """
import json,sys,torch
from tiny_transformer import Config,Trainer,publish,recover
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
t=Trainer(Config(dropout=0.1))
recover(t,sys.argv[1])
losses=[t.update() for _ in range(5)]
p=publish(t,sys.argv[1])
print(json.dumps({'generation':str(p),'losses':losses}))
"""
        run = subprocess.run([sys.executable, "-c", code, str(root / "split")],
                             cwd=Path(__file__).parent, check=True, capture_output=True,
                             text=True, timeout=30)
        result = json.loads(run.stdout)
        actual = verify_generation(Path(result["generation"]))
        equal(expected, actual)
        assert full_losses == prefix_losses + result["losses"]
        print(json.dumps({"fresh_process_resume": True, "updates": 8, "split_at": 3,
                          "dropout": 0.1, "full_state_bitwise_equal": True,
                          "loss_sequence_equal": True, "runtime": torch.__version__}, indent=2))


if __name__ == "__main__":
    main()
