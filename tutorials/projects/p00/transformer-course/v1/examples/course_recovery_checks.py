"""Check full-state continuation, interrupted writes and compatible fallback."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile

import numpy as np
import torch

from course_checkpoint import load_latest, save_generation, _verified
from translation_train import make_state, update


def build(kind):
    if kind == "translation":
        return make_state(seed=7, steps=32)
    from roberta_train import build_training_bundle
    return build_training_bundle(kind, seed=7, total_steps=32)


def step(kind, state):
    if kind == "translation":
        return update(state)
    from roberta_train import training_step
    return training_step(state)


def fields(state):
    identity = state.identity() if callable(state.identity) else state.identity
    generators = ({"mask": state.mask_generator, "batch": state.batch_generator}
                  if hasattr(state, "mask_generator") else {})
    return {"model": state.model, "optimizer": state.optimizer,
            "scheduler": state.scheduler, "identity": identity, "generators": generators}


def random_draws():
    return [random.random(), float(np.random.random()), float(torch.rand(()))]


def save(directory, state):
    return save_generation(directory, step=state.step,
                           data_state={"completed_updates": state.step}, **fields(state))


def restore(directory, state):
    result = load_latest(directory, **fields(state))
    state.step = result["step"]
    assert result["data_state"]["completed_updates"] == state.step
    return result


def compare(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            compare(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            compare(a, b)
    else:
        assert left == right, (left, right)


def resumed_process(kind, directory, output):
    state = build(kind)
    # Setup may consume randomness; the checkpoint must replace it afterward.
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    restore(directory, state)
    draws = random_draws()
    for _ in range(5):
        step(kind, state)
    torch.save({"model": state.model.state_dict(), "optimizer": state.optimizer.state_dict(),
                "scheduler": state.scheduler.state_dict(), "step": state.step,
                "draws": draws}, output)


def fresh_process_parity(kind, root):
    state = build(kind)
    random.seed(101)
    np.random.seed(102)
    for _ in range(3):
        step(kind, state)
    directory = root / kind
    save(directory, state)
    draws = random_draws()
    for _ in range(5):
        step(kind, state)
    output = root / f"{kind}-resumed.pt"
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--resume", kind,
                    "--directory", str(directory), "--output", str(output)], check=True)
    resumed = torch.load(output, weights_only=True)
    compare(state.model.state_dict(), resumed["model"])
    compare(state.optimizer.state_dict(), resumed["optimizer"])
    compare(state.scheduler.state_dict(), resumed["scheduler"])
    assert state.step == resumed["step"] == 8
    assert draws == resumed["draws"]


def failure_and_retention(root):
    state = build("translation")
    directory = root / "failure"
    for _ in range(3):
        step("translation", state)
    older = save(directory, state)
    for _ in range(2):
        step("translation", state)
    newest = save(directory, state)
    with (newest / "state.pt").open("ab") as stream:
        stream.write(b"corrupt")
    partial = directory / "99999999-interrupted"
    partial.mkdir()
    (partial / "state.pt").write_bytes(b"unfinished")
    restored = build("translation")
    result = restore(directory, restored)
    assert result["generation"] == older.name and restored.step == 3
    assert len(result["skipped"]) == 2
    baseline = {name: value.clone() for name, value in restored.model.state_dict().items()}
    incompatible = fields(restored)
    incompatible["identity"] = {"wrong_tokenizer": True}
    try:
        load_latest(directory, **incompatible)
    except ValueError as error:
        assert "compatible" in str(error)
    else:
        raise AssertionError("incompatible tokenizer identity was accepted")
    compare(restored.model.state_dict(), baseline)
    for _ in range(3):
        step("translation", restored)
        save(directory, restored)
    valid = [p for p in directory.iterdir() if p.is_dir() and _verified(p) is not None]
    assert len(valid) == 2


def source_and_schedule_rejection(root):
    state = build("translation")
    step("translation", state)
    directory = root / "identity-boundaries"
    save(directory, state)
    restored = build("translation")
    baseline_model = {name: value.clone() for name, value in restored.model.state_dict().items()}
    baseline_optimizer = deepcopy(restored.optimizer.state_dict())
    for boundary in ("source", "schedule"):
        arguments = fields(restored)
        arguments["identity"] = deepcopy(arguments["identity"])
        if boundary == "source":
            arguments["identity"]["source_sha256"]["translation_train.py"] = "0" * 64
        else:
            arguments["identity"]["protocol"]["scheduler"]["name"] = "different-schedule-v2"
        try:
            load_latest(directory, **arguments)
        except ValueError as error:
            assert "compatible" in str(error)
        else:
            raise AssertionError(f"changed {boundary} identity was accepted")
        compare(restored.model.state_dict(), baseline_model)
        compare(restored.optimizer.state_dict(), baseline_optimizer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", choices=("translation", "mlm", "qa"))
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--project", choices=("translation", "roberta", "all"), default="all")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.resume:
        resumed_process(args.resume, args.directory, args.output)
        return
    with tempfile.TemporaryDirectory(prefix="transformer-recovery-") as temporary:
        root = Path(temporary)
        kinds = ("translation",) if args.project == "translation" else ("mlm", "qa") if args.project == "roberta" else ("translation", "mlm", "qa")
        for kind in kinds:
            fresh_process_parity(kind, root)
        failure_and_retention(root)
        source_and_schedule_rejection(root)
    print(json.dumps({"translation_resume": "translation" in kinds, "mlm_resume": "mlm" in kinds, "qa_resume": "qa" in kinds,
                      "corrupt_fallback": True, "partial_ignored": True,
                      "incompatible_rejected": True, "source_identity_rejected": True,
                      "scheduler_identity_rejected": True, "retained_generations": 2}))


if __name__ == "__main__":
    main()
