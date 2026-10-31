"""Verified immutable generations. Local tests do not prove Drive durability."""
import hashlib
import json
import pickle
from pathlib import Path
import shutil
import tempfile
import uuid
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified(root):
    good, rejected = [], []
    for folder in sorted(Path(root).glob("step-*")):
        try:
            manifest = json.loads((folder / "manifest.json").read_text())
            if (folder / "COMPLETE").read_text() != sha(folder / "manifest.json"):
                raise ValueError("Completion record mismatch")
            file = folder / "state.pt"
            if file.stat().st_size != manifest["bytes"] or sha(file) != manifest["sha256"]:
                raise ValueError("State checksum mismatch")
            state = torch.load(file, map_location="cpu", weights_only=True)
            if manifest["schema"] != 1 or state["schema"] != 1 or state["step"] != manifest["step"] or state["identity"] != manifest["identity"]:
                raise ValueError("State/manifest mismatch")
            good.append((state["step"], folder, state))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
            rejected.append({"generation": folder.name, "reason": str(error)})
    return sorted(good, key=lambda entry: (entry[0], entry[1].name)), rejected


def recover(root, identity):
    good, rejected = verified(root)
    if not good:
        raise ValueError("No verified checkpoint; resume cannot become a new run")
    step, folder, state = good[-1]
    if state["identity"] != identity:
        raise ValueError("Incompatible config/source/data/tokenizer/runtime; use an explicit new run")
    return state, {"generation": folder.name, "restored_step": step, "rejected": rejected}


def publish(root, state, budget_bytes=64*1024*1024):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        local = Path(scratch) / "state.pt"
        torch.save(state, local)
        check = torch.load(local, map_location="cpu", weights_only=True)
        if check["step"] != state["step"]:
            raise ValueError("Local serialization failed")
        size = local.stat().st_size
        used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        if used + size + 16384 > budget_bytes or shutil.disk_usage(root).free < 2*size+16384:
            raise OSError("Checkpoint storage budget exhausted; preserve old states and stop")
        folder = root / f'step-{state["step"]:08d}-{uuid.uuid4().hex[:8]}'
        folder.mkdir()
        shutil.copyfile(local, folder / "state.pt")
        if sha(local) != sha(folder / "state.pt"):
            raise OSError("Persistent copy did not verify")
        manifest = {"schema": 1, "step": state["step"], "bytes": size,
                    "sha256": sha(local), "identity": state["identity"]}
        (folder / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
        # Written last; never depend on mounted-storage rename semantics.
        (folder / "COMPLETE").write_text(sha(folder / "manifest.json"))
    good, rejected = verified(root)
    if not any(path == folder for _, path, _ in good):
        raise OSError("New persistent generation is not loadable")
    keep = {entry[1] for entry in good[-2:]}
    best = [entry for entry in good if entry[2]["best_step"] == entry[0]]
    if best:
        keep.add(min(best, key=lambda entry: entry[2]["best_nll"])[1])
    for _, old, _ in good:
        if old not in keep:
            shutil.rmtree(old)
    return {"generation": folder.name, "step": state["step"], "state_bytes": size,
            "retained": sorted(path.name for path in keep), "rejected": rejected}
