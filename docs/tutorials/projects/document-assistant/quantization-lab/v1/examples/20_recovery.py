"""Small immutable-generation demo; all writes are inside a temporary directory."""
import hashlib
import json
from pathlib import Path
import tempfile


def publish(root, generation, payload):
    staging = root / (generation + ".partial")
    staging.mkdir()
    (staging / "weights.bin").write_bytes(payload)
    manifest = {"schema": 1, "generation": generation, "file": "weights.bin",
                "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
    (staging / "manifest.json").write_text(json.dumps(manifest))
    staging.rename(root / generation)  # Same filesystem; unique immutable destination.


def load_latest(root):
    rejected = []
    for folder in sorted(root.glob("gen-*"), reverse=True):
        if not folder.is_dir() or folder.name.endswith(".partial"):
            continue
        try:
            manifest = json.loads((folder / "manifest.json").read_text())
            if manifest["schema"] != 1 or manifest["generation"] != folder.name:
                raise ValueError("Manifest identity")
            if manifest["file"] != "weights.bin":
                raise ValueError("Unexpected artifact path")
            raw = (folder / "weights.bin").read_bytes()
            if len(raw) != manifest["size"] or hashlib.sha256(raw).hexdigest() != manifest["sha256"]:
                raise ValueError("Artifact integrity")
            return folder.name, raw, rejected
        except (OSError, ValueError, KeyError, TypeError):
            rejected.append(folder.name)
    raise ValueError("No verified generation available")


def main():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        publish(root, "gen-0001", b"verified toy weights")
        publish(root, "gen-0002", b"new toy weights")
        (root / "gen-0002/weights.bin").write_bytes(b"corrupt")
        (root / "gen-0003.partial").mkdir()
        name, raw, rejected = load_latest(root)
        assert name == "gen-0001" and raw == b"verified toy weights"
        print(json.dumps({"loaded": name, "rejected": rejected, "partial_ignored": True,
                          "scope": "integrity demo; not durable model storage"}))


if __name__ == "__main__":
    main()
