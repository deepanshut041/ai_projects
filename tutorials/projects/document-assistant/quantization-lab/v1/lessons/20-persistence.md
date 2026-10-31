## 20. Save a candidate that can be verified after restart

A successful in-memory answer is not a reloadable application. Files can be incomplete, a configuration can point to the wrong tokenizer, and a “latest” directory can contain a corrupt artifact. Persistence needs a verifiable generation boundary rather than a promise that the last write probably finished.

### Separate identity, integrity and compatibility

Identity says which source, recipe, tokenizer, template and runtime produced the artifact. Integrity checks whether the saved bytes match the recorded digest/size. Compatibility checks whether a particular runtime can interpret them as intended. A hash match proves bytes agree with the manifest; it does not prove the model answers correctly or that the manifest's provenance is true. Preserve generation/conversion receipts as well.

A model manifest should include schema, immutable generation ID, source/model revision and hashes, artifact files with size/digest, actual tensor precision inventory, tokenizer/template hashes, conversion/quantizer runtime commit and arguments, calibration identity, accepted context/output/placement/cache settings, selection protocol and quality/resource report identities. Run results point back to this manifest. Avoid mutable “best.gguf” files without a stable identity.

### Publish a complete generation atomically

Write into a new staging directory such as `gen-0002.partial`. Complete artifacts and manifest, verify them, then rename the directory to `gen-0002` on the same filesystem. Consumers ignore staging names. Existing complete generations are immutable. A pointer to the chosen generation should itself be updated atomically after validation, with a documented fallback policy.

Atomic visibility and crash durability are different. A rename can prevent readers observing a half-published directory, while power-loss durability may require file and directory flushes and platform/filesystem-specific handling. Our reference demonstrates visibility/integrity/fallback; it does not implement production durability, concurrent writers or large-model storage. State that limit rather than claiming a short file-writing function solves every recovery case.

![Immutable generation recovery](figures/recovery.svg)

### Complete corruption and partial-write demonstration

Create `20_recovery.py`. All writes happen in a temporary directory. `publish()` creates two small generations with toy bytes. `load_latest()` scans complete generation directories newest first, validates schema/name/path, checks byte count and SHA-256, and falls back if validation fails. We corrupt generation 2 and leave generation 3 partial; generation 1 must load. The path is fixed to `weights.bin` in this fixture, preventing a manifest from redirecting the loader outside the generation directory.

[Download the complete reference](examples/20_recovery.py)

```python
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
```

From the scratch directory containing that file, run:

```bash
python3 20_recovery.py
```

Expected checked output (JSON formatting may differ):

```json
{
  "loaded": "gen-0001",
  "rejected": [
    "gen-0002"
  ],
  "partial_ignored": true,
  "scope": "integrity demo; not durable model storage"
}
```

The output must name `gen-0001`, reject `gen-0002` and ignore the partial directory. If it returns generation 2, inspect hash validation rather than adding a retry. If a corrupt newest generation prevents checking older ones, the exception boundary is wrong. A real loader should record rejected reasons in a recovery receipt and never silently hide fallback from the user/report.

### Fresh-process recovery is a separate acceptance check

An in-process second call can reuse cached state or loaded objects. The final model application must exit, start a new process, reload only retained artifacts and answer the frozen replay questions. Preserve IDs, actual selected generation, hashes, prompts, finish reasons and outputs. Exact reproduction depends on the declared runtime/determinism contract; define whether the acceptance test requires identical tokens, identical parsed answer/status or an explicitly reviewed numerical tolerance.

Test missing tokenizer/config, mismatched template, corrupt artifact, unsupported schema, an incomplete newest generation and no valid generation. In the last case, fail clearly; do not start with a different model and present its answer as the selected candidate. Retain earlier verified generations under a bounded retention policy after checking their identities. Deletion/cleanup is separate from this tutorial; the reference touches only its temporary directory.

[[practice:recovery-transfer]]

The next application step can now load an explicit generation with visible fallback status. We still need a request lifecycle that prevents one document's state from leaking into another and a response boundary that rejects unknown evidence IDs.
