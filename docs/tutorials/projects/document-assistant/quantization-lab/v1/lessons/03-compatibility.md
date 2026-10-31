## 3. A model choice needs an executable compatibility proof

The scaffold identifies a source model, but an identifier is only the start. The proposed primary checkpoint is `google/gemma-4-E4B-it` at revision `ee0ef6023621cff504d758262d4e04895a5af4a2`. We keep this current-generation project target while freezing its bytes for comparisons. “Latest” is useful at intake and harmful as an experiment identity: a moving branch can change underneath two nominally identical runs.

### Read architecture facts before making a memory promise

The [pinned official configuration](source:gemma-config) identifies a 42-layer text decoder with local and full attention, a 512-token local window, distinct local/global head dimensions and KV sharing. The captured [configuration file](evidence/gemma4-config.json) is small metadata, not downloaded weights. Those facts explain why a generic “layers × context × heads” estimate needs architecture-specific corrections. We derive a logical cache model in Lesson 18; actual allocation remains a backend measurement.

The full checkpoint also carries multimodal configuration. Our first application uses text documents. A text-only application does not automatically prove that its exporter omits every audio/vision parameter, nor that all embeddings fit an advertised effective parameter count. Inventory converted tensors and measured memory. The [model card](source:gemma-card) is a source of intended architecture/use, not evidence that our chosen runtime loads it.

### Pin the complete interpretation of the weights

A model artifact needs more than a repository commit. Preserve tokenizer and special-token files, configuration, chat-template bytes, source file hashes, conversion command/build, resulting tensor types, decoding settings and placement. The template translates structured user/system messages into the actual token sequence. If one run adds different control tokens or a different generation prefix, it is answering a different prompt even if the visible question is unchanged.

Create a probe manifest only after running each check. Its acceptance contract has five separate facts: the selected runtime loads the artifact, rendered prompt tokens match the declared template policy, generation stops at recognized end markers, a fresh process reloads it and placement is observed. A field set to `true` by an author is not a measurement; retain command, timestamps, exit status and output beside the manifest. Test a malformed input, a short answer and an unanswerable question. A model that emits valid text but never terminates is not ready for a bounded benchmark.

Context capacity means total tokens, including formatted prompt and generated output. If $P=2048$ prompt tokens and the output allowance is $O=256$, require $P+O\le C$, so $C$ must be at least 2304 under that runtime's definition. A publisher's maximum context is not a local memory budget. Check whether the runtime reserves the whole capacity, grows dynamically, pads to blocks or has separate per-sequence limits.

### A complete gate with synthetic probe results

Create `03_compatibility.py` in the scratch directory. This executable validates a manifest; it does not populate one by claiming to run Gemma. We give it a passing fixture, then remove template evidence and verify rejection. Exact revision syntax and `is True` comparisons prevent accepting a label such as `"probably"` as a completed probe.

[Download the complete reference](examples/03_compatibility.py)

```python
"""A manifest acceptance gate; capability values are deliberately fixtures."""
import json
import re


def require_ready(manifest):
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["model_revision"]):
        raise ValueError("Model revision must be an exact commit")
    required = {"loads", "template_matches", "eos_stops", "reloads", "placement_verified"}
    missing = sorted(name for name in required if manifest["probes"].get(name) is not True)
    if missing:
        raise ValueError("Unproven capabilities: " + ", ".join(missing))
    if manifest["prompt_tokens"] + manifest["output_tokens"] > manifest["context_capacity"]:
        raise ValueError("Context capacity exceeded")
    return "ready_for_bounded_reference"


def main():
    sample = {"model_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
              "prompt_tokens": 2048, "output_tokens": 256, "context_capacity": 2304,
              "probes": dict.fromkeys(["loads", "template_matches", "eos_stops",
                                      "reloads", "placement_verified"], True)}
    assert require_ready(sample) == "ready_for_bounded_reference"
    sample["probes"]["template_matches"] = False
    try:
        require_ready(sample)
    except ValueError as exc:
        rejected = str(exc)
    else:
        raise AssertionError("An unproven template was accepted")
    print(json.dumps({"fixture_only": True, "rejected": rejected}))


if __name__ == "__main__":
    main()
```

From the scratch directory containing that file, run:

```bash
python3 03_compatibility.py
```

Expected checked output (JSON formatting may differ):

```json
{
  "fixture_only": true,
  "rejected": "Unproven capabilities: template_matches"
}
```

The rejection should name `template_matches`. In actual phase 00, set this field only after preserving rendered messages, token IDs and the agreement check. Successful configuration parsing is weaker evidence than successful generation. A generic architecture implementation appearing in a runtime repository is weaker still: conversion, special tensors, kernels, device placement and reload can fail independently.

### Choose the first backend by an explicit gate

The pinned [llama.cpp source](source:llama-gemma) is a candidate for the GGUF route. OpenVINO/Optimum Intel is another candidate for a later supported method comparison. We have not compiled either against this checkpoint in this tutorial. At implementation kickoff, record compiler/runtime versions and run the small gate before choosing dependencies or acquiring large artifacts. If a backend lacks exact architecture support, keep the failed receipt and choose a supported build or a separately declared model fallback. Do not silently replace E4B with E2B inside the same experiment.

The planned fallback is Gemma 4 E2B at the revision already in project configuration. Treat it as a different model requiring a new budget and baseline. Shrinking the model and reducing its precision at once cannot tell us which change caused an answer regression.

[[practice:compatibility-transfer]]

The phase-00 output is a verified compatibility receipt plus a bounded working reference, not a benchmark score. The [source/departure ledger](research.md) records inspected facts and unknowns. Next we compute why a compatible model can still exceed memory once a real request starts.
