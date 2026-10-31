## 11. Turn weight formats into an identifiable comparison

The toy layers explain mechanisms. Phase 02 now needs an actual floating artifact and one eight-bit artifact; phase 03 adds a four-bit artifact. Each must come from the same frozen source and be reloadable. A format label by itself does not identify the artifact or prove that a requested recipe was applied.

### Keep conversion and quantization as separate steps

Conversion translates a source checkpoint into a runtime container. Quantization changes numerical representation. Preserve a converted floating-point reference before deriving quantized files; quantizing an already low-bit source can add another approximation and confound the study. Retain source file hashes, tokenizer/template identities, converter build, arguments, tensor inventory, destination hash and warnings. Treat unsupported tensors or silently skipped options as failed/partial work requiring inspection.

The [pinned llama.cpp quantization documentation](source:llama-quantize) describes `llama-quantize` and Q8_0/Q4_K_M candidates. This is a **candidate protocol**, not a command receipt for our Gemma checkpoint. The relevant source commit is `9b8446b91434e0c0cdc293baf0b29c099eaaa0c7`. At implementation, create a separate runtime checkout, verify its help/build/converter support, freeze dependencies, pass Lesson 3's compatibility gate and only then run full conversion.

For a verified converted reference already present as `artifacts/gemma4-reference-bf16.gguf`, the two quantizer invocations would have this form from that runtime checkout:

```bash
./build/bin/llama-quantize artifacts/gemma4-reference-bf16.gguf artifacts/gemma4-q8.gguf Q8_0
./build/bin/llama-quantize artifacts/gemma4-reference-bf16.gguf artifacts/gemma4-q4.gguf Q4_K_M
```

These commands are conditional instructions for future model work. The named files do not currently exist in this project. Verify input tensor precision and the pinned binary's accepted arguments before use. Keep the conversion/calibration peak separate from inference peaks. Runtime weights and generated artifacts stay under ignored runtime directories, not in this tutorial package.

### Fix the factors that define a mechanism comparison

For a weight-only comparison, hold source/model, tokenizer, rendered prompts, question membership/order, runtime build, placement, cache policy, decoding, input/output caps and batch/concurrency fixed. Change the weight artifact and its declared recipe. Record seeds even under greedy decoding and repeat a bounded agreement check. If BF16 cannot fit at the same placement, its quality reference can still be useful, while the timing row is explicitly a different deployment configuration.

The complete `11_artifact_plan.py` reference compares these matched factors. A model identifier like `source-A` is a fixture identity, not a genuine cryptographic digest. Real manifests use full validated hashes. The first candidate changes only weight format and passes; the second changes tokenizer and placement and is classified as a deployment comparison.

[Download the complete reference](examples/11_artifact_plan.py)

```python
"""Same-runtime mechanism comparisons reject hidden changed factors."""
import json

MATCHED = ("source_sha", "tokenizer_sha", "prompt_sha", "runtime_sha", "placement",
           "cache_dtype", "input_limit", "output_limit", "decoding", "question_ids")


def changed_factors(reference, candidate):
    return [key for key in MATCHED if reference[key] != candidate[key]]


def main():
    baseline = dict(source_sha="source-A", tokenizer_sha="tok-A", prompt_sha="prompt-A",
                    runtime_sha="runtime-A", placement="cpu", cache_dtype="bf16",
                    input_limit=2048, output_limit=256, decoding="greedy", question_ids=["q1"])
    candidate = dict(baseline, weight_format="Q4_K_M")
    assert changed_factors(baseline, candidate) == []
    candidate["placement"] = "gpu"
    candidate["tokenizer_sha"] = "tok-B"
    assert changed_factors(baseline, candidate) == ["tokenizer_sha", "placement"]
    print(json.dumps({"confounds": changed_factors(baseline, candidate),
                      "classification": "deployment_comparison"}))


if __name__ == "__main__":
    main()
```

From the scratch directory containing that file, run:

```bash
python3 11_artifact_plan.py
```

Expected checked output (JSON formatting may differ):

```json
{
  "confounds": [
    "tokenizer_sha",
    "placement"
  ],
  "classification": "deployment_comparison"
}
```

This checker cannot establish that two actual runs used the fields they claim. Retained commands, build logs, prompt hashes, placement observations and prediction records provide that evidence. A metadata gate is useful because it catches accidental confounds early; it is not a substitute for execution receipts.

### Inspect actual tensor composition

For each artifact, tabulate tensor name, shape, stored type, storage bytes and any higher-precision exceptions. A named mixed preset may use different types for sensitive tensors. Calculate total payload/metadata and compare with file size; document tied/shared storage and alignment so values are not double counted. The [comparison record](project/comparisons.md) separates same-runtime weight changes, calibration changes, cache changes and deployment comparisons.

The first task report should align every development question by ID and retain reference/Q8/Q4 raw and parsed responses, correctness/support/refusal/truncation outcomes, memory and timing receipts. Include failed loads, malformed output and OOM as explicit statuses. Dropping a failing candidate's hardest questions would change the denominator and make the score misleading.

[[practice:artifacts-transfer]]

### A bounded stop before a larger experiment

Start with a tiny pilot to validate artifacts, prompt agreement and record completeness. Then freeze the main development subset and protocol. A pilot that succeeds does not establish preserved application quality. A candidate that loads after a process restart still needs end-to-end answer evaluation. Retain the floating reference as the quality anchor while exploring smaller artifacts.

Basic rounding uses weights alone. The layer calculation showed that input magnitudes matter, so our next step is to measure activation distributions using permitted calibration text.
