## 24. Connect every phase to evidence that someone else can inspect

We began by rejecting a correct number with the wrong citation. We can now explain the entire path from canonical document bytes to a supported answer, the numerical mechanisms that compress weights/cache, and the measurements that decide whether a saved candidate is usable. The final task is to connect those explanations to an implementation handoff with no missing or invented evidence.

### Walk one request from the beginning

A source document is acquired with provenance, canonicalized under a versioned policy and split into stable passages. Its question belongs to a frozen document-level role. The chosen tokenizer/template renders the same bounded prompt for each compared artifact; packing records which evidence was actually supplied. A verified backend loads a manifest-identified artifact at a declared placement/cache policy, begins a fresh sequence and generates under the frozen output policy. The service retains raw output/token IDs, parses its answer/refusal structure, validates citation membership and resets request state even on failure.

Evaluation aligns exact question IDs, scores correctness/support/refusal/truncation under the declared rubric and pairs each candidate with the reference. Measurements retain timestamp boundaries, actual token work and host/device memory observations. Selection uses development constraints; final assessment evaluates the frozen choice without further tuning. Restart verification loads the saved generation in a new process and replays its accepted contract, disclosing fallback. Each link can fail independently, so “the model ran” is insufficient as a completion receipt.

### Map the seven phases to their required exit evidence

| Phase | Concrete implementation destination | Evidence required before moving on |
| --- | --- | --- |
| 00 — Compatibility/budget | Backend adapter probe, pinned environment and runtime manifest | Exact source/template identities, successful bounded floating generation, EOS/reload/placement checks, measured feasible memory/context budget |
| 01 — Documents | `src/data/` canonicalization, passages, membership and packing | Provenance/licenses, stable spans, frozen disjoint roles, reviewed answerability/support labels, complete-prompt token checks |
| 02 — Reference/Q8 | `src/quantization/` artifact protocol and `src/evaluation/` aligned runner | Preserved floating artifact, Q8 identity/composition, matched predictions/metrics/resources, failures and fresh-process reload |
| 03 — Q4 | Same runner with one low-bit recipe | Actual tensor precision, same prompts/runtime factors, per-question regressions and storage/peak-memory tradeoff |
| 04 — Calibration/method | Collector plus one supported advanced recipe at a time | Frozen calibration tokens/statistics, completed layer coverage, matched budget/method comparisons, task-level evidence |
| 05 — Context | Architecture-aware cache policy and bounded length runner | Fixed weight hash, supported actual cache behavior, memory/latency/support by length/evidence position, failed policies retained |
| 06 — Saved application | Manifest loader, answer service and local interface | Visible artifact/fallback identity, real document interactions, cancellation/error cleanup, restart replay, frozen selection/final report |

These are build contracts. The shared tutorial does not silently create those production modules. The [implementation checklist](project/implementation.md) makes the next phase and its exact checks resumable. The [readiness matrix](project/readiness.md) keeps author references, model runs, learner explanation and hosted execution separate.

### Execute the completion gate

Create `24_handoff.py`. It deliberately accepts only evidence marked `verified` for required model/application milestones. Scaffold and teaching-reference checks do not satisfy those keys. The output lists ten missing evidence categories and keeps phase 00 as the next implementation step. This is the expected truthful state after writing a tutorial.

[Download the complete reference](examples/24_handoff.py)

```python
"""Evidence-led completion gate; no invented model or learner receipts."""
import json

REQUIRED = ["compatibility", "frozen_data", "float_reference", "q8_comparison",
            "q4_comparison", "calibration_study", "context_study", "saved_application",
            "fresh_process_recovery", "final_evaluation"]


def completion(evidence):
    missing = [name for name in REQUIRED if evidence.get(name) != "verified"]
    return {"complete": not missing, "missing": missing}


def main():
    actual = {"scaffold": "verified", "teaching_references": "verified"}
    result = completion(actual)
    assert result["complete"] is False and len(result["missing"]) == 10
    print(json.dumps(result | {"next_phase": "00 — compatibility and budget",
                              "learner_review": "pending"}))


if __name__ == "__main__":
    main()
```

From the scratch directory containing that file, run:

```bash
python3 24_handoff.py
```

Expected checked output (JSON formatting may differ):

```json
{
  "complete": false,
  "missing": [
    "compatibility",
    "frozen_data",
    "float_reference",
    "q8_comparison",
    "q4_comparison",
    "calibration_study",
    "context_study",
    "saved_application",
    "fresh_process_recovery",
    "final_evaluation"
  ],
  "next_phase": "00 — compatibility and budget",
  "learner_review": "pending"
}
```

Do not change these fields to pass the gate without attaching receipts. Conversely, a future verified model result should update its own manifest/run record and readiness state, not require rewriting the mathematical teaching examples. Software source, model results and learner answers have different owners and lifetimes.

### Explain the design without reciting tool names

Try to explain why valid citations do not guarantee support, why a large activation changes the cost of an equal weight error, why scaling both sides preserves the float layer but changes quantized reconstruction, why correlated channels permit compensation, and why low-bit weights do not remove context memory. Trace the first-byte nibble example and the key-versus-value attention example with numbers. Predict a changed condition before revealing the answer.

Use the [explanation questions](practice/defense.md), [learning points](project/learning-points.md) and [comparison guide](project/comparisons.md) for review. They consolidate the route; they do not store an invented learner score. The [reference-run record](project/reference-runs.md) and [numerical audit](evidence/numerical-audit.json) contain author checks with their limits.

[[practice:handoff-transfer]]

### What begins next

For implementation, begin with phase 00: recheck releases/support, freeze one runtime and execute a tiny source/reference compatibility probe under a measured budget. Do not start a full QA sweep before the probe can produce complete identities and reload evidence. After phase 00 passes, implement the phase-01 document contract with reviewed data. Expand and review one bounded implementation phase at a time while using this complete course as the teaching route.

The sibling Compressed Semantic Search project can later provide versioned retrieval results, but its encoder/index changes remain separate from weight/cache studies. Videos can be produced after written-text review; none are implied by this course. A Colab/notebook demonstration, if later requested, needs actual hosted execution/recovery receipts rather than a locally valid file. The written route is complete when its explanations and references are verified; learner understanding and the model-backed assistant remain their own future evidence gates.

[Reference outputs](evidence/reference-outputs.json) · [Complete outline](project/course-outline.md) · [Source/departure ledger](research.md)
