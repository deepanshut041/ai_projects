## 16. Score the answer the application promises

Layer reconstruction helps diagnose compression, but the user sees an answer and its support. We return to the Lumen contract and build a transparent evaluator. The first rule is alignment: compare the same question IDs exactly once. A smaller denominator must not make a failing candidate look better.

### Separate outcomes with different denominators

For answerable questions, record correct answers, supported correct answers, false refusals and malformed/truncated outputs. For unanswerable questions, record correct refusals and unsupported answers. An answered response on an answerable question can still be unsupported if its fact or citation is wrong. Keep raw counts alongside rates.

Let $N_A$ and $N_U$ denote answerable and unanswerable counts. Supported-answer rate is $S/N_A$; false-refusal rate is $F/N_A$; correct-refusal rate is $R/N_U$. An overall unsupported-answer rate can use all questions as denominator, but name that choice explicitly. It answers a different question from unsupported fraction **among attempted answers**. A denominator of zero should produce an undefined/not-applicable rate, not an invented zero.

Our fixture has three questions: charging time, adapter voltage and an unavailable warranty. Predictions answer charging correctly with its evidence, refuse the voltage question and guess a warranty while reaching the output limit. Counts are: one supported correct answer out of two answerable questions, one false refusal out of two, no correct refusal out of one unanswerable question, one unsupported answer out of three and one truncation out of three. The errors are different and may overlap; the truncated warranty is also unsupported.

### Complete evaluator and its explicit simplifications

Create `16_quality.py`. It rejects missing, extra or duplicate prediction IDs before scoring. It uses declared exact aliases and evidence IDs for this fixture. It does not perform semantic entailment, detect every contradictory extra sentence or score citation precision. A production rubric must cover those behaviors rather than claiming exact-match code establishes grounding.

[Download the complete reference](examples/16_quality.py)

```python
"""A transparent exact-match fixture evaluator with distinct denominators."""
import json


def evaluate(gold, predictions):
    expected = {x["id"] for x in gold}
    by_id = {x["id"]: x for x in predictions}
    if len(by_id) != len(predictions) or set(by_id) != expected:
        raise ValueError("Prediction IDs must match exactly, without duplicates")
    counts = dict(total=len(gold), answerable=0, unanswerable=0, correct=0,
                  supported=0, unsupported_answer=0, false_refusal=0, correct_refusal=0,
                  truncated=0)
    for item in gold:
        pred = by_id[item["id"]]
        refusal = pred["status"] == "not_in_document"
        counts["truncated"] += pred["finish_reason"] == "length"
        if item["answerable"]:
            counts["answerable"] += 1
            correct = not refusal and pred["answer"].casefold() in item["aliases"]
            supported = correct and set(item["evidence"]) <= set(pred["citations"])
            counts["correct"] += correct
            counts["supported"] += supported
            counts["false_refusal"] += refusal
            counts["unsupported_answer"] += not refusal and not supported
        else:
            counts["unanswerable"] += 1
            counts["correct_refusal"] += refusal
            counts["unsupported_answer"] += not refusal
    return counts


def main():
    gold = [{"id": "q1", "answerable": True, "aliases": ["3 hours"], "evidence": ["p1"]},
            {"id": "q2", "answerable": True, "aliases": ["12 v"], "evidence": ["p1"]},
            {"id": "q3", "answerable": False}]
    preds = [{"id": "q1", "status": "answered", "answer": "3 hours", "citations": ["p1"], "finish_reason": "eos"},
             {"id": "q2", "status": "not_in_document", "answer": "", "citations": [], "finish_reason": "eos"},
             {"id": "q3", "status": "answered", "answer": "5 years", "citations": ["p1"], "finish_reason": "length"}]
    result = evaluate(gold, preds)
    assert result["supported"] == result["false_refusal"] == result["unsupported_answer"] == 1
    print(json.dumps(result))


if __name__ == "__main__":
    main()
```

From the scratch directory containing that file, run:

```bash
python3 16_quality.py
```

Expected checked output (JSON formatting may differ):

```json
{
  "total": 3,
  "answerable": 2,
  "unanswerable": 1,
  "correct": 1,
  "supported": 1,
  "unsupported_answer": 1,
  "false_refusal": 1,
  "correct_refusal": 0,
  "truncated": 1
}
```

A malformed response should retain `parse_error` plus raw text and enter the declared failure category; it must not disappear from `predictions`. The tiny reference assumes well-formed records so that outcome counting is visible. Phase-02 implementation should add schema validation before this scorer and fixture tests for parse errors, duplicate IDs, zero-sized slices and alternative valid evidence sets.

### Review support at the claim level

For a real answer such as “A full charge takes 3 hours, and the warranty lasts five years,” the first fact is supported and the second is not. A complete-answer support label must fail even though the expected short answer appears. Check every material claim, contradiction, unit and qualifier. A valid passage ID proves only that a reference exists. A cited charging passage does not support an invented warranty.

If several passages provide equivalent evidence, label alternative valid sets. If a question needs two facts, require both or define partial credit before scoring. Citation precision can be evaluated separately: citing every passage is not as useful as citing the relevant support. Human review should preserve rubric, reviewer decisions, disagreements and adjudication. A model judge can assist a declared workflow, but its confidence is not ground truth.

### Compare paired outputs, then inspect regressions

For each question $i$, let $s_i^{(c)}$ and $s_i^{(r)}$ be candidate/reference supported-answer indicators. Paired difference $d_i=s_i^{(c)}-s_i^{(r)}$ is -1, 0 or +1. Average $\bar d$ counts improvements and regressions on matched questions. Equal average rates can conceal a swapped failure set: one lost fact and one newly correct fact cancel numerically while remaining useful diagnostic evidence.

Group failures by answerability, evidence position, input length, numerical facts, multi-passage requirements and truncation. A slice with two questions is descriptive evidence with high uncertainty, not a robust performance claim. Keep the chosen metrics/thresholds fixed while exploring; reserve final questions for the frozen decision in Lesson 23.

[[practice:quality-transfer]]

Perplexity or average token loss may reveal numerical shifts, but neither replaces this application contract. We now know how to compare answer quality. Next we measure time with equally explicit boundaries so a warm decode speed cannot masquerade as a fast end-to-end application.
