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
