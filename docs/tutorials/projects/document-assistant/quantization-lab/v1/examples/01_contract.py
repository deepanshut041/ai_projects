"""An authored manual fixture, not a language model or production evaluator."""
import json

PASSAGES = {
    "lumen-1": "A full charge takes 3 hours. Use the supplied 12 V adapter.",
    "lumen-2": "Hold the power button for 2 seconds to start the Lumen lamp.",
    "lumen-3": "The Lumen lamp is intended for indoor use only.",
}
QUESTION = {"id": "q-charge", "text": "How long does a full charge take?",
            "answerable": True, "answer": "3 hours", "evidence": {"lumen-1"}}


def score(record, gold, passages):
    """Exact fixture checks; semantic entailment needs a separate review rubric."""
    answer = record["answer"].strip().casefold()
    cited = set(record["citations"])
    correct = answer == gold["answer"].casefold()
    valid_ids = bool(cited) and cited <= passages.keys()
    supported = correct and valid_ids and gold["evidence"] <= cited
    return {"correct": correct, "valid_ids": valid_ids, "supported": supported}


def main():
    good = score({"answer": "3 hours", "citations": ["lumen-1"]}, QUESTION, PASSAGES)
    wrong_source = score({"answer": "3 hours", "citations": ["lumen-2"]}, QUESTION, PASSAGES)
    assert good == {"correct": True, "valid_ids": True, "supported": True}
    assert wrong_source["correct"] and not wrong_source["supported"]
    print(json.dumps({"good": good, "wrong_source": wrong_source}))


if __name__ == "__main__":
    main()
