## 1. Start with an answer we can defend

Read this invented product-manual passage: **“A full charge takes 3 hours. Use the supplied 12 V adapter.”** The question is “How long does a full charge take?” An assistant replies “3 hours” and cites the passage about the power button. Is the answer acceptable? Its words are correct, but its supporting reference is wrong. This small distinction determines what we will preserve when compressing a model.

Our project is a local document assistant under a memory budget. The result we eventually want is an application that takes a document and a question, returns a supported answer or an explicit “not in this document,” and can reload a chosen compressed model after restart. We will learn why each compression mechanism works, implement small complete references, and design the measurements that decide whether it helps this application. A smaller file alone does not answer that question.

### One fixture for the entire route

The Lumen lamp manual is authored teaching data, not a published benchmark. Its three passages are:

| Stable teaching ID | Text | What it can support |
| --- | --- | --- |
| lumen-1 | A full charge takes 3 hours. Use the supplied 12 V adapter. | Charging time; adapter voltage |
| lumen-2 | Hold the power button for 2 seconds to start the Lumen lamp. | Starting procedure |
| lumen-3 | The Lumen lamp is intended for indoor use only. | Intended environment |

“What is the warranty duration?” is unanswerable from this manual. We do not reward a plausible warranty guessed from the model's general knowledge. The target is an answer **from the supplied evidence**. Later we will add long distractors, evidence at different positions, paraphrases and deliberately unavailable facts. We keep these questions separate from text used to calibrate compression.

### Define the output before choosing a runtime

For each request, preserve the question ID, exact prompt identity, supplied passage IDs, answer text, cited passage IDs, answer/refusal status, finish reason and artifact/run identity. The initial answer contract has three different checks. Correctness asks whether the claim matches the labeled fact. Citation validity asks whether cited IDs exist in the supplied material. Support asks whether the referenced text establishes the claim. A real evaluator also checks that extra claims and extra citations do not smuggle in unsupported content; the first reference below intentionally checks only the tiny fixture.

Let $C$, $V$ and $E$ denote correctness, valid references and evidence support. A supported correct answer requires their conjunction $C\land V\land E$. These are Boolean conditions, not probabilities. Knowing $C=1$ does not imply $E=1$. In our opening example, the words pass correctness, the existing power-button ID passes validity, and the evidence check fails. If the source never contained a charging time, a refusal would be appropriate even if a memorized answer happened to match another manual.

![Answer evidence flow](figures/evidence-flow.svg)

The three checks belong beside each other because a compression regression can affect any of them: a number may change, a citation ID may be malformed, or a model may become more willing to guess. We retain raw responses so a parser failure remains visible rather than silently becoming an empty answer.

### Run the first complete reference

Create `01_contract.py` in a new scratch directory, or download the linked file. This and subsequent teaching files use Python's standard library unless the lesson explicitly says C++. They do not load a model. Every helper used below is defined before its call. The file is deliberately small enough to inspect the entire contract.

[Download the complete reference](examples/01_contract.py)

```python
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
```

`score()` compares case-folded fixture text, verifies passage membership and checks the expected evidence ID. `main()` constructs the correct and wrong-source cases rather than asking an LLM to grade itself. Notice why a substring match would be dangerous: “not 3 hours” contains the expected words and contradicts them. Exact matching also has a limit: “three hours” needs a declared alias or human semantic review. Lesson 16 extends the evaluator without pretending these two issues disappear.

From the scratch directory containing that file, run:

```bash
python3 01_contract.py
```

Expected checked output (JSON formatting may differ):

```json
{
  "good": {
    "correct": true,
    "valid_ids": true,
    "supported": true
  },
  "wrong_source": {
    "correct": true,
    "valid_ids": true,
    "supported": false
  }
}
```

The output should mark both answers correct but only the first supported. If both are supported, inspect the evidence-set comparison. If both fail validity, inspect IDs before changing any model setting. This is already a useful debugging order: separate data/schema errors from inference errors.

[[practice:contract-transfer]]

### How to use this long course

The first session can stop after Lessons 1–4: you will have the answer contract, understand the current launcher, know what a compatible model probe must prove and calculate a memory budget. Continue in order for document preparation, reference generation, integer quantization, calibration-aware methods, task/timing evaluation, cache studies and a recoverable application. The [saved outline](project/course-outline.md) maps all 24 lessons to the project's seven phases; the [implementation record](project/implementation.md) is a return-to-work checklist, not another mandatory course.

At authoring time the **real project is runnable boilerplate**. Status, setup checks and configuration work. Model loading, quantization, answering and measured comparisons are pending. The complete references in this course establish arithmetic and component contracts; they are not Gemma benchmark results or evidence that the application exists. The [readiness record](project/readiness.md) keeps those evidence levels separate. Next we inspect the actual source so that every command has an honest meaning.
