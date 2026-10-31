## 21. Give inference a request and response boundary

Saved artifacts solve one kind of persistence. A running service has another kind of state: an active request, its document evidence and its generation cache. We need a boundary that owns this lifecycle and rejects an output that does not fit the answer contract, independently of which backend eventually runs Gemma.

### Define the request before integrating a backend

A request contains question ID/text, supplied passage IDs/text and output allowance, plus a run context identifying artifact, prompt policy and input budget. The small dataclass below omits the full run manifest to keep the lifecycle visible. Its `frozen=True` prevents field reassignment but does not deeply freeze the passage dictionary; a production boundary should copy/validate immutable input rather than assuming shallow freezing is enough.

The backend adapter has three responsibilities: begin a fresh sequence, generate from the declared request/prompt and release/reset request state. It must expose raw output/token IDs, finish reason, token counts and timing/instrumentation through the actual runtime integration. `FixtureAdapter` instead returns a declared fixed answer. It exercises control flow; it is not a model, retriever or evidence reasoner.

### Inspect every stage before the complete listing

`answer()` validates a positive output cap and nonempty evidence, begins a session, calls the adapter, validates the result and resets in `finally`. This ensures a malformed citation also triggers cleanup. If `begin()` itself fails because another session is active, it should not blindly clear that unrelated active session; a real concurrent system needs request-specific handles and ownership rather than one global Boolean.

The result distinguishes `answered` from `not_in_document`. An answer requires nonempty text and citations; a refusal requires empty answer/citations under this schema. Every citation must identify supplied evidence. These are structural checks. They cannot establish that “3 hours” follows from passage text; that remains semantic evaluation. A valid ID attached to the wrong claim still fails Lesson 1's support criterion.

### Complete fixture boundary and reset checks

Create `21_answer_boundary.py`. All classes/functions used by `main()` are present before their calls. The first two requests return the same fixture response with separate resets. A third supplies only an unrelated passage ID; response validation rejects the adapter's `lumen-1` citation and the `finally` block still resets. Three resets prove the exercised cleanup path, not general backend resource correctness.

[Download the complete reference](examples/21_answer_boundary.py)

```python
"""Complete fixture service boundary; the adapter does not infer or retrieve."""
from dataclasses import dataclass
import json


@dataclass(frozen=True)
class Request:
    question_id: str
    question: str
    passages: dict[str, str]
    max_output_tokens: int = 256


class FixtureAdapter:
    def __init__(self):
        self.active = False
        self.reset_count = 0

    def begin(self):
        if self.active:
            raise RuntimeError("Previous request state is still active")
        self.active = True

    def generate(self, request):
        if not self.active:
            raise RuntimeError("No active request")
        return {"status": "answered", "answer": "3 hours", "citations": ["lumen-1"],
                "finish_reason": "eos", "backend": "fixture"}

    def reset(self):
        self.active = False
        self.reset_count += 1


def answer(adapter, request):
    if request.max_output_tokens <= 0 or not request.passages:
        raise ValueError("Invalid request budget or evidence")
    adapter.begin()
    try:
        result = adapter.generate(request)
        required = {"status", "answer", "citations", "finish_reason", "backend"}
        if set(result) != required or not isinstance(result["citations"], list):
            raise ValueError("Malformed response")
        if result["status"] not in {"answered", "not_in_document"}:
            raise ValueError("Invalid answer status")
        if result["status"] == "answered" and (not result["answer"].strip() or not result["citations"]):
            raise ValueError("Answer requires text and citations")
        if result["status"] == "not_in_document" and (result["answer"] or result["citations"]):
            raise ValueError("Refusal must not contain an answer")
        if not all(isinstance(c, str) and c in request.passages for c in result["citations"]):
            raise ValueError("Citation outside supplied passages")
        return result
    finally:
        adapter.reset()


def main():
    adapter = FixtureAdapter()
    request = Request("q-charge", "How long does a full charge take?",
                      {"lumen-1": "A full charge takes 3 hours."})
    first = answer(adapter, request)
    second = answer(adapter, request)
    assert first == second and adapter.reset_count == 2 and not adapter.active
    try:
        answer(adapter, Request("q", "?", {"other": "No charge information."}))
    except ValueError:
        rejected = True
    else:
        raise AssertionError("Unknown evidence accepted")
    assert adapter.reset_count == 3
    print(json.dumps({"backend": "fixture", "reset_count": 3, "invalid_citation_rejected": rejected}))


if __name__ == "__main__":
    main()
```

From the scratch directory containing that file, run:

```bash
python3 21_answer_boundary.py
```

Expected checked output (JSON formatting may differ):

```json
{
  "backend": "fixture",
  "reset_count": 3,
  "invalid_citation_rejected": true
}
```

The output should show `backend: fixture`, three resets and citation rejection. If the third request leaves `active=True`, an error path skipped cleanup. If two requests concatenate their histories, inspect sequence/KV ownership and template construction before treating the difference as a quantization regression. Real error handling must also account for cancellation, timeout, partial streaming and runtime exceptions.

### Specify the real adapter's acceptance contract

The future destination is `src/core/answering.py` plus a backend-specific module selected after phase 00. Create no undocumented live dependency on a sibling project. Its tokenizer/template adapter must render the exact frozen prompt and expose token counts before inference. It must reject excess input rather than silently truncate evidence. The generation adapter must bind the verified artifact/placement/cache settings, reset sequence state for independent requests, enforce output limits and return a typed failure without fabricating an answer.

Test with a supplied known-answer prompt, unavailable fact, invalid citation response, malformed JSON, over-budget document, cancellation and two documents asked sequentially. If retrieval is later connected, retain retrieved IDs/scores and the actual subset packed into the prompt. A retrieved fact dropped during packing is not evidence the model received. Version retrieval separately so a changed index is not mistaken for a weight-method improvement.

[[practice:boundary-transfer]]

### Keep document instructions in their data role

The prompt should clearly delimit evidence and say which answer schema/policy applies. Include quoted document instructions as adversarial fixtures; a document asking the assistant to ignore its contract should not be promoted to a system message. Schema validation alone cannot prevent a fluent but unsupported answer, which is why raw output, evidence review and unanswerability tests remain essential.

We have a complete, tested **fixture** service boundary and a concrete real-adapter contract. Next we connect a local interface while keeping the backend identity visible to its user.
