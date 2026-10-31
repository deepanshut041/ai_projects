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
