"""Declare future entry points without simulating their completion."""

PENDING_ACTIONS = ("prepare-data","quantize","evaluate","answer",)


def pending_result(action: str) -> dict:
    if action not in PENDING_ACTIONS:
        raise ValueError(f"Unknown pending action: {action}")
    return {
        "project": "document-assistant",
        "action": action,
        "status": "not_implemented",
        "message": "Document ingestion, quantization, evaluation and answering remain pending. No model or output was created.",
    }
