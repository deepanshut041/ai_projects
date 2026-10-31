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
