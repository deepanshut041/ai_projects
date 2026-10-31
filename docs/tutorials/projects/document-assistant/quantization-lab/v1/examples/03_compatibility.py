"""A manifest acceptance gate; capability values are deliberately fixtures."""
import json
import re


def require_ready(manifest):
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["model_revision"]):
        raise ValueError("Model revision must be an exact commit")
    required = {"loads", "template_matches", "eos_stops", "reloads", "placement_verified"}
    missing = sorted(name for name in required if manifest["probes"].get(name) is not True)
    if missing:
        raise ValueError("Unproven capabilities: " + ", ".join(missing))
    if manifest["prompt_tokens"] + manifest["output_tokens"] > manifest["context_capacity"]:
        raise ValueError("Context capacity exceeded")
    return "ready_for_bounded_reference"


def main():
    sample = {"model_revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
              "prompt_tokens": 2048, "output_tokens": 256, "context_capacity": 2304,
              "probes": dict.fromkeys(["loads", "template_matches", "eos_stops",
                                      "reloads", "placement_verified"], True)}
    assert require_ready(sample) == "ready_for_bounded_reference"
    sample["probes"]["template_matches"] = False
    try:
        require_ready(sample)
    except ValueError as exc:
        rejected = str(exc)
    else:
        raise AssertionError("An unproven template was accepted")
    print(json.dumps({"fixture_only": True, "rejected": rejected}))


if __name__ == "__main__":
    main()
