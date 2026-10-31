"""Same-runtime mechanism comparisons reject hidden changed factors."""
import json

MATCHED = ("source_sha", "tokenizer_sha", "prompt_sha", "runtime_sha", "placement",
           "cache_dtype", "input_limit", "output_limit", "decoding", "question_ids")


def changed_factors(reference, candidate):
    return [key for key in MATCHED if reference[key] != candidate[key]]


def main():
    baseline = dict(source_sha="source-A", tokenizer_sha="tok-A", prompt_sha="prompt-A",
                    runtime_sha="runtime-A", placement="cpu", cache_dtype="bf16",
                    input_limit=2048, output_limit=256, decoding="greedy", question_ids=["q1"])
    candidate = dict(baseline, weight_format="Q4_K_M")
    assert changed_factors(baseline, candidate) == []
    candidate["placement"] = "gpu"
    candidate["tokenizer_sha"] = "tok-B"
    assert changed_factors(baseline, candidate) == ["tokenizer_sha", "placement"]
    print(json.dumps({"confounds": changed_factors(baseline, candidate),
                      "classification": "deployment_comparison"}))


if __name__ == "__main__":
    main()
