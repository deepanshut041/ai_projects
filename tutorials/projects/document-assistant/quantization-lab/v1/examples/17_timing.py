"""Timestamp arithmetic only; the timestamps below are invented fixtures."""
import json
import math


def summarize(start, submitted, first, last, tokens, prompt_tokens, prefill_seconds):
    if not start <= submitted <= first <= last or tokens < 2 or prefill_seconds <= 0:
        raise ValueError("Invalid timestamp/count contract")
    return {"end_to_end_s": last-start, "runtime_ttft_s": first-submitted,
            "user_ttft_s": first-start, "decode_tokens_s": (tokens-1)/(last-first),
            "prefill_tokens_s": prompt_tokens/prefill_seconds}


def nearest_rank(values, probability):
    if not values or not 0 < probability <= 1:
        raise ValueError("Need samples and a probability in (0,1]")
    ordered = sorted(values)
    return ordered[math.ceil(probability*len(values))-1]


def main():
    result = summarize(0, .2, .7, 2.7, 5, 100, .4)
    assert result["decode_tokens_s"] == 2 and result["prefill_tokens_s"] == 250
    print(json.dumps({k: round(v, 6) for k, v in result.items()} |
                     {"p95_fixture_s": nearest_rank([1, 2, 3, 4, 9], .95), "measured": False}))


if __name__ == "__main__":
    main()
