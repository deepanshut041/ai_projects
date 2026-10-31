"""Freeze membership by document before attaching question labels."""
import hashlib
import json


def split(doc_id, seed=17):
    value = int(hashlib.sha256(f"{seed}:{doc_id}".encode()).hexdigest(), 16) % 10
    return "calibration" if value < 4 else "development" if value < 8 else "final"


def pack(passages, count_tokens, max_input, overhead, output, capacity):
    chosen = []
    for passage in passages:
        candidate = chosen + [passage]
        used = overhead + count_tokens("\n".join(candidate))
        if used <= max_input and used + output <= capacity:
            chosen = candidate
    return chosen


def main():
    docs = [f"manual-{i}" for i in range(30)]
    sets = {name: {d for d in docs if split(d) == name}
            for name in ["calibration", "development", "final"]}
    assert set.union(*sets.values()) == set(docs)
    assert not sets["calibration"] & sets["final"]
    # Whitespace counting is ONLY a fixture tokenizer, not a model tokenizer.
    chosen = pack(["charge takes three hours", "indoor only"], lambda s: len(s.split()),
                  max_input=8, overhead=3, output=4, capacity=12)
    assert chosen == ["charge takes three hours"]
    print(json.dumps({"disjoint": True, "packed": chosen, "seed": 17}))


if __name__ == "__main__":
    main()
