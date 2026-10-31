"""Illustrative arithmetic; no device capacity or model memory is measured."""
import json
import math

GIB = 2 ** 30


def weight_bytes(parameters, bits, group, scale_bytes):
    return math.ceil(parameters * bits / 8) + math.ceil(parameters / group) * scale_bytes


def main():
    # Hypothetical 8-billion stored scalars, not a Gemma tensor inventory.
    count = 8_000_000_000
    bf16 = count * 2
    q4 = weight_bytes(count, 4, 32, 2)
    reserve, workspace, cache = 2 * GIB, 1 * GIB, 2 * GIB
    capacity = 12 * GIB
    total = q4 + reserve + workspace + cache
    assert q4 == 4_500_000_000 and total < capacity < bf16
    print(json.dumps({"bf16_gib": round(bf16 / GIB, 3), "q4_gib": round(q4 / GIB, 3),
                      "total_gib": round(total / GIB, 3), "fits": total <= capacity,
                      "longer_cache_fits": q4 + reserve + workspace + 8 * GIB <= capacity}))


if __name__ == "__main__":
    main()
