"""Count explicit cache owners; architecture fixture is not Gemma's layout."""
import json


def cache_bytes(layers, length, batch, bytes_per_scalar):
    owners, total = set(), 0
    for layer in layers:
        owner = layer["owner"]
        if owner in owners:
            continue
        owners.add(owner)
        retained = min(length, layer["window"]) if layer["window"] else length
        total += 2 * batch * retained * layer["kv_heads"] * layer["head_dim"] * bytes_per_scalar
    return total


def main():
    # Layers 1 and 2 share owner 's'; layer 3 owns a full-context cache 'f'.
    layers = [{"owner": "s", "window": 512, "kv_heads": 2, "head_dim": 64},
              {"owner": "s", "window": 512, "kv_heads": 2, "head_dim": 64},
              {"owner": "f", "window": None, "kv_heads": 2, "head_dim": 128}]
    sizes = [cache_bytes(layers, n, 1, 2) for n in [512, 2048, 4096]]
    assert sizes == [786432, 2359296, 4456448]
    assert cache_bytes(layers, 2048, 2, 2) == 2*sizes[1]
    print(json.dumps({"lengths": [512, 2048, 4096], "bytes": sizes,
                      "scope": "deduplicated logical cache fixture"}))


if __name__ == "__main__":
    main()
