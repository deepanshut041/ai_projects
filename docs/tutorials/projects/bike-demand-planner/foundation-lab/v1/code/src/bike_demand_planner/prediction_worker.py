"""Fresh-process replay using only a saved predictor and target-free requests."""

import argparse
from pathlib import Path

from .artifacts import json_bytes, read_json
from .saved_predictor import SavedPredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictor = SavedPredictor(args.bundle, expected_manifest_sha256=args.expected_sha256)
    results = [predictor.predict(request) for request in read_json(args.requests)]
    with args.output.open("xb") as output:
        output.write(json_bytes(results))


if __name__ == "__main__":
    main()
