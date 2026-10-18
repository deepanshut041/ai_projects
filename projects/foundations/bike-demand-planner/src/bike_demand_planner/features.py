"""Fixed calendar categories and an ordered binary representation."""

import json
from collections.abc import Mapping

from .data import sha256


CATEGORY_DOMAINS = (
    ("hr", tuple(range(24))),
    ("weekday", tuple(range(7))),
    ("mnth", tuple(range(1, 13))),
)
FLAG_FIELDS = ("holiday", "workingday")
INPUT_FIELDS = tuple(name for name, _ in CATEGORY_DOMAINS) + FLAG_FIELDS
FEATURE_NAMES = tuple(
    f"{name}_{category:02d}"
    for name, categories in CATEGORY_DOMAINS
    for category in categories
) + FLAG_FIELDS


def encode_calendar(calendar: Mapping[str, int]) -> list[float]:
    """Encode exactly the declared fields; reject unknown or malformed inputs."""
    if set(calendar) != set(INPUT_FIELDS):
        raise ValueError(f"Expected exactly these calendar inputs: {INPUT_FIELDS}.")
    for name, categories in (*CATEGORY_DOMAINS, *((name, (0, 1)) for name in FLAG_FIELDS)):
        if type(calendar[name]) is not int or calendar[name] not in categories:
            raise ValueError(f"Invalid calendar value for {name}: {calendar[name]!r}.")
    expected_workingday = int(calendar["weekday"] in range(1, 6) and calendar["holiday"] == 0)
    if calendar["workingday"] != expected_workingday:
        raise ValueError("Working-day flag disagrees with weekday/holiday flags.")
    encoded = [
        float(calendar[name] == category)
        for name, categories in CATEGORY_DOMAINS
        for category in categories
    ]
    return encoded + [float(calendar[name]) for name in FLAG_FIELDS]


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def calendar_schema() -> dict:
    """Describe the transform without fitting categories or statistics to data."""
    schema = {
        "version": "calendar-one-hot-v1",
        "input_fields": list(INPUT_FIELDS),
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "category_domains": {name: list(values) for name, values in CATEGORY_DOMAINS},
        "binary_flags": list(FLAG_FIELDS),
        "domain_source": "Fixed calendar definitions; not inferred from held-out observations.",
        "unknown_missing_extra_policy": "reject",
        "scaling": "none; every encoded value is 0.0 or 1.0",
        "learned_transform": False,
        "constant_feature_policy": "Retain fixed columns; report training-only constants; never divide by a zero scale.",
        "unused_raw_calendar_fields": {
            "yr": "Constant in the 2011 training window; no learned year-to-year effect is claimed.",
            "season": "Month is retained; the coarser source code and disputed labels are omitted.",
        },
        "coefficient_policy": "Full one-hot groups and flags can be dependent with an intercept; individual coefficients need not be unique.",
        "future_fit_policy": "Any later learned transform must fit only the relevant training partition, including each fold.",
    }
    return {**schema, "schema_sha256": sha256(canonical_json(schema))}
