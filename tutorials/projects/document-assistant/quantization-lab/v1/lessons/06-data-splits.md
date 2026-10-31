## 6. Separate data roles and fit the actual prompt

Stable passages make it possible to freeze membership. They do not prevent the same manual from appearing in calibration, development and final evaluation. Different questions from one manual can still expose the same facts and phrasing. We therefore assign whole documents to roles before building labeled examples.

### Three roles answer three different questions

Calibration text estimates numerical distributions for a compression method. Development questions compare recipes, thresholds and application behavior. Final questions assess the frozen choice once. Calibration generally does not need answer labels. Reusing final document text for calibration can still make evaluation transductive even if labels are hidden; this tutorial chooses document-disjoint roles to avoid that ambiguity. Related versions or copied manuals should be grouped before splitting, because different hashes do not guarantee different content.

The complete reference hashes `seed:document_id` and reduces the integer modulo ten: buckets 0–3 calibration, 4–7 development, 8–9 final. The proportions are a declared fixture policy, not a guarantee for 30 documents. Hash assignment is deterministic and easy to audit; it does not ensure balanced domains or answerability. For a real small dataset, inspect counts and freeze a stratified/grouped membership manifest before tuning. If you change membership to repair imbalance, make a new dataset version rather than searching split seeds for a favorable score.

### Label a question so disagreements can be reviewed

For each question preserve ID, document identity, question text, answerability, accepted aliases/semantic facts, minimal supporting spans, allowed evidence alternatives and a scoring note. An unavailable warranty has `answerable=false`, empty gold answer and an explanation that no supplied passage establishes the duration. “How do I start it?” expects both the action and the two-second duration. A response that names the button but omits duration can be partially informative while failing the declared complete-answer criterion.

Two independent reviews are useful for ambiguous labels; retain disagreement and adjudication. A candidate public source such as [LongBench](source:longbench) supplies task examples and its own definitions, but a selected subset is not the full benchmark. Check license, fields, document provenance and permitted use, declare IDs before evaluation, and name the subset honestly.

### Token budgeting includes every wrapper

Let $I$ be instruction/template/question overhead, $D$ the passage token contribution and $O$ the output allowance. We require both $I+D\le P_{\max}$ and $I+D+O\le C$. These bounds are distinct: the application may deliberately permit only 2048 prompt tokens even if the runtime supports more. Tokenization at joins can change counts, so the real implementation must tokenize the **complete rendered prompt**, not simply add precomputed individual counts. The fixture below uses whitespace counting solely to make the arithmetic visible.

With overhead 3, the first passage has four words and the second two. Prompt budget 8 allows at most five passage words. The first fits: $3+4=7$; adding the second would give 9 and is rejected. With output allowance 4 and total capacity 12, the retained prompt also satisfies $7+4=11\le12$. Rejection is recorded, not hidden truncation. For an oracle-evidence weight study, ensure the labeled evidence was supplied; otherwise a refusal can be caused by packing rather than compression.

[Download the complete reference](examples/06_splits.py)

```python
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
```

From the scratch directory containing that file, run:

```bash
python3 06_splits.py
```

Expected checked output (JSON formatting may differ):

```json
{
  "disjoint": true,
  "packed": [
    "charge takes three hours"
  ],
  "seed": 17
}
```

The greedy packer considers passages in a fixed order and can skip an overlarge passage while admitting a later small one. That is an explicit policy, not an optimal evidence selector. Actual phase 01 must retain selected IDs, dropped IDs, complete prompt hash, token count and a reason such as `input_budget`. A prompt exceeding capacity should fail before inference rather than relying on a backend's implicit left/right truncation.

[[practice:splits-transfer]]

### Make the first experiment identifiable

Freeze a manifest of canonical documents, role membership, question IDs, labels, tokenizer/template identity and packing policy. Keep final labels out of normal tuning reports. Save a tiny development pilot separately from the main declared comparison; do not quietly delete failed examples. The first controlled study supplies the same passages to every weight variant. Retrieval enters later through a separately versioned interface, since a missing relevant passage and a corrupted generation are different failure causes.

We now have fixed inputs and a correctness target. Next we establish the floating-point behavior that compressed weights will be compared against, beginning with exactly how logits become an answer token.
