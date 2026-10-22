## Lesson 3 — freeze the text and tokenizer

A language model can appear to improve because the evaluation text leaked into training. Fix the split before tokenization or windows. We will use one small, named corpus and keep its bytes, boundaries and tokenizer identity explicit.

### Choose a bounded text source

The worked corpus is Lewis Carroll's *Alice's Adventures in Wonderland*, Project Gutenberg ebook 11. The [catalog](source:alice) labels its copyright status public domain in the USA and provides its usage terms. Record the catalog, retrieval date and terms with the acquisition instructions; a hash records which bytes you used, not usage permission. This is a one-book educational task. Its writing style and vocabulary cannot represent general language ability.

Use chapters I–IX for training, X for validation, and XI–XII for a final untouched test. Validation guides debugging/choices. Test is opened only after those choices are fixed. Splitting chapters keeps adjacent windows of a chapter in one partition; randomly splitting already overlapping windows would allow near-duplicate context across partitions. Exact-document deduplication is necessary but does not prove an absence of near duplicates or shared phrases. Record these limits.

[[diagram:data-flow]]

The raw file is checked against SHA-256 `01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e`. SHA-256 maps bytes to a 256-bit digest. If any byte differs, stop and inspect the source/parser before declaring a new data version. Do not edit the expected hash merely to silence the check. We retain the selected chapter text's exact CRLF/newline and Unicode content; no normalization is hidden in preparation.

### Reuse a frozen tokenizer source

Save [gpt_tokenizer.py](examples/gpt_tokenizer.py) as `gpt_tokenizer.py` in the learner root. It is the byte-for-byte P00 `ByteBPE` educational reference snapshot, whose source digest is recorded in the [sources record](research.md). This deliberate reuse is why we do not write BPE again in P01.

Run `python gpt_tokenizer.py`. Expect deterministic merges `[[97,98],[260,260]]`, encoded `abab` as `[261]`, and true Unicode/boundary/rank checks. Open the file at `encode`, `payload` and `load`: encoding uses learned merge rank; payload records boundaries/special IDs; loading rejects altered or unsupported policies. Confirm these match your P00 handoff before replacing the reference with your own implementation.

The baseline uses `ByteBPE.train(train_documents, cap=260)`, so it learns no merges: 256 bytes plus four structural IDs. Even this artifact records the training-document hash. The same API lets you make a separate BPE experiment with cap=512 later. Fit merges on the training chapters only; freeze them before encoding validation/test. Set model V to the artifact's **actual** size, which may be below the cap.

[[practice:tokenizer]]

### Write acquisition and identity checks

Create `gpt_data.py`. First add its imports, source pin and document preparation/load functions below. Later in the lesson route we append windows, sampling and the CLI. The functions are callable already; the full file download includes the later parts too.

<!-- file:gpt_data.py mode:write -->
```python
"""Split documents first; freeze tokenizer; then make bounded next-token windows."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import torch
from gpt_tokenizer import ByteBPE, BOS, EOS, PAD, digest

URL = "https://www.gutenberg.org/ebooks/11.txt.utf-8"
RAW_SHA = "01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e"


def alice_splits(raw):
    if hashlib.sha256(raw).hexdigest() != RAW_SHA:
        raise ValueError("Source bytes changed: inspect/re-pin explicitly; do not train")
    text = raw.decode("utf-8-sig")
    body = text.split("*** END OF THE PROJECT GUTENBERG EBOOK", 1)[0]
    starts = list(re.finditer(r"^CHAPTER [IVX]+\.\r?$", body, re.M))
    if len(starts) != 12:
        raise ValueError("Expected twelve chapter headings; parser needs review")
    documents = [body[m.start():starts[i+1].start() if i+1 < 12 else len(body)]
                 for i, m in enumerate(starts)]
    # Keep exact chapter text including CRLF; normalization is part of identity.
    return {"train": documents[:9], "validation": documents[9:10],
            "test": documents[10:]}


def prepare(directory, cap=260):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("Use a fresh data directory; never replace a frozen split")
    raw = urllib.request.urlopen(URL, timeout=30).read()
    splits = alice_splits(raw)
    tok = ByteBPE.train(splits["train"], cap=cap)
    directory.mkdir(parents=True)
    (directory / "alice.txt").write_bytes(raw)
    (directory / "splits.json").write_text(json.dumps(splits, ensure_ascii=False))
    tok.save(directory / "tokenizer.json")
    manifest = {"schema": 1, "url": URL, "raw_sha256": RAW_SHA,
                "catalog": "https://www.gutenberg.org/ebooks/11",
                "rights_label": "Public domain in the USA (catalog); retain source terms",
                "partition": "chapters I-IX / X / XI-XII; exact UTF-8; no normalization",
                "split_hashes": {k: digest(v) for k, v in splits.items()},
                "tokenizer_sha256": tok.identity, "vocab": tok.size}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_data(directory):
    directory = Path(directory)
    splits = json.loads((directory / "splits.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    tok = ByteBPE.load(directory / "tokenizer.json")
    if manifest["schema"] != 1 or tok.identity != manifest["tokenizer_sha256"]:
        raise ValueError("Data/tokenizer identity mismatch")
    if {k: digest(v) for k, v in splits.items()} != manifest["split_hashes"]:
        raise ValueError("Split bytes changed")
    hashes = [set(digest(doc) for doc in splits[k]) for k in ("train", "validation", "test")]
    if any(hashes[i] & hashes[j] for i in range(3) for j in range(i)):
        raise ValueError("An exact document crosses splits")
    if tok.train_hash != digest(splits["train"]):
        raise ValueError("Tokenizer was fitted on different documents")
    return splits, tok, manifest
```

`alice_splits` verifies bytes before decoding and finds chapter headings anchored to whole lines. `prepare` refuses an existing directory so a new data preparation cannot overwrite a frozen version. `load_data` checks the saved tokenizer, every split hash and the tokenizer's training split. It also rejects exact duplicate documents across partitions.

At this stage call the function directly; `data/alice-byte` must not exist:

```bash
python -c "from gpt_data import prepare; import json; print(json.dumps(prepare('data/alice-byte'), indent=2))"
```

Expect V=260 and three named split hashes. Keep the resulting manifest with the run's provenance. If the network is unavailable, obtain the same file separately, verify its raw hash, then call `alice_splits(raw_bytes)`; do not substitute a different text under the same identity. Raw data stays ignored.

Check a non-ASCII input:

```bash
python -c "from gpt_data import load_data; s,t,m=load_data('data/alice-byte'); text='café 🙂'; print(t.encode(text)); assert t.decode(t.encode(text))==text"
```

The token count exceeds the number of displayed characters because UTF-8 uses several bytes for some characters. This matters later: byte-token perplexity cannot be compared directly with a BPE-token perplexity from a different tokenizer. We will preserve the denominator and compare the same raw held-out text with an explicit boundary/context policy.

### A controlled BPE follow-up

After the byte baseline and window lesson work, use a new data directory: `python gpt_data.py data/alice-bpe --vocab 512`. Read the resulting manifest's actual vocabulary size, put that V in a new configuration and use a new run directory. Never load byte-model weights/optimizer state through strict resume into a different vocabulary. Keep the same chapter split and report tokens per raw byte before comparing model loss. This experiment tests a changed representation and sequence length; it is not GPT-2 tokenizer compatibility.

### Your data record

Create `docs/data.md` with source/terms, raw hash, parser version, split chapter IDs/hashes, exact normalization policy, tokenizer size/hash, and a statement that test chapters have not been used for tuning. Add a small round-trip table for plain ASCII, an emoji, empty text and a literal `<BOS>` spelling. The spelling must encode as ordinary bytes; only the pipeline inserts structural BOS.

The text identities are now fixed. The next lesson converts each chapter into input/label arrays without crossing a boundary or losing its last prediction.

[Continue to Lesson 4](lessons/04-windows.md) · [Sources and choices](research.md)

### Complete files from this lesson

[gpt_data.py](examples/gpt_data.py)
