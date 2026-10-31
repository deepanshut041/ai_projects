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


def windows(documents, tok, context):
    if context < 1:
        raise ValueError("Positive context required")
    xs, ys = [], []
    for document in documents:
        ids = [BOS] + tok.encode(document) + [EOS]
        # Each target occurs once. Reset context at each window and document.
        for start in range(0, len(ids)-1, context):
            x = ids[start:start+context]
            y = ids[start+1:start+context+1]
            x = x[:len(y)]
            xs.append(x + [PAD] * (context-len(x)))
            ys.append(y + [PAD] * (context-len(y)))
    if not xs:
        raise ValueError("At least one document required")
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


class Batches:
    """One explicit CPU generator; random windows with replacement, no workers."""
    def __init__(self, x, y, seed):
        self.x, self.y = x, y
        self.rng = torch.Generator().manual_seed(seed)
        self.draws = 0

    def next(self, size, device="cpu"):
        indices = torch.randint(len(self.x), (size,), generator=self.rng)
        self.draws += size
        return self.x[indices].to(device), self.y[indices].to(device)

    def state(self):
        return {"rng": self.rng.get_state(), "draws": self.draws}

    def restore(self, state):
        self.rng.set_state(state["rng"])
        self.draws = state["draws"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--vocab", type=int, default=260)
    args = parser.parse_args()
    print(json.dumps(prepare(args.directory, args.vocab), indent=2))
