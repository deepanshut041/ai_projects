"""Educational document-bounded byte BPE. Not a GPT-compatible tokenizer.

Bytes 0..255; PAD/BOS/EOS/SEP 256..259; merge IDs begin at 260.
No normalization, regex pre-tokenization, or special-spelling interpretation.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

PAD, BOS, EOS, SEP = 256, 257, 258, 259
SPECIALS = {"PAD": PAD, "BOS": BOS, "EOS": EOS, "SEP": SEP}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class ByteBPE:
    def __init__(self, merges=(), train_hash="untrained"):
        self.merges = [tuple(pair) for pair in merges]
        self.train_hash = train_hash
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.ranks = {}
        for rank, pair in enumerate(self.merges):
            new_id = 260 + rank
            if pair in self.ranks or any(i not in self.vocab for i in pair):
                raise ValueError("Duplicate, forward, special or invalid merge")
            self.ranks[pair] = rank
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

    @classmethod
    def train(cls, documents, cap=272):
        if cap < 260:
            raise ValueError("Vocabulary cap must include structural specials")
        documents = list(documents)
        streams = [list(text.encode("utf-8")) for text in documents]
        pairs = []
        while 260 + len(pairs) < cap:
            counts = Counter(pair for ids in streams for pair in zip(ids, ids[1:]))
            if not counts or max(counts.values()) < 2:
                break
            pair = min(counts, key=lambda p: (-counts[p], p))
            new_id = 260 + len(pairs)
            pairs.append(pair)
            streams = [merge(ids, pair, new_id) for ids in streams]
        return cls(pairs, digest(documents))

    @property
    def size(self):
        return 260 + len(self.merges)

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        # Choose the lowest learned rank currently present, not frequency.
        while len(ids) >= 2:
            candidates = [(self.ranks[pair], pair) for pair in zip(ids, ids[1:])
                          if pair in self.ranks]
            if not candidates:
                break
            rank, pair = min(candidates)
            ids = merge(ids, pair, 260 + rank)
        return ids

    def decode_bytes(self, ids):
        try:
            return b"".join(self.vocab[i] for i in ids)
        except KeyError as error:
            raise ValueError("Decode content IDs only; remove structural tokens explicitly") from error

    def decode(self, ids, errors="strict"):
        return self.decode_bytes(ids).decode("utf-8", errors=errors)

    def payload(self):
        return {"schema": 1, "algorithm": "document-byte-bpe-v1", "specials": SPECIALS,
                "normalization": "none", "boundaries": "documents-and-structural-specials",
                "tie_break": "lexicographic-token-pair", "training_sha256": self.train_hash,
                "merges": [list(pair) for pair in self.merges]}

    @property
    def identity(self):
        return digest(self.payload())

    def save(self, path):
        Path(path).write_bytes(canonical({"payload": self.payload(), "sha256": self.identity}))

    @classmethod
    def load(cls, path):
        saved = json.loads(Path(path).read_text())
        p = saved["payload"]
        if saved["sha256"] != digest(p):
            raise ValueError("Tokenizer checksum mismatch")
        tokenizer = cls(p["merges"], p["training_sha256"])
        if tokenizer.payload() != p:
            raise ValueError("Unsupported tokenizer schema or policy")
        return tokenizer


def selfcheck():
    import tempfile
    docs = ["abab", "abab"]
    tok = ByteBPE.train(docs, cap=262)
    assert tok.merges == [(97, 98), (260, 260)]
    assert tok.encode("abab") == [261]
    for text in ["", "aaaaa", "café", "你好🙂", "<BOS>", "e\u0301", "é"]:
        assert tok.decode(tok.encode(text)) == text
    assert ByteBPE.train(["a", "b"]).merges == []  # No document seam merge.
    assert ByteBPE.train(["abab", "cdcd"], 261).merges == [(97, 98)]
    assert ByteBPE.train(docs, 262).identity == tok.identity
    # Frozen rank 0 wins even when the new text makes rank 1 more frequent.
    ranked = ByteBPE([(97, 98), (98, 99)])
    assert ranked.encode("abc") == [260, 99]
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "tokenizer.json"
        tok.save(path)
        assert ByteBPE.load(path).identity == tok.identity
    print(json.dumps({"merges": [list(p) for p in tok.merges], "encoded": tok.encode("abab"),
                      "unicode_roundtrip": True, "boundary_safe": True, "rank_safe": True,
                      "size": tok.size}))


if __name__ == "__main__":
    selfcheck()
