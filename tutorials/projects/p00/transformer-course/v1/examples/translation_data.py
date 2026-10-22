"""Separate vocabularies, aligned records and shifted bilingual batches."""

from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
import re
import unicodedata

import torch
from torch.nn.utils.rnn import pad_sequence


TRAIN_PAIRS = (
    ("i hate cats", "je déteste les chats"),
    ("i hate dogs", "je déteste les chiens"),
    ("you hate cats", "tu détestes les chats"),
    ("you hate dogs", "tu détestes les chiens"),
    ("i see cats", "je vois les chats"),
    ("you see dogs", "tu vois les chiens"),
)
VALIDATION_PAIRS = (
    ("i see dogs", "je vois les chiens"),
    ("you see cats", "tu vois les chats"),
)


def words(text):
    return re.findall(r"\w+|[^\w\s]", unicodedata.normalize("NFC", text).lower())


@dataclass
class Vocabulary:
    tokens: tuple

    @classmethod
    def fit(cls, texts, *, source=False):
        special = ("[PAD]", "[UNK]", "[BOS]", "[EOS]") if source else (
            "[UNK]", "[PAD]", "[BOS]", "[EOS]")
        observed = sorted({word for text in texts for word in words(text)} - set(special))
        return cls(tuple(special) + tuple(observed))

    @property
    def pad_token_id(self):
        return self.tokens.index("[PAD]")

    @property
    def bos_token_id(self):
        return self.tokens.index("[BOS]")

    @property
    def eos_token_id(self):
        return self.tokens.index("[EOS]")

    @property
    def vocab_size(self):
        return len(self.tokens)

    def encode(self, text, *, max_length=32):
        ids = {token: i for i, token in enumerate(self.tokens)}
        encoded = [self.bos_token_id] + [ids.get(word, ids["[UNK]"]) for word in words(text)]
        encoded += [self.eos_token_id]
        if len(encoded) > max_length:
            raise ValueError("pair exceeds the configured length; filter both sides together")
        return encoded

    def decode(self, ids):
        result = []
        for token_id in ids:
            token = self.tokens[int(token_id)]
            if token == "[EOS]":
                break
            if token not in {"[PAD]", "[BOS]"}:
                result.append(token)
        return " ".join(result)


def aligned_lines(english_path, french_path):
    """Never silently truncate a misaligned parallel corpus."""
    with Path(english_path).open(encoding="utf-8") as english, Path(french_path).open(encoding="utf-8") as french:
        for line_number, (src, tgt) in enumerate(zip_longest(english, french), start=1):
            if src is None or tgt is None:
                raise ValueError(f"parallel file length mismatch at line {line_number}")
            src, tgt = src.strip(), tgt.strip()
            if src and tgt:
                yield src, tgt


def encode_pairs(pairs, src_tokenizer, tgt_tokenizer, *, max_length=32):
    records = []
    for source, target in pairs:
        source_ids = src_tokenizer.encode(source)
        target_ids = tgt_tokenizer.encode(target)
        if not source_ids or len(target_ids) < 2:
            raise ValueError("source must be nonempty and target must contain start/end")
        if max(len(source_ids), len(target_ids)) > max_length:
            raise ValueError("pair exceeds maximum length; do not truncate either side silently")
        records.append({"src_ids": source_ids, "tgt_ids": target_ids})
    return records


def collate_translation(records, src_pad_id, tgt_pad_id):
    if not records:
        raise ValueError("a batch must contain at least one parallel example")
    if any(not r["src_ids"] or len(r["tgt_ids"]) < 2 for r in records):
        raise ValueError("source must be nonempty and target must contain start/end")
    src = pad_sequence([torch.tensor(r["src_ids"], dtype=torch.long) for r in records],
                       batch_first=True, padding_value=src_pad_id)
    target = pad_sequence([torch.tensor(r["tgt_ids"], dtype=torch.long) for r in records],
                          batch_first=True, padding_value=tgt_pad_id)
    decoder_input = target[:, :-1].clone()
    labels = target[:, 1:].clone()
    labels[labels == tgt_pad_id] = -100
    return {"src_input_ids": src, "src_pad_mask": src != src_pad_id,
            "tgt_input_ids": decoder_input, "tgt_pad_mask": decoder_input != tgt_pad_id,
            "tgt_outputs": labels}


def make_fixture():
    src = Vocabulary.fit((s for s, _ in TRAIN_PAIRS), source=True)
    tgt = Vocabulary.fit((t for _, t in TRAIN_PAIRS))
    batch = collate_translation(encode_pairs(TRAIN_PAIRS, src, tgt), src.pad_token_id, tgt.pad_token_id)
    return src, tgt, batch


def train_french_wordpiece(training_texts, output_path, *, vocab_size=32000):
    """Train/save an actual subword tokenizer on the training split only."""
    from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors, trainers
    from tokenizers.models import WordPiece
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFC(), normalizers.Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordPieceTrainer(vocab_size=vocab_size,
                                       special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"])
    tokenizer.train_from_iterator(training_texts, trainer=trainer)
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]", special_tokens=[("[BOS]", tokenizer.token_to_id("[BOS]")),
                                               ("[EOS]", tokenizer.token_to_id("[EOS]"))])
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(destination))
    return tokenizer


def main():
    records = [{"src_ids": [2, 4, 3], "tgt_ids": [2, 8, 9, 3]},
               {"src_ids": [2, 5, 6, 3], "tgt_ids": [2, 7, 3]}]
    for name, value in collate_translation(records, 0, 1).items():
        print(name, value.tolist())


if __name__ == "__main__":
    main()
