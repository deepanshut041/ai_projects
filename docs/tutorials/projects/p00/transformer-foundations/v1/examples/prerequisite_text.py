"""Inspect character IDs, UTF-8 bytes and original-text coordinates."""

from dataclasses import dataclass
import hashlib
import json

PAD, UNK = 0, 1


def utf8_ids(text):
    return list(text.encode("utf-8"))


def utf8_text(ids):
    values = list(ids)
    if any(type(value) is not int or not 0 <= value < 256 for value in values):
        raise ValueError("Byte IDs must be integers in [0,255]")
    return bytes(values).decode("utf-8", errors="strict")


def byte_character_offsets(text):
    """Return one original character interval for each emitted UTF-8 byte."""
    return [(index, index + 1) for index, character in enumerate(text)
            for _ in character.encode("utf-8")]


@dataclass(frozen=True)
class CharacterVocabulary:
    characters: tuple[str, ...]

    def __post_init__(self):
        if any(not isinstance(char, str) or len(char) != 1 for char in self.characters):
            raise ValueError("Each vocabulary entry must contain one character")
        if len(set(self.characters)) != len(self.characters):
            raise ValueError("Vocabulary entries must be unique")

    @classmethod
    def fit(cls, training_texts):
        return cls(tuple(sorted(set("".join(training_texts)))))

    def encode(self, text):
        mapping = {character: index + 2 for index, character in enumerate(self.characters)}
        return [mapping.get(character, UNK) for character in text]

    def decode(self, ids):
        pieces = []
        for value in ids:
            if type(value) is not int or not 0 <= value < len(self.characters) + 2:
                raise ValueError("Character ID is outside this vocabulary")
            if value == UNK:
                pieces.append("<UNK>")
            elif value != PAD:
                pieces.append(self.characters[value - 2])
        return "".join(pieces)

    def to_dict(self):
        return {"kind": "character-vocabulary-v1", "pad": PAD, "unknown": UNK,
                "characters": list(self.characters)}

    @classmethod
    def from_dict(cls, value):
        if value.get("kind") != "character-vocabulary-v1" or value.get("pad") != PAD or value.get("unknown") != UNK:
            raise ValueError("Incompatible character-vocabulary contract")
        return cls(tuple(value["characters"]))

    def fingerprint(self):
        raw = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()


def checked_results():
    from byte_bpe import ByteBPE
    text = "café"
    vocabulary = CharacterVocabulary.fit(["cat", "car"])
    assert vocabulary.characters == ("a", "c", "r", "t")
    bpe = ByteBPE.train(["abab", "abab"], cap=262)
    bpe_ids = bpe.encode("abab")
    assert bpe_ids == [261] and bpe.decode(bpe_ids) == "abab"
    assert utf8_text(utf8_ids(text)) == text
    return {"utf8": utf8_ids(text), "characters": len(text), "bytes": len(utf8_ids(text)),
            "byte_character_offsets": byte_character_offsets(text),
            "cat_ids": vocabulary.encode("cat"), "unseen_ids": vocabulary.encode("cab"),
            "unseen_decode": vocabulary.decode(vocabulary.encode("cab")),
            "original_span": "The café is open."[4:8], "merged_abab": bpe_ids}


if __name__ == "__main__":
    print(json.dumps(checked_results(), ensure_ascii=False))
