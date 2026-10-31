"""Known coordinates, boundary failures and tokenizer identity checks."""

import unittest
import unicodedata

from byte_bpe import ByteBPE
from prerequisite_text import CharacterVocabulary, byte_character_offsets, utf8_ids, utf8_text


class TextContracts(unittest.TestCase):
    def test_exact_utf8_and_character_coordinates(self):
        self.assertEqual(utf8_ids("café"), [99, 97, 102, 195, 169])
        self.assertEqual(byte_character_offsets("café"), [(0, 1), (1, 2), (2, 3), (3, 4), (3, 4)])

    def test_unicode_roundtrips(self):
        for text in ("", "café", "🙂", "漢字", "cafe\u0301"):
            self.assertEqual(utf8_text(utf8_ids(text)), text)

    def test_invalid_or_partial_bytes_rejected(self):
        for invalid in ([256], [-1], [True], [1.0], ["99"]):
            with self.assertRaises(ValueError):
                utf8_text(invalid)
        with self.assertRaises(UnicodeDecodeError):
            utf8_text([195])

    def test_original_character_slice_differs_from_byte_slice(self):
        text = "The café is open."
        self.assertEqual(text[4:8], "café")
        self.assertEqual(text.encode()[4:9].decode(), "café")
        with self.assertRaises(UnicodeDecodeError):
            text.encode()[4:8].decode()

    def test_normalization_changes_coordinates(self):
        text = "cafe\u0301"
        normalized = unicodedata.normalize("NFC", text)
        self.assertEqual(normalized, "café")
        self.assertEqual((len(text), len(normalized)), (5, 4))

    def test_training_vocabulary_unknown_and_padding_roles(self):
        vocab = CharacterVocabulary.fit(["cat", "car"])
        self.assertEqual(vocab.encode("cat"), [3, 2, 5])
        self.assertEqual(vocab.encode("cab"), [3, 2, 1])
        self.assertEqual(vocab.decode([0, 3, 2, 5, 0]), "cat")
        self.assertEqual(vocab.decode([3, 2, 1]), "ca<UNK>")

    def test_artifact_reload_identity_and_mapping_order(self):
        vocab = CharacterVocabulary.fit(["cat", "car"])
        loaded = CharacterVocabulary.from_dict(vocab.to_dict())
        self.assertEqual(loaded.fingerprint(), vocab.fingerprint())
        self.assertEqual(loaded.encode("car"), [3, 2, 4])
        reordered = CharacterVocabulary(tuple(reversed(vocab.characters)))
        self.assertNotEqual(reordered.fingerprint(), vocab.fingerprint())
        self.assertNotEqual(reordered.encode("car"), vocab.encode("car"))

    def test_invalid_artifacts_and_ids_rejected(self):
        for entries in (("aa",), ("a", "a")):
            with self.assertRaises(ValueError):
                CharacterVocabulary(entries)
        vocab = CharacterVocabulary.fit(["a"])
        for values in ([True], [-1], [3]):
            with self.assertRaises(ValueError):
                vocab.decode(values)
        changed = vocab.to_dict() | {"pad": 9}
        with self.assertRaises(ValueError):
            CharacterVocabulary.from_dict(changed)

    def test_merges_stay_within_documents(self):
        combined = ByteBPE.train(["ab", "ab"], cap=261)
        separated = ByteBPE.train(["a", "b", "a", "b"], cap=261)
        self.assertEqual(combined.merges, [(97, 98)])
        self.assertEqual(separated.merges, [])

    def test_byte_bpe_falls_back_and_preserves_structural_specials(self):
        vocab = ByteBPE.train(["abab", "abab"], cap=262)
        self.assertEqual(vocab.encode("abab"), [261])
        self.assertEqual(vocab.decode(vocab.encode("café🙂")), "café🙂")
        self.assertEqual(vocab.encode("<PAD>"), list(b"<PAD>"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
