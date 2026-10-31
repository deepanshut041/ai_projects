"""Byte-pair tokens, masked-token targets and character-aligned QA windows."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import re

import torch


class ByteBPETokenizer:
    """Train byte merges within Unicode word, punctuation and whitespace chunks."""

    bos_id, pad_id, eos_id, unk_id, mask_id = range(5)
    special_ids = (0, 1, 2, 3, 4)
    pattern = re.compile(r"\s*\w+|\s*[^\w\s]|\s+", re.UNICODE)

    def __init__(self, merges: list[tuple[tuple[int, ...], tuple[int, ...]]] | None = None):
        self.merges = list(merges or [])
        self.symbols = [tuple([byte]) for byte in range(256)]
        for left, right in self.merges:
            symbol = left + right
            if symbol not in self.symbols:
                self.symbols.append(symbol)
        self.ids = {symbol: index + 5 for index, symbol in enumerate(self.symbols)}

    @property
    def vocab_size(self) -> int:
        return len(self.symbols) + 5

    @staticmethod
    def _merge(sequence, pair):
        result = []
        index = 0
        while index < len(sequence):
            if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                result.append(sequence[index] + sequence[index + 1])
                index += 2
            else:
                result.append(sequence[index])
                index += 1
        return result

    @classmethod
    def train(cls, texts: list[str], merge_count: int = 40) -> "ByteBPETokenizer":
        chunks = Counter(match.group().encode("utf-8") for text in texts for match in cls.pattern.finditer(text))
        sequences = {chunk: [tuple([byte]) for byte in chunk] for chunk in chunks}
        merges = []
        for _ in range(merge_count):
            counts = Counter()
            for chunk, sequence in sequences.items():
                for pair in zip(sequence, sequence[1:]):
                    counts[pair] += chunks[chunk]
            if not counts:
                break
            pair = min(counts, key=lambda candidate: (-counts[candidate], candidate))
            merges.append(pair)
            sequences = {chunk: cls._merge(sequence, pair) for chunk, sequence in sequences.items()}
        return cls(merges)

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        ids, offsets = [], []
        for match in self.pattern.finditer(text):
            byte_positions = []
            for index, character in enumerate(match.group(), start=match.start()):
                byte_positions.extend([(index, index + 1)] * len(character.encode("utf-8")))
            symbols = [tuple([byte]) for byte in match.group().encode("utf-8")]
            for pair in self.merges:
                symbols = self._merge(symbols, pair)
            cursor = 0
            for symbol in symbols:
                start = byte_positions[cursor][0]
                end = byte_positions[cursor + len(symbol) - 1][1]
                cursor += len(symbol)
                while start < end and text[start].isspace():
                    start += 1
                ids.append(self.ids[symbol])
                offsets.append((start, end))
        return ids, offsets

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids, _ = self.encode_with_offsets(text)
        return [self.bos_id, *ids, self.eos_id] if add_special_tokens else ids

    def decode(self, ids: list[int]) -> str:
        values = [byte for token in ids if token >= 5 for byte in self.symbols[token - 5]]
        return bytes(values).decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        return {"kind": "course-byte-bpe-v1", "merges": [[list(left), list(right)] for left, right in self.merges]}

    @classmethod
    def from_dict(cls, state: dict) -> "ByteBPETokenizer":
        if state.get("kind") != "course-byte-bpe-v1":
            raise ValueError("Tokenizer format does not match")
        return cls([(tuple(left), tuple(right)) for left, right in state["merges"]])

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def pad_sequences(sequences: list[list[int]], pad_id: int = 1) -> dict[str, torch.Tensor]:
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("Every sequence must contain at least one token")
    length = max(map(len, sequences))
    ids = torch.full((len(sequences), length), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        ids[row, : len(sequence)] = torch.tensor(sequence)
        mask[row, : len(sequence)] = True
    return {"input_ids": ids, "attention_mask": mask}


def pack_documents(documents: list[str], tokenizer: ByteBPETokenizer, context_length=64) -> list[list[int]]:
    """Retain every token while grouping an already-split corpus into bounded chunks."""
    if context_length < 2:
        raise ValueError("The context must fit at least two tokens")
    stream = [token for document in documents for token in tokenizer.encode(document)]
    return [stream[start : start + context_length] for start in range(0, len(stream), context_length)]


def corrupt_mlm(
    input_ids: torch.Tensor,
    tokenizer: ByteBPETokenizer,
    generator: torch.Generator,
    probability: float = 0.15,
    *,
    selection_draws: torch.Tensor | None = None,
    replacement_draws: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select ordinary tokens; preserve original labels through all 80/10/10 branches."""
    if input_ids.device.type != "cpu" or not 0 <= probability <= 1:
        raise ValueError("The reference collator uses CPU tensors and a probability in [0,1]")
    selected_draws = selection_draws if selection_draws is not None else torch.rand(input_ids.shape, generator=generator)
    branch_draws = replacement_draws if replacement_draws is not None else torch.rand(input_ids.shape, generator=generator)
    if selected_draws.shape != input_ids.shape or branch_draws.shape != input_ids.shape:
        raise ValueError("Controlled draws must match the token matrix")
    eligible = ~torch.isin(input_ids, torch.tensor(tokenizer.special_ids))
    selected = eligible & (selected_draws < probability)
    labels = torch.full_like(input_ids, -100)
    labels[selected] = input_ids[selected]
    corrupted = input_ids.clone()
    corrupted[selected & (branch_draws < 0.8)] = tokenizer.mask_id
    random_positions = selected & (branch_draws >= 0.8) & (branch_draws < 0.9)
    ordinary = torch.arange(5, tokenizer.vocab_size)
    draws = torch.randint(len(ordinary), input_ids.shape, generator=generator)
    corrupted[random_positions] = ordinary[draws[random_positions]]
    return corrupted, labels


def mlm_batch(sequences, tokenizer, generator, probability=0.15):
    batch = pad_sequences(sequences, tokenizer.pad_id)
    batch["input_ids"], batch["labels"] = corrupt_mlm(batch["input_ids"], tokenizer, generator, probability)
    return batch


@dataclass
class QAFeature:
    input_ids: list[int]
    attention_mask: list[bool]
    offsets: list[tuple[int, int] | None]
    start_position: int
    end_position: int
    window_start: int


def character_span_to_tokens(offsets, start_char: int, end_char: int) -> tuple[int, int] | None:
    """Map a nonempty half-open character interval into an inclusive token span."""
    if start_char < 0 or end_char <= start_char:
        return None
    candidates = [(index, span) for index, span in enumerate(offsets) if span is not None and span[0] < span[1]]
    if not candidates or start_char < candidates[0][1][0] or end_char > candidates[-1][1][1]:
        return None
    starts = [index for index, (left, right) in candidates if left <= start_char < right]
    ends = [index for index, (left, right) in candidates if left < end_char <= right]
    if not starts or not ends or starts[0] > ends[-1]:
        return None
    return starts[0], ends[-1]


def qa_features(
    question: str,
    context: str,
    tokenizer: ByteBPETokenizer,
    *,
    context_length: int = 64,
    overlap: int = 12,
    answer_start: int | None = None,
    answer_text: str | None = None,
) -> list[QAFeature]:
    question_ids = tokenizer.encode(question.strip(), add_special_tokens=False)
    context_ids, context_offsets = tokenizer.encode_with_offsets(context)
    budget = context_length - len(question_ids) - 4
    if budget <= 0:
        raise ValueError("Question and four special tokens leave no context space")
    if not 0 <= overlap < budget:
        raise ValueError("Window overlap must be smaller than the context token budget")
    if answer_start is not None:
        if not answer_text or context[answer_start : answer_start + len(answer_text)] != answer_text:
            raise ValueError("Answer text must exactly match its half-open character interval")
    elif answer_text is not None:
        raise ValueError("An answer needs both text and character start")
    features = []
    for window_start in range(0, max(1, len(context_ids)), budget - overlap):
        part = context_ids[window_start : window_start + budget]
        ids = [tokenizer.bos_id, *question_ids, tokenizer.eos_id, tokenizer.eos_id, *part, tokenizer.eos_id]
        prefix = len(question_ids) + 3
        offsets = [None] * prefix + context_offsets[window_start : window_start + budget] + [None]
        real_length = len(ids)
        ids += [tokenizer.pad_id] * (context_length - real_length)
        offsets += [None] * (context_length - len(offsets))
        span = None if answer_start is None else character_span_to_tokens(offsets, answer_start, answer_start + len(answer_text))
        start, end = span if span is not None else (-100, -100)
        features.append(QAFeature(ids, [index < real_length for index in range(context_length)], offsets, start, end, window_start))
        if window_start + budget >= len(context_ids):
            break
    return features


def qa_batch(features: list[QAFeature]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([feature.input_ids for feature in features]),
        "attention_mask": torch.tensor([feature.attention_mask for feature in features]),
        "start_positions": torch.tensor([feature.start_position for feature in features]),
        "end_positions": torch.tensor([feature.end_position for feature in features]),
    }


def best_span(start_logits, end_logits, offsets, context: str, max_answer_tokens: int = 16) -> dict | None:
    """Search context-only, ordered spans using start plus end logit scores."""
    if max_answer_tokens < 1:
        raise ValueError("Answer length limit must be positive")
    start_logits, end_logits = torch.as_tensor(start_logits).flatten(), torch.as_tensor(end_logits).flatten()
    if len(offsets) != len(start_logits) or start_logits.shape != end_logits.shape:
        raise ValueError("Logits and offsets must describe the same sequence")
    best = None
    for start, left_span in enumerate(offsets):
        if left_span is None or left_span[0] >= left_span[1]:
            continue
        for end in range(start, min(len(offsets), start + max_answer_tokens)):
            right_span = offsets[end]
            if right_span is None or right_span[0] >= right_span[1]:
                continue
            if any(span is None for span in offsets[start : end + 1]):
                continue
            score = float(start_logits[start] + end_logits[end])
            if best is None or score > best["score"]:
                left, right = left_span[0], right_span[1]
                best = {"start_token": start, "end_token": end, "start_char": left, "end_char": right, "text": context[left:right], "score": score}
    return best


def normalize_answer(text: str) -> str:
    import string

    text = "".join(character for character in text.lower() if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_metrics(prediction: str, gold_answers: list[str]) -> dict[str, float]:
    if not gold_answers:
        raise ValueError("At least one gold answer is required")
    normalized = normalize_answer(prediction)
    predicted_tokens = normalized.split()
    exact, f1 = 0.0, 0.0
    for gold in gold_answers:
        gold = normalize_answer(gold)
        exact = max(exact, float(normalized == gold))
        gold_tokens = gold.split()
        overlap = sum((Counter(predicted_tokens) & Counter(gold_tokens)).values())
        if not predicted_tokens or not gold_tokens:
            score = float(predicted_tokens == gold_tokens)
        elif overlap == 0:
            score = 0.0
        else:
            precision, recall = overlap / len(predicted_tokens), overlap / len(gold_tokens)
            score = 2 * precision * recall / (precision + recall)
        f1 = max(f1, score)
    return {"exact_match": exact, "f1": f1}
