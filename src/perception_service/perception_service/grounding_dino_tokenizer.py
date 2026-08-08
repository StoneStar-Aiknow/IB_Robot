"""Small BERT WordPiece tokenizer for fixed-sequence Grounding-DINO bundles."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np


class BertWordPieceTokenizer:
    """Build the integer and attention tensors expected by exported GDINO graphs."""

    def __init__(self, vocab_path: str | Path, *, sequence_length: int):
        path = Path(vocab_path).expanduser().resolve(strict=True)
        tokens = path.read_text(encoding="utf-8").splitlines()
        self._vocab = {token: index for index, token in enumerate(tokens)}
        self._tokens = tokens
        self.sequence_length = int(sequence_length)
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        required = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", ".", "?")
        missing = [token for token in required if token not in self._vocab]
        if missing:
            raise ValueError(f"BERT vocabulary is missing required tokens: {missing}")

    def encode(self, text: str) -> dict[str, np.ndarray]:
        caption = text.lower().strip()
        if not caption:
            raise ValueError("text_prompt must not be empty")
        if not caption.endswith("."):
            caption += "."
        pieces = ["[CLS]"]
        for token in self._basic_tokens(caption):
            pieces.extend(self._wordpiece(token))
        pieces.append("[SEP]")
        if len(pieces) > self.sequence_length:
            raise ValueError(
                f"text_prompt needs {len(pieces)} BERT tokens but this model supports {self.sequence_length}"
            )
        valid_length = len(pieces)
        pieces.extend(["[PAD]"] * (self.sequence_length - valid_length))
        input_ids = np.asarray([[self._vocab[piece] for piece in pieces]], dtype=np.int64)
        token_mask = np.zeros((1, self.sequence_length), dtype=np.int64)
        token_mask[:, :valid_length] = 1
        attention, position_ids = self._build_attention(input_ids)
        return {
            "gdino.input_ids": input_ids,
            "gdino.token_type_ids": np.zeros_like(input_ids),
            "gdino.position_ids": position_ids,
            "gdino.text_self_attention_masks": attention,
            "gdino.text_token_mask": token_mask,
            "text_token_mask": token_mask,
        }

    def decode_probability_mask(self, input_ids: np.ndarray, probabilities: np.ndarray, threshold: float) -> str:
        ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)
        probability_values = np.asarray(probabilities).reshape(-1)
        if probability_values.size < ids.size:
            raise ValueError(f"probability vector has {probability_values.size} values for {ids.size} input tokens")
        selected = probability_values[: ids.size] > threshold
        special_ids = {self._vocab[token] for token in ("[PAD]", "[CLS]", "[SEP]", ".", "?")}
        tokens = [
            self._tokens[int(token_id)]
            for token_id, keep in zip(ids, selected, strict=True)
            if keep and token_id not in special_ids
        ]
        words: list[str] = []
        for token in tokens:
            if token.startswith("##") and words:
                words[-1] += token[2:]
            else:
                words.append(token)
        return " ".join(words)

    def _build_attention(self, input_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sequence_length = input_ids.shape[1]
        attention = np.eye(sequence_length, dtype=np.int64)[None]
        position_ids = np.zeros((1, sequence_length), dtype=np.int64)
        special_ids = {self._vocab[token] for token in ("[CLS]", "[SEP]", ".", "?")}
        previous = 0
        for column in np.flatnonzero(np.isin(input_ids[0], tuple(special_ids))):
            column = int(column)
            if column == 0 or column == sequence_length - 1:
                attention[0, column, column] = 1
                position_ids[0, column] = 0
            else:
                attention[0, previous + 1 : column + 1, previous + 1 : column + 1] = 1
                position_ids[0, previous + 1 : column + 1] = np.arange(column - previous, dtype=np.int64)
            previous = column
        return attention, position_ids

    def _wordpiece(self, token: str) -> list[str]:
        if token in self._vocab:
            return [token]
        if len(token) > 100:
            return ["[UNK]"]
        pieces: list[str] = []
        start = 0
        while start < len(token):
            end = len(token)
            match = None
            while start < end:
                candidate = token[start:end]
                if start:
                    candidate = "##" + candidate
                if candidate in self._vocab:
                    match = candidate
                    break
                end -= 1
            if match is None:
                return ["[UNK]"]
            pieces.append(match)
            start = end
        return pieces

    @classmethod
    def _basic_tokens(cls, text: str) -> list[str]:
        cleaned: list[str] = []
        for character in text:
            code = ord(character)
            category = unicodedata.category(character)
            if code in {0, 0xFFFD} or category in {"Cc", "Cf"}:
                continue
            if character.isspace():
                cleaned.append(" ")
            elif cls._is_chinese(code):
                cleaned.extend((" ", character, " "))
            else:
                cleaned.append(character)
        tokens: list[str] = []
        for token in "".join(cleaned).split():
            normalized = "".join(
                character
                for character in unicodedata.normalize("NFD", token)
                if unicodedata.category(character) != "Mn"
            )
            current: list[str] = []
            for character in normalized:
                if unicodedata.category(character).startswith("P"):
                    if current:
                        tokens.append("".join(current))
                        current.clear()
                    tokens.append(character)
                else:
                    current.append(character)
            if current:
                tokens.append("".join(current))
        return tokens

    @staticmethod
    def _is_chinese(code: int) -> bool:
        return any(
            start <= code <= end
            for start, end in (
                (0x4E00, 0x9FFF),
                (0x3400, 0x4DBF),
                (0x20000, 0x2A6DF),
                (0x2A700, 0x2B73F),
                (0x2B740, 0x2B81F),
                (0x2B820, 0x2CEAF),
                (0xF900, 0xFAFF),
                (0x2F800, 0x2FA1F),
            )
        )
