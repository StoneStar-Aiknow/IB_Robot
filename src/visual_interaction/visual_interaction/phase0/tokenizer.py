"""Qwen2 byte-level BPE built from the tokenizer data files alone.

Counting prompt tokens needs a tokenizer, not a model and not the board demo:
``vocab.json``, ``merges.txt`` and ``added_tokens.json`` are data, and the
tokenization they describe is the published Qwen2 one — NFC, split on the added
tokens, the Qwen2 pre-tokenize regex, byte-level BPE. This module assembles
exactly that from ``tokenizers``, so a count runs on any host with no NPU, no
CANN and nothing imported from the demo directory.

Verified against the board tokenizer's own output: plain Chinese text, the
``<img>``/``<IMG_CONTEXT>``/``</img>`` block and the ``<|im_start|>`` chat
markers all give identical ids. ``added_tokens.json`` carries the ids, so the
added tokens are registered at those ids rather than appended — appending
would renumber them and silently change every count.
"""

from __future__ import annotations

import json
from pathlib import Path

# The published Qwen2 pre-tokenization pattern (HF ``Qwen2Tokenizer``).
PRETOKENIZE_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|"
    r"\s*[\r\n]+|\s+(?!\S)|\s+"
)
DATA_FILES = ("vocab.json", "merges.txt", "added_tokens.json")


class TokenizerDataError(FileNotFoundError):
    """The directory does not hold the three tokenizer data files."""


class PromptTokenizer:
    """Encodes text to ids the board's tokenizer would produce."""

    def __init__(self, directory: str | Path) -> None:
        from tokenizers import AddedToken, Regex, Tokenizer, models, normalizers, pre_tokenizers

        self.directory = Path(directory).resolve()
        missing = [name for name in DATA_FILES if not (self.directory / name).exists()]
        if missing:
            raise TokenizerDataError(f"{self.directory} is missing {', '.join(missing)}")

        vocab = json.loads((self.directory / "vocab.json").read_text(encoding="utf-8"))
        self.added: dict[str, int] = json.loads((self.directory / "added_tokens.json").read_text(encoding="utf-8"))
        merges = [
            tuple(line.split(" "))
            for line in (self.directory / "merges.txt").read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#version")
        ]

        # The added tokens go into the vocabulary at their declared ids before
        # the model is built, so registering them cannot renumber anything.
        full = dict(vocab)
        full.update(self.added)
        tokenizer = Tokenizer(models.BPE(full, merges))
        tokenizer.normalizer = normalizers.NFC()
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(PRETOKENIZE_REGEX), "isolated"),
                # The regex above has already split; byte-level only maps bytes.
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        tokenizer.add_special_tokens(
            # Longest first, so ``<img>`` cannot win over a longer marker.
            [AddedToken(name, special=True, normalized=False) for name in sorted(self.added, key=len, reverse=True)]
        )
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def data_files(self) -> dict[str, Path]:
        """The files a count depended on, for the report's asset digests."""
        return {f"tokenizer:{name}": self.directory / name for name in DATA_FILES}


__all__ = ["DATA_FILES", "PRETOKENIZE_REGEX", "PromptTokenizer", "TokenizerDataError"]
