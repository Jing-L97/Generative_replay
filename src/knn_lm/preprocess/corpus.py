"""Chunk a raw text corpus into token / phrase / sequence units for datastore build."""

from __future__ import annotations

import re
from typing import Literal

from transformers import PreTrainedTokenizerBase

Granularity = Literal["token", "phrase", "sequence"]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_PHRASE_SPLIT = re.compile(r"(?<=[,;:—])\s+|(?<=[.!?])\s+")


def _split_units(text: str, granularity: Granularity) -> list[str]:
    if granularity == "sequence":
        units = _SENT_SPLIT.split(text)
    elif granularity == "phrase":
        units = _PHRASE_SPLIT.split(text)
    else:  # token
        units = [text]
    return [u.strip() for u in units if u.strip()]


def chunk_corpus(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    granularity: Granularity = "sequence",
    tau: int = 3,
) -> list[list[int]]:
    """Return a list of token-id chunks for `text`.

    For token-granularity, this returns sliding windows of length `tau`. For
    phrase / sequence, each unit is tokenized and truncated to `tau` tokens.
    """
    if tau < 1:
        raise ValueError(f"tau must be >= 1, got {tau}")

    chunks: list[list[int]] = []
    if granularity == "token":
        all_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        # tau-length sliding windows; the last position is the value, prior tokens are context.
        for end in range(1, len(all_ids) + 1):
            start = max(0, end - tau)
            window = all_ids[start:end]
            if len(window) >= 2:  # need at least one context token + one value token
                chunks.append(window)
        return chunks

    for unit in _split_units(text, granularity):
        ids = tokenizer(unit, add_special_tokens=False)["input_ids"]
        if len(ids) >= 2:
            chunks.append(ids[:tau] if len(ids) > tau else ids)
    return chunks


def chunk_corpus_file(
    path: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    granularity: Granularity = "sequence",
    tau: int = 3,
) -> list[list[int]]:
    """Read a UTF-8 text file and chunk it. Lines are joined into one document."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return chunk_corpus(text, tokenizer, granularity=granularity, tau=tau)
