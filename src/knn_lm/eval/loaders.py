"""Load syntactic-contrast minimal pairs from BLiMP / Zorro / BIG-Bench."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

# Content words are everything except a short stoplist.
_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "am",
    "do",
    "does",
    "did",
    "done",
    "doing",
    "have",
    "has",
    "had",
    "having",
    "will",
    "would",
    "shall",
    "should",
    "can",
    "could",
    "may",
    "might",
    "must",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "from",
    "as",
    "and",
    "or",
    "but",
    "if",
    "then",
    "than",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "they",
    "them",
    "their",
    "he",
    "she",
    "his",
    "her",
    "him",
    "we",
    "you",
    "i",
    "me",
    "my",
    "your",
    "what",
    "who",
    "whom",
    "whose",
    "which",
    "where",
    "when",
    "why",
    "how",
    "not",
    "no",
    "yes",
}


def _content_words(sentence: str) -> list[str]:
    import re

    tokens = re.findall(r"[A-Za-z]+", sentence.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file already in the canonical schema."""
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    for item in out:
        if "content_words" not in item:
            item["content_words"] = _content_words(item["good"])
    return out


def load_blimp(path: str | Path) -> Iterator[dict]:
    """BLiMP json: {sentence_good, sentence_bad, ...}."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            yield {
                "good": ex["sentence_good"],
                "bad": ex["sentence_bad"],
                "content_words": _content_words(ex["sentence_good"]),
            }


def load_zorro(path: str | Path) -> Iterator[dict]:
    """Zorro txt format: alternating bad / good lines."""
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    for bad, good in zip(lines[0::2], lines[1::2], strict=False):
        yield {"good": good, "bad": bad, "content_words": _content_words(good)}


def load_bigbench(path: str | Path) -> Iterator[dict]:
    """BIG-Bench task json: {examples: [{target, ... target_scores: {tgt: 1/0}}]}.

    For minimal-pair tasks, scores 1 → grammatical, 0 → ungrammatical, and the
    input is the same for both.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for ex in data.get("examples", []):
        scores = ex.get("target_scores", {})
        good = next((t for t, s in scores.items() if s == 1), None)
        bad = next((t for t, s in scores.items() if s == 0), None)
        if good and bad:
            yield {"good": good, "bad": bad, "content_words": _content_words(good)}
