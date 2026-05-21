"""Split eval items into high- vs low-frequency strata.

Manuscript definition (Section 2.2):
    high-frequency: mean content-word frequency f > 10^4
    low-frequency : mean content-word frequency f < 10^3
    (items in between are dropped)
"""

from __future__ import annotations

import json
from pathlib import Path

HIGH_THRESHOLD = 1e4
LOW_THRESHOLD = 1e3


def load_freq(path: str | Path) -> dict[str, float]:
    """Load a JSON mapping word → corpus frequency."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {k.lower(): float(v) for k, v in d.items()}


def mean_content_freq(content_words: list[str], freq: dict[str, float]) -> float:
    """Mean of `freq[w]` over content_words; OOV words count as 0."""
    if not content_words:
        return 0.0
    return sum(freq.get(w.lower(), 0.0) for w in content_words) / len(content_words)


def stratify(items: list[dict], freq: dict[str, float]) -> dict[str, list[dict]]:
    """Return {"high": [...], "low": [...], "mid": [...]}.

    Asserts both `high` and `low` are non-empty — empty bins almost always mean
    the freq dict doesn't cover the eval set.
    """
    high, low, mid = [], [], []
    for item in items:
        f = mean_content_freq(item["content_words"], freq)
        item = {**item, "mean_freq": f}
        if f > HIGH_THRESHOLD:
            high.append(item)
        elif f < LOW_THRESHOLD:
            low.append(item)
        else:
            mid.append(item)
    if not high:
        raise AssertionError("Empty high-frequency stratum — check freq dict coverage.")
    if not low:
        raise AssertionError("Empty low-frequency stratum — check freq dict coverage.")
    return {"high": high, "low": low, "mid": mid}
