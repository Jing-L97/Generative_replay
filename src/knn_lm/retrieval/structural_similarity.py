"""Structural-similarity metric used by `structural_filter.StructuralFilteredKNNWrapper`."""

from __future__ import annotations

import re
from typing import Literal

Phenomenon = Literal["SV", "Wh", "RC"]

_WH_WORDS = {"what", "who", "whom", "whose", "which", "where", "when", "why", "how"}
_AUX = {"do", "does", "did", "is", "are", "was", "were", "has", "have", "had", "can", "could", "will", "would"}
_RELATIVIZERS = {"that", "which", "who", "whom", "whose"}


_FINITE_VERBS = {
    "run",
    "runs",
    "ran",
    "laugh",
    "laughs",
    "laughed",
    "watch",
    "watches",
    "watched",
    "chase",
    "chases",
    "chased",
    "like",
    "likes",
    "liked",
    "see",
    "sees",
    "saw",
    "eat",
    "eats",
    "ate",
    "play",
    "plays",
    "played",
    "read",
    "reads",
    "sleep",
    "sleeps",
    "slept",
}


def _tokens(s: str) -> list[str]:
    return re.findall(r"[A-Za-z]+", s.lower())


def has_sv_pattern(s: str) -> bool:
    """`(det) (adj)* noun (aux)? verb` — present-tense subject-verb agreement."""
    toks = _tokens(s)
    if len(toks) < 2:
        return False
    # Look for a verb token preceded by something noun-like (anything that isn't a wh-word/aux).
    for i, tok in enumerate(toks[1:], start=1):
        if tok in _FINITE_VERBS:
            prev = toks[i - 1]
            if prev not in _WH_WORDS and prev not in _AUX:
                return True
    return False


def has_wh_pattern(s: str) -> bool:
    """Wh-fronted clause; we require a wh-word in the first 2 positions and an aux."""
    toks = _tokens(s)
    if len(toks) < 3:
        return False
    if not any(t in _WH_WORDS for t in toks[:2]):
        return False
    return any(t in _AUX for t in toks[:5])


def has_rc_pattern(s: str) -> bool:
    """`NP <relativizer> <verb> ... <verb>` — a relative clause modifying a subject."""
    toks = _tokens(s)
    if len(toks) < 5:
        return False
    for i, tok in enumerate(toks):
        if tok in _RELATIVIZERS and 0 < i < len(toks) - 2:
            # Need a verb both inside the RC and afterwards.
            tail = toks[i + 1 :]
            verbs_in_tail = [t for t in tail if t in _FINITE_VERBS]
            if len(verbs_in_tail) >= 2:
                return True
    return False


_DETECTORS = {"SV": has_sv_pattern, "Wh": has_wh_pattern, "RC": has_rc_pattern}


def has_structure(sentence: str, phenomenon: Phenomenon) -> bool:
    """True iff `sentence` instantiates the target syntactic phenomenon."""
    return _DETECTORS[phenomenon](sentence)


def structural_similarity(a: str, b: str, phenomenon: Phenomenon) -> float:
    """Binary similarity: 1.0 if both share the target structure, else 0.0.

    A continuous variant could weight by overlap of POS-tag n-grams; the binary
    version is what the *w/* vs. *w/o* split in Figure `fig:datastore` needs.
    """
    return 1.0 if has_structure(a, phenomenon) and has_structure(b, phenomenon) else 0.0
