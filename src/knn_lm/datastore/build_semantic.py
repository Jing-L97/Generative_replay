"""Build the semantics-only datastore variants"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from transformers import PreTrainedTokenizerBase

from knn_lm.datastore.build_index import build_index, get_dstore_path
from knn_lm.preprocess.corpus import Granularity

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Plain-text word2vec / GloVe / fastText loading
# --------------------------------------------------------------------------- #
def load_word_embeddings(path: str | Path) -> dict[str, np.ndarray]:
    """Load a word2vec-format text file (`word v1 v2 ...` per line)."""
    vectors: dict[str, np.ndarray] = {}
    with open(path, encoding="utf-8") as f:
        first = f.readline().strip().split()
        # Optional header `vocab_size dim`.
        if len(first) == 2 and all(tok.isdigit() for tok in first):
            pass
        else:
            word, *vals = first
            vectors[word] = np.asarray([float(v) for v in vals], dtype=np.float32)
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            word, *vals = parts
            vectors[word] = np.asarray([float(v) for v in vals], dtype=np.float32)
    if not vectors:
        raise ValueError(f"No vectors loaded from {path}")
    return vectors


def _embed_unit(unit: str, embeddings: dict[str, np.ndarray], dim: int) -> np.ndarray | None:
    """Sum word embeddings over the tokens of `unit`. Returns None if no token found."""
    words = unit.lower().split()
    hits = [embeddings[w] for w in words if w in embeddings]
    if not hits:
        return None
    return np.sum(hits, axis=0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Variant 1: word-embedding datastore (GloVe / fastText, summed)
# --------------------------------------------------------------------------- #
def build_word_embedding_datastore(
    corpus_path: str,
    tokenizer: PreTrainedTokenizerBase,
    embeddings_paths: list[str],
    *,
    dstore_dir: str,
    granularity: Granularity = "sequence",
    tiny_mode: bool = True,
    model_type: str = "wordemb",
) -> tuple[str, int, int]:
    """Build a datastore whose keys are summed word embeddings of each corpus unit.

    Values are the **last** token id of each unit (a reasonable convention for
    "what comes next here" retrieval at sequence/phrase granularity).

    Returns (index_path, dstore_size, dimension).
    """
    from knn_lm.preprocess.corpus import _split_units  # local import to avoid cycle at module load

    # Combine multiple embedding files (e.g. GloVe + fastText) by summation.
    embeddings_per_file = [load_word_embeddings(p) for p in embeddings_paths]
    vocab = set().union(*(emb.keys() for emb in embeddings_per_file))
    dim = next(iter(embeddings_per_file[0].values())).shape[0]
    if not all(next(iter(e.values())).shape[0] == dim for e in embeddings_per_file):
        raise ValueError("All embedding files must share the same dimensionality.")
    combined: dict[str, np.ndarray] = {}
    for w in vocab:
        v = np.zeros(dim, dtype=np.float32)
        for emb in embeddings_per_file:
            if w in emb:
                v = v + emb[w]
        combined[w] = v

    with open(corpus_path, encoding="utf-8") as f:
        text = f.read()

    keys: list[np.ndarray] = []
    vals: list[int] = []
    for unit in _split_units(text, granularity):
        vec = _embed_unit(unit, combined, dim)
        if vec is None:
            continue
        ids = tokenizer(unit, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        keys.append(vec)
        vals.append(ids[-1])

    if not keys:
        raise ValueError("Built an empty word-embedding datastore — no overlap between corpus and embeddings.")

    return _write_and_index(
        keys=np.stack(keys).astype(np.float16),
        vals=np.asarray(vals, dtype=np.int32).reshape(-1, 1),
        dstore_dir=dstore_dir,
        model_type=model_type,
        dimension=dim,
        tiny_mode=tiny_mode,
    )


# --------------------------------------------------------------------------- #
# Position-averaged contextualized embeddings
# --------------------------------------------------------------------------- #
def build_averaged_datastore(
    full_dstore_dir: str,
    full_model_type: str,
    full_dstore_size: int,
    full_dimension: int,
    *,
    window: int,
    dstore_dir: str,
    model_type: str | None = None,
    tiny_mode: bool = True,
) -> tuple[str, int, int]:
    """Take a previously-built full datastore and average keys over `window` positions.

    Values are inherited from the **last** position of each window.
    """
    full_prefix = get_dstore_path(full_dstore_dir, full_model_type, full_dstore_size, full_dimension)
    keys_in = np.memmap(f"{full_prefix}_keys.npy", dtype=np.float16, mode="r", shape=(full_dstore_size, full_dimension))
    vals_in = np.memmap(f"{full_prefix}_vals.npy", dtype=np.int32, mode="r", shape=(full_dstore_size, 1))

    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    n_windows = max(0, full_dstore_size - window + 1)
    if n_windows == 0:
        raise ValueError(f"Datastore too small ({full_dstore_size}) for window={window}.")

    keys_out = np.empty((n_windows, full_dimension), dtype=np.float16)
    vals_out = np.empty((n_windows, 1), dtype=np.int32)
    for i in range(n_windows):
        keys_out[i] = keys_in[i : i + window].astype(np.float32).mean(axis=0).astype(np.float16)
        vals_out[i] = vals_in[i + window - 1]

    return _write_and_index(
        keys=keys_out,
        vals=vals_out,
        dstore_dir=dstore_dir,
        model_type=model_type or f"{full_model_type}_avg{window}",
        dimension=full_dimension,
        tiny_mode=tiny_mode,
    )


def _write_and_index(
    keys: np.ndarray,
    vals: np.ndarray,
    *,
    dstore_dir: str,
    model_type: str,
    dimension: int,
    tiny_mode: bool,
) -> tuple[str, int, int]:
    Path(dstore_dir).mkdir(parents=True, exist_ok=True)
    dstore_size = keys.shape[0]
    prefix = get_dstore_path(dstore_dir, model_type, dstore_size, dimension)
    keys_mm = np.memmap(f"{prefix}_keys.npy", dtype=np.float16, mode="w+", shape=keys.shape)
    keys_mm[:] = keys
    keys_mm.flush()
    vals_mm = np.memmap(f"{prefix}_vals.npy", dtype=np.int32, mode="w+", shape=vals.shape)
    vals_mm[:] = vals
    vals_mm.flush()

    index_path = build_index(
        keys=keys_mm, dstore_dir=dstore_dir, model_type=model_type, dimension=dimension, tiny_mode=tiny_mode
    )
    return index_path, dstore_size, dimension
