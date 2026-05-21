from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Literal

import faiss
import numpy as np
import torch

from knn_lm.retrieval.knn_wrapper import KNNWrapper

logger = logging.getLogger(__name__)

Condition = Literal["w", "wo", "w+r", "wo+r"]


class StructuralFilteredKNNWrapper(KNNWrapper):
    """Retrieve only from datastore indices belonging to (or excluded from) a structural subset.

    Args:
        structural_ids: 1-D array of indices into the datastore that *match* the
            target syntactic phenomenon (used by `w` / `w+r`).
        non_structural_ids: 1-D array of the complementary subset.
        condition: which split to retrieve from.
        rng_seed: RNG seed for `+r` random retrieval.

    """

    def __init__(
        self,
        *args,
        structural_ids: Iterable[int],
        non_structural_ids: Iterable[int],
        condition: Condition = "w",
        rng_seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._structural_ids = np.asarray(list(structural_ids), dtype=np.int64)
        self._non_structural_ids = np.asarray(list(non_structural_ids), dtype=np.int64)
        self.condition = condition
        self._rng = np.random.default_rng(rng_seed)

    def _active_ids(self) -> np.ndarray:
        return self._structural_ids if self.condition in {"w", "w+r"} else self._non_structural_ids

    def _is_random(self) -> bool:
        return self.condition in {"w+r", "wo+r"}

    def get_knns(self, queries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        active = self._active_ids()
        if active.size == 0:
            # Degenerate split: emit -1 ids; KNNWrapper.knns_to_log_prob will neginf them.
            n = queries.shape[0]
            dists = torch.full((n, self.k), float("inf"), device=self.device)
            knns = torch.full((n, self.k), -1, dtype=torch.long, device=self.device)
            return dists, knns

        if self._is_random():
            return self._random_knns(queries, active)
        return self._filtered_knns(queries, active)

    def _random_knns(self, queries: torch.Tensor, active: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        n = queries.shape[0]
        sampled = self._rng.choice(active, size=(n, self.k), replace=active.size < self.k)
        # Recompute distances if requested, else use uniform zeros (softmax → uniform weighting).
        knns = torch.from_numpy(sampled).long().to(self.device)
        if self.keys is not None:
            keys = torch.from_numpy(np.asarray(self.keys[sampled], dtype=np.float32)).to(self.device)
            dists = torch.sum((queries.unsqueeze(-2) - keys) ** 2, dim=-1)
        else:
            dists = torch.zeros_like(knns, dtype=torch.float32, device=self.device)
        return dists, knns

    def _filtered_knns(self, queries: torch.Tensor, active: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        sel = faiss.IDSelectorBatch(active.size, faiss.swig_ptr(active.astype(np.int64)))
        params = faiss.SearchParameters(sel=sel) if hasattr(faiss, "SearchParameters") else None
        q_np = queries.detach().cpu().numpy().astype(np.float32)
        if params is not None:
            try:
                dists, knns = self.index.search(q_np, self.k, params=params)
            except TypeError:
                # Some FAISS builds don't accept SearchParameters on IndexFlat — fall back.
                dists, knns = self._brute_force(queries, active)
        else:
            dists, knns = self._brute_force(queries, active)
        if isinstance(dists, np.ndarray):
            dists = torch.from_numpy(dists).to(self.device)
            knns = torch.from_numpy(knns.astype(np.int64)).to(self.device)
        return dists, knns

    def _brute_force(self, queries: torch.Tensor, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exact restricted L2 search by reading keys from memmap. Fine for smoke tests."""
        if self.keys is None:
            raise RuntimeError("StructuralFilteredKNNWrapper requires `keys` loaded (no_load_keys=False).")
        keys = np.asarray(self.keys[active], dtype=np.float32)
        q = queries.detach().cpu().numpy().astype(np.float32)
        d2 = ((q[:, None, :] - keys[None, :, :]) ** 2).sum(axis=-1)
        k = min(self.k, d2.shape[1])
        idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(d2.shape[0])[:, None]
        sorted_pos = np.argsort(d2[rows, idx], axis=1)
        idx_sorted = np.take_along_axis(idx, sorted_pos, axis=1)
        dists = np.take_along_axis(d2, idx_sorted, axis=1)
        ids = active[idx_sorted]
        # Pad to self.k if fewer neighbors were available.
        if k < self.k:
            pad_d = np.full((dists.shape[0], self.k - k), np.inf, dtype=dists.dtype)
            pad_i = np.full((ids.shape[0], self.k - k), -1, dtype=ids.dtype)
            dists = np.concatenate([dists, pad_d], axis=1)
            ids = np.concatenate([ids, pad_i], axis=1)
        return dists, ids


def partition_datastore_by_structure(
    unit_texts: list[str],
    phenomenon: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Given per-position unit texts, return (structural_ids, non_structural_ids).

    For sequence-level datastores, `unit_texts[i]` is the sentence containing
    datastore position `i`. For phrase/token, it's the smallest enclosing
    phrase/sentence.
    """
    from knn_lm.retrieval.structural_similarity import has_structure  # avoid cycle

    mask = np.array([has_structure(t, phenomenon) for t in unit_texts], dtype=bool)  # type: ignore[arg-type]
    structural = np.where(mask)[0]
    non_structural = np.where(~mask)[0]
    return structural, non_structural
