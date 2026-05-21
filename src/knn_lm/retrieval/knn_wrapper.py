"""kNN-LM inference wrapper."""

from __future__ import annotations

import logging
import time
from enum import Enum, auto

import faiss
import faiss.contrib.torch_utils
import numpy as np
import torch
from torch import nn

from knn_lm.datastore.build_index import get_dstore_path, read_index
from knn_lm.datastore.capturer import MODEL_LAYER_TO_CAPTURE, ActivationCapturer, KeyType, get_model_last_layer

logger = logging.getLogger(__name__)


class Dist(Enum):
    l2 = auto()
    dot = auto()

    @staticmethod
    def from_string(s: str) -> Dist:
        return Dist[s.lower()]


def _l2(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    return torch.sum((query.unsqueeze(-2) - keys) ** 2, dim=-1)


def _dotprod(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    return torch.sum(query.unsqueeze(-2) * keys, dim=-1)


_DIST_FUNCS = {Dist.l2: _l2, Dist.dot: _dotprod}


class KNNWrapper:
    """Wrap an HF causal LM so each forward pass interpolates with kNN retrieval."""

    def __init__(
        self,
        dstore_size: int,
        dstore_dir: str,
        dimension: int,
        *,
        knn_sim_func: Dist | None = None,
        knn_keytype: KeyType | None = None,
        no_load_keys: bool = False,
        move_dstore_to_mem: bool = False,
        knn_gpu: bool = True,
        recompute_dists: bool = False,
        k: int = 1024,
        lmbda: float = 0.25,
        knn_temp: float = 1.0,
        probe: int = 32,
    ) -> None:
        self.dstore_size = dstore_size
        self.dstore_dir = dstore_dir
        self.dimension = dimension
        self.lmbda = lmbda
        self.k = k
        self.knn_temperature = knn_temp
        self.probe = probe
        self.knn_sim_func = Dist.l2 if knn_sim_func is None else knn_sim_func
        self.knn_keytype = KeyType.last_ffn_input if knn_keytype is None else knn_keytype
        self.no_load_keys = no_load_keys
        self.recompute_dists = recompute_dists
        self.move_dstore_to_mem = move_dstore_to_mem
        self.knn_gpu = knn_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels: torch.Tensor | None = None
        self.keys: np.memmap | np.ndarray | None = None
        self.vals: np.memmap | torch.Tensor | None = None
        self.model: nn.Module | None = None
        self.vocab_size: int | None = None
        self.activation_capturer: ActivationCapturer | None = None
        self.is_encoder_decoder: bool | None = None
        self.hook_handles: list = []

        self.dist_func = _DIST_FUNCS[self.knn_sim_func]

    def setup_faiss(self) -> tuple[faiss.Index, faiss.Index]:
        if not self.dstore_dir:
            raise ValueError("Cannot build a datastore without the data.")

        t0 = time.time()
        cpu_index = read_index(
            self.dstore_dir,
            self.model.config.model_type,
            self.dstore_size,
            self.dimension,
            probe=self.probe,
            move_to_gpu=False,
        )
        logger.info("Reading datastore took %.1fs", time.time() - t0)

        if self.knn_gpu:
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index, co)
        else:
            gpu_index = cpu_index

        # IndexFlatL2 has no make_direct_map (and doesn't need one).
        if hasattr(cpu_index, "make_direct_map"):
            try:
                cpu_index.make_direct_map()
            except RuntimeError:
                pass  # already mapped

        prefix = get_dstore_path(self.dstore_dir, self.model.config.model_type, self.dstore_size, self.dimension)
        if not self.no_load_keys:
            self.keys = np.memmap(
                f"{prefix}_keys.npy", dtype=np.float16, mode="r", shape=(self.dstore_size, self.dimension)
            )
        self.vals = np.memmap(f"{prefix}_vals.npy", dtype=np.int32, mode="r", shape=(self.dstore_size, 1))

        if self.move_dstore_to_mem:
            if not self.no_load_keys:
                self.keys = np.array(self.keys, dtype=np.float16)
            vals_arr = np.array(self.vals, dtype=np.int32)
            self.vals = torch.from_numpy(vals_arr).long().to(self.device)

        return cpu_index, gpu_index

    def break_into(self, model: nn.Module) -> None:
        self.model = model
        model.broken_into = True
        self.reconstruct_index, self.index = self.setup_faiss()
        self.is_encoder_decoder = model.config.is_encoder_decoder

        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook

        layer_fn, capture_input = MODEL_LAYER_TO_CAPTURE[model.config.model_type][self.knn_keytype]
        layer = layer_fn(model)
        self.activation_capturer = ActivationCapturer(layer, capture_input=capture_input)
        self._register(layer, self.activation_capturer)

        final_layer = get_model_last_layer(model.config.model_type)(model)
        self._register(final_layer, self.post_forward_hook)
        self.vocab_size = final_layer.out_features

    def get_knns(self, queries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.knn_gpu:
            queries = queries.cpu()
        dists, knns = self.index.search(queries, self.k)
        return dists.to(self.device), knns.to(self.device)

    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        self.labels = labels
        return self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)

    def post_forward_hook(self, module, input, output):  # noqa: A002 — torch hook signature
        batch, time_dim, _ = output.shape
        shift = 0 if self.is_encoder_decoder else 1
        lm_log_probs = torch.nn.functional.log_softmax(output, dim=-1)
        queries = self.activation_capturer.captured

        if self.labels is None:
            nonpad_mask = torch.cat(
                [
                    torch.zeros([batch, time_dim - 1], dtype=torch.bool),
                    torch.ones([batch, 1], dtype=torch.bool),
                ],
                dim=-1,
            ).to(self.device)
        else:
            nonpad_mask = torch.cat(
                [
                    self.labels[:, shift:] != -100,
                    torch.zeros([self.labels.shape[0], shift], dtype=torch.bool).to(self.device),
                ],
                dim=-1,
            )

        lm_log_probs = lm_log_probs[nonpad_mask]
        queries = queries[nonpad_mask]

        dists, knns = self.get_knns(queries)
        if self.recompute_dists:
            knns_vecs = torch.from_numpy(self.keys[knns.cpu().numpy()]).to(self.device)
            dists = self.dist_func(queries, knns_vecs)

        knn_log_probs, _ = self._knns_to_log_prob(knns, -dists)
        output[nonpad_mask] = self.interpolate(knn_log_probs, lm_log_probs, self.lmbda)
        return output

    def _knns_to_log_prob(self, knns: torch.Tensor, neg_dists: torch.Tensor):
        probs = torch.nn.functional.softmax(neg_dists / self.knn_temperature, dim=-1)
        vals_at_knns = self._vals_tensor()[knns].squeeze(-1)
        knn_log_probs = (
            torch.full(vals_at_knns.shape[:-1] + (self.vocab_size,), 0.0, device=self.device)
            .scatter_add(dim=-1, index=vals_at_knns, src=probs)
            .log()
        )
        return torch.nan_to_num(knn_log_probs, nan=None, neginf=-10000.0), vals_at_knns

    def _vals_tensor(self) -> torch.Tensor:
        if isinstance(self.vals, torch.Tensor):
            return self.vals
        return torch.from_numpy(np.array(self.vals)).long().to(self.device)

    def _register(self, layer: nn.Module, func, *, pre: bool = False) -> None:
        handle = layer.register_forward_pre_hook(func) if pre else layer.register_forward_hook(func)
        self.hook_handles.append(handle)

    def break_out(self) -> None:
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()
        if self.model is not None and getattr(self.model, "broken_into", None) is not None:
            self.model.forward = self.original_forward_func
            self.model.broken_into = None

    def get_metrics(self) -> dict:
        return {}

    @staticmethod
    def interpolate(knn_log_probs: torch.Tensor, lm_log_probs: torch.Tensor, lmbda: float) -> torch.Tensor:
        return torch.logaddexp(lm_log_probs + np.log(1 - lmbda), knn_log_probs + np.log(lmbda))
