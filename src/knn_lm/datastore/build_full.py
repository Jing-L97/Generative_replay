"""Build the **full** contextualized datastore."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from knn_lm.datastore.build_index import build_index, get_dstore_path
from knn_lm.datastore.capturer import MODEL_LAYER_TO_CAPTURE, ActivationCapturer, KeyType, get_model_last_layer

logger = logging.getLogger(__name__)


class KNNSaver:
    """Forward-hook-driven writer for (key, value) pairs into np.memmap files."""

    def __init__(
        self,
        dstore_size: int,
        dstore_dir: str,
        dimension: int,
        *,
        knn_keytype: KeyType | None = None,
    ) -> None:
        self.dstore_size = dstore_size
        self.dstore_dir = dstore_dir
        self.dimension = dimension
        self.knn_keytype = KeyType.last_ffn_input if knn_keytype is None else knn_keytype

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module | None = None
        self.activation_capturer: ActivationCapturer | None = None
        self.is_encoder_decoder: bool | None = None
        self.dstore_idx = 0
        self.dstore_keys: np.memmap | None = None
        self.dstore_vals: np.memmap | None = None
        self.labels: torch.Tensor | None = None
        self.hook_handles: list = []

    def break_into(self, model: nn.Module) -> None:
        self.model = model
        model.broken_into = True
        self.is_encoder_decoder = model.config.is_encoder_decoder

        layer_fn, capture_input = MODEL_LAYER_TO_CAPTURE[model.config.model_type][self.knn_keytype]
        layer = layer_fn(model)
        self.activation_capturer = ActivationCapturer(layer, capture_input=capture_input)
        self._register(layer, self.activation_capturer)

        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook

        final_layer = get_model_last_layer(model.config.model_type)(model)
        self._register(final_layer, self.post_forward_hook)

        prefix = get_dstore_path(self.dstore_dir, model.config.model_type, self.dstore_size, self.dimension)
        keys_filename = f"{prefix}_keys.npy"
        vals_filename = f"{prefix}_vals.npy"
        Path(keys_filename).parent.mkdir(parents=True, exist_ok=True)
        mode = "r+" if Path(keys_filename).exists() and Path(vals_filename).exists() else "w+"
        self.dstore_keys = np.memmap(
            keys_filename, dtype=np.float16, mode=mode, shape=(self.dstore_size, self.dimension)
        )
        self.dstore_vals = np.memmap(vals_filename, dtype=np.int32, mode=mode, shape=(self.dstore_size, 1))

    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if labels is None:
            raise ValueError("labels must be provided when saving a datastore.")
        self.labels = labels
        return self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)

    def post_forward_hook(self, module, input, output):  # noqa: A002 — torch hook signature
        shift = 0 if self.is_encoder_decoder else 1
        captured_keys = self.activation_capturer.captured
        if shift == 1:
            captured_keys = captured_keys[:, :-shift]
        captured_keys = captured_keys.flatten(0, 1)
        captured_vals = self.labels[:, shift:].flatten(0, 1)

        nonpad = captured_vals != -100
        keys = captured_keys[nonpad]
        vals = captured_vals[nonpad]

        n = keys.shape[0]
        if self.dstore_idx + n > self.dstore_size:
            n = max(self.dstore_size - self.dstore_idx, 0)
            keys = keys[:n]
            vals = vals[:n]
        self.dstore_keys[self.dstore_idx : self.dstore_idx + n] = keys.cpu().numpy().astype(np.float16)
        self.dstore_vals[self.dstore_idx : self.dstore_idx + n] = vals.unsqueeze(-1).cpu().numpy().astype(np.int32)
        self.dstore_idx += n
        return output

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


class _ChunkDataset(Dataset):
    def __init__(self, chunks: list[list[int]], pad_id: int) -> None:
        self.chunks = chunks
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict:
        ids = self.chunks[idx]
        return {"input_ids": torch.tensor(ids, dtype=torch.long)}


def _collate(batch: list[dict], pad_id: int) -> dict:
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for i, b in enumerate(batch):
        ids = b["input_ids"]
        input_ids[i, : ids.shape[0]] = ids
        attention_mask[i, : ids.shape[0]] = 1
        labels[i, : ids.shape[0]] = ids
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_full_datastore(
    model: nn.Module,
    chunks: list[list[int]],
    *,
    dstore_dir: str,
    dimension: int,
    pad_id: int,
    knn_keytype: KeyType | None = None,
    batch_size: int = 8,
    tiny_mode: bool = False,
    ncentroids: int = 4096,
    code_size: int = 64,
    probe: int = 32,
) -> tuple[str, int]:
    """Run `model` over `chunks`, persist (key,value) memmaps, build FAISS index.

    Returns:
        (index_path, dstore_size)

    """
    model.eval()
    # After the autoregressive shift, each chunk of length L contributes L-1
    # (key, value) pairs. Match that exactly so we don't index garbage trailing rows.
    dstore_size = sum(max(0, len(c) - 1) for c in chunks)
    if dstore_size == 0:
        raise ValueError("Empty corpus — nothing to store.")

    saver = KNNSaver(dstore_size, dstore_dir, dimension, knn_keytype=knn_keytype)
    saver.break_into(model)
    try:
        loader = DataLoader(
            _ChunkDataset(chunks, pad_id=pad_id),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: _collate(b, pad_id),
        )
        device = next(model.parameters()).device
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                model(**batch)
    finally:
        saver.break_out()

    if saver.dstore_idx != dstore_size:
        logger.warning(
            "dstore_idx=%d but expected dstore_size=%d. Padding interaction unexpected.",
            saver.dstore_idx,
            dstore_size,
        )
    # Force memmap to disk before indexing.
    saver.dstore_keys.flush()
    saver.dstore_vals.flush()

    index_path = build_index(
        keys=saver.dstore_keys,
        dstore_dir=dstore_dir,
        model_type=model.config.model_type,
        dimension=dimension,
        tiny_mode=tiny_mode,
        ncentroids=ncentroids,
        code_size=code_size,
        probe=probe,
    )
    return index_path, dstore_size
