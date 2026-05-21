"""Hydra CLI: build a full kNN-LM datastore from a text corpus.

Example:
    python -m scripts.datastore.build_full \
        --config-path ../../experiments/configs \
        --config-name tests/tiny_full

The config must define: model.name_or_path, corpus.path, datastore.dir,
retrieval.granularity, retrieval.tau, datastore.tiny_mode.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from knn_lm.datastore.build_full import build_full_datastore
from knn_lm.datastore.capturer import KeyType
from knn_lm.preprocess.corpus import chunk_corpus_file

logger = logging.getLogger(__name__)


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    chunks = chunk_corpus_file(
        cfg.corpus.path,
        tokenizer,
        granularity=cfg.retrieval.granularity,
        tau=cfg.retrieval.tau,
    )
    logger.info("Chunked corpus into %d units (granularity=%s, τ=%d)",
                len(chunks), cfg.retrieval.granularity, cfg.retrieval.tau)

    Path(cfg.datastore.dir).mkdir(parents=True, exist_ok=True)
    dimension = _infer_key_dim(model)
    index_path, dstore_size = build_full_datastore(
        model,
        chunks,
        dstore_dir=cfg.datastore.dir,
        dimension=dimension,
        pad_id=tokenizer.pad_token_id,
        knn_keytype=KeyType.from_string(cfg.datastore.get("keytype", "last_ffn_input")),
        batch_size=cfg.datastore.get("batch_size", 8),
        tiny_mode=bool(cfg.datastore.get("tiny_mode", False)),
        ncentroids=cfg.datastore.get("ncentroids", 4096),
        code_size=cfg.datastore.get("code_size", 64),
        probe=cfg.datastore.get("probe", 32),
    )
    logger.info("Wrote index=%s, dstore_size=%d, dim=%d", index_path, dstore_size, dimension)


def _infer_key_dim(model) -> int:
    """Last-FFN-input dim. For GPT-2 this is hidden_size; for other archs, override in config."""
    return model.config.hidden_size


if __name__ == "__main__":
    main()
