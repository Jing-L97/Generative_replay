"""Hydra CLI: build a semantics-only datastore.

Selects variant via `datastore.variant`:
    - `glove`      : sum GloVe vectors over each corpus unit
    - `fasttext`   : sum fastText vectors
    - `glove+fasttext` : sum both
    - `averaged`   : position-average a previously built full datastore
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from knn_lm.datastore.build_semantic import build_averaged_datastore, build_word_embedding_datastore

logger = logging.getLogger(__name__)


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    variant = cfg.datastore.variant
    if variant in {"glove", "fasttext", "glove+fasttext"}:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        paths = (
            cfg.datastore.embeddings if isinstance(cfg.datastore.embeddings, list) else [cfg.datastore.embeddings]
        )
        index_path, n, d = build_word_embedding_datastore(
            corpus_path=cfg.corpus.path,
            tokenizer=tokenizer,
            embeddings_paths=list(paths),
            dstore_dir=cfg.datastore.dir,
            granularity=cfg.retrieval.granularity,
            tiny_mode=bool(cfg.datastore.get("tiny_mode", True)),
            model_type=cfg.datastore.get("model_type", variant.replace("+", "_")),
        )
    elif variant == "averaged":
        index_path, n, d = build_averaged_datastore(
            full_dstore_dir=cfg.datastore.full_dir,
            full_model_type=cfg.datastore.full_model_type,
            full_dstore_size=cfg.datastore.full_dstore_size,
            full_dimension=cfg.datastore.full_dimension,
            window=cfg.datastore.window,
            dstore_dir=cfg.datastore.dir,
            model_type=cfg.datastore.get("model_type", None),
            tiny_mode=bool(cfg.datastore.get("tiny_mode", True)),
        )
    else:
        raise ValueError(f"Unknown semantic variant: {variant}")

    logger.info("Wrote semantic index=%s, n=%d, dim=%d", index_path, n, d)


if __name__ == "__main__":
    main()
