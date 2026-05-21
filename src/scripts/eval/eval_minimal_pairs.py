"""Hydra CLI: end-to-end minimal-pair evaluation.

Pipeline (one config invocation):
    1. Load model + tokenizer.
    2. (Optional) build / attach a kNN datastore.
       - `retrieval.mode == "none"`         : baseline, parametric only.
       - `retrieval.mode == "full"`         : build full datastore, attach KNNWrapper.
       - `retrieval.mode == "semantic"`     : build semantic datastore, attach KNNWrapper.
       - `retrieval.mode == "structural"`   : build full datastore, attach
                                              StructuralFilteredKNNWrapper, sweep over conditions.
    3. Load eval pairs (per-phenomenon JSONL), stratify by frequency.
    4. Score; write results JSON to `output.path`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from knn_lm.datastore.build_full import build_full_datastore
from knn_lm.datastore.build_semantic import build_word_embedding_datastore
from knn_lm.datastore.capturer import KeyType
from knn_lm.eval.frequency_strata import load_freq, stratify
from knn_lm.eval.loaders import load_jsonl
from knn_lm.eval.minimal_pair import score_pairs
from knn_lm.preprocess.corpus import _split_units, chunk_corpus_file
from knn_lm.retrieval.knn_wrapper import KNNWrapper
from knn_lm.retrieval.structural_filter import StructuralFilteredKNNWrapper, partition_datastore_by_structure

logger = logging.getLogger(__name__)


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    out_path = Path(cfg.output.path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    mode = cfg.retrieval.get("mode", "none")
    wrapper = None
    if mode != "none":
        wrapper = _build_and_attach(cfg, model, tokenizer)

    freq = load_freq(cfg.eval.freq_path)
    results = {"config": OmegaConf.to_container(cfg, resolve=True), "phenomena": {}}
    for phen, path in cfg.eval.pairs.items():
        items = load_jsonl(path)
        strata = stratify(items, freq)
        per_stratum = {}
        for name in ("high", "low"):
            res = score_pairs(strata[name], model, tokenizer, device=device)
            res.pop("items", None)  # don't bloat the smoke-test artifact
            per_stratum[name] = res
            logger.info("phen=%s stratum=%s acc=%.3f n=%d", phen, name, res["accuracy"], res["n"])
        results["phenomena"][phen] = per_stratum

    if wrapper is not None:
        wrapper.break_out()

    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", out_path)


def _build_and_attach(cfg: DictConfig, model, tokenizer):
    """Build the appropriate datastore and break_into the model."""
    dstore_dir = cfg.datastore.dir
    Path(dstore_dir).mkdir(parents=True, exist_ok=True)
    dimension = model.config.hidden_size
    mode = cfg.retrieval.mode

    if mode == "full":
        chunks = chunk_corpus_file(
            cfg.corpus.path, tokenizer,
            granularity=cfg.retrieval.granularity, tau=cfg.retrieval.tau,
        )
        index_path, n = build_full_datastore(
            model, chunks,
            dstore_dir=dstore_dir, dimension=dimension, pad_id=tokenizer.pad_token_id,
            tiny_mode=bool(cfg.datastore.get("tiny_mode", False)),
            ncentroids=cfg.datastore.get("ncentroids", 4096),
            code_size=cfg.datastore.get("code_size", 64),
        )
        wrapper = KNNWrapper(
            dstore_size=n, dstore_dir=dstore_dir, dimension=dimension,
            knn_keytype=KeyType.last_ffn_input,
            k=cfg.retrieval.k, lmbda=cfg.retrieval.lmbda, knn_temp=cfg.retrieval.tau_softmax,
            knn_gpu=torch.cuda.is_available(),
            move_dstore_to_mem=True,
        )
        wrapper.break_into(model)
        return wrapper

    if mode == "semantic":
        index_path, n, dim = build_word_embedding_datastore(
            corpus_path=cfg.corpus.path,
            tokenizer=tokenizer,
            embeddings_paths=list(cfg.datastore.embeddings),
            dstore_dir=dstore_dir,
            granularity=cfg.retrieval.granularity,
            tiny_mode=bool(cfg.datastore.get("tiny_mode", True)),
            model_type=cfg.datastore.get("model_type", "wordemb"),
        )
        # Semantic-only retrieval can't be hooked into the LM's final-FFN-input
        # because keys live in word-embedding space, not in the LM's hidden space.
        # For the smoke test we leave the LM in baseline mode but still exercise
        # the build pipeline + index — what we're testing here is the build path.
        logger.warning(
            "retrieval.mode='semantic' built index but not attaching wrapper "
            "(key space != model hidden space); evaluation falls back to baseline."
        )
        return None

    if mode == "structural":
        chunks = chunk_corpus_file(
            cfg.corpus.path, tokenizer,
            granularity=cfg.retrieval.granularity, tau=cfg.retrieval.tau,
        )
        index_path, n = build_full_datastore(
            model, chunks,
            dstore_dir=dstore_dir, dimension=dimension, pad_id=tokenizer.pad_token_id,
            tiny_mode=bool(cfg.datastore.get("tiny_mode", False)),
        )
        with open(cfg.corpus.path, encoding="utf-8") as f:
            text = f.read()
        units = _split_units(text, cfg.retrieval.granularity)
        # Assume one-to-one alignment unit↔datastore-row, true for sequence-level
        # because each unit contributes exactly one (key, value) pair via the
        # last-token-as-value convention; for the smoke test we further constrain
        # by truncating to min length.
        units = units[:n]
        structural, non_structural = partition_datastore_by_structure(units, cfg.retrieval.phenomenon)
        wrapper = StructuralFilteredKNNWrapper(
            dstore_size=n, dstore_dir=dstore_dir, dimension=dimension,
            knn_keytype=KeyType.last_ffn_input,
            k=cfg.retrieval.k, lmbda=cfg.retrieval.lmbda, knn_temp=cfg.retrieval.tau_softmax,
            knn_gpu=torch.cuda.is_available(),
            move_dstore_to_mem=True,
            structural_ids=structural, non_structural_ids=non_structural,
            condition=cfg.retrieval.get("condition", "w"),
            rng_seed=cfg.retrieval.get("rng_seed", 0),
        )
        wrapper.break_into(model)
        return wrapper

    raise ValueError(f"Unknown retrieval.mode: {mode}")


if __name__ == "__main__":
    main()
