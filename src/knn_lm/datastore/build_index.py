"""FAISS index construction"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def get_dstore_path(dstore_dir: str | Path, model_type: str, dstore_size: int, dimension: int) -> str:
    return f"{dstore_dir}/dstore_{model_type}_{dstore_size}_{dimension}"


def get_index_path(dstore_dir: str | Path, model_type: str, dstore_size: int, dimension: int) -> str:
    return f"{dstore_dir}/index_{model_type}_{dstore_size}_{dimension}.indexed"


def build_index(
    keys: np.memmap | np.ndarray,
    dstore_dir: str | Path,
    model_type: str,
    dimension: int,
    *,
    num_keys_to_add_at_a_time: int = 1_000_000,
    ncentroids: int = 4096,
    seed: int = 1,
    code_size: int = 64,
    probe: int = 32,
    tiny_mode: bool = False,
) -> str:
    """Build a FAISS index over `keys` and write it to disk.

    Args:
        keys: (N, dim) array of fp16 keys.
        tiny_mode: If True, use `IndexFlatL2` (no training, exact search). Required
            for corpora with fewer than ~39 * ncentroids vectors. The query API is
            identical to the IVFPQ path.

    Returns:
        Path to the written index file.

    """
    dstore_size = keys.shape[0]
    index_name = get_index_path(dstore_dir, model_type, dstore_size, dimension)
    Path(index_name).parent.mkdir(parents=True, exist_ok=True)

    if tiny_mode:
        logger.info("Building IndexFlatL2 (tiny_mode); N=%d, dim=%d", dstore_size, dimension)
        index = faiss.IndexFlatL2(dimension)
        # Faiss does not accept fp16 directly.
        index.add(np.ascontiguousarray(keys[:].astype(np.float32)))
        faiss.write_index(index, index_name)
        return index_name

    logger.info("Building IndexIVFPQ; N=%d, dim=%d, ncentroids=%d", dstore_size, dimension, ncentroids)
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFPQ(quantizer, dimension, ncentroids, code_size, 8)
    index.nprobe = probe

    np.random.seed(seed)
    sample_size = min(1_000_000, dstore_size)
    random_sample = np.random.choice(np.arange(dstore_size), size=[sample_size], replace=False)
    t0 = time.time()
    index.train(keys[random_sample].astype(np.float32))
    logger.info("Training took %.1fs", time.time() - t0)

    t0 = time.time()
    start = 0
    while start < dstore_size:
        end = min(dstore_size, start + num_keys_to_add_at_a_time)
        to_add = keys[start:end].astype(np.float32)
        index.add_with_ids(to_add, np.arange(start, end))
        start = end
    logger.info("Adding %d keys took %.1fs", dstore_size, time.time() - t0)

    faiss.write_index(index, index_name)
    return index_name


def read_index(
    dstore_dir: str | Path,
    model_type: str,
    dstore_size: int,
    dimension: int,
    *,
    probe: int = 32,
    move_to_gpu: bool = False,
) -> faiss.Index:
    """Load a previously-built FAISS index. Returns CPU index unless `move_to_gpu`."""
    index_name = get_index_path(dstore_dir, model_type, dstore_size, dimension)
    cpu_index = faiss.read_index(index_name, faiss.IO_FLAG_ONDISK_SAME_DIR)
    if hasattr(cpu_index, "nprobe"):
        cpu_index.nprobe = probe
    if move_to_gpu:
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        return faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index, co)
    return cpu_index
