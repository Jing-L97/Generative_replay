"""Minimal-pair scoring: P(good) > P(bad)"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@torch.no_grad()
def sentence_logprob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sentence: str,
    *,
    device: torch.device | None = None,
) -> float:
    """Sum of log P(x_t | x_<t) under `model`. Per-token labels are set so the
    KNNWrapper hook treats them as non-pad.
    """
    device = device or next(model.parameters()).device
    enc = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    if input_ids.numel() < 2:
        return 0.0
    labels = input_ids.clone()
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    # Use logits directly to avoid HF's mean-over-tokens reduction.
    log_probs = torch.nn.functional.log_softmax(out.logits, dim=-1)
    shifted_logp = log_probs[:, :-1, :]
    shifted_tgt = input_ids[:, 1:]
    token_lp = shifted_logp.gather(-1, shifted_tgt.unsqueeze(-1)).squeeze(-1)
    return float(token_lp.sum().item())


def score_pairs(
    items: Iterable[dict],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    device: torch.device | None = None,
) -> dict:
    """For each {good, bad} pair, return per-item correctness + aggregate accuracy."""
    device = device or next(model.parameters()).device
    correct, total = 0, 0
    per_item = []
    for item in items:
        lp_good = sentence_logprob(model, tokenizer, item["good"], device=device)
        lp_bad = sentence_logprob(model, tokenizer, item["bad"], device=device)
        ok = lp_good > lp_bad
        per_item.append({**item, "lp_good": lp_good, "lp_bad": lp_bad, "correct": bool(ok)})
        correct += int(ok)
        total += 1
    accuracy = correct / total if total else float("nan")
    return {"accuracy": accuracy, "n": total, "n_correct": correct, "items": per_item}


def compute_ppl(token_lps: list[float], n_tokens: int) -> float:
    """Perplexity from a list of summed log-probs and total token count."""
    if n_tokens == 0:
        return float("nan")
    return math.exp(-sum(token_lps) / n_tokens)
