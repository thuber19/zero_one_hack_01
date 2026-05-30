"""Shared evaluation metrics — frozen contract for Specs 001, 002, 005.

compute(logits, targets, mask, variant_ids, vocab) -> dict
"""
from __future__ import annotations

import math

import torch


def compute(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    variant_ids: torch.Tensor | None = None,
    vocab=None,
    top_k: tuple[int, ...] = (1, 5),
) -> dict:
    """Aggregate metrics over a batch (or whole eval set if streamed).

    logits  [N, V]  flattened token-level logits at scored positions
    targets [N]     ground truth ids
    mask    [N]     1 for scored positions, 0 otherwise (already applied if you pass N=scored count)
    variant_ids [N] per-token variant tag for breakdown (optional)
    """
    mask_b = mask.bool()
    if mask_b.any():
        logits = logits[mask_b]
        targets = targets[mask_b]
        if variant_ids is not None:
            variant_ids = variant_ids[mask_b]
    n = targets.numel()
    out: dict = {"n_tokens": int(n)}
    if n == 0:
        for k in top_k:
            out[f"top{k}_accuracy"] = 0.0
        out["perplexity"] = float("inf")
        out["mrr"] = 0.0
        return out

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    out["perplexity"] = float(torch.exp(nll.mean()).item())

    sorted_ids = logits.argsort(dim=-1, descending=True)
    for k in top_k:
        topk = sorted_ids[:, :k]
        out[f"top{k}_accuracy"] = float((topk == targets.unsqueeze(1)).any(dim=1).float().mean().item())

    # MRR over top-100 to bound cost
    cutoff = min(100, logits.size(-1))
    topc = sorted_ids[:, :cutoff]
    match = (topc == targets.unsqueeze(1)).float()
    ranks = (match * (1.0 / torch.arange(1, cutoff + 1, device=logits.device).float())).sum(dim=1)
    out["mrr"] = float(ranks.mean().item())

    if variant_ids is not None:
        by_variant = {}
        for vid in variant_ids.unique().tolist():
            sel = variant_ids == vid
            if sel.any():
                tk1 = (sorted_ids[sel, :1] == targets[sel].unsqueeze(1)).any(dim=1).float().mean().item()
                by_variant[int(vid)] = {"top1_accuracy": float(tk1), "n": int(sel.sum().item())}
        out["by_variant"] = by_variant

    return out


class StreamingAccumulator:
    """Collect (logits, targets, mask, variant_ids) chunks then call compute() once."""

    def __init__(self):
        self.parts: list[tuple] = []

    def add(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, variant_ids: torch.Tensor | None = None):
        mask_b = mask.bool()
        if not mask_b.any():
            return
        self.parts.append(
            (
                logits[mask_b].detach().cpu(),
                targets[mask_b].detach().cpu(),
                variant_ids[mask_b].detach().cpu() if variant_ids is not None else None,
            )
        )

    def finalize(self, top_k=(1, 5)) -> dict:
        if not self.parts:
            return {"n_tokens": 0}
        logits = torch.cat([p[0] for p in self.parts], dim=0)
        targets = torch.cat([p[1] for p in self.parts], dim=0)
        vids = torch.cat([p[2] for p in self.parts], dim=0) if self.parts[0][2] is not None else None
        mask = torch.ones(targets.size(0), dtype=torch.bool)
        return compute(logits, targets, mask, vids, top_k=top_k)
