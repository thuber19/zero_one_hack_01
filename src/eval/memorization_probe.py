"""Honest-evaluation probe: perturbed-sequence score ratio.

If the model has learned real process logic, its log-likelihood on a true
test sequence should be much higher than on a randomly-shuffled version of
the same sequence. Ratio < 5 → suspicious memorization or trivial baseline.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F


@torch.no_grad()
def sequence_nll(model, token_ids: list[int], device: torch.device) -> float:
    if len(token_ids) < 2:
        return 0.0
    x = torch.tensor([token_ids], device=device, dtype=torch.long)
    logits = model(x)
    log_probs = F.log_softmax(logits[:, :-1].float(), dim=-1)
    targets = x[:, 1:]
    nll = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    return float(nll.mean().item())


@torch.no_grad()
def perturbed_score_ratio(
    model,
    test_token_lists: list[list[int]],
    device: torch.device,
    n: int = 100,
    seed: int = 42,
) -> dict:
    """Returns dict with 'true_mean_nll', 'perturbed_mean_nll', 'ratio'."""
    rng = random.Random(seed)
    sampled = test_token_lists[:n] if len(test_token_lists) >= n else test_token_lists
    true_nlls, perm_nlls = [], []
    for seq in sampled:
        if len(seq) < 4:
            continue
        true_nlls.append(sequence_nll(model, seq, device))
        # shuffle middle (preserve variant + BOS at start, EOS at end)
        head, mid, tail = seq[:2], seq[2:-1], seq[-1:]
        mid_perm = mid[:]
        rng.shuffle(mid_perm)
        perm_nlls.append(sequence_nll(model, head + mid_perm + tail, device))
    if not true_nlls:
        return {"true_mean_nll": 0.0, "perturbed_mean_nll": 0.0, "ratio": 0.0, "n": 0}
    t = sum(true_nlls) / len(true_nlls)
    p = sum(perm_nlls) / len(perm_nlls)
    ratio = (p / t) if t > 0 else float("inf")
    return {
        "true_mean_nll": t,
        "perturbed_mean_nll": p,
        "ratio": ratio,
        "n": len(true_nlls),
    }
