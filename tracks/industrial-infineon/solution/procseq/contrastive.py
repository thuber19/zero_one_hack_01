"""Supervised contrastive loss (Khosla et al. 2020) for the anomaly encoder.

Our anomaly_inject produces minimal-edit hard negatives: a valid sequence and
its rule-broken twin differ by a single rule violation. Pulling valid-sequence
embeddings together and pushing rule-violating ones away — with those twins as
the hardest in-batch negatives — teaches the encoder *why* a sequence is invalid
in representation space, rather than memorizing surface tokens. This complements
the BCE classification heads and tends to help F1 and OOD.
"""
import torch
import torch.nn.functional as F


def supcon_loss(embeddings: torch.Tensor, labels: torch.Tensor,
                temperature: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss over L2-normalized embeddings.

    embeddings: (N, D) — projection-head outputs (normalized internally).
    labels:     (N,)   — class id per sample (here: 0 = valid, 1 = invalid).
    Anchors with no same-label positive in the batch are skipped. Returns a
    differentiable scalar; 0 (graph-preserving) if no anchor has a positive.
    """
    device = embeddings.device
    z = F.normalize(embeddings, dim=1)
    n = z.size(0)
    sim = (z @ z.t()) / temperature                      # (N, N)
    # numerical stability: subtract per-row max (detached)
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.eye(n, dtype=torch.bool, device=device)
    labels = labels.contiguous().view(-1, 1)
    pos_mask = (labels == labels.t()) & ~self_mask       # same label, not self

    exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos_counts = pos_mask.sum(dim=1)
    has_pos = pos_counts > 0
    if not bool(has_pos.any()):
        return (embeddings * 0.0).sum()                  # keep graph, zero loss
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)[has_pos] / pos_counts[has_pos]
    return -mean_log_prob_pos.mean()
