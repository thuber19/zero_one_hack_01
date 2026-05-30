"""PyTorch Dataset over packed token shards."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class PackedShardDataset(Dataset):
    """Loads .pt shards (each a LongTensor [N, max_len]) and concatenates."""

    def __init__(self, shard_paths: list[str | Path]):
        tensors = [torch.load(p, map_location="cpu", weights_only=True) for p in shard_paths]
        self.data: torch.Tensor = torch.cat(tensors, dim=0).long()

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def loss_mask(tokens: torch.Tensor, pad_id: int = 0, variant_ids: tuple[int, ...] = (4, 5, 6)) -> torch.Tensor:
    """Returns mask of 1s where we want to compute next-step loss (positions whose TARGET is informative).

    Shape: tokens [B, T] → mask [B, T-1] aligned with targets = tokens[:, 1:].
    We exclude positions where TARGET is PAD or VARIANT.
    """
    targets = tokens[:, 1:]
    mask = (targets != pad_id)
    for vid in variant_ids:
        mask &= (targets != vid)
    return mask
