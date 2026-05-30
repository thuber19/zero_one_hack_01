"""LSTMModel — vanilla unidirectional LSTM for fab-process next-step prediction.

Frozen API contract for Spec 003 consumers:
    model.next_step_logits(input_ids) -> Tensor[B, V]  # logits for next token given prefix
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.embed_drop = nn.Dropout(dropout)
        # PyTorch LSTM applies inter-layer dropout natively when num_layers > 1
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_drop = nn.Dropout(dropout)
        # embed_dim != hidden_size so weight tying is not applicable here
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids [B, T] → logits [B, T, V]"""
        x = self.embed_drop(self.embedding(input_ids))
        lstm_out, _ = self.lstm(x)
        lstm_out = self.out_drop(lstm_out)
        return self.lm_head(lstm_out)

    def next_step_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Frozen API contract — logits [B, V] for the next token after the last position."""
        return self.forward(input_ids)[:, -1, :]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
