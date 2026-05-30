"""
LSTM model for process sequence modeling.

Alternative to the transformer — simpler architecture that may
perform well on structured sequential data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProcessLSTM(nn.Module):
    """
    LSTM-based sequence model for semiconductor process sequences.

    Parameters
    ----------
    vocab_size : int
        Number of tokens (process steps + special tokens).
    d_model : int
        Embedding and hidden dimension.
    n_layers : int
        Number of LSTM layers.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : (B, T) long tensor
        attention_mask : (B, T) long tensor, 1=attend, 0=ignore

        Returns
        -------
        logits : (B, T, vocab_size)
        """
        x = self.token_emb(input_ids)
        x = self.drop(x)

        # Pack padded sequences for efficiency
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            output, _ = self.lstm(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True, total_length=input_ids.shape[1]
            )
        else:
            x, _ = self.lstm(x)

        x = self.ln(x)
        logits = self.head(x)
        return logits

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.forward(input_ids, attention_mask)
        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = target_ids.reshape(-1)
        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=0)
        return loss

    @torch.no_grad()
    def get_next_step_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.forward(input_ids, attention_mask)
        if attention_mask is not None:
            last_idx = attention_mask.sum(dim=1) - 1
            logits_last = logits[0, last_idx[0]]
        else:
            logits_last = logits[0, -1]
        return F.softmax(logits_last, dim=-1)

    @torch.no_grad()
    def sequence_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> float:
        B, T = input_ids.shape
        logits = self.forward(input_ids, attention_mask)
        logits = logits[:, :-1]
        targets = input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_losses = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        if attention_mask is not None:
            mask = attention_mask[:, 1:]
            token_losses = token_losses * mask
            avg_loss = token_losses.sum() / mask.sum()
        else:
            avg_loss = token_losses.mean()
        return avg_loss.item()


def create_lstm_model(
    vocab_size: int,
    size: str = "small",
) -> ProcessLSTM:
    """
    Factory for different LSTM sizes.

    Sizes:
      tiny:   2 layers, 128 dim  (~200K params)
      small:  2 layers, 256 dim  (~1.5M params)
      medium: 3 layers, 512 dim  (~10M params)
    """
    configs = {
        "tiny": dict(d_model=128, n_layers=2, dropout=0.1),
        "small": dict(d_model=256, n_layers=2, dropout=0.1),
        "medium": dict(d_model=512, n_layers=3, dropout=0.1),
    }
    if size not in configs:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(configs.keys())}")

    return ProcessLSTM(vocab_size=vocab_size, **configs[size])
