"""
Small causal transformer for process sequence modeling.

Architecture: GPT-style decoder-only transformer trained from scratch
with a custom ~130 token vocabulary.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProcessTransformer(nn.Module):
    """
    Small GPT-style transformer for semiconductor process sequences.

    Parameters
    ----------
    vocab_size : int
        Number of tokens (process steps + special tokens).
    d_model : int
        Hidden dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer decoder layers.
    max_seq_len : int
        Maximum sequence length.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        max_seq_len: int = 200,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
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
        B, T = input_ids.shape
        device = input_ids.device

        positions = torch.arange(T, device=device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.drop(x)

        # Causal mask: prevent attending to future tokens
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=device, dtype=x.dtype
        )

        # Padding mask for transformer: True = ignore
        if attention_mask is not None:
            pad_mask = attention_mask == 0  # True where padded
        else:
            pad_mask = None

        # TransformerDecoder needs memory — use dummy for decoder-only
        # Actually, let's use nn.TransformerEncoder instead for cleaner decoder-only
        # But we already set up TransformerDecoder, so pass x as both tgt and memory
        x = self.transformer(
            tgt=x,
            memory=torch.zeros(B, 1, self.d_model, device=device),
            tgt_mask=causal_mask,
            tgt_key_padding_mask=pad_mask,
        )

        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss, ignoring padded positions."""
        logits = self.forward(input_ids, attention_mask)
        # Flatten for cross-entropy
        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = target_ids.reshape(-1)
        # Ignore padding (target=0)
        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=0)
        return loss

    @torch.no_grad()
    def get_next_step_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Get probability distribution over next token.

        Parameters
        ----------
        input_ids : (1, T) — single sequence
        attention_mask : (1, T)

        Returns
        -------
        probs : (vocab_size,) — probability for each token
        """
        logits = self.forward(input_ids, attention_mask)
        # Take last real token's logits
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
        """
        Compute per-token loss for a single sequence.
        Useful for anomaly detection: high loss = unusual sequence.
        """
        B, T = input_ids.shape
        logits = self.forward(input_ids, attention_mask)
        # Shift: predict position t+1 from position t
        logits = logits[:, :-1]  # (B, T-1, V)
        targets = input_ids[:, 1:]  # (B, T-1)

        log_probs = F.log_softmax(logits, dim=-1)
        token_losses = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

        if attention_mask is not None:
            mask = attention_mask[:, 1:]  # align with targets
            token_losses = token_losses * mask
            avg_loss = token_losses.sum() / mask.sum()
        else:
            avg_loss = token_losses.mean()

        return avg_loss.item()


def create_model(
    vocab_size: int,
    size: str = "small",
) -> ProcessTransformer:
    """
    Factory for different model sizes.

    Sizes:
      tiny:  2 layers, 128 dim, 4 heads (~200K params)
      small: 6 layers, 256 dim, 8 heads (~3M params)
      medium: 8 layers, 512 dim, 8 heads (~20M params)
    """
    configs = {
        "tiny": dict(d_model=128, n_heads=4, n_layers=2, dropout=0.1),
        "small": dict(d_model=256, n_heads=8, n_layers=6, dropout=0.1),
        "medium": dict(d_model=512, n_heads=8, n_layers=8, dropout=0.1),
    }
    if size not in configs:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(configs.keys())}")

    return ProcessTransformer(vocab_size=vocab_size, **configs[size])
