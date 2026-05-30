"""BertMLMEncoder — 6-layer bidirectional encoder with MLM head.

Architecture: Pre-LN, learned positional embeddings, bidirectional attention.
Adapted from fab_gpt.py; key difference: no causal mask, MLM head instead of LM head.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BertMLMConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    max_len: int = 100
    dropout: float = 0.1
    pad_id: int = 0


class BidirectionalSelfAttention(nn.Module):
    def __init__(self, cfg: BertMLMConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, T, D]

        if attention_mask is not None:
            # attention_mask: [B, T] with 1=attend, 0=ignore (PAD)
            # expand to [B, 1, 1, T] for broadcasting over heads and query positions
            additive = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -1e9
        else:
            additive = None

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=additive,
            is_causal=False,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: BertMLMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = BidirectionalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class BertMLMEncoder(nn.Module):
    def __init__(self, cfg: BertMLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.mlm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=True)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        if T > self.cfg.max_len:
            raise ValueError(f"sequence length {T} > max_len {self.cfg.max_len}")
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        return self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return last-hidden-state [B, T, d_model] (no MLM head)."""
        x = self._embed(input_ids)
        for block in self.blocks:
            x = block(x, attention_mask)
        return self.ln_f(x)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return logits [B, T, vocab_size]."""
        h = self.encode(input_ids, attention_mask)
        return self.mlm_head(h)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
