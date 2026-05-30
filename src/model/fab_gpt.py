"""FabGPT — decoder-only Transformer, Pre-LN, learned positional embeddings.

Frozen contract for Spec 003 consumers:
    model.next_step_logits(input_ids) -> Tensor[B, V]   # logits for next token given prefix
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FabGPTConfig:
    vocab_size: int
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    d_ff: int = 2048
    max_len: int = 256
    dropout: float = 0.1
    tie_embeddings: bool = True
    pad_id: int = 0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: FabGPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # [B, T, H, D]
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, T, D]
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: FabGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class FabGPT(nn.Module):
    def __init__(self, cfg: FabGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        if cfg.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _logits_from_hidden(self, h: torch.Tensor) -> torch.Tensor:
        if self.lm_head is not None:
            return self.lm_head(h)
        return F.linear(h, self.tok_emb.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        if T > self.cfg.max_len:
            raise ValueError(f"sequence length {T} > max_len {self.cfg.max_len}")
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self._logits_from_hidden(x)

    @torch.no_grad()
    def next_step_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Frozen API for Spec 003. input_ids [B, T] → logits [B, V] for next token."""
        was_training = self.training
        self.eval()
        T = input_ids.size(1)
        if T > self.cfg.max_len:
            input_ids = input_ids[:, -self.cfg.max_len :]
        logits = self.forward(input_ids)
        out = logits[:, -1, :]
        if was_training:
            self.train()
        return out

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        eos_id: int = 2,
    ) -> torch.Tensor:
        self.eval()
        seq = prompt
        for _ in range(max_new_tokens):
            logits = self.next_step_logits(seq)
            if temperature <= 0:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            seq = torch.cat([seq, next_tok], dim=1)
            if (next_tok == eos_id).all():
                break
        return seq

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
