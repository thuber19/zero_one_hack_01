"""From-scratch DeBERTa-v2-style encoder with binary + multi-label rule heads."""
import torch
import torch.nn as nn
from transformers import DebertaV2Config, DebertaV2Model

SIZES = {
    "tiny":  dict(hidden_size=128, intermediate_size=256, num_hidden_layers=4,
                  num_attention_heads=4),
    "small": dict(hidden_size=256, intermediate_size=768, num_hidden_layers=6,
                  num_attention_heads=8),
    "base":  dict(hidden_size=512, intermediate_size=1536, num_hidden_layers=8,
                  num_attention_heads=8),
}

class ProcessAnomalyModel(nn.Module):
    def __init__(self, cfg, n_rules, proj_dim=128):
        super().__init__()
        self.encoder = DebertaV2Model(cfg)
        h = cfg.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.invalid_head = nn.Linear(h, 1)
        self.rule_head = nn.Linear(h, n_rules)
        # Projection head for the supervised-contrastive objective (SupCon).
        self.proj = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, proj_dim))

    def forward(self, input_ids, attention_mask=None, **_):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]      # [CLS]
        pooled = self.dropout(cls)
        return {"invalid_logit": self.invalid_head(pooled).squeeze(-1),
                "rule_logits": self.rule_head(pooled),
                "embed": self.proj(cls)}       # contrastive embedding (unnormalized)

def build_encoder(size, tokenizer, n_rules, max_position_embeddings=256, proj_dim=128):
    p = SIZES[size]
    cfg = DebertaV2Config(
        vocab_size=len(tokenizer), max_position_embeddings=max_position_embeddings,
        pad_token_id=tokenizer.pad_token_id, relative_attention=True,
        pos_att_type=["p2c", "c2p"], **p,
    )
    return ProcessAnomalyModel(cfg, n_rules, proj_dim=proj_dim)
