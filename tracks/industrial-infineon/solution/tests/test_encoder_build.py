# tests/test_encoder_build.py
import torch
from procseq.tokenizer import build_tokenizer
from procseq.models.encoder import build_encoder
from procseq.grammar import RULE_IDS

def test_encoder_forward_shapes():
    tok = build_tokenizer()
    model = build_encoder("tiny", tok, n_rules=len(RULE_IDS))
    ids = torch.tensor([[tok.cls_token_id, 5, 6, tok.sep_token_id]])
    mask = torch.ones_like(ids)
    out = model(input_ids=ids, attention_mask=mask)
    assert out["invalid_logit"].shape == (1,)
    assert out["rule_logits"].shape == (1, len(RULE_IDS))
