# tests/test_infer_anomaly.py
import torch
from procseq.tokenizer import build_tokenizer
from procseq.models.encoder import build_encoder
from procseq.grammar import RULE_IDS
from procseq.infer_anomaly import classify_sequence

def test_classify_returns_triple():
    tok = build_tokenizer()
    model = build_encoder("tiny", tok, n_rules=len(RULE_IDS))
    steps = ["RECEIVE WAFER LOT", "THERMAL OXIDATION", "SHIP LOT"]
    is_valid, score, rule = classify_sequence(model, tok, steps, "MOSFET", RULE_IDS)
    assert is_valid in (0, 1)
    assert 0.0 <= score <= 1.0
    assert rule in RULE_IDS or rule == ""
