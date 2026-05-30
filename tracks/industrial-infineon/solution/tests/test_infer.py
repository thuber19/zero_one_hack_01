# tests/test_infer.py
from procseq.tokenizer import build_tokenizer
from procseq.models.decoder import build_decoder
from procseq import infer

def test_nextstep_returns_5_valid_steps():
    tok = build_tokenizer()
    model = build_decoder("tiny", tok)
    partial = ["RECEIVE WAFER LOT", "LOT IDENTIFICATION"]
    ranked = infer.predict_next_step(model, tok, partial, "MOSFET", k=5)
    assert len(ranked) == 5
    assert all(isinstance(s, str) and "_" not in s for s in ranked)  # decoded to spaces

def test_completion_stops_and_returns_suffix():
    tok = build_tokenizer()
    model = build_decoder("tiny", tok)
    partial = ["RECEIVE WAFER LOT"]
    suffix = infer.complete_sequence(model, tok, partial, "IC", max_new=10)
    assert isinstance(suffix, list)
    assert len(suffix) <= 10
