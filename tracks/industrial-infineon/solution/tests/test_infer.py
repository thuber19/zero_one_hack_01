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

def test_constrained_completion_adds_no_local_violation():
    # Every grammar rule fires at its trigger step's index, so a correctly
    # vetoed completion must introduce no violation at any generated position.
    from procseq.grammar import validate_sequence
    tok = build_tokenizer()
    model = build_decoder("tiny", tok)
    prefix = ["RECEIVE WAFER LOT", "LOT IDENTIFICATION", "INITIAL WAFER INSPECTION"]
    suffix = infer.complete_sequence(model, tok, prefix, "MOSFET",
                                     max_new=12, constrain=True)
    full = prefix + suffix
    start = len(prefix)
    assert all(v.step_index < start for v in validate_sequence(full))

def test_constrained_nextstep_still_returns_k():
    tok = build_tokenizer()
    model = build_decoder("tiny", tok)
    partial = ["RECEIVE WAFER LOT", "LOT IDENTIFICATION"]
    ranked = infer.predict_next_step(model, tok, partial, "MOSFET", k=5, constrain=True)
    assert len(ranked) == 5
