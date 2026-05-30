# tests/test_tokenizer.py
from procseq.tokenizer import build_tokenizer, encode_sequence, decode_to_steps

def test_each_step_is_one_token():
    tok = build_tokenizer()
    steps = ["RECEIVE WAFER LOT", "LOT IDENTIFICATION", "SHIP LOT"]
    ids = encode_sequence(tok, steps, family="MOSFET", add_bos_eos=True)
    # [BOS][FAM_MOSFET] + 3 steps + [EOS] = 6 tokens
    assert len(ids) == 6

def test_roundtrip_steps():
    tok = build_tokenizer()
    steps = ["RECEIVE WAFER LOT", "THERMAL OXIDATION", "SHIP LOT"]
    ids = encode_sequence(tok, steps, family="IGBT", add_bos_eos=False)
    assert decode_to_steps(tok, ids) == steps
