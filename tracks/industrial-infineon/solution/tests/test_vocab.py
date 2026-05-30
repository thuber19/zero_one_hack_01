# tests/test_vocab.py
from procseq import vocab

def test_step_token_roundtrip():
    assert vocab.step_to_token("RECEIVE WAFER LOT") == "RECEIVE_WAFER_LOT"
    assert vocab.token_to_step("RECEIVE_WAFER_LOT") == "RECEIVE WAFER LOT"

def test_build_vocab_contains_core_steps_and_specials():
    v = vocab.build_vocab()
    assert "RECEIVE_WAFER_LOT" in v
    assert "SHIP_LOT" in v
    assert "ALIGN_MASK_LEVEL_3" in v
    for s in vocab.SPECIAL_TOKENS:
        assert s in v
    # specials occupy the first ids
    assert v["[PAD]"] == 0
    assert len(v) > 150
