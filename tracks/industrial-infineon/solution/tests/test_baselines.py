# tests/test_baselines.py
from procseq import baselines

def test_ngram_predicts_seen_transition():
    seqs = [["A", "B", "C"], ["A", "B", "D"], ["A", "B", "C"]]
    ng = baselines.NgramModel(n=2).fit(seqs)
    top = ng.predict_next(["A", "B"], k=5)
    assert top[0] == "C"  # C seen twice after A,B vs D once

def test_rule_oracle_flags_invalid():
    seq = ["RECEIVE WAFER LOT", "SHIP LOT", "WAFER SORT TEST"]
    is_valid, rule = baselines.rule_oracle(seq)
    assert is_valid == 0 and rule == "RULE_SHIP_BEFORE_TEST"
