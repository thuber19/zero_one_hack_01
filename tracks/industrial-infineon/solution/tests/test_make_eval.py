# tests/test_make_eval.py
import csv, random
from pathlib import Path
from procseq import make_eval

def test_valid_mirror_shapes(tmp_path):
    seqs = {f"seq_{i:04d}": ["RECEIVE WAFER LOT"] + ["A"] * 8 + ["SHIP LOT"]
            for i in range(10)}
    rows = make_eval.build_valid_rows(seqs, family="MOSFET",
                                      fractions=(0.6, 0.8), rng=random.Random(0))
    assert len(rows) == 20  # 10 seqs x 2 fractions
    r = rows[0]
    assert set(r) >= {"EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION",
                      "PARTIAL_SEQUENCE", "FULL_SEQUENCE"}
    assert "|" in r["PARTIAL_SEQUENCE"]

def test_anomaly_mirror_balance(tmp_path):
    valid = {f"v_{i}": ["RECEIVE WAFER LOT", "PRE CLEAN WAFER", "THERMAL OXIDATION",
                        "CURE PASSIVATION", "WAFER SORT TEST", "SHIP LOT"]
             for i in range(20)}
    rows = make_eval.build_anomaly_rows(valid, family="MOSFET",
                                        n_valid=10, n_invalid=8, rng=random.Random(0))
    assert len(rows) == 18
    n_inv = sum(1 for r in rows if r["IS_VALID"] == 0)
    assert n_inv == 8
    assert all(r["PREDICTED_RULE"] for r in rows if r["IS_VALID"] == 0)
