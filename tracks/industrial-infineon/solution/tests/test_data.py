# tests/test_data.py
from procseq import data

def test_load_provided_returns_1000_each():
    d = data.load_provided("MOSFET")
    assert len(d) == 1000
    assert all(s[0] == "RECEIVE WAFER LOT" and s[-1] == "SHIP LOT"
               for s in list(d.values())[:5])

def test_split_is_deterministic_and_disjoint():
    seqs = {f"seq_{i:04d}": ["RECEIVE WAFER LOT", "SHIP LOT"] for i in range(100)}
    tr, va, te = data.split_ids(list(seqs), val_frac=0.1, test_frac=0.1, seed=7)
    tr2, va2, te2 = data.split_ids(list(seqs), val_frac=0.1, test_frac=0.1, seed=7)
    assert (tr, va, te) == (tr2, va2, te2)
    assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)
    assert len(tr) + len(va) + len(te) == 100

def test_ucbs_buckets_balance_lengths():
    seqs = [["a"] * 10] * 5 + [["a"] * 100] * 5
    weights = data.ucbs_weights(seqs, n_buckets=2)
    assert len(weights) == 10
    # short and long get equal total weight
    assert abs(sum(weights[:5]) - sum(weights[5:])) < 1e-6
