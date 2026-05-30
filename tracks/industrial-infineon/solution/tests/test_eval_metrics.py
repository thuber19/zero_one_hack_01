# tests/test_eval_metrics.py
from procseq import eval_metrics as em

def test_mrr_and_topk():
    preds = {"a": ["X", "Y", "Z", "P", "Q"], "b": ["Y", "X", "Z", "P", "Q"]}
    gold = {"a": "X", "b": "X"}
    r = em.score_nextstep(preds, gold)
    assert r["top1"] == 0.5
    assert r["top3"] == 1.0
    assert abs(r["mrr"] - ((1.0 + 0.5) / 2)) < 1e-9

def test_normalized_edit_distance_identity():
    assert em.normalized_edit_distance(["a", "b"], ["a", "b"]) == 1.0
    assert em.normalized_edit_distance(["a", "b"], ["a", "c"]) == 0.5

def test_completion_exact_match_and_token_acc():
    preds = {"a": ["c", "d"], "b": ["c", "x"]}
    gold = {"a": ["c", "d"], "b": ["c", "d"]}
    r = em.score_completion(preds, gold)
    assert r["exact_match"] == 0.5
    assert abs(r["token_accuracy"] - 0.75) < 1e-9

def test_anomaly_f1_and_auc():
    # IS_VALID convention: 1 valid, 0 invalid; positive class = invalid
    pred = {"a": (0, 0.1, "RULE_DEP_NO_CLEAN"), "b": (1, 0.9, ""),
            "c": (0, 0.2, "RULE_ETCH_NO_MASK"), "d": (1, 0.8, "")}
    gold = {"a": (0, "RULE_DEP_NO_CLEAN"), "b": (1, ""),
            "c": (0, "RULE_CMP_NO_DEP"), "d": (1, "")}
    r = em.score_anomaly(pred, gold)
    assert r["binary_accuracy"] == 1.0
    assert r["f1"] == 1.0
    assert r["rule_attribution_accuracy"] == 0.5  # 1 of 2 invalids matched
    assert 0.0 <= r["roc_auc"] <= 1.0
