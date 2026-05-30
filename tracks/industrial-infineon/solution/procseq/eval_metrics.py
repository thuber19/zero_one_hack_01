"""Self-scoring harness for Tasks 1-3 + OOD + logic probe. numpy-only."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def _cat(step: str) -> str:
    """Functional category of a step (lazy import; identity-ish fallback)."""
    try:
        from procseq.external import category_of
        return category_of(step)
    except Exception:
        return step.split()[0] if step else ""


# ---------- Task 1: next-step ----------
def score_nextstep(preds: dict[str, list[str]], gold: dict[str, str]) -> dict:
    """Lexical Top-k/MRR plus category-level Top-1/MRR.

    Category metrics ask "did the model predict the right *kind* of step"
    (CLEAN/DEPOSIT/ETCH/...). They degrade far more gracefully than lexical
    accuracy under synonym/OOD shift, so they are the clearest signal that the
    model learned process logic rather than surface tokens.
    """
    ids = [i for i in gold if i in preds]
    top1 = top3 = top5 = 0.0
    mrr = 0.0
    cat_top1 = 0.0
    cat_mrr = 0.0
    for i in ids:
        ranked = preds[i]
        g = gold[i]
        rank = ranked.index(g) + 1 if g in ranked else None
        if rank == 1: top1 += 1
        if rank and rank <= 3: top3 += 1
        if rank and rank <= 5: top5 += 1
        mrr += (1.0 / rank) if rank else 0.0
        # category-level
        gcat = _cat(g)
        rcats = [_cat(s) for s in ranked]
        if rcats and rcats[0] == gcat:
            cat_top1 += 1
        crank = next((k + 1 for k, c in enumerate(rcats) if c == gcat), None)
        cat_mrr += (1.0 / crank) if crank else 0.0
    n = max(1, len(ids))
    return {"n": len(ids), "top1": top1/n, "top3": top3/n, "top5": top5/n,
            "mrr": mrr/n, "top1_category": cat_top1/n, "mrr_category": cat_mrr/n}

# ---------- Task 2: completion ----------
def _levenshtein(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + (a[i-1] != b[j-1]))
            prev = cur
    return dp[n]

def normalized_edit_distance(a, b):
    if not a and not b: return 1.0
    return 1.0 - _levenshtein(a, b) / max(len(a), len(b))

def _block_signature(steps):
    """Collapse consecutive duplicate category prefixes into ordered blocks."""
    sig = []
    for s in steps:
        cat = s.split()[0]
        if not sig or sig[-1] != cat:
            sig.append(cat)
    return sig

def score_completion(preds, gold):
    ids = [i for i in gold if i in preds]
    em = ned = tok = blk = catok = 0.0
    for i in ids:
        p, g = preds[i], gold[i]
        em += 1.0 if p == g else 0.0
        ned += normalized_edit_distance(p, g)
        L = max(1, len(g))
        tok += sum(1 for k in range(min(len(p), len(g))) if p[k] == g[k]) / L
        # category-level token accuracy: right kind of step at each position
        catok += sum(1 for k in range(min(len(p), len(g)))
                     if _cat(p[k]) == _cat(g[k])) / L
        blk += normalized_edit_distance(_block_signature(p), _block_signature(g))
    n = max(1, len(ids))
    return {"n": len(ids), "exact_match": em/n, "normalized_edit_distance": ned/n,
            "category_token_accuracy": catok/n,
            "token_accuracy": tok/n, "block_accuracy": blk/n}

# ---------- Task 3: anomaly ----------
def _auc(scores, labels):
    """scores = P(valid); positive class = invalid (label 0). AUC for detecting invalid."""
    inv = np.array([1 - l for l in labels])  # 1 = invalid (positive)
    s = -np.array(scores)                     # higher => more likely invalid
    order = np.argsort(s)
    inv = inv[order]
    P = inv.sum(); N = len(inv) - P
    if P == 0 or N == 0: return 0.5
    ranks = np.arange(1, len(inv) + 1)
    auc = (ranks[inv == 1].sum() - P * (P + 1) / 2) / (P * N)
    return float(auc)

def score_anomaly(pred, gold):
    ids = [i for i in gold if i in pred]
    tp = fp = tn = fn = 0
    rule_hit = rule_tot = 0
    scores, labels = [], []
    for i in ids:
        pv, ps, pr = pred[i]
        gv, gr = gold[i]
        scores.append(ps); labels.append(gv)
        if gv == 0 and pv == 0: tp += 1
        elif gv == 1 and pv == 0: fp += 1
        elif gv == 1 and pv == 1: tn += 1
        elif gv == 0 and pv == 1: fn += 1
        if gv == 0:
            rule_tot += 1
            if pv == 0 and pr == gr: rule_hit += 1
    n = max(1, len(ids))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"n": len(ids), "binary_accuracy": (tp + tn) / n,
            "precision": prec, "recall": rec, "f1": f1,
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "roc_auc": _auc(scores, labels),
            "rule_attribution_accuracy": rule_hit / max(1, rule_tot)}

# ---------- logic probe ----------
def logic_validity_rate(generated_full_sequences):
    """Fraction of generated full sequences with zero rule violations."""
    from procseq.grammar import validate_sequence
    ok = sum(1 for s in generated_full_sequences if not validate_sequence(s))
    return ok / max(1, len(generated_full_sequences))

# ---------- CLI ----------
def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["nextstep", "completion", "anomaly"], required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    gt = _read_csv(a.ground_truth); pr = _read_csv(a.predictions)
    if a.task == "nextstep":
        gold = {r["EXAMPLE_ID"]: r["NEXT_STEP"] for r in gt}
        preds = {r["EXAMPLE_ID"]: [r[f"RANK_{k}"] for k in range(1, 6)] for r in pr}
        res = score_nextstep(preds, gold)
    elif a.task == "completion":
        gold = {r["EXAMPLE_ID"]: r["SUFFIX"].split("|") for r in gt}
        preds = {r["EXAMPLE_ID"]: r["PREDICTED_SEQUENCE"].split("|") for r in pr}
        res = score_completion(preds, gold)
    else:
        gold = {r["EXAMPLE_ID"]: (int(r["IS_VALID"]), r["PREDICTED_RULE"]) for r in gt}
        preds = {r["EXAMPLE_ID"]: (int(r["IS_VALID"]), float(r.get("SCORE", 0.5) or 0.5),
                                   r.get("PREDICTED_RULE", "")) for r in pr}
        res = score_anomaly(preds, gold)
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
