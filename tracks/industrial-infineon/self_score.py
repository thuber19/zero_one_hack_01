#!/usr/bin/env python3
"""
self_score.py — score the three submission CSVs against a ground-truth answer key
(answers.json produced by make_sample_eval.py). Lets us benchmark the FULL
pipeline on held-out data without the organizers' scorer.

NOTE: this is OUR re-implementation of the documented metrics on a SAMPLE set;
the official numbers come from the organizers' eval_metrics.py. Use for sanity
+ regression, not as the final score.

Usage: python self_score.py --submission-dir outputs_M1/submissions --answers sample_eval/answers.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


def norm_edit_distance(pred, gold):
    """Token-level Levenshtein / max(len) — the documented T2 metric (lower=better)."""
    n, m = len(pred), len(gold)
    if n == 0 and m == 0:
        return 0.0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (pred[i - 1] != gold[j - 1]))
            prev = cur
    return dp[m] / max(n, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", default="outputs_M1/submissions")
    ap.add_argument("--answers", default="sample_eval/answers.json")
    args = ap.parse_args()
    sub = Path(args.submission_dir)
    ans = json.load(open(args.answers, encoding="utf-8"))

    def rd(name):
        return list(csv.DictReader(open(sub / name, encoding="utf-8-sig")))

    # ── Task 1 ──
    rows = rd("nextstep.csv")
    t1 = t3 = t5 = mrr = 0
    for r in rows:
        gt = ans["nextstep"][r["EXAMPLE_ID"]]
        ranks = [r[f"RANK_{k}"] for k in range(1, 6)]
        if gt in ranks:
            rk = ranks.index(gt) + 1
            mrr += 1.0 / rk
            t5 += 1
            t3 += rk <= 3
            t1 += rk == 1
    n = len(rows)
    print(f"T1 next-step : Top-1={t1/n:.3f} Top-3={t3/n:.3f} Top-5={t5/n:.3f} MRR={mrr/n:.3f} (n={n})")

    # ── Task 2 ──
    rows = rd("completion.csv")
    neds, exact = [], 0
    for r in rows:
        gold = ans["completion"][r["EXAMPLE_ID"]]
        pred = [s for s in r["PREDICTED_SEQUENCE"].split("|") if s]
        neds.append(norm_edit_distance(pred, gold))
        exact += pred == gold
    print(f"T2 completion: NormEditDist={statistics.mean(neds):.3f} (lower=better) "
          f"ExactMatch={exact/len(rows):.3f} (n={len(rows)})")

    # ── Task 3 ──
    rows = rd("anomaly.csv")
    tp = fp = tn = fn = ratt = rtot = 0
    for r in rows:
        gt = ans["anomaly"][r["EXAMPLE_ID"]]
        pv, gv = int(r["IS_VALID"]), gt["is_valid"]
        if gv == 0 and pv == 0: tp += 1
        elif gv == 1 and pv == 0: fp += 1
        elif gv == 1 and pv == 1: tn += 1
        else: fn += 1
        if gv == 0:
            rtot += 1
            ratt += r["PREDICTED_RULE"] == gt["rule"]
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"T3 anomaly   : Acc={(tp+tn)/len(rows):.3f} Prec={prec:.3f} Rec={rec:.3f} "
          f"F1={f1:.3f} RuleAttr={ratt}/{rtot} | TP={tp} FP={fp} TN={tn} FN={fn}")


if __name__ == "__main__":
    main()
