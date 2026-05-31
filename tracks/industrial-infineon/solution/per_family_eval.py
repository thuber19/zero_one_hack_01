#!/usr/bin/env python3
"""
per_family_eval.py - per-family (MOSFET / IGBT / IC) breakdown of the submission.

MIRRORS procseq/run_eval.py exactly (same gold construction from
eval_valid_groundtruth.csv + eval_input_valid.csv, same procseq.eval_metrics
scorers, same prediction files), but partitions every EXAMPLE_ID by its family
prefix and scores each family separately. Because it reuses the identical
code+data that produced metrics.json / metrics_hybrid.json, the per-family rows
aggregate (n-weighted) back to those headline numbers.

  * Task 1 (next-step)  - submission_task1_hybrid.csv  (submitted hybrid)
  * Task 2 (completion) - submission_task2.csv          (decoder + physics)
  * Task 3 (anomaly)    - submission_task3_hybrid.csv   (submitted hybrid)

Writes artifacts/metrics_per_family.json + artifacts/per_family_scores.txt.

Usage:
  python per_family_eval.py --run ../results/procseq_base_d20000_s16001_seed11101
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # .../solution
sys.path.insert(0, str(_HERE))
from procseq import eval_metrics as em           # the scorer that produced metrics.json

FAMS = ["MOSFET", "IGBT", "IC"]
_fam = lambda eid: eid.split("_")[0].upper()


def _rows(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def compute(run: Path) -> dict:
    # ---- gold, built EXACTLY as run_eval.py does ----
    full = {r["EXAMPLE_ID"]: r["FULL_SEQUENCE"].split("|") for r in _rows(run / "eval_valid_groundtruth.csv")}
    partial = {r["EXAMPLE_ID"]: r["PARTIAL_SEQUENCE"].split("|") for r in _rows(run / "eval_input_valid.csv")}
    ns_gold = {eid: full[eid][len(partial[eid])] for eid in partial if len(full[eid]) > len(partial[eid])}
    comp_gold = {eid: full[eid][len(partial[eid]):] for eid in partial}
    anom_gold = {r["EXAMPLE_ID"]: (int(r["IS_VALID"]), r["PREDICTED_RULE"])
                 for r in _rows(run / "eval_anomaly_labels.csv")}

    # ---- predictions (submitted variants) ----
    t1 = {r["EXAMPLE_ID"]: [r[f"RANK_{k}"] for k in range(1, 6)]
          for r in _rows(run / "submission_task1_hybrid.csv")}
    t2 = {r["EXAMPLE_ID"]: (r["PREDICTED_SEQUENCE"].split("|") if r["PREDICTED_SEQUENCE"] else [])
          for r in _rows(run / "submission_task2.csv")}
    t3 = {r["EXAMPLE_ID"]: (int(r["IS_VALID"]), float(r["SCORE"] or 0.5), r["PREDICTED_RULE"])
          for r in _rows(run / "submission_task3_hybrid.csv")}

    res = {"task1_nextstep": {}, "task2_completion": {}, "task3_anomaly": {}}
    for fam in FAMS:
        gn = {i: v for i, v in ns_gold.items() if _fam(i) == fam}
        gc = {i: v for i, v in comp_gold.items() if _fam(i) == fam}
        ga = {i: v for i, v in anom_gold.items() if _fam(i) == fam}
        res["task1_nextstep"][fam] = em.score_nextstep({i: t1[i] for i in gn if i in t1}, gn)
        res["task2_completion"][fam] = em.score_completion({i: t2[i] for i in gc if i in t2}, gc)
        res["task3_anomaly"][fam] = em.score_anomaly({i: t3[i] for i in ga if i in t3}, ga)
    return {"run": run.name,
            "note": "Per-family breakdown - mirrors procseq/run_eval.py (same gold + "
                    "procseq.eval_metrics scorers + prediction files), partitioned by "
                    "family. Task1+Task3 = submitted hybrid; Task2 = decoder+physics. "
                    "Per-family rows aggregate (n-weighted) back to metrics.json / "
                    "metrics_hybrid.json.",
            **res}


def _g(d, k):
    return d.get(k, float("nan"))


def _table(res: dict) -> str:
    L = [f"Per-family breakdown - run {res['run']}", ""]
    L.append("TASK 1  next-step (hybrid)      n   Top-1  Top-5   MRR   cat-T1")
    for f in FAMS:
        d = res["task1_nextstep"].get(f)
        if d: L.append("  %-9s %18d  %.3f  %.3f  %.3f  %.3f" % (
            f, d["n"], d["top1"], d["top5"], d["mrr"], d["top1_category"]))
    L += ["", "TASK 2  completion              n   block  token  cat-tok  exact"]
    for f in FAMS:
        d = res["task2_completion"].get(f)
        if d: L.append("  %-9s %18d  %.3f  %.3f  %.3f   %.3f" % (
            f, d["n"], d["block_accuracy"], d["token_accuracy"],
            d["category_token_accuracy"], d["exact_match"]))
    L += ["", "TASK 3  anomaly (hybrid)        n   binAcc  F1    ruleAttr"]
    for f in FAMS:
        d = res["task3_anomaly"].get(f)
        if d: L.append("  %-9s %18d  %.3f   %.3f  %.3f" % (
            f, d["n"], _g(d, "binary_accuracy"), _g(d, "f1"),
            _g(d, "rule_attribution_accuracy")))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--json-out", default="artifacts/metrics_per_family.json", type=Path)
    ap.add_argument("--txt-out", default="artifacts/per_family_scores.txt", type=Path)
    args = ap.parse_args()
    run = args.run if args.run.is_absolute() else (_HERE / args.run)
    res = compute(run)
    jo = args.json_out if args.json_out.is_absolute() else (_HERE / args.json_out)
    to = args.txt_out if args.txt_out.is_absolute() else (_HERE / args.txt_out)
    jo.parent.mkdir(parents=True, exist_ok=True)
    jo.write_text(json.dumps(res, indent=2))
    table = _table(res)
    to.write_text(table)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(table)
    print(f"wrote {jo}")
    print(f"wrote {to}")


if __name__ == "__main__":
    main()
