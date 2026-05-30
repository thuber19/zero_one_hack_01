"""Score procseq submissions with the OFFICIAL data/eval_metrics.py — ONE
authoritative scorer for the decision (resolves the "two scorers" discrepancy).

It builds the official-format ground truth from our eval mirrors, then runs the
official scorer on each variant (pure neural vs. physics hybrid) for every task,
so "which variant wins" is decided by the same scorer the organizers use — never
our convenience harness.

    python -m procseq.score_official --config <cfg>
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

from procseq import grammar
from procseq.config import load_config


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _build_valid_gt(art, out):
    """Official next-step + completion GT from our mirror (NEXT_STEP derived)."""
    partial = {r["EXAMPLE_ID"]: r for r in _rows(art / "eval_input_valid.csv")}
    full = {r["EXAMPLE_ID"]: r["FULL_SEQUENCE"] for r in _rows(art / "eval_valid_groundtruth.csv")}
    fields = ["EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION",
              "PARTIAL_SEQUENCE", "FULL_SEQUENCE", "NEXT_STEP"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for eid, r in partial.items():
            pl = r["PARTIAL_SEQUENCE"].split("|")
            fl = full.get(eid, "").split("|")
            w.writerow({"EXAMPLE_ID": eid, "FAMILY": r["FAMILY"],
                        "COMPLETION_FRACTION": r["COMPLETION_FRACTION"],
                        "PARTIAL_SEQUENCE": r["PARTIAL_SEQUENCE"],
                        "FULL_SEQUENCE": full.get(eid, ""),
                        "NEXT_STEP": fl[len(pl)] if len(fl) > len(pl) else ""})


def _build_anomaly_gt(art, forbidden_out, supp_out):
    """Official forbidden set (IS_VALID=0 + VIOLATION_RULE) + valid supplement."""
    forb, supp = [], []
    for r in _rows(art / "eval_anomaly_labels.csv"):
        if int(r["IS_VALID"]) == 0:
            forb.append({"EXAMPLE_ID": r["EXAMPLE_ID"], "VIOLATION_RULE": r.get("PREDICTED_RULE", "")})
        else:
            supp.append({"EXAMPLE_ID": r["EXAMPLE_ID"]})
    with open(forbidden_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["EXAMPLE_ID", "VIOLATION_RULE"]); w.writeheader(); w.writerows(forb)
    with open(supp_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["EXAMPLE_ID"]); w.writeheader(); w.writerows(supp)


def _run(scorer, task, gt, pred, extra=None):
    cmd = [sys.executable, str(scorer), "--task", task,
           "--ground-truth", str(gt), "--predictions", str(pred)] + (extra or [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    art = Path(cfg["artifacts"])
    scorer = Path(grammar.TRAINING_DATA_DIR) / "eval_metrics.py"
    if not scorer.exists():
        raise SystemExit(f"official scorer not found at {scorer}")

    gt_valid = art / "_gt_valid.csv"
    gt_forb = art / "_gt_forbidden.csv"
    gt_supp = art / "_gt_valid_supp.csv"
    _build_valid_gt(art, gt_valid)
    _build_anomaly_gt(art, gt_forb, gt_supp)

    print("=" * 70)
    print("  OFFICIAL data/eval_metrics.py — pure neural vs. physics hybrid")
    print("=" * 70)
    for label, suf in [("pure", ""), ("hybrid", "_hybrid")]:
        p1 = art / f"submission_task1{suf}.csv"
        p2 = art / f"submission_task2{suf}.csv"
        p3 = art / f"submission_task3{suf}.csv"
        if p1.exists():
            print(f"\n----- TASK 1 next-step [{label}] -----\n{_run(scorer, 'next-step', gt_valid, p1)}")
        if p2.exists():
            print(f"\n----- TASK 2 completion [{label}] -----\n{_run(scorer, 'completion', gt_valid, p2)}")
        if p3.exists():
            print(f"\n----- TASK 3 anomaly [{label}] -----\n"
                  f"{_run(scorer, 'anomaly', gt_forb, p3, ['--valid-supplement', str(gt_supp)])}")


if __name__ == "__main__":
    main()
