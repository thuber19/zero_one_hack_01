#!/usr/bin/env python3
"""
make_sample_eval.py — build a SPEC-FORMAT *sample* eval set so the FULL
submission pipeline can be dry-run and self-scored locally, WITHOUT the
organizers' (not-yet-distributed) eval files.

Writes, into --out-dir:
  eval_input_valid.csv    (EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE)
  eval_input_anomaly.csv  (EXAMPLE_ID, FAMILY, SEQUENCE)
  answers.json            ground-truth key (next step, full completion, is_valid, rule)

CLEARLY A STAND-IN: these are sampled from the PROVIDED in-vocabulary data and
labelled by the reference checker. Real scoring uses the organizers' files +
eval_metrics.py. This only lets us prove the pipeline runs end-to-end, the
output format is spec-clean, and the numbers are plausible on held-out data.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "training_data"))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from generate_sequences import read_csv_sequences
import bad_data_generator as BDG

_FILES = {"mosfet": "MOSFET_variants.csv", "igbt": "IGBT_variants.csv", "ic": "IC_variants.csv"}


def load_valid_by_family():
    out = {}
    for fam, fname in _FILES.items():
        p = _ROOT / "training_data" / fname
        seqs = list(read_csv_sequences(p).values()) if p.exists() else []
        out[fam] = seqs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-family", type=int, default=6)
    ap.add_argument("--out-dir", default="sample_eval")
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = _ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    valids = load_valid_by_family()

    valid_rows, anomaly_rows = [], []
    answers = {"nextstep": {}, "completion": {}, "anomaly": {}}
    vid = 0

    # ── T1/T2 source: cut real sequences at 0.6 and 0.8 ──
    for fam, seqs in valids.items():
        rng.shuffle(seqs)
        for seq in seqs[: args.n_per_family]:
            for frac in (0.6, 0.8):
                cut = max(4, int(len(seq) * frac))
                if cut >= len(seq):
                    continue
                vid += 1
                eid = f"valid_{vid:04d}"
                partial, nxt, rest = seq[:cut], seq[cut], seq[cut:]
                valid_rows.append({
                    "EXAMPLE_ID": eid, "FAMILY": fam,
                    "COMPLETION_FRACTION": frac,
                    "PARTIAL_SEQUENCE": "|".join(partial)})
                answers["nextstep"][eid] = nxt
                answers["completion"][eid] = rest

    # ── T3 source: valid (label 1) + reference-labelled bad (label 0) ──
    bad_records, _neg = BDG.build(per_combo=1, seed=args.seed)
    rng.shuffle(bad_records)
    aid = 0
    # valid anomaly rows
    for fam, seqs in valids.items():
        for seq in seqs[args.n_per_family: args.n_per_family * 2]:
            aid += 1
            eid = f"valid_{aid:04d}"
            anomaly_rows.append({"EXAMPLE_ID": eid, "FAMILY": fam,
                                 "SEQUENCE": "|".join(seq)})
            answers["anomaly"][eid] = {"is_valid": 1, "rule": ""}
    # invalid anomaly rows
    for rec in bad_records[: args.n_per_family * 3]:
        aid += 1
        eid = f"forbidden_{aid:04d}"
        anomaly_rows.append({"EXAMPLE_ID": eid, "FAMILY": rec["family"].lower(),
                             "SEQUENCE": "|".join(rec["steps"])})
        answers["anomaly"][eid] = {"is_valid": 0, "rule": rec["first_rule"]}
    rng.shuffle(anomaly_rows)

    with open(out / "eval_input_valid.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION", "PARTIAL_SEQUENCE"])
        w.writeheader(); w.writerows(valid_rows)
    with open(out / "eval_input_anomaly.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["EXAMPLE_ID", "FAMILY", "SEQUENCE"])
        w.writeheader(); w.writerows(anomaly_rows)
    (out / "answers.json").write_text(json.dumps(answers, indent=2))

    print(f"Sample eval written to {out}")
    print(f"  eval_input_valid.csv   : {len(valid_rows)} rows (T1 + T2)")
    print(f"  eval_input_anomaly.csv : {len(anomaly_rows)} rows "
          f"({sum(1 for v in answers['anomaly'].values() if v['is_valid']==1)} valid / "
          f"{sum(1 for v in answers['anomaly'].values() if v['is_valid']==0)} invalid)")
    print(f"  answers.json           : ground-truth key for self-scoring")


if __name__ == "__main__":
    main()
