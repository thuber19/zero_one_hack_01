"""Build organizer-format internal eval mirrors from held-out sequences."""
import csv
import math
import random
from pathlib import Path
from procseq.anomaly_inject import inject_random

def _cut(steps: list[str], frac: float) -> int:
    return max(1, int(math.floor(len(steps) * frac)))

def build_valid_rows(seqs, family, fractions, rng):
    rows = []
    for sid, steps in seqs.items():
        for frac in fractions:
            k = _cut(steps, frac)
            rows.append({
                "EXAMPLE_ID": f"{family}_{sid}_{int(frac*100)}",
                "FAMILY": family,
                "COMPLETION_FRACTION": frac,
                "PARTIAL_SEQUENCE": "|".join(steps[:k]),
                "FULL_SEQUENCE": "|".join(steps),
            })
    return rows

def build_anomaly_rows(seqs, family, n_valid, n_invalid, rng):
    items = list(seqs.values())
    rng.shuffle(items)
    rows = []
    for i in range(min(n_valid, len(items))):
        rows.append({"EXAMPLE_ID": f"{family}_valid_{i:04d}", "FAMILY": family,
                     "SEQUENCE": "|".join(items[i]), "IS_VALID": 1, "PREDICTED_RULE": ""})
    pool = items[n_valid:] or items
    for i in range(n_invalid):
        base = pool[i % len(pool)]
        seq, rule = inject_random(base, rng)
        rows.append({"EXAMPLE_ID": f"{family}_inval_{i:04d}", "FAMILY": family,
                     "SEQUENCE": "|".join(seq), "IS_VALID": 0, "PREDICTED_RULE": rule})
    rng.shuffle(rows)
    return rows

def write_valid_files(rows, input_path: Path, gt_path: Path):
    with input_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION", "PARTIAL_SEQUENCE"])
        for r in rows:
            w.writerow([r["EXAMPLE_ID"], r["FAMILY"], r["COMPLETION_FRACTION"],
                        r["PARTIAL_SEQUENCE"]])
    with gt_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "FULL_SEQUENCE"])
        for r in rows:
            w.writerow([r["EXAMPLE_ID"], r["FULL_SEQUENCE"]])

def write_anomaly_files(rows, input_path: Path, labels_path: Path):
    with input_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "FAMILY", "SEQUENCE"])
        for r in rows:
            w.writerow([r["EXAMPLE_ID"], r["FAMILY"], r["SEQUENCE"]])
    with labels_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EXAMPLE_ID", "IS_VALID", "PREDICTED_RULE"])
        for r in rows:
            w.writerow([r["EXAMPLE_ID"], r["IS_VALID"], r["PREDICTED_RULE"]])
