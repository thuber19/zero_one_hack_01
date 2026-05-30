#!/usr/bin/env python3
"""Update threshold.json for a new product variant CSV.

Usage:
  python scripts/recalibrate_threshold.py \\
      --data IGBT_newvariant.csv \\
      --checkpoint $WORK/checkpoints/002/best_model.pt \\
      --output $WORK/checkpoints/002/threshold_igbt_v2.json

Warns if the reference CSV has < 10 sequences but continues (does not abort).
Runs on CPU in < 2 min for 50 sequences (SC-007).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.infer import load_model, pseudo_perplexity_batch


def _load_csv_sequences(path: Path) -> list[list[str]]:
    from collections import defaultdict
    seqs: dict[str, list[str]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("SEQUENCE_ID", "").strip()
            step = row.get("STEP", "").strip()
            if sid and step:
                seqs[sid].append(step)
    return list(seqs.values())


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    sorted_v = sorted(values)
    idx = max(0, int(len(sorted_v) * p / 100) - 1)
    return sorted_v[idx]


def _infer_variant(path: Path) -> str:
    stem = path.stem.upper()
    if "IGBT" in stem:
        return "IGBT"
    if "MOSFET" in stem:
        return "MOSFET"
    return "IC"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to reference variant CSV")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    data_path = Path(args.data)
    device = torch.device(args.device)
    model, tokenizer = load_model(args.checkpoint, device)

    step_lists = _load_csv_sequences(data_path)
    if len(step_lists) < 10:
        warnings.warn(
            f"Reference CSV has only {len(step_lists)} sequences (< 10); "
            "threshold estimates may be unreliable."
        )
    print(f"Recalibrating on {len(step_lists)} sequences from {data_path.name} ...")

    variant = _infer_variant(data_path)
    max_len = model.cfg.max_len

    per_step_losses: list[float] = []
    seq_mean_losses: list[float] = []
    model.eval()
    for steps in step_lists:
        token_ids = tokenizer.encode_mlm(variant, steps, max_len=max_len)
        losses = pseudo_perplexity_batch(model, tokenizer, token_ids, device)
        scored = [l for l in losses if l > 0.0]
        if scored:
            per_step_losses.extend(scored)
            seq_mean_losses.append(sum(scored) / len(scored))

    p95 = _percentile(per_step_losses, 95)
    p99 = _percentile(per_step_losses, 99)
    ood_p99 = _percentile(seq_mean_losses, 99)

    threshold = {
        "p95_loss": p95,
        "p99_loss": p99,
        "ood_p99": ood_p99,
        "calibration_n": len(step_lists),
        "calibration_date": time.strftime("%Y-%m-%d"),
        "variant_tag": data_path.stem,
        "seed": args.seed,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(threshold, f, indent=2)
    print(f"Wrote {out_path}: p95={p95:.4f} p99={p99:.4f} ood_p99={ood_p99:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
