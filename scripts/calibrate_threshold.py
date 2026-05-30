#!/usr/bin/env python3
"""Derive p95/p99/ood_p99 thresholds from the clean validation split.

Usage:
  python scripts/calibrate_threshold.py \\
      --checkpoint $WORK/checkpoints/002/best_model.pt \\
      --splits $WORK/data/fab_sequences/splits.json \\
      --data-dir $TMPDIR/fab_sequences \\
      --output $WORK/checkpoints/002/threshold.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.infer import load_model, pseudo_perplexity_batch
from src.data.sequences import load_all_variants


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    sorted_v = sorted(values)
    idx = max(0, int(len(sorted_v) * p / 100) - 1)
    return sorted_v[idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Loading model from {args.checkpoint} ...")
    model, tokenizer = load_model(args.checkpoint, device)

    data_dir = Path(args.data_dir)
    csv_candidates = {
        "IC":    [data_dir / "IC_large.csv",    data_dir / "IC_variants.csv"],
        "IGBT":  [data_dir / "IGBT_large.csv",  data_dir / "IGBT_variants.csv"],
        "MOSFET":[data_dir / "MOSFET_large.csv", data_dir / "MOSFET_variants.csv"],
    }
    csv_paths = {}
    for variant, candidates in csv_candidates.items():
        for p in candidates:
            if p.exists():
                csv_paths[variant] = p
                break
    if not csv_paths:
        raise FileNotFoundError(f"No variant CSVs found in {data_dir}")
    records = load_all_variants(csv_paths)
    by_key = {(v, sid): (v, steps) for v, sid, steps in records}

    with open(args.splits) as f:
        splits_data = json.load(f)
    val_keys = [tuple(x) for x in splits_data.get("val", [])]
    val_seqs = [(v, sid, by_key[(v, sid)][1]) for v, sid in val_keys if (v, sid) in by_key]
    print(f"Calibrating on {len(val_seqs)} val sequences ...")

    max_len = model.cfg.max_len
    per_step_losses: list[float] = []
    seq_mean_losses: list[float] = []

    model.eval()
    for i, (variant, sid, steps) in enumerate(val_seqs):
        token_ids = tokenizer.encode_mlm(variant, steps, max_len=max_len)
        losses = pseudo_perplexity_batch(model, tokenizer, token_ids, device)
        scored = [l for l in losses if l > 0.0]
        if scored:
            per_step_losses.extend(scored)
            seq_mean_losses.append(sum(scored) / len(scored))
        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(val_seqs)}")

    p95 = _percentile(per_step_losses, 95)
    p99 = _percentile(per_step_losses, 99)
    ood_p99 = _percentile(seq_mean_losses, 99)

    threshold = {
        "p95_loss": p95,
        "p99_loss": p99,
        "ood_p99": ood_p99,
        "calibration_n": len(val_seqs),
        "n_per_step_samples": len(per_step_losses),
        "calibration_date": time.strftime("%Y-%m-%d"),
        "variant_tag": "all_variants",
        "seed": args.seed,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(threshold, f, indent=2)
    print(f"Wrote threshold.json: p95={p95:.4f} p99={p99:.4f} ood_p99={ood_p99:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
