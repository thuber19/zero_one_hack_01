#!/usr/bin/env python3
"""Generate splits.json from local training data.

Usage:
  python scripts/generate_splits.py
  python scripts/generate_splits.py --data-dir tracks/industrial-infineon/training_data --output checkpoints/002/splits.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.sequences import load_all_variants, build_splits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="tracks/industrial-infineon/training_data")
    ap.add_argument("--output", default="checkpoints/002/splits.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    csv_candidates = {
        "IC":     [data_dir / "IC_large.csv",     data_dir / "IC_variants.csv"],
        "IGBT":   [data_dir / "IGBT_large.csv",   data_dir / "IGBT_variants.csv"],
        "MOSFET": [data_dir / "MOSFET_large.csv",  data_dir / "MOSFET_variants.csv"],
    }
    csv_paths = {}
    for variant, candidates in csv_candidates.items():
        for p in candidates:
            if p.exists():
                csv_paths[variant] = p
                break

    if not csv_paths:
        print(f"ERROR: no variant CSVs found in {data_dir}", file=sys.stderr)
        return 1

    print(f"Loading from: {[str(v) for v in csv_paths.values()]}")
    records = load_all_variants(csv_paths)
    splits = build_splits(records, seed=args.seed)

    out = {k: [[v, sid] for v, sid in items] for k, items in splits.items()}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {out_path}: " + ", ".join(f"{k}={len(v)}" for k, v in out.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
