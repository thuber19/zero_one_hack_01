#!/usr/bin/env python3
"""Generate large synthetic CSV datasets for all three product families.

Run on the login node (CPU-only, ~2-4 min for 21k per family).
Idempotent: skips families whose output CSV already exists.

Usage:
  python scripts/generate_data.py
  python scripts/generate_data.py --count 5000          # quick smoke test
  python scripts/generate_data.py --out_dir /tmp/data   # custom output dir
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Directory name contains a hyphen so it can't be imported as a package;
# load the module directly from its file path instead.
_gen_path = ROOT / "tracks/industrial-infineon/training_data/generate_sequences.py"
_spec = importlib.util.spec_from_file_location("generate_sequences", _gen_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
generate_dataset = _mod.generate_dataset
write_csv = _mod.write_csv

FAMILIES = ["mosfet", "igbt", "ic"]
DEFAULT_COUNT = 21_000  # 21k × 3 = 63k total → ~50.4k training rows after 80/10/10 split
DEFAULT_OUT_DIR = ROOT / "tracks/industrial-infineon/training_data"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=DEFAULT_COUNT,
                    help=f"Sequences per family (default: {DEFAULT_COUNT})")
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="Directory for output CSVs")
    ap.add_argument("--seed", type=int, default=123,
                    help="Base random seed (default: 123, distinct from original 42)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing CSV files")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for i, family in enumerate(FAMILIES):
        out_path = args.out_dir / f"{family.upper()}_large.csv"
        if out_path.exists() and not args.force:
            print(f"[generate_data] {out_path.name} already exists — skipping (use --force to overwrite)")
            continue
        print(f"[generate_data] generating {args.count:,} {family.upper()} sequences (seed={args.seed + i}) ...")
        seqs = generate_dataset(family, args.count, seed=args.seed + i, validate=True)
        write_csv(out_path, seqs)

    print("[generate_data] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
