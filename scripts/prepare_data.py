#!/usr/bin/env python3
"""One-shot data prep: build tokenizer, splits, and packed shards into $WORK.

Run on the login node (CPU only, < 5 min). Idempotent.

Usage:
  python scripts/prepare_data.py --config configs/train_gpt_fab.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.sequences import (
    build_splits,
    encode_split,
    load_all_variants,
    pack_sequences,
    vocab_hash,
)
from src.data.tokenizer import FabTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_gpt_fab.yaml")
    ap.add_argument("--work_dir", default=os.environ.get("WORK", str(ROOT / "_work")))
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    csv_paths = {k: ROOT / v for k, v in cfg["data"]["csv_paths"].items()}
    for p in csv_paths.values():
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 1

    out_dir = Path(args.work_dir) / cfg["data"]["work_subdir"]
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prepare_data] loading CSVs from {csv_paths}")
    records = load_all_variants(csv_paths)
    print(f"[prepare_data] loaded {len(records)} sequences across {len(csv_paths)} variants")

    print("[prepare_data] building tokenizer...")
    tokenizer = FabTokenizer.build(csv_paths)
    tokenizer.save(out_dir / "tokenizer.json")
    print(f"[prepare_data] vocab_size={tokenizer.vocab_size} hash={vocab_hash(tokenizer)}")

    print("[prepare_data] building splits...")
    splits = build_splits(records, tuple(cfg["data"]["split_ratio"]), seed=cfg["seed"])
    with open(out_dir / "splits.json", "w") as f:
        json.dump({k: [list(t) for t in v] for k, v in splits.items()}, f, indent=2)
    print(f"[prepare_data] splits: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    max_len = cfg["data"]["max_len"]
    pack = bool(cfg["data"].get("pack", True))

    n_train_shards = 8
    for split_name, keys in splits.items():
        encoded = encode_split(records, keys, tokenizer)
        if pack:
            tensors = pack_sequences(encoded, max_len)
        else:
            from src.data.tokenizer import PAD_ID

            tensors = []
            for seq in encoded:
                seq = seq[:max_len]
                seq = seq + [PAD_ID] * (max_len - len(seq))
                tensors.append(seq)
        t = torch.tensor(tensors, dtype=torch.long)
        if split_name == "train":
            chunks = torch.chunk(t, n_train_shards, dim=0)
            for i, c in enumerate(chunks):
                torch.save(c.contiguous(), shards_dir / f"train_{i:03d}.pt")
            print(f"[prepare_data] train: {t.size(0)} packed rows -> {n_train_shards} shards")
        else:
            torch.save(t.contiguous(), shards_dir / f"{split_name}.pt")
            print(f"[prepare_data] {split_name}: {t.size(0)} packed rows")

    # Save raw test token lists (unpacked) for memorization probe
    test_encoded = encode_split(records, splits["test"], tokenizer)
    with open(out_dir / "test_sequences.json", "w") as f:
        json.dump(test_encoded, f)

    print(f"[prepare_data] DONE. Artifacts in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
