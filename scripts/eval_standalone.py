#!/usr/bin/env python3
"""Standalone test-eval script — no DDP required.

Loads a saved checkpoint, runs the test split eval and memorization probe,
and writes eval_report.json next to the checkpoint. Run this to recover
eval_report.json after a training job that crashed during final evaluation.

Usage (login node — CPU smoke check):
  python scripts/eval_standalone.py \
      --checkpoint $WORK/checkpoints/001-gpt-fab/checkpoint_best.pt \
      --data_dir   $WORK/data/fab_sequences \
      --config     configs/train_gpt_fab.yaml

Usage (GPU node via eval_standalone.sh):
  sbatch scripts/slurm/eval_standalone.sh
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import PackedShardDataset, loss_mask
from src.data.sequences import vocab_hash
from src.data.tokenizer import FabTokenizer, PAD_ID
from src.eval.memorization_probe import perturbed_score_ratio
from src.eval.sequence_metrics import StreamingAccumulator
from src.model.fab_gpt import FabGPT, FabGPTConfig


def evaluate_single(model, loader, device, pad_id, variant_ids, top_k):
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_tokens = torch.tensor(0.0, device=device)
    acc = StreamingAccumulator()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(batch)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = batch[:, 1:].contiguous()
            mask = loss_mask(batch, pad_id=pad_id, variant_ids=variant_ids)
            row_variant = batch[:, 0]
            vids = row_variant.unsqueeze(1).expand_as(shift_targets)
            lp = F.log_softmax(shift_logits.float(), dim=-1)
            nll = -lp.gather(2, shift_targets.unsqueeze(-1)).squeeze(-1)
            total_loss += (nll * mask).sum()
            total_tokens += mask.sum()
            acc.add(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_targets.reshape(-1),
                mask.reshape(-1),
                vids.reshape(-1),
            )
    avg_loss = (total_loss / total_tokens.clamp(min=1)).item()
    out = {
        "val_loss": avg_loss,
        "val_perplexity": float(math.exp(min(20.0, avg_loss))),
    }
    out.update(acc.finalize(top_k=top_k))
    model.train()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to checkpoint_best.pt")
    ap.add_argument("--data_dir", required=True, help="dir with test.pt and tokenizer.json")
    ap.add_argument("--config", default="configs/train_gpt_fab.yaml")
    ap.add_argument("--output", default=None, help="path for eval_report.json (default: next to checkpoint)")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    out_path = Path(args.output) if args.output else ckpt_path.parent / "eval_report.json"

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}  checkpoint={ckpt_path}", flush=True)

    print("[eval] loading checkpoint...", flush=True)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)

    tokenizer_path = data_dir / "tokenizer.json"
    tokenizer = FabTokenizer.load(tokenizer_path)
    vocab_h = vocab_hash(tokenizer)

    model_cfg = ck.get("config", {}).get("model", cfg["model"])
    fab_cfg = FabGPTConfig(
        vocab_size=ck.get("vocab_size", tokenizer.vocab_size),
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        d_ff=model_cfg["d_ff"],
        max_len=model_cfg["max_len"],
        dropout=0.0,
        tie_embeddings=model_cfg.get("tie_embeddings", True),
    )
    model = FabGPT(fab_cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[eval] model_params={model.num_params()/1e6:.2f}M  vocab_size={tokenizer.vocab_size}  hash={vocab_h}", flush=True)

    pad_id = PAD_ID
    variant_ids = tuple(tokenizer.variant_ids) if hasattr(tokenizer, "variant_ids") else (4, 5, 6)
    bsz = cfg["train"]["per_device_batch"]
    top_k = tuple(cfg["eval"]["top_k"])

    # prepare_data.py saves shards into data_dir/shards/; try both locations
    test_path = data_dir / "shards" / "test.pt"
    if not test_path.exists():
        test_path = data_dir / "test.pt"

    test_metrics: dict = {}
    if test_path.exists():
        print(f"[eval] running test eval on {test_path}...", flush=True)
        from torch.utils.data import DataLoader
        test_ds = PackedShardDataset([test_path])
        test_loader = DataLoader(test_ds, batch_size=bsz, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
        test_metrics = evaluate_single(model, test_loader, device, pad_id, variant_ids, top_k)
        print(f"[eval] test_metrics={json.dumps(test_metrics, indent=None)}", flush=True)
    else:
        print(f"[eval] WARNING: test shard not found at {test_path}", flush=True)

    probe: dict = {}
    ts_path = data_dir / "test_sequences.json"
    if ts_path.exists():
        print("[eval] running memorization probe...", flush=True)
        with open(ts_path) as f:
            test_seqs = json.load(f)
        probe = perturbed_score_ratio(model, test_seqs, device, n=cfg["eval"]["memorization_n"], seed=cfg["seed"])
        print(f"[eval] memorization_probe={probe}", flush=True)
    else:
        print(f"[eval] WARNING: test_sequences.json not found at {ts_path}", flush=True)

    report = {
        "checkpoint_path": str(ckpt_path),
        "vocab_hash": vocab_h,
        "vocab_size": tokenizer.vocab_size,
        "model_params_M": model.num_params() / 1e6,
        "epochs_trained": ck.get("epoch", "unknown"),
        "test_metrics": test_metrics,
        "memorization_probe": probe,
        "eval_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[eval] WROTE {out_path}", flush=True)

    top1 = test_metrics.get("top1_accuracy", 0.0)
    ratio = probe.get("ratio", 0.0)
    print(f"\n[eval] SUMMARY", flush=True)
    print(f"  top1_accuracy : {top1:.4f}  {'✅ ≥0.80' if top1 >= 0.80 else '❌ <0.80'}", flush=True)
    print(f"  memo ratio    : {ratio:.2f}  {'✅ ≥5.0' if ratio >= 5.0 else '❌ <5.0'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
