#!/usr/bin/env python3
"""Single-GPU training loop for LSTMModel (Spec 005).

Usage:
  python src/train/train_lstm.py \\
      --config configs/lstm_baseline.yaml \\
      --data_dir $WORK/data/fab_sequences/shards \\
      --tokenizer $WORK/data/fab_sequences/tokenizer.json \\
      --test_sequences $WORK/data/fab_sequences/test_sequences.json \\
      --output_dir $WORK/checkpoints/005-lstm-baseline

Debug smoke (1 epoch, small batch):
  python src/train/train_lstm.py --config configs/lstm_baseline.yaml ... --debug
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import PackedShardDataset, loss_mask
from src.data.sequences import vocab_hash
from src.data.tokenizer import FabTokenizer, PAD_ID
from src.eval.memorization_probe import perturbed_score_ratio
from src.eval.sequence_metrics import StreamingAccumulator
from src.model.lstm import LSTMModel

VARIANT_IDS: tuple[int, ...] = (4, 5, 6)


# ---------------------------------------------------------------------------
# Utilities (mirrored from train.py for consistency)
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[lstm {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cosine_warmup(step: int, total: int, warmup: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * step / max(1, warmup)
    if step >= total:
        return lr_min
    p = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * p))


def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "ln" in n.lower() or "norm" in n.lower() or "emb" in n.lower() or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def atomic_save(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def prune_old_checkpoints(out_dir: Path, keep: int) -> None:
    epoch_ckpts = sorted(out_dir.glob("checkpoint_epoch*.pt"))
    for old in epoch_ckpts[:-keep]:
        old.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: LSTMModel,
    loader: DataLoader,
    device: torch.device,
    use_autocast: bool,
    top_k: tuple[int, ...],
) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    acc = StreamingAccumulator()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            if use_autocast:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits = model(batch)
            else:
                logits = model(batch)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = batch[:, 1:].contiguous()
            mask = loss_mask(batch, pad_id=PAD_ID, variant_ids=VARIANT_IDS)
            row_variant = batch[:, 0]
            vids = row_variant.unsqueeze(1).expand_as(shift_targets)
            lp = F.log_softmax(shift_logits.float(), dim=-1)
            nll = -lp.gather(2, shift_targets.unsqueeze(-1)).squeeze(-1)
            total_loss += float((nll * mask).sum().item())
            total_tokens += int(mask.sum().item())
            acc.add(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_targets.reshape(-1),
                mask.reshape(-1),
                vids.reshape(-1),
            )
    avg_loss = total_loss / max(1, total_tokens)
    out = {
        "val_loss": avg_loss,
        "val_perplexity": float(math.exp(min(20.0, avg_loss))),
    }
    out.update(acc.finalize(top_k=top_k))
    model.train()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Train LSTMModel on fab-process sequences.")
    ap.add_argument("--config", required=True, help="Path to lstm_baseline.yaml")
    ap.add_argument("--data_dir", required=True, help="Dir with train_*.pt, val.pt, test.pt shards")
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json (default: data_dir/../tokenizer.json)")
    ap.add_argument("--test_sequences", default=None, help="test_sequences.json for memorization probe")
    ap.add_argument("--output_dir", required=True, help="Checkpoint and metrics output dir")
    ap.add_argument("--resume", action="store_true", help="Resume from latest epoch checkpoint")
    ap.add_argument("--max_epochs", type=int, default=None, help="Override config epochs")
    ap.add_argument("--batch_size", type=int, default=None, help="Override config batch size")
    ap.add_argument("--debug", action="store_true", help="Shorthand: 1 epoch, batch=32")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.debug:
        cfg["train"]["epochs"] = 1
        cfg["train"]["per_device_batch"] = 32
        cfg["train"]["soft_walltime_hours"] = 0.4
    if args.max_epochs is not None:
        cfg["train"]["epochs"] = args.max_epochs
    if args.batch_size is not None:
        cfg["train"]["per_device_batch"] = args.batch_size

    seed_everything(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    tokenizer_path = Path(args.tokenizer) if args.tokenizer else data_dir.parent / "tokenizer.json"
    splits_path = tokenizer_path.parent / "splits.json"

    # Fail-fast prerequisite checks
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Spec 001 tokenizer not found at {tokenizer_path}. "
            "Run prepare_data.py first, or pass --tokenizer."
        )
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Spec 001 splits.json not found at {splits_path}. "
            "Run prepare_data.py first."
        )

    tokenizer_sha = sha256_file(tokenizer_path)
    split_sha = sha256_file(splits_path)
    log(f"tokenizer_sha={tokenizer_sha[:16]}...  split_sha={split_sha[:16]}...")

    tokenizer = FabTokenizer.load(tokenizer_path)
    vh = vocab_hash(tokenizer)
    log(f"vocab_size={tokenizer.vocab_size}  vocab_hash={vh}")

    # Shards
    train_shards = sorted(data_dir.glob("train_*.pt"))
    if not train_shards:
        raise FileNotFoundError(f"No train shards found in {data_dir}")
    train_ds = PackedShardDataset(train_shards)
    val_ds = PackedShardDataset([data_dir / "val.pt"])
    log(f"train_rows={len(train_ds)}  val_rows={len(val_ds)}")

    bsz = cfg["train"]["per_device_batch"]
    train_loader = DataLoader(
        train_ds, batch_size=bsz, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bsz, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=False,
    )

    # Model
    mcfg = cfg["model"]
    model = LSTMModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=mcfg["embed_dim"],
        hidden_size=mcfg["hidden_size"],
        num_layers=mcfg["num_layers"],
        dropout=mcfg["dropout"],
        pad_id=PAD_ID,
    ).to(device)
    log(f"model_params={model.num_params() / 1e6:.2f}M")

    use_autocast = cfg["train"].get("precision", "fp32") == "bfloat16"

    opt_cfg = cfg["optim"]
    optimizer = torch.optim.AdamW(
        param_groups(model, opt_cfg["weight_decay"]),
        lr=opt_cfg["lr"],
        betas=tuple(opt_cfg["betas"]),
        eps=opt_cfg["eps"],
    )

    total_steps = len(train_loader) * cfg["train"]["epochs"]
    warmup_steps = max(1, int(total_steps * opt_cfg["warmup_frac"]))

    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    epochs_no_improve = 0

    if args.resume:
        epoch_ckpts = sorted(out_dir.glob("checkpoint_epoch*.pt"))
        if epoch_ckpts:
            ck_path = epoch_ckpts[-1]
            log(f"resuming from {ck_path}")
            ck = torch.load(ck_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            start_epoch = ck["epoch"] + 1
            global_step = ck["step"]
            best_val = ck.get("best_val_loss", float("inf"))

    job_start = time.time()
    soft_deadline = job_start + cfg["train"]["soft_walltime_hours"] * 3600
    metrics_log: list[dict] = []

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        ep_loss = 0.0
        ep_tokens = 0
        optimizer.zero_grad(set_to_none=True)
        t_ep = time.time()

        for step, batch in enumerate(train_loader):
            batch = batch.to(device, non_blocking=True)
            if use_autocast:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits = model(batch)
            else:
                logits = model(batch)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = batch[:, 1:].contiguous()
            mask = loss_mask(batch, pad_id=PAD_ID, variant_ids=VARIANT_IDS)
            lp = F.log_softmax(shift_logits.float(), dim=-1)
            nll = -lp.gather(2, shift_targets.unsqueeze(-1)).squeeze(-1)
            n_tok = mask.sum().clamp(min=1)
            loss = (nll * mask).sum() / n_tok

            loss.backward()

            lr = cosine_warmup(global_step, total_steps, warmup_steps, opt_cfg["lr"], opt_cfg["lr_min"])
            for g in optimizer.param_groups:
                g["lr"] = lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt_cfg["grad_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            ep_loss += loss.item() * int(n_tok.item())
            ep_tokens += int(n_tok.item())

            if global_step % cfg["train"]["log_every_steps"] == 0:
                log(f"epoch={epoch} step={global_step}/{total_steps} loss={loss.item():.4f} lr={lr:.2e}")

            if time.time() > soft_deadline:
                log("soft walltime reached during epoch — breaking")
                break

        train_loss = ep_loss / max(1, ep_tokens)
        log(f"epoch {epoch} train_loss={train_loss:.4f} time={time.time() - t_ep:.0f}s")

        if (epoch + 1) % cfg["train"]["val_every_epochs"] == 0:
            val_metrics = evaluate(model, val_loader, device, use_autocast, tuple(cfg["eval"]["top_k"]))
            log(f"epoch {epoch} val={json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in val_metrics.items() if k != 'by_variant'})}")
            metrics_log.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
            with open(out_dir / "metrics.json", "w") as f:
                json.dump(metrics_log, f, indent=2, default=str)

            improved = val_metrics["val_loss"] < best_val - 1e-4
            if improved:
                best_val = val_metrics["val_loss"]
                epochs_no_improve = 0
                atomic_save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "best_val_loss": best_val,
                    "config": cfg,
                    "tokenizer_sha": tokenizer_sha,
                    "split_sha": split_sha,
                    "vocab_hash": vh,
                    "seed": cfg["seed"],
                }, out_dir / "checkpoint_best.pt")
                log(f"saved checkpoint_best.pt  val_loss={best_val:.4f}")
            else:
                epochs_no_improve += 1

            if (epoch + 1) % cfg["train"]["ckpt_every_epochs"] == 0:
                atomic_save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "best_val_loss": best_val,
                    "config": cfg,
                    "tokenizer_sha": tokenizer_sha,
                    "split_sha": split_sha,
                    "vocab_hash": vh,
                    "seed": cfg["seed"],
                }, out_dir / f"checkpoint_epoch{epoch:03d}.pt")
                prune_old_checkpoints(out_dir, cfg["train"]["keep_recent_ckpts"])

            if epochs_no_improve >= cfg["train"]["early_stop_patience"]:
                log(f"early stopping at epoch {epoch} (no improvement for {epochs_no_improve} epochs)")
                break

        if time.time() > soft_deadline:
            log("soft walltime reached at epoch end")
            atomic_save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "step": global_step,
                "best_val_loss": best_val,
                "config": cfg,
                "tokenizer_sha": tokenizer_sha,
                "split_sha": split_sha,
                "vocab_hash": vh,
                "seed": cfg["seed"],
            }, out_dir / "checkpoint_final.pt")
            break

    # Final test evaluation on best checkpoint
    best_ckpt = out_dir / "checkpoint_best.pt"
    if not best_ckpt.exists():
        # Fallback: no validation improvement was ever recorded (e.g., 1-epoch debug run)
        best_ckpt = out_dir / "checkpoint_final.pt"
        # If also missing (very short debug), save current state
        if not best_ckpt.exists():
            atomic_save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "step": global_step,
                "best_val_loss": best_val,
                "config": cfg,
                "tokenizer_sha": tokenizer_sha,
                "split_sha": split_sha,
                "vocab_hash": vh,
                "seed": cfg["seed"],
            }, best_ckpt)

    log(f"loading {best_ckpt} for test eval...")
    ck = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])

    test_metrics: dict = {}
    test_path = data_dir / "test.pt"
    if test_path.exists():
        test_ds = PackedShardDataset([test_path])
        test_loader = DataLoader(test_ds, batch_size=bsz, shuffle=False, num_workers=2, pin_memory=True)
        test_metrics = evaluate(model, test_loader, device, use_autocast, tuple(cfg["eval"]["top_k"]))
        log(f"test_metrics={json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in test_metrics.items() if k != 'by_variant'})}")
    else:
        log(f"WARNING: test.pt not found at {test_path} — skipping test eval")

    probe: dict = {}
    ts_path = Path(args.test_sequences) if args.test_sequences else data_dir.parent / "test_sequences.json"
    if ts_path.exists():
        log("running memorization probe...")
        with open(ts_path) as f:
            test_seqs = json.load(f)
        probe = perturbed_score_ratio(model, test_seqs, device, n=cfg["eval"]["memorization_n"], seed=cfg["seed"])
        log(f"probe={probe}")
    else:
        log(f"WARNING: test_sequences.json not found at {ts_path} — skipping memorization probe")

    metrics_lstm = {
        "model": "lstm-baseline",
        "date": datetime.datetime.utcnow().isoformat() + "Z",
        "top1_accuracy": test_metrics.get("top1_accuracy"),
        "top5_accuracy": test_metrics.get("top5_accuracy"),
        "perplexity": test_metrics.get("perplexity"),
        "mrr": test_metrics.get("mrr"),
        "probe_score": probe.get("ratio"),
        "probe_n": probe.get("n"),
        "n_tokens": test_metrics.get("n_tokens"),
        "by_variant": test_metrics.get("by_variant"),
        "tokenizer_sha": tokenizer_sha,
        "split_sha": split_sha,
        "vocab_hash": vh,
        "seed": cfg["seed"],
        "model_params_M": model.num_params() / 1e6,
        "elapsed_hours": (time.time() - job_start) / 3600,
        "config": cfg,
    }
    metrics_path = out_dir / "metrics_lstm.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_lstm, f, indent=2, default=str)
    log(f"WROTE {metrics_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
