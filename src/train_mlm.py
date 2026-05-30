#!/usr/bin/env python3
"""DDP MLM training for Spec 002 BertMLMEncoder. Launched via torchrun.

Usage (single node, 4 GPUs):
  torchrun --nnodes=1 --nproc_per_node=4 src/train_mlm.py \\
      --config configs/002_mlm.yaml \\
      --data-dir $TMPDIR/fab_sequences \\
      --splits $WORK/data/fab_sequences/splits.json \\
      --vocab $WORK/artifacts/001/vocab.json \\
      --output-dir $WORK/checkpoints/002 \\
      --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tokenizer import MLMTokenizer
from src.model.bert_mlm import BertMLMEncoder, BertMLMConfig
from src.data.sequences import load_all_variants, build_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDP utilities (mirrors src/train/train.py)
# ---------------------------------------------------------------------------

def setup_ddp() -> tuple[int, int, int, torch.device]:
    if "RANK" in os.environ:
        try:
            dist.init_process_group(backend="nccl")
        except Exception:
            warnings.warn("NCCL init failed, falling back to gloo")
            dist.init_process_group(backend="gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def log_main(rank: int, msg: str) -> None:
    if is_main(rank):
        print(f"[rank0 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_everything(seed: int, rank: int) -> None:
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


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
        if p.ndim < 2 or any(x in n.lower() for x in ("ln", "norm", "emb", "bias")):
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FabMLMDataset(Dataset):
    """Each item is a fixed-length token sequence (padded to max_len=100)."""

    def __init__(self, sequences: list[list[int]]):
        self.data = [torch.tensor(s, dtype=torch.long) for s in sequences]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


# ---------------------------------------------------------------------------
# Masking collator
# ---------------------------------------------------------------------------

class MLMMaskingCollator:
    def __init__(self, tokenizer: MLMTokenizer, masking_cfg: dict):
        self.tok = tokenizer
        self.mask_prob = masking_cfg.get("mask_prob", 0.15)
        self.mask_token_frac = masking_cfg.get("mask_token_frac", 0.80)
        self.random_token_frac = masking_cfg.get("random_token_frac", 0.10)
        self.strategy = masking_cfg.get("strategy", "random")
        self.span_max_len = masking_cfg.get("span_max_len", 5)
        self.min_seq_len_for_span = masking_cfg.get("min_seq_len_for_span", 10)

        if self.strategy == "param":
            raise NotImplementedError(
                "Masking strategy 'param' is reserved for future extension; "
                "parameter sub-tokenisation is not yet defined in the shared vocab."
            )

        # tokens that must NOT be masked
        self._special_ids = {
            tokenizer.pad_id, tokenizer.cls_id, tokenizer.sep_id,
            # variant tokens
            tokenizer.variant_id("IC"), tokenizer.variant_id("IGBT"), tokenizer.variant_id("MOSFET"),
        }
        # non-special token IDs for random replacement
        self._non_special_ids = [
            i for i in range(tokenizer.vocab_size) if i not in self._special_ids
        ]

    def __call__(self, batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        input_ids = torch.stack(batch)  # [B, T]
        labels = torch.full_like(input_ids, -100)
        masked_input = input_ids.clone()
        B, T = input_ids.shape

        for b in range(B):
            eligible = [
                i for i in range(T)
                if input_ids[b, i].item() not in self._special_ids
            ]
            if not eligible:
                continue

            if self.strategy == "span" and len(eligible) >= self.min_seq_len_for_span:
                selected = self._span_mask(eligible)
            else:
                n_mask = max(1, int(len(eligible) * self.mask_prob))
                selected = random.sample(eligible, min(n_mask, len(eligible)))

            # All-masked guard
            if len(selected) == T:
                log.warning("all-masked guard triggered — skipping batch item")
                continue

            for pos in selected:
                labels[b, pos] = input_ids[b, pos]
                r = random.random()
                if r < self.mask_token_frac:
                    masked_input[b, pos] = self.tok.mask_id
                elif r < self.mask_token_frac + self.random_token_frac:
                    masked_input[b, pos] = random.choice(self._non_special_ids)
                # else: keep original (10% unchanged)

        attention_mask = (input_ids != self.tok.pad_id).long()
        return {"input_ids": masked_input, "attention_mask": attention_mask, "labels": labels}

    def _span_mask(self, eligible: list[int]) -> list[int]:
        budget = max(1, int(len(eligible) * self.mask_prob))
        selected = set()
        attempts = 0
        while len(selected) < budget and attempts < 20:
            attempts += 1
            span_len = random.randint(1, min(self.span_max_len, budget - len(selected), len(eligible)))
            start_idx = random.randint(0, len(eligible) - span_len)
            for k in range(span_len):
                selected.add(eligible[start_idx + k])
        return list(selected)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def _prune_epoch_ckpts(out_dir: Path, keep: int) -> None:
    ckpts = sorted(out_dir.glob("checkpoint_epoch*.pt"))
    for old in ckpts[:-keep]:
        try:
            old.unlink()
        except OSError:
            pass


def save_checkpoint(out_dir: Path, tag: str, model, optimizer, epoch: int,
                    step: int, best_val_acc: float, cfg: dict, vocab_path: str,
                    vocab_size: int, seed: int) -> None:
    raw = model.module if isinstance(model, DDP) else model
    payload = {
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val_acc": best_val_acc,
        "config": cfg,
        "vocab_path": vocab_path,
        "vocab_size": vocab_size,
        "seed": seed,
    }
    atomic_save(payload, out_dir / f"checkpoint_{tag}.pt")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def evaluate(model, loader: DataLoader, collator: MLMMaskingCollator,
             device: torch.device, rank: int, world_size: int,
             amp_dtype: torch.dtype) -> dict:
    model.eval()
    total_correct = torch.tensor(0.0, device=device)
    total_masked = torch.tensor(0.0, device=device)
    total_loss = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for batch_tensors in loader:
            batch = collator(batch_tensors)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            dev_type = "cuda" if device.type == "cuda" else "cpu"
            with torch.autocast(device_type=dev_type, dtype=amp_dtype):
                logits = model(input_ids, attention_mask)

            mask = labels != -100
            n = mask.sum().float()
            if n == 0:
                continue
            loss = F.cross_entropy(logits[mask], labels[mask], reduction="mean")
            preds = logits[mask].argmax(dim=-1)
            total_correct += (preds == labels[mask]).float().sum()
            total_masked += n
            total_loss += loss * n

    if dist.is_initialized() and world_size > 1:
        dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_masked, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)

    n_masked = total_masked.item()
    acc = (total_correct / max(n_masked, 1)).item()
    avg_loss = (total_loss / max(n_masked, 1)).item()
    model.train()
    return {"val_masked_acc": acc, "val_mlm_loss": avg_loss, "val_n_masked": int(n_masked)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_sequences(data_dir: Path, splits_path: Path, tokenizer: MLMTokenizer,
                    max_len: int, debug: bool) -> dict[str, list[list[int]]]:
    # Prefer large CSVs if present (generated by generate_data.py), else fall back to variants
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
        raise FileNotFoundError(
            f"No variant CSVs found in {data_dir}. "
            "Expected IC_variants.csv / IC_large.csv (and IGBT/MOSFET equivalents). "
            "Stage them from tracks/industrial-infineon/training_data/ first."
        )

    records = load_all_variants(csv_paths)

    if splits_path.exists():
        with open(splits_path) as f:
            splits_data = json.load(f)
        # splits_data: {split: [[variant, sid], ...]}
        by_key = {(v, sid): (v, steps) for v, sid, steps in records}
        result: dict[str, list[list[int]]] = {}
        for split_name in ("train", "val", "test"):
            keys = [tuple(x) for x in splits_data.get(split_name, [])]
            seqs = []
            for key in keys:
                if key in by_key:
                    v, steps = by_key[key]
                    seqs.append(tokenizer.encode_mlm(v, steps, max_len=max_len))
            result[split_name] = seqs
    else:
        # build splits on the fly and write splits.json
        splits = build_splits(records, seed=42)
        splits_data_out = {
            k: [[v, sid] for v, sid in items] for k, items in splits.items()
        }
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        with open(splits_path, "w") as f:
            json.dump(splits_data_out, f, indent=2)
        by_key = {(v, sid): (v, steps) for v, sid, steps in records}
        result = {}
        for split_name in ("train", "val", "test"):
            keys = splits[split_name]
            seqs = []
            for v, sid in keys:
                if (v, sid) in by_key:
                    _, steps = by_key[(v, sid)]
                    seqs.append(tokenizer.encode_mlm(v, steps, max_len=max_len))
            result[split_name] = seqs

    if debug:
        for k in result:
            result[k] = result[k][:200]

    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--debug", action="store_true", help="limit to 200 sequences, fast smoke test")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.max_epochs is not None:
        cfg["train"]["epochs"] = args.max_epochs

    rank, world_size, local_rank, device = setup_ddp()
    seed_everything(cfg["seed"], rank)

    out_dir = Path(args.output_dir)
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = Path(args.vocab)
    if vocab_path.exists():
        tokenizer = MLMTokenizer.load(vocab_path)
        log_main(rank, f"Loaded tokenizer from {vocab_path} vocab_size={tokenizer.vocab_size}")
    else:
        log_main(rank, f"Vocab not found at {vocab_path}; building from CSVs")
        data_dir = Path(args.data_dir)
        csv_paths = {
            "IC": data_dir / "IC_variants.csv",
            "IGBT": data_dir / "IGBT_variants.csv",
            "MOSFET": data_dir / "MOSFET_variants.csv",
        }
        tokenizer = MLMTokenizer.build({k: v for k, v in csv_paths.items() if v.exists()})
        fallback = Path(args.vocab).parent.parent / "002" / "vocab.json"
        if is_main(rank):
            tokenizer.save(fallback)
            log_main(rank, f"Built tokenizer vocab_size={tokenizer.vocab_size}, saved to {fallback}")

    if is_main(rank):
        tokenizer.save(out_dir / "tokenizer.json")

    # Load and split sequences
    max_len = cfg["model"]["max_len"]
    split_seqs = _load_sequences(
        Path(args.data_dir), Path(args.splits), tokenizer, max_len, args.debug
    )
    log_main(rank, f"train={len(split_seqs['train'])} val={len(split_seqs['val'])} test={len(split_seqs['test'])}")

    train_ds = FabMLMDataset(split_seqs["train"])
    val_ds = FabMLMDataset(split_seqs["val"])

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg["seed"]) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    bsz = cfg["train"]["per_device_batch"]
    train_loader = DataLoader(
        train_ds, batch_size=bsz, sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=2, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bsz, sampler=val_sampler, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )

    masking_cfg = cfg.get("masking", {})
    collator = MLMMaskingCollator(tokenizer, masking_cfg)

    mcfg = cfg["model"]
    model_cfg = BertMLMConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=mcfg["d_model"],
        n_layers=mcfg["n_layers"],
        n_heads=mcfg["n_heads"],
        d_ff=mcfg["d_ff"],
        max_len=mcfg["max_len"],
        dropout=mcfg["dropout"],
        pad_id=tokenizer.pad_id,
    )
    model = BertMLMEncoder(model_cfg).to(device)
    log_main(rank, f"model_params={model.num_params() / 1e6:.2f}M")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    opt_cfg = cfg["optim"]
    raw_model = model.module if isinstance(model, DDP) else model
    optimizer = torch.optim.AdamW(
        param_groups(raw_model, opt_cfg["weight_decay"]),
        lr=opt_cfg["lr"], betas=tuple(opt_cfg["betas"]), eps=opt_cfg["eps"],
    )

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * cfg["train"]["epochs"]
    warmup_steps = max(1, int(total_steps * opt_cfg["warmup_frac"]))

    precision = cfg["train"].get("precision", "bfloat16")
    if precision == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        warnings.warn("bfloat16 not supported on this device; falling back to fp16")
        precision = "fp16"
    amp_dtype = {"bfloat16": torch.bfloat16, "fp16": torch.float16}.get(precision, torch.float32)

    scaler = torch.cuda.amp.GradScaler() if precision == "fp16" and device.type == "cuda" else None

    start_epoch = 0
    global_step = 0
    best_val_acc = 0.0
    epochs_no_improve = 0

    if args.resume:
        ckpts = sorted(out_dir.glob("checkpoint_epoch*.pt"))
        if ckpts:
            ck = torch.load(ckpts[-1], map_location=device, weights_only=False)
            raw_model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            start_epoch = ck["epoch"] + 1
            global_step = ck["step"]
            best_val_acc = ck.get("best_val_acc", 0.0)
            log_main(rank, f"Resumed from {ckpts[-1]}")

    job_start = time.time()
    soft_deadline = job_start + cfg["train"]["soft_walltime_hours"] * 3600
    metrics_log: list[dict] = []

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        ep_loss = 0.0
        ep_tokens = 0
        optimizer.zero_grad(set_to_none=True)
        t_ep = time.time()

        for step, batch_tensors in enumerate(train_loader):
            batch = collator(batch_tensors)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            dev_type = "cuda" if device.type == "cuda" else "cpu"
            with torch.autocast(device_type=dev_type, dtype=amp_dtype):
                logits = model(input_ids, attention_mask)
                mask = labels != -100
                n_tok = mask.sum().clamp(min=1)
                loss = F.cross_entropy(logits[mask], labels[mask], reduction="mean")

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            lr = cosine_warmup(global_step, total_steps, warmup_steps, opt_cfg["lr"], opt_cfg["lr_min"])
            for g in optimizer.param_groups:
                g["lr"] = lr

            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt_cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt_cfg["grad_clip"])
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            ep_loss += loss.item() * int(n_tok.item())
            ep_tokens += int(n_tok.item())

            if is_main(rank) and global_step % cfg["train"]["log_every_steps"] == 0:
                log_main(rank, f"epoch={epoch} step={global_step}/{total_steps} loss={loss.item():.4f} lr={lr:.2e}")

            if is_main(rank) and global_step % cfg["train"].get("ckpt_every_steps", 500) == 0:
                save_checkpoint(out_dir, f"step{global_step:07d}", model, optimizer, epoch,
                                global_step, best_val_acc, cfg, str(vocab_path),
                                tokenizer.vocab_size, cfg["seed"])

            if time.time() > soft_deadline:
                log_main(rank, "soft walltime reached during epoch — breaking")
                break

        train_loss = ep_loss / max(1, ep_tokens)
        log_main(rank, f"epoch {epoch} train_loss={train_loss:.4f} time={time.time() - t_ep:.0f}s")

        if (epoch + 1) % cfg["train"]["val_every_epochs"] == 0:
            val_metrics = evaluate(model, val_loader, collator, device, rank, world_size, amp_dtype)
            log_main(rank, f"epoch {epoch} val={json.dumps(val_metrics)}")
            metrics_log.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
            if is_main(rank):
                with open(out_dir / "metrics.json", "w") as f:
                    json.dump(metrics_log, f, indent=2)

            val_acc = val_metrics["val_masked_acc"]
            if val_acc > best_val_acc + 1e-4:
                best_val_acc = val_acc
                epochs_no_improve = 0
                if is_main(rank):
                    save_checkpoint(out_dir, "best", model, optimizer, epoch, global_step,
                                    best_val_acc, cfg, str(vocab_path), tokenizer.vocab_size, cfg["seed"])
                    best_link = out_dir / "best_model.pt"
                    if best_link.is_symlink():
                        best_link.unlink()
                    try:
                        best_link.symlink_to(out_dir / "checkpoint_best.pt")
                    except OSError:
                        pass
            else:
                epochs_no_improve += 1

            if is_main(rank):
                save_checkpoint(out_dir, f"epoch{epoch:03d}", model, optimizer, epoch, global_step,
                                best_val_acc, cfg, str(vocab_path), tokenizer.vocab_size, cfg["seed"])
                _prune_epoch_ckpts(out_dir, keep=cfg["train"].get("keep_recent_ckpts", 3))

            if epochs_no_improve >= cfg["train"]["early_stop_patience"]:
                log_main(rank, f"early stop at epoch {epoch} (no improvement for {epochs_no_improve} epochs)")
                break

        if time.time() > soft_deadline:
            log_main(rank, "soft walltime reached at epoch end")
            if is_main(rank):
                save_checkpoint(out_dir, "final", model, optimizer, epoch, global_step,
                                best_val_acc, cfg, str(vocab_path), tokenizer.vocab_size, cfg["seed"])
            break

    if is_main(rank) and not (out_dir / "checkpoint_final.pt").exists():
        save_checkpoint(out_dir, "final", model, optimizer, epoch, global_step,
                        best_val_acc, cfg, str(vocab_path), tokenizer.vocab_size, cfg["seed"])

    if dist.is_initialized():
        dist.barrier()

    cleanup_ddp()
    return 0


if __name__ == "__main__":
    sys.exit(main())
