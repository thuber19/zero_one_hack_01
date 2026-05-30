#!/usr/bin/env python3
"""DDP training loop for FabGPT. Launched via torchrun.

Usage (single node, 4 GPUs):
  torchrun --nnodes=1 --nproc_per_node=4 src/train/train.py \
      --config configs/train_gpt_fab.yaml \
      --data_dir $TMPDIR/shards \
      --output_dir $WORK/checkpoints/001-gpt-fab
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import PackedShardDataset, loss_mask
from src.data.sequences import vocab_hash
from src.data.tokenizer import FabTokenizer, PAD_ID
from src.eval.memorization_probe import perturbed_score_ratio
from src.eval.sequence_metrics import StreamingAccumulator
from src.model.fab_gpt import FabGPT, FabGPTConfig


def setup_ddp() -> tuple[int, int, int, torch.device]:
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def log(rank: int, msg: str) -> None:
    if is_main(rank):
        print(f"[rank0 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_everything(seed: int, rank: int) -> None:
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def save_checkpoint(
    out_dir: Path, tag: str, model: DDP, optimizer, scheduler_state: dict, epoch: int, step: int, cfg: dict, vocab_h: str, vocab_size: int
) -> None:
    payload = {
        "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler_state,
        "epoch": epoch,
        "step": step,
        "config": cfg,
        "vocab_hash": vocab_h,
        "vocab_size": vocab_size,
    }
    atomic_save(payload, out_dir / f"checkpoint_{tag}.pt")


def find_resume(out_dir: Path) -> Path | None:
    ckpts = sorted(out_dir.glob("checkpoint_epoch*.pt"))
    return ckpts[-1] if ckpts else None


def evaluate(
    model: DDP,
    loader: DataLoader,
    device: torch.device,
    rank: int,
    world_size: int,
    pad_id: int,
    variant_ids_tuple: tuple[int, ...],
    top_k: tuple[int, ...],
) -> dict:
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_tokens = torch.tensor(0.0, device=device)
    acc = StreamingAccumulator()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16):
                logits = model(batch)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = batch[:, 1:].contiguous()
            mask = loss_mask(batch, pad_id=pad_id, variant_ids=variant_ids_tuple)
            # variant per token = variant token at position 0 of the packed row (best-effort)
            row_variant = batch[:, 0]  # variant token id for the row
            vids = row_variant.unsqueeze(1).expand_as(shift_targets)
            lp = F.log_softmax(shift_logits.float(), dim=-1)
            nll = -lp.gather(2, shift_targets.unsqueeze(-1)).squeeze(-1)
            total_loss += (nll * mask).sum()
            total_tokens += mask.sum()
            # streaming metrics on rank0 only to bound memory
            if rank == 0:
                acc.add(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_targets.reshape(-1),
                    mask.reshape(-1),
                    vids.reshape(-1),
                )
    if dist.is_initialized() and world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    avg_loss = (total_loss / total_tokens.clamp(min=1)).item()
    out = {"val_loss": avg_loss, "val_perplexity": float(math.exp(min(20.0, avg_loss)))}
    if rank == 0:
        out.update(acc.finalize(top_k=top_k))
    model.train()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_dir", required=True, help="dir with train_*.pt, val.pt, test.pt")
    ap.add_argument("--tokenizer", default=None, help="path to tokenizer.json (defaults to data_dir/../tokenizer.json)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--test_sequences", default=None, help="path to test_sequences.json for memorization probe")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rank, world_size, local_rank, device = setup_ddp()
    seed_everything(cfg["seed"], rank)

    out_dir = Path(args.output_dir)
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "runs").mkdir(exist_ok=True)

    data_dir = Path(args.data_dir)
    tokenizer_path = Path(args.tokenizer) if args.tokenizer else data_dir.parent / "tokenizer.json"
    tokenizer = FabTokenizer.load(tokenizer_path)
    vocab_h = vocab_hash(tokenizer)
    if is_main(rank):
        # copy tokenizer next to checkpoints for inference portability
        tokenizer.save(out_dir / "tokenizer.json")

    log(rank, f"vocab_size={tokenizer.vocab_size} hash={vocab_h} world_size={world_size}")

    train_shards = sorted(data_dir.glob("train_*.pt"))
    if not train_shards:
        raise FileNotFoundError(f"no train shards in {data_dir}")
    train_ds = PackedShardDataset(train_shards)
    val_ds = PackedShardDataset([data_dir / "val.pt"])
    log(rank, f"train_rows={len(train_ds)} val_rows={len(val_ds)}")

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg["seed"]) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    bsz = cfg["train"]["per_device_batch"]
    train_loader = DataLoader(
        train_ds, batch_size=bsz, sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bsz, sampler=val_sampler, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=False,
    )

    mcfg = cfg["model"]
    model_cfg = FabGPTConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=mcfg["d_model"],
        n_layers=mcfg["n_layers"],
        n_heads=mcfg["n_heads"],
        d_ff=mcfg["d_ff"],
        max_len=mcfg["max_len"],
        dropout=mcfg["dropout"],
        tie_embeddings=mcfg["tie_embeddings"],
        pad_id=PAD_ID,
    )
    model = FabGPT(model_cfg).to(device)
    log(rank, f"model_params={model.num_params() / 1e6:.2f}M")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False, static_graph=True)

    opt_cfg = cfg["optim"]
    optimizer = torch.optim.AdamW(
        param_groups(model.module if isinstance(model, DDP) else model, opt_cfg["weight_decay"]),
        lr=opt_cfg["lr"], betas=tuple(opt_cfg["betas"]), eps=opt_cfg["eps"],
    )

    steps_per_epoch = max(1, len(train_loader) // cfg["train"]["grad_accum"])
    total_steps = steps_per_epoch * cfg["train"]["epochs"]
    warmup_steps = max(1, int(total_steps * opt_cfg["warmup_frac"]))

    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    epochs_no_improve = 0

    if args.resume:
        ck_path = find_resume(out_dir)
        if ck_path is not None:
            log(rank, f"resuming from {ck_path}")
            ck = torch.load(ck_path, map_location=device)
            (model.module if isinstance(model, DDP) else model).load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            start_epoch = ck["epoch"] + 1
            global_step = ck["step"]

    job_start = time.time()
    soft_deadline = job_start + cfg["train"]["soft_walltime_hours"] * 3600
    variant_ids_tuple = (4, 5, 6)
    pad_id = PAD_ID

    metrics_log: list[dict] = []

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        ep_loss = 0.0
        ep_tokens = 0
        accum = cfg["train"]["grad_accum"]
        optimizer.zero_grad(set_to_none=True)
        t_ep = time.time()
        for step, batch in enumerate(train_loader):
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16):
                logits = model(batch)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = batch[:, 1:].contiguous()
                mask = loss_mask(batch, pad_id=pad_id, variant_ids=variant_ids_tuple)
                lp = F.log_softmax(shift_logits.float(), dim=-1)
                nll = -lp.gather(2, shift_targets.unsqueeze(-1)).squeeze(-1)
                n_tok = mask.sum().clamp(min=1)
                loss = (nll * mask).sum() / n_tok

            (loss / accum).backward()

            if (step + 1) % accum == 0:
                lr = cosine_warmup(global_step, total_steps, warmup_steps, opt_cfg["lr"], opt_cfg["lr_min"])
                for g in optimizer.param_groups:
                    g["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt_cfg["grad_clip"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if is_main(rank) and global_step % cfg["train"]["log_every_steps"] == 0:
                    log(rank, f"epoch={epoch} step={global_step}/{total_steps} loss={loss.item():.4f} lr={lr:.2e}")
            ep_loss += loss.item() * int(n_tok.item())
            ep_tokens += int(n_tok.item())

            if time.time() > soft_deadline:
                log(rank, "soft walltime reached during epoch — breaking")
                break

        train_loss = ep_loss / max(1, ep_tokens)
        log(rank, f"epoch {epoch} train_loss={train_loss:.4f} time={time.time() - t_ep:.0f}s")

        if (epoch + 1) % cfg["train"]["val_every_epochs"] == 0:
            val_metrics = evaluate(
                model, val_loader, device, rank, world_size,
                pad_id, variant_ids_tuple, tuple(cfg["eval"]["top_k"]),
            )
            log(rank, f"epoch {epoch} val={json.dumps(val_metrics)}")
            metrics_log.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
            if is_main(rank):
                with open(out_dir / "metrics.json", "w") as f:
                    json.dump(metrics_log, f, indent=2)

            improved = val_metrics["val_loss"] < best_val - 1e-4
            if improved:
                best_val = val_metrics["val_loss"]
                epochs_no_improve = 0
                if is_main(rank):
                    save_checkpoint(out_dir, "best", model, optimizer, {}, epoch, global_step, cfg, vocab_h, tokenizer.vocab_size)
            else:
                epochs_no_improve += 1

            if (epoch + 1) % cfg["train"]["ckpt_every_epochs"] == 0 and is_main(rank):
                save_checkpoint(out_dir, f"epoch{epoch:03d}", model, optimizer, {}, epoch, global_step, cfg, vocab_h, tokenizer.vocab_size)

            if epochs_no_improve >= cfg["train"]["early_stop_patience"]:
                log(rank, f"early stop at epoch {epoch}")
                break

        if time.time() > soft_deadline:
            log(rank, "soft walltime reached at epoch end")
            if is_main(rank):
                save_checkpoint(out_dir, "final", model, optimizer, {}, epoch, global_step, cfg, vocab_h, tokenizer.vocab_size)
            break

    if is_main(rank):
        save_checkpoint(out_dir, "final", model, optimizer, {}, epoch, global_step, cfg, vocab_h, tokenizer.vocab_size)

    # Park ranks 1-3 here so they don't destroy the process group while rank 0
    # runs test eval (which would cause the NCCL ALLREDUCE to hang/timeout).
    if dist.is_initialized():
        dist.barrier()

    if is_main(rank):
        # Final test eval on best checkpoint
        test_path = data_dir / "test.pt"
        if test_path.exists():
            log(rank, "loading best checkpoint for test eval...")
            ck = torch.load(out_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
            (model.module if isinstance(model, DDP) else model).load_state_dict(ck["model"])
            test_ds = PackedShardDataset([test_path])
            test_loader = DataLoader(test_ds, batch_size=bsz, shuffle=False, num_workers=2, pin_memory=True)
            test_metrics = evaluate(
                model, test_loader, device, 0, 1, pad_id, variant_ids_tuple, tuple(cfg["eval"]["top_k"]),
            )
        else:
            test_metrics = {}

        probe = {}
        ts_path = Path(args.test_sequences) if args.test_sequences else data_dir.parent / "test_sequences.json"
        if ts_path.exists():
            log(rank, "running memorization probe...")
            with open(ts_path) as f:
                test_seqs = json.load(f)
            single = model.module if isinstance(model, DDP) else model
            probe = perturbed_score_ratio(single, test_seqs, device, n=cfg["eval"]["memorization_n"], seed=cfg["seed"])

        report = {
            "checkpoint_path": str(out_dir / "checkpoint_best.pt"),
            "vocab_hash": vocab_h,
            "vocab_size": tokenizer.vocab_size,
            "model_params_M": (model.module if isinstance(model, DDP) else model).num_params() / 1e6,
            "epochs_trained": epoch + 1,
            "test_metrics": test_metrics,
            "memorization_probe": probe,
            "metrics_history": metrics_log,
            "elapsed_hours": (time.time() - job_start) / 3600,
        }
        with open(out_dir / "eval_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        log(rank, f"WROTE {out_dir / 'eval_report.json'}")

    cleanup_ddp()
    return 0


if __name__ == "__main__":
    sys.exit(main())
