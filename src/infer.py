#!/usr/bin/env python3
"""Pseudo-perplexity inference for Spec 002 BERT MLM anomaly detection.

Usage:
  python src/infer.py \\
      --checkpoint $WORK/checkpoints/002/best_model.pt \\
      --threshold $WORK/checkpoints/002/threshold.json \\
      --input sequences.json \\
      --output results/002/anomaly_scores.json

Input JSON format: list of {"variant": "IC", "steps": ["STEP1", "STEP2", ...]}
Output JSON: list of AnomalyReport dicts
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.tokenizer import MLMTokenizer
from src.model.bert_mlm import BertMLMEncoder, BertMLMConfig


@dataclass
class AnomalyReport:
    sequence_id: str
    per_step_raw_loss: list[float]
    per_step_zscore: list[float]
    seq_score_max: float
    seq_score_mean: float
    is_anomalous: bool
    is_ood: bool
    anomalous_steps: list[int]
    threshold_used: dict
    warnings: list[str] = field(default_factory=list)


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[BertMLMEncoder, MLMTokenizer]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    vocab_path = ckpt.get("vocab_path")

    # Prefer tokenizer saved next to checkpoint
    ckpt_dir = Path(checkpoint_path).parent
    local_tok = ckpt_dir / "tokenizer.json"
    if local_tok.exists():
        tokenizer = MLMTokenizer.load(local_tok)
    elif vocab_path and Path(vocab_path).exists():
        tokenizer = MLMTokenizer.load(vocab_path)
    else:
        raise FileNotFoundError(
            f"Cannot find tokenizer: tried {local_tok} and {vocab_path}"
        )

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
    model = BertMLMEncoder(model_cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, tokenizer


def pseudo_perplexity_batch(
    model: BertMLMEncoder,
    tokenizer: MLMTokenizer,
    token_ids: list[int],
    device: torch.device,
) -> list[float]:
    """Batch pseudo-perplexity: construct N×T matrix with MASK at each position.

    Returns per-step cross-entropy loss for each non-PAD/CLS/SEP/VARIANT position.
    Positions that are special tokens are returned as 0.0.
    """
    T = len(token_ids)
    ids_tensor = torch.tensor(token_ids, dtype=torch.long)

    # Identify scored positions (non-special tokens)
    special_ids = {
        tokenizer.pad_id, tokenizer.cls_id, tokenizer.sep_id,
        tokenizer.variant_id("IC"), tokenizer.variant_id("IGBT"), tokenizer.variant_id("MOSFET"),
    }
    scored_positions = [i for i in range(T) if token_ids[i] not in special_ids]

    if not scored_positions:
        return [0.0] * T

    N = len(scored_positions)
    # Build N×T matrix: row i has MASK at scored_positions[i]
    batch_ids = ids_tensor.unsqueeze(0).expand(N, T).clone()  # [N, T]
    for i, pos in enumerate(scored_positions):
        batch_ids[i, pos] = tokenizer.mask_id

    attention_mask = (ids_tensor != tokenizer.pad_id).long().unsqueeze(0).expand(N, T)

    batch_ids = batch_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        logits = model(batch_ids, attention_mask)  # [N, T, V]

    losses = [0.0] * T
    for i, pos in enumerate(scored_positions):
        logit_at_pos = logits[i, pos, :].float()
        true_id = token_ids[pos]
        loss = F.cross_entropy(logit_at_pos.unsqueeze(0), torch.tensor([true_id], device=device))
        losses[pos] = loss.item()

    return losses


def pseudo_perplexity_sequential(
    model: BertMLMEncoder,
    tokenizer: MLMTokenizer,
    token_ids: list[int],
    device: torch.device,
) -> list[float]:
    """Sequential pseudo-perplexity: N forward passes (slower, for debugging)."""
    T = len(token_ids)
    ids_tensor = torch.tensor(token_ids, dtype=torch.long).to(device)
    attention_mask = (ids_tensor != tokenizer.pad_id).long().unsqueeze(0)
    special_ids = {
        tokenizer.pad_id, tokenizer.cls_id, tokenizer.sep_id,
        tokenizer.variant_id("IC"), tokenizer.variant_id("IGBT"), tokenizer.variant_id("MOSFET"),
    }
    losses = [0.0] * T
    with torch.no_grad():
        for pos in range(T):
            if token_ids[pos] in special_ids:
                continue
            masked = ids_tensor.clone().unsqueeze(0)
            masked[0, pos] = tokenizer.mask_id
            logits = model(masked, attention_mask)
            loss = F.cross_entropy(
                logits[0, pos, :].float().unsqueeze(0),
                torch.tensor([token_ids[pos]], device=device),
            )
            losses[pos] = loss.item()
    return losses


def score_sequence(
    model: BertMLMEncoder,
    tokenizer: MLMTokenizer,
    variant: str,
    steps: list[str],
    threshold: dict,
    seq_id: str = "unknown",
    device: torch.device = torch.device("cpu"),
    batch_scoring: bool = True,
) -> AnomalyReport:
    max_len = model.cfg.max_len
    token_ids = tokenizer.encode_mlm(variant, steps, max_len=max_len)

    if batch_scoring:
        per_step_losses = pseudo_perplexity_batch(model, tokenizer, token_ids, device)
    else:
        per_step_losses = pseudo_perplexity_sequential(model, tokenizer, token_ids, device)

    # Only score non-zero positions (i.e. positions where we have real steps)
    scored_losses = [l for l in per_step_losses if l > 0.0]

    if not scored_losses:
        return AnomalyReport(
            sequence_id=seq_id,
            per_step_raw_loss=per_step_losses,
            per_step_zscore=[0.0] * len(per_step_losses),
            seq_score_max=0.0,
            seq_score_mean=0.0,
            is_anomalous=False,
            is_ood=False,
            anomalous_steps=[],
            threshold_used=threshold,
            warnings=["no scored positions"],
        )

    import statistics
    mean_loss = statistics.mean(scored_losses)
    std_loss = statistics.stdev(scored_losses) if len(scored_losses) > 1 else 0.0

    z_scores = [(l - mean_loss) / (std_loss + 1e-8) for l in per_step_losses]

    seq_score_max = max(per_step_losses)
    seq_score_mean = mean_loss

    p99 = threshold.get("p99_loss", float("inf"))
    p95 = threshold.get("p95_loss", float("inf"))
    ood_p99 = threshold.get("ood_p99", float("inf"))

    anomalous_steps = [i for i, l in enumerate(per_step_losses) if l >= p95]
    is_anomalous = seq_score_max >= p95
    is_ood = seq_score_mean >= ood_p99

    return AnomalyReport(
        sequence_id=seq_id,
        per_step_raw_loss=per_step_losses,
        per_step_zscore=z_scores,
        seq_score_max=seq_score_max,
        seq_score_mean=seq_score_mean,
        is_anomalous=is_anomalous,
        is_ood=is_ood,
        anomalous_steps=anomalous_steps,
        threshold_used=threshold,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--threshold", required=True)
    ap.add_argument("--input", required=True, help="JSON file: list of {variant, steps, id?}")
    ap.add_argument("--output", required=True)
    ap.add_argument("--sequential-scoring", action="store_true", dest="sequential")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, tokenizer = load_model(args.checkpoint, device)

    with open(args.threshold) as f:
        threshold = json.load(f)

    with open(args.input) as f:
        records = json.load(f)

    reports = []
    for rec in records:
        seq_id = rec.get("id", rec.get("sequence_id", "unknown"))
        variant = rec["variant"]
        steps = rec["steps"]
        report = score_sequence(
            model, tokenizer, variant, steps, threshold,
            seq_id=seq_id, device=device, batch_scoring=not args.sequential,
        )
        reports.append(asdict(report))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"Wrote {len(reports)} anomaly reports to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
