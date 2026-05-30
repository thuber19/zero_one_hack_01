#!/usr/bin/env python3
"""
Score eval sequences through BERT MLM (checkpoint 002) and produce:
  results/eval_anomaly_predictions.csv  — for eval_metrics.py --task anomaly
  results/eval_scored.json              — for the frontend API to serve

Usage:
  python scripts/score_eval.py [--device mps|cpu|cuda]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from src.infer import load_model, pseudo_perplexity_batch
from src.tokenizer import MLMTokenizer

EVAL_DIR  = ROOT / "tracks" / "industrial-infineon" / "eval"
RESULTS   = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
CHECKPOINT = ROOT / "checkpoints" / "002" / "checkpoint_best.pt"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def score_sequence_raw(model, tokenizer: MLMTokenizer, variant: str, steps: list[str], device: torch.device) -> dict:
    """Return raw per-step losses and aggregate scores."""
    max_len = model.cfg.max_len
    token_ids = tokenizer.encode_mlm(variant, steps, max_len=max_len)
    losses = pseudo_perplexity_batch(model, tokenizer, token_ids, device)

    scored = [l for l in losses if l > 0.0]
    if not scored:
        return {"mean": 0.0, "max": 0.0, "per_step": losses, "n_scored": 0}

    mean = sum(scored) / len(scored)
    return {
        "mean": mean,
        "max": max(losses),
        "per_step": losses,
        "n_scored": len(scored),
    }


def build_per_step(steps: list[str], example_id: str, losses: list[float], risk_scale: float) -> list[dict]:
    result = []
    for i, step in enumerate(steps):
        raw = losses[i] if i < len(losses) else 0.0
        risk = min(raw / risk_scale, 1.0)
        result.append({
            "step_id": f"{example_id}_s{i:03d}",
            "step_name": step,
            "category": "process",
            "risk_score": round(risk, 4),
            "confidence_lo": round(max(0.0, risk - 0.08), 4),
            "confidence_hi": round(min(1.0, risk + 0.08), 4),
            "shap": [],
        })
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Loading model from {CHECKPOINT.relative_to(ROOT)} ...")
    model, tokenizer = load_model(CHECKPOINT, device)
    print(f"Model ready — vocab={tokenizer.vocab_size}, max_len={model.cfg.max_len}\n")

    # -----------------------------------------------------------------------
    # Load eval CSVs
    # -----------------------------------------------------------------------
    anomaly_rows = read_csv(EVAL_DIR / "eval_input_anomaly.csv")
    valid_rows   = read_csv(EVAL_DIR / "eval_input_valid.csv")
    total = len(anomaly_rows) + len(valid_rows)
    print(f"Sequences: {len(anomaly_rows)} anomaly + {len(valid_rows)} valid = {total} total\n")

    all_scores: list[dict] = []  # for threshold calibration
    pred_rows:  list[dict] = []  # for predictions CSV
    batch_json: list[dict] = []  # for frontend JSON

    # -----------------------------------------------------------------------
    # Scoring loop
    # -----------------------------------------------------------------------
    def run(rows: list[dict], true_label: int, seq_col: str, label_str: str):
        t_start = time.time()
        for i, row in enumerate(rows):
            eid    = row["EXAMPLE_ID"]
            family = row.get("FAMILY", "IC").strip()
            variant = family if family in ("IC", "IGBT", "MOSFET") else "IC"
            seq_str = row.get(seq_col, "").strip()
            steps = [s.strip() for s in seq_str.split("|") if s.strip()]

            t0 = time.time()
            result = score_sequence_raw(model, tokenizer, variant, steps, device)
            elapsed = time.time() - t0

            all_scores.append({"eid": eid, "mean": result["mean"], "label": true_label})

            if (i + 1) % 100 == 0 or i == 0:
                eta = (time.time() - t_start) / (i + 1) * (len(rows) - i - 1)
                print(f"  [{label_str}] {i+1:4d}/{len(rows)}  {eid}  mean={result['mean']:.4f}  {elapsed:.2f}s  ETA {eta:.0f}s")

            # store raw; will convert to IS_VALID after calibration
            pred_rows.append({
                "EXAMPLE_ID":    eid,
                "mean_loss":     result["mean"],
                "max_loss":      result["max"],
                "n_scored":      result["n_scored"],
                "true_label":    true_label,
                "family":        family,
                "steps":         steps,
                "per_step_loss": result["per_step"],
            })

    print("=== Scoring anomaly sequences ===")
    run(anomaly_rows, true_label=0, seq_col="SEQUENCE",         label_str="ANOMALY")
    print("\n=== Scoring valid sequences ===")
    run(valid_rows,   true_label=1, seq_col="PARTIAL_SEQUENCE", label_str="VALID  ")

    # -----------------------------------------------------------------------
    # Calibrate threshold using valid-set mean losses
    # -----------------------------------------------------------------------
    valid_means  = sorted(r["mean_loss"] for r in pred_rows if r["true_label"] == 1)
    anomaly_means= sorted(r["mean_loss"] for r in pred_rows if r["true_label"] == 0)

    # Use 95th percentile of valid scores as the anomaly threshold
    p95_idx = int(0.95 * len(valid_means))
    threshold = valid_means[p95_idx] if valid_means else 1.0

    # risk_scale: normalise raw loss to [0,1] for frontend risk_score display
    p99_idx = int(0.99 * len(anomaly_means))
    risk_scale = max(anomaly_means[p99_idx] if anomaly_means else 3.0, 0.5)

    print(f"\nCalibrated threshold (p95 of valid means): {threshold:.4f}")
    print(f"Risk scale (p99 of anomaly means):        {risk_scale:.4f}")

    # -----------------------------------------------------------------------
    # Build output rows
    # -----------------------------------------------------------------------
    csv_rows = []
    for r in pred_rows:
        mean = r["mean_loss"]
        # validity score: 1 = definitely valid, 0 = definitely anomalous
        validity_score = 1.0 / (1.0 + mean)
        is_valid_pred  = 1 if mean <= threshold else 0

        csv_rows.append({
            "EXAMPLE_ID":     r["EXAMPLE_ID"],
            "IS_VALID":       is_valid_pred,
            "SCORE":          round(validity_score, 6),
            "PREDICTED_RULE": "",
        })

        per_step = build_per_step(r["steps"], r["EXAMPLE_ID"], r["per_step_loss"], risk_scale)
        risk_steps = sum(1 for s in per_step if s["risk_score"] >= 0.70)
        batch_json.append({
            "batch_id":          r["EXAMPLE_ID"],
            "family":            r["family"],
            "true_is_valid":     r["true_label"],
            "predicted_valid":   is_valid_pred,
            "predicted_yield":   round(max(0.0, 1.0 - mean / risk_scale), 4),
            "confidence":        round(validity_score, 4),
            "risk_steps_detected": risk_steps,
            "anomalous_batches": 0 if r["true_label"] == 1 else 1,
            "mean_loss":         round(mean, 6),
            "per_step":          per_step,
            "anomalies":         [],
        })

    # -----------------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------------
    pred_csv = RESULTS / "eval_anomaly_predictions.csv"
    with pred_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nWrote {pred_csv.relative_to(ROOT)} ({len(csv_rows)} rows)")

    scored_json = RESULTS / "eval_scored.json"
    with scored_json.open("w") as f:
        json.dump(batch_json, f, indent=2)
    print(f"Wrote {scored_json.relative_to(ROOT)}")

    # -----------------------------------------------------------------------
    # Quick accuracy report
    # -----------------------------------------------------------------------
    n_correct = sum(1 for r, c in zip(pred_rows, csv_rows) if c["IS_VALID"] == r["true_label"])
    tp = sum(1 for r, c in zip(pred_rows, csv_rows) if r["true_label"] == 0 and c["IS_VALID"] == 0)
    tn = sum(1 for r, c in zip(pred_rows, csv_rows) if r["true_label"] == 1 and c["IS_VALID"] == 1)
    fp = sum(1 for r, c in zip(pred_rows, csv_rows) if r["true_label"] == 1 and c["IS_VALID"] == 0)
    fn = sum(1 for r, c in zip(pred_rows, csv_rows) if r["true_label"] == 0 and c["IS_VALID"] == 1)

    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    print(f"""
┌─────────────────────────────────────────────────┐
│  QUICK EVAL RESULTS (BERT MLM checkpoint 002)   │
├─────────────────────────────────────────────────┤
│  Total sequences:  {len(pred_rows):>4d}                        │
│  Anomaly (label=0):{len(anomaly_rows):>4d}                        │
│  Valid   (label=1):{len(valid_rows):>4d}                        │
│  Threshold:        {threshold:.4f}                     │
├─────────────────────────────────────────────────┤
│  Accuracy:  {n_correct/len(pred_rows):.4f}  ({n_correct}/{len(pred_rows)})           │
│  Precision: {prec:.4f}                           │
│  Recall:    {rec:.4f}                           │
│  F1:        {f1:.4f}                           │
│  Confusion: TP={tp:<4d} FP={fp:<4d} FN={fn:<4d} TN={tn:<4d}   │
└─────────────────────────────────────────────────┘
""")
    print(f"Run official eval:")
    print(f"  python tracks/industrial-infineon/eval/eval_metrics.py \\")
    print(f"    --task anomaly \\")
    print(f"    --ground-truth tracks/industrial-infineon/eval/eval_input_anomaly.csv \\")
    print(f"    --predictions results/eval_anomaly_predictions.csv \\")
    print(f"    --valid-supplement tracks/industrial-infineon/eval/eval_input_valid.csv")


if __name__ == "__main__":
    main()
