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
from src.infer import load_model, score_sequence as _score_seq
from src.tokenizer import MLMTokenizer

EVAL_DIR   = ROOT / "tracks" / "industrial-infineon" / "eval"
RESULTS    = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
CHECKPOINT = ROOT / "checkpoints" / "002" / "checkpoint_best.pt"
THRESHOLD_FILE = ROOT / "checkpoints" / "002" / "threshold.json"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def build_per_step(steps: list[str], example_id: str, losses: list[float], p99_loss: float) -> list[dict]:
    result = []
    for i, step in enumerate(steps):
        raw = losses[i] if i < len(losses) else 0.0
        # Normalize against calibrated p99 so risk_score is meaningful
        risk = min(raw / max(p99_loss, 0.01), 1.0)
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
    print(f"Model ready — vocab={tokenizer.vocab_size}, max_len={model.cfg.max_len}")

    threshold = json.loads(THRESHOLD_FILE.read_text())
    p95  = threshold["p95_loss"]
    p99  = threshold["p99_loss"]
    ood  = threshold["ood_p99"]
    print(f"Calibrated thresholds — p95={p95:.4f}  p99={p99:.4f}  ood_p99={ood:.4f}  "
          f"(n={threshold['calibration_n']} seqs, {threshold['n_per_step_samples']} steps)\n")

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
            report = _score_seq(
                model, tokenizer,
                variant=variant,
                steps=steps,
                threshold=threshold,
                seq_id=eid,
                device=device,
            )
            elapsed = time.time() - t0

            all_scores.append({"eid": eid, "mean": report.seq_score_mean, "label": true_label})

            if (i + 1) % 100 == 0 or i == 0:
                eta = (time.time() - t_start) / (i + 1) * (len(rows) - i - 1)
                flag = "🚨" if (report.is_anomalous or report.is_ood) else "✓"
                print(f"  [{label_str}] {i+1:4d}/{len(rows)}  {eid}  "
                      f"mean={report.seq_score_mean:.4f}  max={report.seq_score_max:.4f}  "
                      f"{flag}  {elapsed:.2f}s  ETA {eta:.0f}s")

            pred_rows.append({
                "EXAMPLE_ID":    eid,
                "mean_loss":     report.seq_score_mean,
                "max_loss":      report.seq_score_max,
                "is_anomalous":  report.is_anomalous,
                "is_ood":        report.is_ood,
                "anomalous_steps": report.anomalous_steps,
                "true_label":    true_label,
                "family":        family,
                "steps":         steps,
                "per_step_loss": report.per_step_raw_loss,
            })

    print("=== Scoring anomaly sequences ===")
    run(anomaly_rows, true_label=0, seq_col="SEQUENCE",         label_str="ANOMALY")
    print("\n=== Scoring valid sequences ===")
    run(valid_rows,   true_label=1, seq_col="PARTIAL_SEQUENCE", label_str="VALID  ")

    # -----------------------------------------------------------------------
    # Build output rows using calibrated thresholds
    # -----------------------------------------------------------------------
    # is_valid=0 if model says anomalous OR OOD
    print(f"\nBuilding predictions with calibrated thresholds...")
    csv_rows = []
    for r in pred_rows:
        mean = r["mean_loss"]
        # validity score: 1 = definitely valid, 0 = definitely anomalous
        # Use normalized score against ood_p99 so AUC is properly calibrated
        validity_score = max(0.0, 1.0 - mean / (ood * 2))
        is_valid_pred  = 0 if (r["is_anomalous"] or r["is_ood"]) else 1

        csv_rows.append({
            "EXAMPLE_ID":     r["EXAMPLE_ID"],
            "IS_VALID":       is_valid_pred,
            "SCORE":          round(validity_score, 6),
            "PREDICTED_RULE": "",
        })

        per_step = build_per_step(r["steps"], r["EXAMPLE_ID"], r["per_step_loss"], p99)
        risk_steps = sum(1 for s in per_step if s["risk_score"] >= 0.70)
        batch_json.append({
            "batch_id":          r["EXAMPLE_ID"],
            "family":            r["family"],
            "true_is_valid":     r["true_label"],
            "predicted_valid":   is_valid_pred,
            "predicted_yield":   round(max(0.0, 1.0 - mean / max(p99, 0.01)), 4),
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
┌─────────────────────────────────────────────────────┐
│  QUICK EVAL RESULTS (BERT MLM + calibrated thresh)  │
├─────────────────────────────────────────────────────┤
│  Total sequences:  {len(pred_rows):>4d}                          │
│  Anomaly (label=0):{len(anomaly_rows):>4d}                          │
│  Valid   (label=1):{len(valid_rows):>4d}                          │
│  p95_loss={p95:.4f}  p99={p99:.4f}  ood_p99={ood:.4f}   │
├─────────────────────────────────────────────────────┤
│  Accuracy:  {n_correct/len(pred_rows):.4f}  ({n_correct}/{len(pred_rows)})             │
│  Precision: {prec:.4f}                             │
│  Recall:    {rec:.4f}                             │
│  F1:        {f1:.4f}                             │
│  TP={tp:<4d} FP={fp:<4d} FN={fn:<4d} TN={tn:<4d}              │
└─────────────────────────────────────────────────────┘
""")
    print(f"Run official eval:")
    print(f"  python tracks/industrial-infineon/eval/eval_metrics.py \\")
    print(f"    --task anomaly \\")
    print(f"    --ground-truth tracks/industrial-infineon/eval/eval_input_anomaly.csv \\")
    print(f"    --predictions results/eval_anomaly_predictions.csv \\")
    print(f"    --valid-supplement tracks/industrial-infineon/eval/eval_input_valid.csv")


if __name__ == "__main__":
    main()
