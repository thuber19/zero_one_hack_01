#!/usr/bin/env python3
"""Aggregate per-model metrics into a comparison report for the Spec 004 GUI leaderboard.

Reads:
  $WORK/checkpoints/005-lstm-baseline/metrics_lstm.json   (Spec 005 schema)
  $WORK/checkpoints/001-gpt-fab/eval_report.json          (Spec 001 schema)

Writes:
  --output <path>  (default: $WORK/reports/comparison.json)

Schema per entry:
  {"model": str, "metric": str, "value": float|null, "ci_low": float|null, "ci_high": float|null}
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


METRICS = ["top1_accuracy", "top5_accuracy", "perplexity", "probe_score"]

# ---- Wilson CI for proportions (accuracy metrics) -------------------------

def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n < 2:
        return (None, None)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def ci_for_metric(metric: str, value: float, n_tokens: int | None) -> tuple[float | None, float | None]:
    """Return (ci_low, ci_high) where available; null otherwise."""
    if value is None or n_tokens is None or n_tokens < 100:
        return (None, None)
    if metric in ("top1_accuracy", "top5_accuracy"):
        return wilson_ci(value, n_tokens)
    # perplexity and probe_score: no closed-form CI without per-sample data
    return (None, None)


# ---- Schema readers --------------------------------------------------------

def read_lstm_metrics(path: Path) -> dict | None:
    """Read Spec 005 metrics_lstm.json."""
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    return {
        "top1_accuracy": d.get("top1_accuracy"),
        "top5_accuracy": d.get("top5_accuracy"),
        "perplexity": d.get("perplexity"),
        "probe_score": d.get("probe_score"),
        "n_tokens": d.get("n_tokens"),
    }


def read_gpt_metrics(path: Path) -> dict | None:
    """Read Spec 001 eval_report.json."""
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    tm = d.get("test_metrics", {})
    probe = d.get("memorization_probe", {})
    return {
        "top1_accuracy": tm.get("top1_accuracy"),
        "top5_accuracy": tm.get("top5_accuracy"),
        "perplexity": tm.get("val_perplexity"),
        "probe_score": probe.get("ratio"),
        "n_tokens": tm.get("n_tokens"),
    }


# ---- Report builder --------------------------------------------------------

def build_rows(model_name: str, metrics: dict | None) -> list[dict]:
    rows = []
    for metric in METRICS:
        if metrics is None:
            value, ci_low, ci_high = None, None, None
        else:
            value = metrics.get(metric)
            ci_low, ci_high = ci_for_metric(metric, value, metrics.get("n_tokens"))
        rows.append({
            "model": model_name,
            "metric": metric,
            "value": value,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate leaderboard comparison report.")
    ap.add_argument("--output", default=None, help="Output path (default: $WORK/reports/comparison.json)")
    ap.add_argument("--work_dir", default=None, help="Override $WORK base dir")
    args = ap.parse_args()

    work = Path(args.work_dir) if args.work_dir else Path(os.environ.get("WORK", "."))

    lstm_path = work / "checkpoints/005-lstm-baseline/metrics_lstm.json"
    gpt_path = work / "checkpoints/001-gpt-fab/eval_report.json"

    print(f"Reading LSTM metrics from: {lstm_path}")
    print(f"Reading GPT  metrics from: {gpt_path}")

    lstm_m = read_lstm_metrics(lstm_path)
    gpt_m = read_gpt_metrics(gpt_path)

    if lstm_m is None:
        print(f"WARNING: {lstm_path} not found — LSTM rows will be null")
    if gpt_m is None:
        print(f"WARNING: {gpt_path} not found — GPT rows will be null")

    rows = []
    rows.extend(build_rows("lstm-baseline", lstm_m))
    rows.extend(build_rows("transformer", gpt_m))

    report = {"models": rows}

    out_path = Path(args.output) if args.output else work / "reports/comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"WROTE {out_path}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
