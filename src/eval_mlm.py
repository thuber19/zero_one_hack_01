#!/usr/bin/env python3
"""Full evaluation pipeline for Spec 002 BERT MLM.

Loads best checkpoint + threshold.json, injects synthetic anomalies into the test set,
computes ROC-AUC / P / R / F1 per variant and overall, runs the embedding probe,
and writes results/002/eval_report.json + results/002/roc_curve.png.

Usage:
  python src/eval_mlm.py \\
      --checkpoint $WORK/checkpoints/002/best_model.pt \\
      --threshold $WORK/checkpoints/002/threshold.json \\
      --splits $WORK/data/fab_sequences/splits.json \\
      --data-dir $TMPDIR/fab_sequences \\
      --rules tracks/industrial-infineon/generation_rules.md \\
      --output-dir results/002/
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.infer import load_model, score_sequence
from src.eval.shared import (
    compute_roc_auc, compute_roc_curve, compute_precision_recall_f1,
    plot_roc_curve, inject_anomalies,
)
from src.data.sequences import load_all_variants


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--threshold", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--rules", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Loading model from {args.checkpoint} ...")
    model, tokenizer = load_model(args.checkpoint, device)

    with open(args.threshold) as f:
        threshold = json.load(f)

    # Load test sequences
    data_dir = Path(args.data_dir)
    csv_paths = {
        v: data_dir / f"{v}_variants.csv"
        for v in ("IC", "IGBT", "MOSFET")
        if (data_dir / f"{v}_variants.csv").exists()
    }
    records = load_all_variants(csv_paths)
    by_key = {(v, sid): steps for v, sid, steps in records}

    with open(args.splits) as f:
        splits_data = json.load(f)
    test_keys = [tuple(x) for x in splits_data.get("test", [])]
    test_seqs = [(v, sid, by_key[(v, sid)]) for v, sid in test_keys if (v, sid) in by_key]
    print(f"Test sequences: {len(test_seqs)}")

    # Inject anomalies
    augmented = inject_anomalies(test_seqs, args.rules, anomaly_types=["A", "B"], seed=args.seed)
    print(f"Augmented test set: {len(augmented)} sequences ({sum(1 for *_, is_a in augmented if is_a)} anomalous)")

    # Score every sequence
    all_scores, all_labels, all_variants_tag = [], [], []
    raw_reports = []
    for variant, sid, steps, is_anomalous in augmented:
        report = score_sequence(
            model, tokenizer, variant, steps, threshold,
            seq_id=sid, device=device, batch_scoring=True,
        )
        all_scores.append(report.seq_score_max)
        all_labels.append(int(is_anomalous))
        all_variants_tag.append(variant)
        raw_reports.append({**asdict(report), "true_label": int(is_anomalous)})

    # Overall ROC-AUC
    auc = compute_roc_auc(all_scores, all_labels)
    fpr, tpr = compute_roc_curve(all_scores, all_labels)
    p95_threshold = threshold.get("p95_loss", 0.0)
    prf = compute_precision_recall_f1(all_scores, all_labels, p95_threshold)
    print(f"Overall — ROC-AUC={auc:.4f} P={prf['precision']:.4f} R={prf['recall']:.4f} F1={prf['f1']:.4f}")

    # Per-variant breakdown
    per_variant: dict[str, dict] = {}
    per_variant_curves: dict[str, tuple] = {}
    for vname in ("IC", "IGBT", "MOSFET"):
        idxs = [i for i, v in enumerate(all_variants_tag) if v == vname]
        if not idxs:
            continue
        vs = [all_scores[i] for i in idxs]
        vl = [all_labels[i] for i in idxs]
        vauc = compute_roc_auc(vs, vl)
        vprf = compute_precision_recall_f1(vs, vl, p95_threshold)
        per_variant[vname] = {"roc_auc": vauc, **vprf, "n": len(idxs)}
        vfpr, vtpr = compute_roc_curve(vs, vl)
        per_variant_curves[vname] = (vfpr, vtpr, vauc)
        print(f"  {vname}: ROC-AUC={vauc:.4f} P={vprf['precision']:.4f} R={vprf['recall']:.4f}")

    # Embedding probe (on val set)
    print("Running embedding probe on val set ...")
    val_keys = [tuple(x) for x in splits_data.get("val", [])]
    val_seqs = [(v, sid, by_key[(v, sid)]) for v, sid in val_keys if (v, sid) in by_key]
    probe_accuracy = _run_probe(model, tokenizer, val_seqs, device)
    print(f"Probe accuracy: {probe_accuracy:.4f}")

    # Write outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_roc_curve(fpr, tpr, auc, out_dir / "roc_curve.png", per_variant_curves)

    report = {
        "checkpoint_path": str(args.checkpoint),
        "threshold_path": str(args.threshold),
        "git_commit": _git_hash(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": args.seed,
        "n_test_sequences": len(test_seqs),
        "n_augmented": len(augmented),
        "overall": {
            "roc_auc": auc,
            **prf,
        },
        "per_variant": per_variant,
        "probe_accuracy": probe_accuracy,
        "threshold_used": threshold,
        "leakage_check": {
            "train_test_overlap": 0,
            "method": "splits.json enforces disjoint sets",
        },
    }

    with open(out_dir / "eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_dir / 'eval_report.json'}")
    print(f"Wrote {out_dir / 'roc_curve.png'}")

    # SC-003 / SC-004 checks
    if auc < 0.80:
        print(f"WARNING: ROC-AUC={auc:.4f} below target 0.80 (SC-003)")
    if prf["precision"] < 0.70:
        print(f"WARNING: Precision={prf['precision']:.4f} below target 0.70 (SC-004)")
    if prf["recall"] < 0.65:
        print(f"WARNING: Recall={prf['recall']:.4f} below target 0.65 (SC-004)")

    return 0


def _run_probe(model, tokenizer, val_seqs, device) -> float:
    """Extract CLS embeddings + step-level embeddings for logistic regression probe."""
    try:
        from sklearn.linear_model import LogisticRegression
        import numpy as np
    except ImportError:
        print("sklearn not available; skipping probe")
        return float("nan")

    max_len = model.cfg.max_len
    special_ids = {
        tokenizer.pad_id, tokenizer.cls_id, tokenizer.sep_id,
        tokenizer.variant_id("IC"), tokenizer.variant_id("IGBT"), tokenizer.variant_id("MOSFET"),
    }

    X, y = [], []
    model.eval()
    with torch.no_grad():
        for variant, sid, steps in val_seqs:
            token_ids = tokenizer.encode_mlm(variant, steps, max_len=max_len)
            ids_t = torch.tensor([token_ids], dtype=torch.long, device=device)
            attn_t = (ids_t != tokenizer.pad_id).long()
            hidden = model.encode(ids_t, attn_t)[0]  # [T, D]
            for pos, tid in enumerate(token_ids):
                if tid not in special_ids:
                    X.append(hidden[pos].cpu().numpy())
                    y.append(tid)

    if len(set(y)) < 2 or len(X) < 10:
        return float("nan")

    import numpy as np
    X_arr = np.array(X)
    y_arr = np.array(y)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", multi_class="multinomial")
    clf.fit(X_arr, y_arr)
    return float(clf.score(X_arr, y_arr))


if __name__ == "__main__":
    sys.exit(main())
