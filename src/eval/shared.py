"""Shared evaluation utilities — used by Specs 001, 002, 005.

Public API:
  compute_roc_auc(scores, labels) -> float
  compute_precision_recall_f1(scores, labels, threshold) -> dict
  plot_roc_curve(fpr, tpr, auc, output_path, per_variant_curves=None)
  inject_anomalies(sequences, rules_path, anomaly_types, seed) -> list[tuple[list[str], bool]]
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence


def compute_roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Compute ROC-AUC via trapezoidal rule (no sklearn required at import time)."""
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tp = fp = 0
    prev_fpr = prev_tpr = 0.0
    auc = 0.0
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        fpr = fp / n_neg
        tpr = tp / n_pos
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr, prev_tpr = fpr, tpr
    return float(auc)


def compute_roc_curve(scores: Sequence[float], labels: Sequence[int]) -> tuple[list[float], list[float]]:
    """Return (fpr_list, tpr_list) sorted by ascending threshold (for plotting)."""
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return [0.0, 1.0], [0.0, 1.0]
    fpr_list, tpr_list = [0.0], [0.0]
    tp = fp = 0
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        fpr_list.append(fp / n_neg)
        tpr_list.append(tp / n_pos)
    return fpr_list, tpr_list


def compute_precision_recall_f1(
    scores: Sequence[float], labels: Sequence[int], threshold: float
) -> dict:
    tp = fp = fn = 0
    for s, l in zip(scores, labels):
        pred = int(s >= threshold)
        if pred == 1 and l == 1:
            tp += 1
        elif pred == 1 and l == 0:
            fp += 1
        elif pred == 0 and l == 1:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def plot_roc_curve(
    fpr: list[float],
    tpr: list[float],
    auc: float,
    output_path: str | Path,
    per_variant_curves: dict[str, tuple[list[float], list[float], float]] | None = None,
) -> None:
    """Save ROC curve plot as PNG. Requires matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f"Overall (AUC={auc:.3f})")
    if per_variant_curves:
        for variant, (vfpr, vtpr, vauc) in per_variant_curves.items():
            ax.plot(vfpr, vtpr, lw=1.5, linestyle="--", label=f"{variant} (AUC={vauc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Anomaly Detection (Spec 002)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def inject_anomalies(
    sequences: list[tuple[str, str, list[str]]],
    rules_path: str | Path,
    anomaly_types: list[str] | None = None,
    seed: int = 42,
) -> list[tuple[str, str, list[str], bool]]:
    """Return list of (variant, seq_id, steps, is_anomalous).

    Delegates to AnomalyInjector to parse generation_rules.md and inject rule violations.
    """
    from src.eval.anomaly_injector import AnomalyInjector

    if anomaly_types is None:
        anomaly_types = ["A", "B"]
    injector = AnomalyInjector(rules_path, seed=seed)
    return injector.inject(sequences, anomaly_types=anomaly_types)
