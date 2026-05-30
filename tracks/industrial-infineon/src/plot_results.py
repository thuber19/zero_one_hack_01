"""
Plotting script: generates visualizations from training history and eval results.

Usage:
    python src/plot_results.py --output-dir outputs
    python src/plot_results.py --output-dir outputs --scaling-dirs outputs/tiny outputs/small outputs/medium
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history_path: Path, save_dir: Path):
    """Plot loss and accuracy curves from training history."""
    with open(history_path) as f:
        data = json.load(f)

    history = data["transformer_history"]
    config = data["config"]
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_acc = [h["val_accuracy"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_loss, label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Training & Validation Loss ({config['model_size']} model)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(epochs, val_acc, label="Val Accuracy", linewidth=2, color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Validation Accuracy ({config['model_size']} model)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved training_curves.png")


def plot_eval_metrics(metrics_path: Path, save_dir: Path):
    """Plot evaluation metrics as bar charts."""
    with open(metrics_path) as f:
        metrics = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Task 1: Next-step prediction
    ax = axes[0]
    task1 = metrics["task1_nextstep"]
    names = ["Top-1", "Top-3", "Top-5", "MRR"]
    values = [task1["top1_accuracy"], task1["top3_accuracy"], task1["top5_accuracy"], task1["mrr"]]
    bars = ax.bar(names, values, color=["#2196F3", "#42A5F5", "#64B5F6", "#90CAF9"])
    ax.set_ylim(0, 1)
    ax.set_title("Task 1: Next-Step Prediction")
    ax.set_ylabel("Score")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Task 2: Sequence completion
    ax = axes[1]
    task2 = metrics["task2_completion"]
    names = ["Exact\nMatch", "Token\nAccuracy", "1 - Edit\nDistance"]
    values = [task2["exact_match_rate"], task2["token_accuracy"],
              1 - task2["normalized_edit_distance"]]
    bars = ax.bar(names, values, color=["#4CAF50", "#66BB6A", "#81C784"])
    ax.set_ylim(0, 1)
    ax.set_title("Task 2: Sequence Completion")
    ax.set_ylabel("Score")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Task 3: Anomaly detection
    ax = axes[2]
    task3 = metrics["task3_anomaly"]
    names = ["Accuracy", "Precision", "Recall", "F1", "Rule\nAttrib."]
    values = [task3["binary_accuracy"], task3["precision"], task3["recall"],
              task3["f1"], task3["rule_attribution_accuracy"]]
    bars = ax.bar(names, values, color=["#FF9800", "#FFA726", "#FFB74D", "#FFCC80", "#FFE0B2"])
    ax.set_ylim(0, 1)
    ax.set_title("Task 3: Anomaly Detection")
    ax.set_ylabel("Score")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_dir / "eval_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved eval_metrics.png")


def plot_scaling_comparison(scaling_dirs: list[Path], save_dir: Path):
    """Compare metrics across model sizes (tiny/small/medium)."""
    results = {}
    for d in scaling_dirs:
        history_path = d / "training_history.json"
        if not history_path.exists():
            continue
        with open(history_path) as f:
            data = json.load(f)
        size = data["config"]["model_size"]
        n_params = data["config"]["n_params"]
        best_val_loss = min(h["val_loss"] for h in data["transformer_history"])
        best_val_acc = max(h["val_accuracy"] for h in data["transformer_history"])
        results[size] = {
            "n_params": n_params,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "history": data["transformer_history"],
        }

    if len(results) < 2:
        print("  Need at least 2 model sizes for scaling comparison, skipping.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"tiny": "#F44336", "small": "#2196F3", "medium": "#4CAF50"}

    # Loss curves overlay
    ax = axes[0]
    for size, r in sorted(results.items(), key=lambda x: x[1]["n_params"]):
        epochs = [h["epoch"] for h in r["history"]]
        val_loss = [h["val_loss"] for h in r["history"]]
        ax.plot(epochs, val_loss, label=f"{size} ({r['n_params']:,} params)",
                linewidth=2, color=colors.get(size, "gray"))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Scaling: Val Loss by Model Size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy curves overlay
    ax = axes[1]
    for size, r in sorted(results.items(), key=lambda x: x[1]["n_params"]):
        epochs = [h["epoch"] for h in r["history"]]
        val_acc = [h["val_accuracy"] for h in r["history"]]
        ax.plot(epochs, val_acc, label=f"{size} ({r['n_params']:,} params)",
                linewidth=2, color=colors.get(size, "gray"))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Scaling: Val Accuracy by Model Size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Params vs best accuracy
    ax = axes[2]
    sizes = sorted(results.keys(), key=lambda s: results[s]["n_params"])
    params = [results[s]["n_params"] for s in sizes]
    accs = [results[s]["best_val_acc"] for s in sizes]
    ax.plot(params, accs, "o-", linewidth=2, markersize=10, color="#9C27B0")
    for s, p, a in zip(sizes, params, accs):
        ax.annotate(s, (p, a), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Best Val Accuracy")
    ax.set_title("Scaling: Parameters vs Accuracy")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "scaling_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved scaling_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Plot training and evaluation results")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--scaling-dirs", type=Path, nargs="*", default=None,
                        help="Directories for scaling comparison (e.g., outputs/tiny outputs/small)")
    args = parser.parse_args()

    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Training curves
    history_path = args.output_dir / "training_history.json"
    if history_path.exists():
        print("Plotting training curves...")
        plot_training_curves(history_path, plots_dir)

    # Eval metrics
    metrics_path = args.output_dir / "eval_results" / "self_eval_metrics.json"
    if metrics_path.exists():
        print("Plotting eval metrics...")
        plot_eval_metrics(metrics_path, plots_dir)

    # Scaling comparison
    if args.scaling_dirs:
        print("Plotting scaling comparison...")
        plot_scaling_comparison(args.scaling_dirs, plots_dir)

    print(f"\nAll plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
