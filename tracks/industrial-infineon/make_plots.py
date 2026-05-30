#!/usr/bin/env python3
"""
make_plots.py — loss/accuracy curves + method comparison (required by the rubric).

Reads training_history.json from each model dir and writes PNGs to plots/.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SUB = Path(__file__).resolve().parent
MODELS = {"M0 baseline": "outputs_test", "M1 continue+integ": "outputs_M1",
          "M2 scratch+integ": "outputs_M2"}
OUT = _SUB / "plots"; OUT.mkdir(exist_ok=True)


def load_hist(d):
    p = _SUB / d / "training_history.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main():
    hists = {name: load_hist(d) for name, d in MODELS.items()}
    hists = {k: v for k, v in hists.items() if v}
    if not hists:
        print("no histories found"); return

    # 1) loss curves
    plt.figure(figsize=(8, 5))
    for name, h in hists.items():
        th = h["transformer_history"]
        ep = [r["epoch"] for r in th]
        plt.plot(ep, [r["train_loss"] for r in th], "--", label=f"{name} train")
        plt.plot(ep, [r["val_loss"] for r in th], "-", label=f"{name} val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Transformer loss curves")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(OUT / "loss_curves.png", dpi=120); plt.close()

    # 2) val accuracy curves
    plt.figure(figsize=(8, 5))
    for name, h in hists.items():
        th = h["transformer_history"]
        plt.plot([r["epoch"] for r in th], [r["val_accuracy"] for r in th], "-o", label=name)
    plt.xlabel("epoch"); plt.ylabel("val next-step accuracy")
    plt.title("Validation accuracy by training method")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(OUT / "val_accuracy.png", dpi=120); plt.close()

    # 3) method comparison (best val acc + RF top-15)
    names = list(hists)
    best_acc = [max(r["val_accuracy"] for r in hists[n]["transformer_history"]) for n in names]
    rf = [hists[n].get("rf_metrics", {}).get("top_15_accuracy", 0) for n in names]
    x = range(len(names))
    plt.figure(figsize=(8, 5))
    plt.bar([i - 0.2 for i in x], best_acc, 0.4, label="transformer best val acc")
    plt.bar([i + 0.2 for i in x], rf, 0.4, label="RF top-15 acc")
    plt.xticks(list(x), names, rotation=15, fontsize=8)
    plt.ylabel("accuracy"); plt.title("Training-method comparison"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "method_comparison.png", dpi=120); plt.close()

    print(f"wrote: {', '.join(p.name for p in OUT.glob('*.png'))} -> {OUT}/")


if __name__ == "__main__":
    main()
