"""
Training script: trains random forest + sequence model (transformer or LSTM).

Usage:
    python train.py --arch transformer --model-size small --epochs 50 --batch-size 64
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SRC_DIR.parent
_DATA_DIR = _PROJECT_DIR / "data"
for _p in (str(_SRC_DIR), str(_DATA_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_pipeline import load_train_csv, ProcessSequenceDataset
from tokenizer import StepTokenizer
from transformer_model import create_model as create_transformer
from lstm_model import create_lstm_model
from random_forest import StepCandidateForest


OUTPUT_DIR = Path(os.environ.get(
    "OUTPUT_DIR",
    str(_PROJECT_DIR / "outputs"),
))


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 3e-4,
    device: str = "cpu",
    save_dir: Path = OUTPUT_DIR,
    patience: int = 20,
) -> list[dict]:
    """Train any sequence model and return loss history."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            loss = model.compute_loss(input_ids, target_ids, attn_mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / n_batches
        elapsed = time.time() - t0

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                target_ids = batch["target_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)

                loss = model.compute_loss(input_ids, target_ids, attn_mask)
                val_loss += loss.item()
                n_val += 1

                logits = model(input_ids, attn_mask)
                preds = logits.argmax(dim=-1)
                mask = attn_mask[:, :preds.shape[1]].bool()
                target_trimmed = target_ids[:, :preds.shape[1]]
                val_correct += ((preds == target_trimmed) & mask).sum().item()
                val_total += mask.sum().item()

        val_loss /= n_val
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "lr": scheduler.get_last_lr()[0],
            "time_s": elapsed,
        }
        history.append(record)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f} | "
                f"time={elapsed:.1f}s"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    torch.save(model.state_dict(), save_dir / "final_model.pt")
    return history


def main():
    parser = argparse.ArgumentParser(description="Train process sequence models")
    parser.add_argument("--arch", choices=["transformer", "lstm"], default="transformer")
    parser.add_argument("--model-size", choices=["tiny", "small", "medium"], default="small")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-rf", action="store_true", help="Skip random forest training")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # ── Step 1: Load training data ──
    print("\n=== Step 1: Loading training data ===")
    train_csv = OUTPUT_DIR / "train_sequences.csv"
    tokenizer = StepTokenizer.load(OUTPUT_DIR / "tokenizer.txt")
    all_pairs = load_train_csv(train_csv)
    print(f"  Tokenizer: {tokenizer.vocab_size} tokens")
    print(f"  Total: {len(all_pairs)} sequences")

    # ── Step 2: Train Random Forest (optional) ──
    rf_metrics = {}
    if not args.no_rf:
        print(f"\n=== Step 2: Training Random Forest ===")
        rf = StepCandidateForest(n_estimators=50, max_depth=15, top_k=15)
        rf_metrics = rf.train(all_pairs, tokenizer)
        rf.save(OUTPUT_DIR / "random_forest.pkl")
        del rf
    else:
        print("\n=== Step 2: Random Forest SKIPPED ===")
    gc.collect()

    # ── Step 3: Train model ──
    print(f"\n=== Step 3: Training {args.arch.upper()} ===")
    dataset = ProcessSequenceDataset(all_pairs, tokenizer, max_len=200)
    print(f"Dataset size: {len(dataset)} sequences")

    del all_pairs
    gc.collect()

    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.arch == "transformer":
        model = create_transformer(tokenizer.vocab_size, size=args.model_size)
    else:
        model = create_lstm_model(tokenizer.vocab_size, size=args.model_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.arch} {args.model_size} ({n_params:,} parameters)")

    history = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, device=device,
        save_dir=OUTPUT_DIR,
    )

    # ── Save config ──
    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump({
            "config": {
                "arch": args.arch,
                "model_size": args.model_size,
                "n_params": n_params,
                "vocab_size": tokenizer.vocab_size,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "device": device,
                "total_sequences": len(dataset),
                "rf_enabled": not args.no_rf,
            },
            "rf_metrics": rf_metrics,
            "history": history,
        }, f, indent=2, default=str)

    print(f"\n=== Training complete ===")
    print(f"  Best val loss: {min(h['val_loss'] for h in history):.4f}")
    print(f"  Best val acc:  {max(h['val_accuracy'] for h in history):.4f}")
    if rf_metrics:
        print(f"  RF top-15 acc: {rf_metrics.get('top_15_accuracy', 'N/A')}")
    print(f"  Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
