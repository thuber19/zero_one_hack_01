"""
Training script: trains both the random forest and transformer.

Usage:
    python train.py --model-size small --epochs 50 --batch-size 64
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from data_pipeline import ProcessSequenceDataset
from tokenizer import StepTokenizer
from transformer_model import create_model, ProcessTransformer
from random_forest import StepCandidateForest


OUTPUT_DIR = Path(os.environ.get(
    "OUTPUT_DIR",
    str(Path(__file__).resolve().parent.parent / "outputs"),
))


def load_pregenerated_data(output_dir: Path) -> tuple[dict[str, list[list[str]]], StepTokenizer]:
    """Load data from generate_data.py output (sequences.json + tokenizer.txt)."""
    seq_path = output_dir / "sequences.json"
    tok_path = output_dir / "tokenizer.txt"

    if not seq_path.exists() or not tok_path.exists():
        raise FileNotFoundError(
            f"Pre-generated data not found in {output_dir}. "
            "Run generate_data.py first."
        )

    print(f"Loading pre-generated data from {output_dir}")
    with open(seq_path) as f:
        family_seqs = json.load(f)

    tokenizer = StepTokenizer.load(tok_path)

    for family, seqs in family_seqs.items():
        print(f"  {family.upper()}: {len(seqs)} sequences")
    print(f"  Tokenizer: {tokenizer.vocab_size} tokens")

    return family_seqs, tokenizer


def train_transformer(
    model: ProcessTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 3e-4,
    device: str = "cpu",
    save_dir: Path = OUTPUT_DIR,
    patience: int = 20,
    unk_dropout: float = 0.0,
) -> list[dict]:
    """Train the transformer and return loss history.

    unk_dropout: probability of replacing a real (non-special) input token with
    [UNK] during training. Teaches the model to predict from CONTEXT when a
    token is unknown — the integration lever for the unseen 4th family (Task 4),
    with no vocabulary growth. Targets are left intact (it must still predict the
    true next step)."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            if unk_dropout > 0.0:
                # replace real step tokens (id >= 7; specials are 0..6) with UNK
                rand = torch.rand_like(input_ids, dtype=torch.float)
                drop = attn_mask.bool() & (input_ids >= 7) & (rand < unk_dropout)
                input_ids = input_ids.masked_fill(drop, 3)  # UNK_ID = 3

            loss = model.compute_loss(input_ids, target_ids, attn_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / n_batches
        elapsed = time.time() - t0

        # ── Validate ──
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

                # Accuracy
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

        # Save best model + early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "best_transformer.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Save final model
    torch.save(model.state_dict(), save_dir / "final_transformer.pt")
    return history


def main():
    parser = argparse.ArgumentParser(description="Train process sequence models")
    parser.add_argument("--model-size", choices=["tiny", "small", "medium"], default="small")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default=None,
                        help="Device (auto-detected if not set)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unk-dropout", type=float, default=0.0,
                        help="prob. of [UNK]-masking a context token (OOD lever).")
    parser.add_argument("--init-from", type=str, default=None,
                        help="checkpoint to continue/fine-tune from (same vocab).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # record the size so inference loads the matching architecture
    (OUTPUT_DIR / "model_config.json").write_text(
        json.dumps({"model_size": args.model_size}))

    # Auto-detect device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # ── Step 1: Load pre-generated data ──
    print("\n=== Step 1: Loading data ===")
    family_seqs, tokenizer = load_pregenerated_data(OUTPUT_DIR)

    # Flatten into (family, steps) pairs
    all_pairs: list[tuple[str, list[str]]] = []
    for family, seqs in family_seqs.items():
        for seq in seqs:
            all_pairs.append((family, seq))

    # ── Step 2: Train Random Forest ──
    print("\n=== Step 2: Training Random Forest ===")
    rf = StepCandidateForest(n_estimators=100, max_depth=20, top_k=15)
    rf_metrics = rf.train(all_pairs, tokenizer)
    rf.save(OUTPUT_DIR / "random_forest.pkl")

    # Free RF memory before transformer training
    del rf
    import gc
    gc.collect()

    # ── Step 3: Train Transformer ──
    print("\n=== Step 3: Training Transformer ===")
    dataset = ProcessSequenceDataset(all_pairs, tokenizer, max_len=200)
    print(f"Dataset size: {len(dataset)} sequences")

    # Free raw data — dataset has its own encoded copy
    del all_pairs, family_seqs
    gc.collect()

    # Split 90/10
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = create_model(tokenizer.vocab_size, size=args.model_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model_size} ({n_params:,} parameters)")

    if args.init_from:
        try:
            sd = torch.load(args.init_from, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"  Continued from {args.init_from} "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        except Exception as e:
            print(f"  WARNING: could not init from {args.init_from} ({e}); "
                  "training from scratch instead.")

    history = train_transformer(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, device=device,
        save_dir=OUTPUT_DIR, unk_dropout=args.unk_dropout,
    )

    # ── Save training history ──
    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump({
            "config": {
                "model_size": args.model_size,
                "n_params": n_params,
                "vocab_size": tokenizer.vocab_size,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "device": device,
                "total_sequences": len(dataset),
            },
            "rf_metrics": rf_metrics,
            "transformer_history": history,
        }, f, indent=2, default=str)

    print(f"\n=== Training complete ===")
    print(f"  Best val loss: {min(h['val_loss'] for h in history):.4f}")
    print(f"  Best val acc:  {max(h['val_accuracy'] for h in history):.4f}")
    print(f"  RF top-15 acc: {rf_metrics['top_15_accuracy']:.4f}")
    print(f"  All outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
