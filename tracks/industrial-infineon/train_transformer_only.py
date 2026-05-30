#!/usr/bin/env python3
"""
train_transformer_only.py — train ONLY the transformer (with the aux category
head) on already-generated data. Skips the Random Forest, which is irrelevant to
the category-understanding measurement and is the heavy/slow part. Lean + light
on the machine.

Usage: OUTPUT_DIR=outputs_M3 python train_transformer_only.py --model-size tiny --epochs 12 --aux-category --unk-dropout 0.15
"""
import argparse, json, os, sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from data_pipeline import ProcessSequenceDataset
from transformer_model import create_model
import train as T  # reuse load_pregenerated_data, build_id2cat, train_transformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-size", default="tiny")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--unk-dropout", type=float, default=0.0)
    ap.add_argument("--aux-category", action="store_true")
    ap.add_argument("--cat-weight", type=float, default=0.3)
    args = ap.parse_args()

    out = Path(os.environ.get("OUTPUT_DIR", "outputs_M3"))
    torch.manual_seed(args.seed)
    (out / "model_config.json").write_text(json.dumps({"model_size": args.model_size}))
    device = "cpu"
    print(f"device={device}  out={out}", flush=True)

    family_seqs, tokenizer = T.load_pregenerated_data(out)
    pairs = [(fam, seq) for fam, seqs in family_seqs.items() for seq in seqs]

    dataset = ProcessSequenceDataset(pairs, tokenizer, max_len=200)
    print(f"Dataset: {len(dataset)} sequences", flush=True)
    n_val = max(1, int(0.1 * len(dataset)))
    train_ds, val_ds = random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    id2cat_t, ncat = None, 0
    if args.aux_category:
        id2cat_list, ncat = T.build_id2cat(tokenizer)
        id2cat_t = torch.tensor(id2cat_list, dtype=torch.long, device=device)
        print(f"Aux category head ON: {ncat} categories, weight={args.cat_weight}", flush=True)

    model = create_model(tokenizer.vocab_size, size=args.model_size, n_categories=ncat)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model {args.model_size}: {n_params:,} params", flush=True)

    history = T.train_transformer(
        model, train_loader, val_loader, epochs=args.epochs, lr=args.lr,
        device=device, save_dir=out, unk_dropout=args.unk_dropout,
        id2cat=id2cat_t, cat_weight=args.cat_weight)

    (out / "training_history.json").write_text(json.dumps(
        {"config": {"model_size": args.model_size, "n_params": n_params,
                    "vocab_size": tokenizer.vocab_size, "epochs": args.epochs,
                    "aux_category": args.aux_category, "n_categories": ncat},
         "transformer_history": history}, indent=2, default=str))
    print(f"\nBest val acc: {max(h['val_accuracy'] for h in history):.4f}", flush=True)
    print("TRANSFORMER DONE", flush=True)


if __name__ == "__main__":
    main()
