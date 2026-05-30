"""
Data generation: creates train/eval split in standard CSV formats.

Outputs:
  train_sequences.csv           — SEQUENCE_ID, FAMILY, STEP (long format, for training)
  eval_input_valid.csv          — EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE
  eval_input_anomaly.csv        — EXAMPLE_ID, FAMILY, SEQUENCE
  eval_truth_nextstep.csv       — EXAMPLE_ID, RANK_1 (ground truth next step)
  eval_truth_completion.csv     — EXAMPLE_ID, PREDICTED_SEQUENCE (ground truth remaining)
  eval_truth_anomaly.csv        — EXAMPLE_ID, IS_VALID, SCORE, PREDICTED_RULE
  tokenizer.txt

Usage:
    python src/generate_data.py --extra-data 10000 --output-dir outputs
"""

import argparse
import csv
import random
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SRC_DIR.parent / "data"
for _p in (str(_SRC_DIR), str(_DATA_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from generate_sequences import generate_dataset, validate_sequence
from data_pipeline import load_existing_sequences
from tokenizer import StepTokenizer


def inject_violation(seq: list[str], rng: random.Random) -> tuple[list[str] | None, str]:
    """Inject a single rule violation into a sequence."""
    from generate_sequences import DEPOSITION_STEPS, CLEAN_STEPS, ETCH_STEPS

    mutated = list(seq)

    for i, step in enumerate(mutated):
        if step in DEPOSITION_STEPS and i > 0:
            for j in range(i - 1, max(0, i - 12) - 1, -1):
                if mutated[j] in CLEAN_STEPS:
                    removed = mutated.pop(j)
                    violations = validate_sequence(mutated)
                    if violations:
                        return mutated, violations[0].rule
                    mutated.insert(j, removed)
                    break

    for i, step in enumerate(mutated):
        if step in ETCH_STEPS:
            for j in range(i - 1, max(0, i - 12) - 1, -1):
                if mutated[j] == "DEVELOP PHOTORESIST":
                    removed = mutated.pop(j)
                    violations = validate_sequence(mutated)
                    if violations:
                        return mutated, violations[0].rule
                    mutated.insert(j, removed)
                    break

    try:
        ship_idx = mutated.index("SHIP LOT")
        sort_idx = mutated.index("WAFER SORT TEST")
        if ship_idx > sort_idx:
            mutated[ship_idx], mutated[sort_idx] = mutated[sort_idx], mutated[ship_idx]
            violations = validate_sequence(mutated)
            if violations:
                return mutated, violations[0].rule
            mutated[ship_idx], mutated[sort_idx] = mutated[sort_idx], mutated[ship_idx]
    except ValueError:
        pass

    return None, ""


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate train/eval data")
    parser.add_argument("--extra-data", type=int, default=10000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-split", type=float, default=0.1)
    parser.add_argument("--ood", type=int, default=0,
                        help="pseudo-family OOD sequences added to TRAINING (Mina's injection)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # ── Generate all sequences ──
    all_seqs: dict[str, list[list[str]]] = load_existing_sequences()
    for family in ["mosfet", "igbt", "ic"]:
        existing = len(all_seqs.get(family, []))
        if existing < args.extra_data:
            extra = generate_dataset(family, args.extra_data - existing, seed=args.seed, validate=True)
            all_seqs.setdefault(family, []).extend(extra)

    # ── Mina's injection: pseudo-family OOD sequences (novelty spectrum), added to
    #    TRAINING ONLY. Their novel vocabulary enters the tokenizer; the held-out
    #    eval below stays the 3 known families, so the benchmark remains clean. ──
    ood_seqs: list = []
    if getattr(args, "ood", 0) and args.ood > 0:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import pseudo_family as PF
        ood_seqs = PF.generate_pseudo_valid(args.ood, rng, structural=True)  # [(tag, steps)]
        print(f"  + {len(ood_seqs)} pseudo-family OOD sequences (TRAINING only)")

    # ── Tokenizer from ALL data (incl. OOD novel vocab) ──
    all_flat = []
    for seqs in all_seqs.values():
        all_flat.extend(seqs)
    all_flat.extend(s for _tag, s in ood_seqs)
    tokenizer = StepTokenizer.from_sequences(all_flat)
    tokenizer.save(args.output_dir / "tokenizer.txt")
    print(f"Tokenizer: {tokenizer.vocab_size} tokens")

    # ── Split train/eval ──
    train_seqs: dict[str, list[list[str]]] = {}
    eval_seqs: dict[str, list[list[str]]] = {}

    for family, seqs in all_seqs.items():
        shuffled = list(seqs)
        rng.shuffle(shuffled)
        n_eval = max(1, int(len(shuffled) * args.eval_split))
        eval_seqs[family] = shuffled[:n_eval]
        train_seqs[family] = shuffled[n_eval:]
        print(f"  {family.upper()}: {len(train_seqs[family])} train, {len(eval_seqs[family])} eval")

    # OOD pseudo-families go to TRAINING ONLY (label = their novel tag; the
    # tokenizer has no family token for it -> encodes as [UNK] family, i.e. the
    # exact Task-4 condition the model must learn to handle).
    for tag, s in ood_seqs:
        train_seqs.setdefault(tag.upper(), []).append(s)

    # ── train_sequences.csv ──
    with open(args.output_dir / "train_sequences.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["SEQUENCE_ID", "FAMILY", "STEP"])
        seq_id = 0
        for family, seqs in train_seqs.items():
            for seq in seqs:
                seq_id += 1
                for step in seq:
                    w.writerow([f"seq_{seq_id:06d}", family.upper(), step])
    print(f"\n  train_sequences.csv: {sum(len(s) for s in train_seqs.values())} sequences")

    # ── eval_input_valid.csv + truth CSVs ──
    eval_valid_rows = []
    truth_valid_rows = []
    example_id = 0

    for family, seqs in eval_seqs.items():
        for seq in seqs:
            for frac in [0.6, 0.8]:
                cut = int(len(seq) * frac)
                if cut == 0 or cut >= len(seq):
                    continue
                example_id += 1
                eid = f"valid_{example_id:04d}"

                eval_valid_rows.append({
                    "EXAMPLE_ID": eid,
                    "FAMILY": family.upper(),
                    "COMPLETION_FRACTION": frac,
                    "PARTIAL_SEQUENCE": "|".join(seq[:cut]),
                })
                # Ground truth for eval_metrics.py (combined format)
                truth_valid_rows.append({
                    "EXAMPLE_ID": eid,
                    "FAMILY": family.upper(),
                    "COMPLETION_FRACTION": frac,
                    "PARTIAL_SEQUENCE": "|".join(seq[:cut]),
                    "FULL_SEQUENCE": "|".join(seq),
                    "NEXT_STEP": seq[cut],
                })

    _write_csv(args.output_dir / "eval_input_valid.csv",
               ["EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION", "PARTIAL_SEQUENCE"],
               eval_valid_rows)
    # Ground truth in format eval_metrics.py expects
    _write_csv(args.output_dir / "eval_set_valid.csv",
               ["EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION", "PARTIAL_SEQUENCE", "FULL_SEQUENCE", "NEXT_STEP"],
               truth_valid_rows)
    print(f"  eval_input_valid.csv: {len(eval_valid_rows)} rows")

    # ── eval_input_anomaly.csv + truth CSV ──
    anomaly_input_rows = []
    truth_forbidden_rows = []  # only invalid sequences, for eval_metrics.py
    anomaly_id = 0

    # Valid sequences (no ground truth entry — eval_metrics uses valid-supplement)
    valid_anomaly_ids = []
    for family, seqs in eval_seqs.items():
        for seq in seqs:
            anomaly_id += 1
            eid = f"valid_{anomaly_id:04d}"
            anomaly_input_rows.append({
                "EXAMPLE_ID": eid, "FAMILY": family.upper(),
                "SEQUENCE": "|".join(seq),
            })
            valid_anomaly_ids.append(eid)

    # Anomalous sequences
    for family, seqs in eval_seqs.items():
        for seq in seqs[:len(seqs) // 2]:
            mutated, rule = inject_violation(seq, rng)
            if mutated is not None:
                anomaly_id += 1
                eid = f"forbidden_{anomaly_id:04d}"
                anomaly_input_rows.append({
                    "EXAMPLE_ID": eid, "FAMILY": family.upper(),
                    "SEQUENCE": "|".join(mutated),
                })
                truth_forbidden_rows.append({
                    "EXAMPLE_ID": eid, "VIOLATION_RULE": rule,
                })

    # Shuffle input (but truth stays aligned by EXAMPLE_ID)
    rng.shuffle(anomaly_input_rows)

    _write_csv(args.output_dir / "eval_input_anomaly.csv",
               ["EXAMPLE_ID", "FAMILY", "SEQUENCE"],
               anomaly_input_rows)
    # Ground truth for eval_metrics.py: only forbidden sequences with VIOLATION_RULE
    _write_csv(args.output_dir / "eval_set_forbidden.csv",
               ["EXAMPLE_ID", "VIOLATION_RULE"],
               truth_forbidden_rows)
    print(f"  eval_input_anomaly.csv: {len(anomaly_input_rows)} rows ({len(truth_forbidden_rows)} forbidden)")

    # Also save the valid eval input separately for --valid-supplement
    valid_supplement = [r for r in anomaly_input_rows if r["EXAMPLE_ID"].startswith("valid_")]
    _write_csv(args.output_dir / "eval_set_valid_supplement.csv",
               ["EXAMPLE_ID", "FAMILY", "SEQUENCE"],
               valid_supplement)

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
