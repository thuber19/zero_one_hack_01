"""
Standalone data generation script. Run on login node (no GPU needed).
Generates training data and saves tokenizer.

Usage:
    python src/generate_data.py --extra-data 10000 --output-dir outputs
"""

import argparse
import json
from pathlib import Path

from data_pipeline import prepare_all_data, build_transition_map


def main():
    parser = argparse.ArgumentParser(description="Generate training data")
    parser.add_argument("--extra-data", type=int, default=10000,
                        help="Sequences per family to generate")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate data + build tokenizer
    family_seqs, tokenizer = prepare_all_data(
        extra_per_family=args.extra_data, seed=args.seed
    )

    # Save tokenizer
    tokenizer.save(args.output_dir / "tokenizer.txt")

    # Save sequences as JSON for fast loading during training
    data = {}
    for family, seqs in family_seqs.items():
        data[family] = seqs
        print(f"  {family.upper()}: {len(seqs)} sequences")

    with open(args.output_dir / "sequences.json", "w") as f:
        json.dump(data, f)

    # Save transition map for RF
    all_pairs = []
    for family, seqs in family_seqs.items():
        for seq in seqs:
            all_pairs.append((family, seq))
    transitions = build_transition_map(all_pairs)
    # Convert sets to lists for JSON serialization
    transitions_json = {k: list(v) for k, v in transitions.items()}
    with open(args.output_dir / "transitions.json", "w") as f:
        json.dump(transitions_json, f)

    print(f"\nSaved to {args.output_dir}:")
    print(f"  tokenizer.txt ({tokenizer.vocab_size} tokens)")
    print(f"  sequences.json ({sum(len(s) for s in family_seqs.values())} sequences)")
    print(f"  transitions.json ({len(transitions)} transition keys)")


if __name__ == "__main__":
    main()
