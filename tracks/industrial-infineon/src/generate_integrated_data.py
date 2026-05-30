"""
generate_integrated_data.py — training corpus for the SECOND (integrated) run.

The first run trained only on the three provided families. This produces a
corpus that also includes physics-verified **pseudo-families** (novel
vocabulary, valid by the engine) so the transformer must learn category-level
regularities rather than memorising the three known vocabularies — the lever for
the unknown 4th family (Task 4).

Output (drop-in for src/train.py): sequences.json, tokenizer.txt, and (if torch
is available) transitions.json for the RF. The stdlib core runs on a login node
with nothing installed; transitions.json (RF) is built only when torch is
present.

Usage:
    python src/generate_integrated_data.py --extra-data 5000 --ood 1500 --output-dir outputs_integrated
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import zlib
from pathlib import Path

_SRC = Path(__file__).resolve().parent
_SUBROOT = _SRC.parent
for _p in (str(_SRC), str(_SUBROOT), str(_SUBROOT / "data")):
    sys.path.insert(0, _p)

from generate_sequences import generate_dataset, validate_sequence
from tokenizer import StepTokenizer
import pseudo_family as PF


def main():
    ap = argparse.ArgumentParser(description="Integrated (real + pseudo-family) data")
    ap.add_argument("--extra-data", type=int, default=5000,
                    help="real sequences per known family")
    ap.add_argument("--ood", type=int, default=1500,
                    help="pseudo-family (novel-vocabulary) sequences total")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs_integrated"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    families = ["mosfet", "igbt", "ic"]
    family_seqs: dict[str, list[list[str]]] = {}

    # 1) real validated sequences per family
    for fam in families:
        # zlib.crc32 is stable across processes (built-in hash() is salted per
        # run, which would silently break --seed reproducibility).
        seqs = generate_dataset(fam, args.extra_data,
                                seed=args.seed + zlib.crc32(fam.encode()) % 9999,
                                validate=True)
        family_seqs[fam] = seqs
        print(f"  {fam.upper()}: {len(seqs)} real sequences")

    # 2) physics-verified pseudo-families (novel vocab, structural variation).
    #    Assigned round-robin to the real family buckets so the existing family
    #    tokens still apply — the model just learns that novel tokens can appear,
    #    forcing category-level generalisation.
    # Bound vocab growth: use only a few novel "families" so the synthetic token
    # count stays modest (each tag adds ~120 novel tokens). The complementary,
    # cleaner OOD lever is UNK-dropout during training (randomly mask known step
    # tokens with [UNK]) — see jobs/train_integrated.sh and REPORT.
    # Use only a few novel "families" (each tag adds ~120 novel tokens) WITHOUT
    # permanently mutating the module global (which would leak into any later
    # import of pseudo_family in the same process).
    _saved_tags = PF.TAGS
    PF.TAGS = PF.TAGS[:3]
    try:
        pseudo = PF.generate_pseudo_valid(args.ood, rng, structural=True)
    finally:
        PF.TAGS = _saved_tags
    for i, (_tag, seq) in enumerate(pseudo):
        family_seqs[families[i % 3]].append(seq)
    print(f"  + {len(pseudo)} pseudo-family (OOD) sequences distributed across families")

    # 3) tokenizer over EVERYTHING (absorbs novel vocab)
    all_seqs = [s for seqs in family_seqs.values() for s in seqs]
    tokenizer = StepTokenizer.from_sequences(all_seqs)
    tokenizer.save(args.output_dir / "tokenizer.txt")

    with open(args.output_dir / "sequences.json", "w") as f:
        json.dump(family_seqs, f)

    # 4) transition map for the RF — only if torch/data_pipeline import succeeds
    try:
        from data_pipeline import build_transition_map
        pairs = [(fam, s) for fam, seqs in family_seqs.items() for s in seqs]
        transitions = build_transition_map(pairs)
        with open(args.output_dir / "transitions.json", "w") as f:
            json.dump({k: list(v) for k, v in transitions.items()}, f)
        print("  transitions.json built (RF enabled)")
    except Exception as e:
        print(f"  [note] transitions.json skipped ({type(e).__name__}); "
              "RF will be disabled — inference falls back to transformer+physics.")

    print(f"\nSaved to {args.output_dir}:  tokenizer ({tokenizer.vocab_size} tokens), "
          f"{len(all_seqs)} sequences")
    # NOTE: model_config.json is intentionally NOT written here — train.py writes
    # it with the ACTUAL --model-size chosen for the run, so inference always
    # loads the architecture that was actually trained (writing a hardcoded
    # "small" here previously could mislead inference if the dir was reused).


if __name__ == "__main__":
    main()
