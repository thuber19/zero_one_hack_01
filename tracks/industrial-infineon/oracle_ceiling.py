#!/usr/bin/env python3
"""
oracle_ceiling.py — measure the Bayes (information-theoretic) ceiling of the
next-step prediction task, MODEL-INDEPENDENTLY, straight from the data.

Why this exists: across many runs (different sizes, epochs, configs) next-token
validation accuracy converges around ~0.81 and refuses to climb. This script
proves *why*: the synthetic generator (generation_rules.md §1 synonyms + §4
variation axes) makes RANDOM valid choices at ~half of all positions, so no model
— of any size, trained for any number of epochs — can predict those better than
the generator's own coin-flip. The "oracle" below is the best predictor that can
possibly exist (it sees the full context distribution); its accuracy is the hard
ceiling.

It reports, for several context orders k (condition on family + last k steps):
  * ORACLE Top-1 / Top-5 (predict the most/5-most frequent next step)
  * average next-step entropy (bits) — the irreducible uncertainty
  * the deterministic vs ambiguous context split
  * the top ambiguous branch points (the named synonym/optional-step coin-flips)
plus an exact-duplicate-sequence check (leakage sanity).

Usage:  python oracle_ceiling.py [--data-dir data]
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "data"))
sys.path.insert(0, str(_ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from generate_sequences import read_csv_sequences


def load_sequences(data_dir: Path):
    files = [("mosfet", "MOSFET_variants.csv"), ("igbt", "IGBT_variants.csv"),
             ("ic", "IC_variants.csv")]
    seqs = []
    for fam, fn in files:
        p = data_dir / fn
        if p.exists():
            for s in read_csv_sequences(p).values():
                seqs.append((fam, s))
    return seqs


def oracle_at(seqs, k: int):
    """Return (top1, top5, avg_entropy_bits) for a last-k context oracle."""
    ctx = defaultdict(Counter)
    for f, s in seqs:
        for i in range(len(s)):
            ctx[(f,) + tuple(s[max(0, i - k):i])][s[i]] += 1
    tot = top1 = top5 = 0
    ent = 0.0
    for f, s in seqs:
        for i in range(len(s)):
            dist = ctx[(f,) + tuple(s[max(0, i - k):i])]
            n = sum(dist.values())
            best = dist.most_common(5)
            tot += 1
            if s[i] == best[0][0]:
                top1 += 1
            if s[i] in [b for b, _ in best]:
                top5 += 1
            ent += -math.log2(dist[s[i]] / n)
    return top1 / tot, top5 / tot, ent / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--context", type=int, default=3, help="k for the branch-point report")
    args = ap.parse_args()
    seqs = load_sequences(_ROOT / args.data_dir if not Path(args.data_dir).is_absolute()
                          else Path(args.data_dir))
    if not seqs:
        print(f"no *_variants.csv found in {args.data_dir}", file=sys.stderr); sys.exit(2)

    total_steps = sum(len(s) for _, s in seqs)
    keys = [tuple([f] + s) for f, s in seqs]
    dups = len(keys) - len(set(keys))
    print(f"sequences: {len(seqs)} | total steps: {total_steps}")
    print(f"exact-duplicate sequences: {dups} ({100*dups/len(keys):.2f}%) "
          f"-> a random split would leak this fraction\n")

    print(f"{'context':>10} {'ORACLE Top-1':>13} {'Top-5':>8} {'avg entropy (bits)':>20}")
    last = None
    for k in (1, 2, 3, 5, 8):
        t1, t5, h = oracle_at(seqs, k)
        print(f"  last-{k:<5} {t1:>11.3f} {t5:>8.3f} {h:>18.3f}")
        last = (t1, t5, h)

    # branch-point report at --context
    k = args.context
    ctx = defaultdict(Counter)
    for f, s in seqs:
        for i in range(len(s)):
            ctx[(f,) + tuple(s[max(0, i - k):i])][s[i]] += 1
    det = sum(1 for c in ctx.values() if len(c) == 1)
    amb = sorted(((sum(c.values()), kk, c) for kk, c in ctx.items() if len(c) > 1), reverse=True)
    print(f"\nlast-{k} contexts: {len(ctx)} | deterministic: {det} "
          f"({100*det/len(ctx):.0f}%) | ambiguous: {len(ctx)-det}")
    print("Top ambiguous branch points (the irreducible choices — synonyms / optional steps):")
    for n, kk, c in amb[:10]:
        nx = ", ".join(f"{a}:{b}" for a, b in c.most_common(4))
        print(f"  after {list(kk[1:])!s:50.50} -> {{{nx}}}")

    t1, t5, h = last
    print("\n=== CEILING VERDICT ===")
    print(f"  Bayes-optimal next-step Top-1 ≈ {t1:.3f}  (Top-5 ≈ {t5:.3f}).")
    print("  A model converging near this Top-1 with Top-5≈1.0 is AT the ceiling:")
    print("  the residual is the generator's random valid choices, not a model flaw.")
    print("  => Optimise the non-saturated metrics (T2 completion, T4 OOD) + the")
    print("     verified physics layer, NOT next-token Top-1.")


if __name__ == "__main__":
    main()
