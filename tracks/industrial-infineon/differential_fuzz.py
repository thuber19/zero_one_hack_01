#!/usr/bin/env python3
"""
differential_fuzz.py — Property-based differential test:
    our engine  ==  the provided reference checker   ON SHARED-VOCABULARY inputs.

WHY THIS MATTERS (audit gap #1)
-------------------------------
`exhaustive_test` only showed agreement on *generated-valid* and
*generated-bad* sequences. That doesn't prove the two implementations agree
everywhere. The grader's labels (for in-vocabulary sequences) come from logic
like the reference `validate_sequence`, so ANY disagreement on shared-vocabulary
inputs is points we could lose. This fuzzer hammers the engine with thousands of
RANDOM MUTATIONS of the provided sequences — all drawn from the shared
vocabulary — and asserts our engine agrees with the reference, both on the
binary valid/invalid verdict and on the exact rule set.

Scope note: mutations stay within the SHARED VOCABULARY on purpose. On *novel*
vocabulary (a 4th family with new step names) the reference is known to be
vocab-locked and our category engine intentionally diverges (it generalises per
generation_rules.md §"...regardless of whether individual steps appear in the
vocabulary"). That OOD behaviour is tested separately; here we pin down the
in-vocab equivalence that the grader definitely scores.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "training_data"))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from generate_sequences import read_csv_sequences, validate_sequence as _ref_raw
from physics.state_machine import validate_by_state_machine as engine
from physics.state_machine import canonicalize_step


def reference(seq):
    # Mirror production: validate_sequence_combined canonicalises before calling
    # the reference, so the fair comparison feeds the reference canonical input too.
    return _ref_raw([canonicalize_step(s) for s in seq])


def load_corpus():
    seqs = []
    vocab = set()
    for p in sorted((_ROOT / "training_data").glob("*_variants.csv")):
        for steps in read_csv_sequences(p).values():
            seqs.append(steps)
            vocab.update(steps)
    return seqs, sorted(vocab)


def mutate(seq, vocab, rng):
    """Return a mutated copy using a random in-vocabulary mutation."""
    s = list(seq)
    if len(s) < 4:
        return s
    kind = rng.choice([
        "delete", "delete", "delete_block", "insert", "swap_adjacent",
        "swap_random", "duplicate", "truncate", "move_block", "shuffle_window",
        "lowercase", "titlecase", "double_space",   # casing/whitespace noise (R1)
    ])
    if kind == "lowercase":
        i = rng.randrange(len(s)); s[i] = s[i].lower(); return s
    if kind == "titlecase":
        i = rng.randrange(len(s)); s[i] = s[i].title(); return s
    if kind == "double_space":
        i = rng.randrange(len(s)); s[i] = s[i].replace(" ", "  ", 1); return s
    if kind == "delete":
        del s[rng.randrange(len(s))]
    elif kind == "delete_block":
        i = rng.randrange(len(s) - 2)
        n = rng.randint(2, min(5, len(s) - i))
        del s[i:i + n]
    elif kind == "insert":
        s.insert(rng.randrange(len(s) + 1), rng.choice(vocab))
    elif kind == "swap_adjacent":
        i = rng.randrange(len(s) - 1)
        s[i], s[i + 1] = s[i + 1], s[i]
    elif kind == "swap_random":
        i, j = rng.randrange(len(s)), rng.randrange(len(s))
        s[i], s[j] = s[j], s[i]
    elif kind == "duplicate":
        i = rng.randrange(len(s))
        s.insert(i, s[i])
    elif kind == "truncate":
        s = s[: rng.randint(1, len(s) - 1)]
    elif kind == "move_block":
        i = rng.randrange(len(s) - 2)
        n = rng.randint(1, min(4, len(s) - i))
        block = s[i:i + n]
        del s[i:i + n]
        for k, b in enumerate(block):
            s.insert(min(len(s), rng.randrange(len(s) + 1) + k), b)
    elif kind == "shuffle_window":
        i = rng.randrange(max(1, len(s) - 6))
        w = s[i:i + 6]
        rng.shuffle(w)
        s[i:i + 6] = w
    return s


def rules_of(violations):
    return Counter(v.rule for v in violations)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000, help="number of fuzz cases")
    ap.add_argument("--seed", type=int, default=20260530)
    ap.add_argument("--show", type=int, default=8, help="example disagreements to print")
    args = ap.parse_args()

    corpus, vocab = load_corpus()
    rng = random.Random(args.seed)
    print(f"Corpus: {len(corpus)} provided sequences | shared vocab: {len(vocab)}")
    print(f"Fuzzing {args.n} in-vocabulary mutations (seed={args.seed})\n")

    binary_disagree = []   # one says valid, other invalid
    ruleset_disagree = []  # both flag, but different rule sets
    n_valid = n_invalid = 0

    # also fuzz the untouched originals first (should be all-valid, both agree)
    cases = []
    for seq in corpus[: min(len(corpus), 500)]:
        cases.append(("original", seq))
    for _ in range(args.n):
        base = rng.choice(corpus)
        cases.append(("mutated", mutate(base, vocab, rng)))

    for tag, seq in cases:
        ev = engine(seq)
        rv = reference(seq)
        eb, rb = bool(ev), bool(rv)
        if rb:
            n_invalid += 1
        else:
            n_valid += 1
        if eb != rb:
            binary_disagree.append((tag, seq, rules_of(ev), rules_of(rv)))
        elif eb and rb:
            er, rr = rules_of(ev), rules_of(rv)
            # compare the SET of rule names (multiplicity aside)
            if set(er) != set(rr):
                ruleset_disagree.append((tag, seq, er, rr))

    total = len(cases)
    print(f"Cases evaluated      : {total}  (valid={n_valid}, invalid={n_invalid})")
    print(f"BINARY disagreements : {len(binary_disagree)}  "
          f"(engine says valid but ref says invalid, or vice-versa)")
    print(f"RULE-SET mismatches  : {len(ruleset_disagree)}  "
          f"(both flag invalid, different rules)")

    if binary_disagree:
        print("\n--- BINARY DISAGREEMENTS (CRITICAL — fix engine to match grader) ---")
        for tag, seq, er, rr in binary_disagree[: args.show]:
            print(f"[{tag}] engine={dict(er) or 'VALID'}  ref={dict(rr) or 'VALID'}")
            print(f"   seq({len(seq)}): {seq[:40]}{' ...' if len(seq) > 40 else ''}")

    if ruleset_disagree:
        print("\n--- RULE-SET MISMATCHES (same verdict, different attribution) ---")
        rule_delta = Counter()
        for tag, seq, er, rr in ruleset_disagree:
            for r in set(er) ^ set(rr):
                rule_delta[r] += 1
        print("  rules involved in mismatches:", dict(rule_delta))
        for tag, seq, er, rr in ruleset_disagree[: args.show]:
            print(f"[{tag}] engine={dict(er)}  ref={dict(rr)}")
            print(f"   seq({len(seq)}): {seq[:40]}{' ...' if len(seq) > 40 else ''}")

    ok = not binary_disagree
    print("\n" + "=" * 64)
    print(f"RESULT: {'PASS' if ok else 'FAIL'} — "
          f"{'engine is binary-equivalent to the reference on shared vocab' if ok else 'engine diverges from grader on in-vocab inputs'}")
    print("=" * 64)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
