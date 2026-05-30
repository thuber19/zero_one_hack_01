#!/usr/bin/env python3
"""
bad_data_generator.py — Known-BAD sequence generator for testing.

The repo ships a generator of VALID sequences (generate_sequences.py). For
rigorous testing we also need the opposite: a generator of INVALID sequences
covering every rule, in every variety we know how to break, all reference-
labelled — so we can measure detection and rule-attribution honestly, and prove
we don't raise false alarms on tricky-but-valid sequences.

What it produces
----------------
1. A KNOWN-BAD dataset: for each of the 10 rules, several distinct injection
   strategies, across all three families. Each example is validated with the
   *reference* checker (generate_sequences.validate_sequence) so its label is
   ground truth, not our assumption.

     TIER 1 — "surgical": the mutation trips exactly ONE rule type.
     TIER 2 — "compound": the mutation trips the target rule plus others
              (still labelled by the reference; FIRST_RULE is the reference's
              first reported rule).

2. A HARD-NEGATIVE set: sequences that *look* suspicious but are genuinely
   VALID (consecutive deposits, consecutive implants, repeated litho level,
   a clean sitting exactly at the 12-step window edge, …). These are the
   false-positive traps — a good detector must pass all of them.

Output files (written to --out-dir, default bad_data/)
------------------------------------------------------
  known_bad.csv         full record table (see columns below)
  hard_negatives.csv    valid-but-suspicious sequences
  eval_input_anomaly.csv  submit_task3-compatible input (bad + hard-negs mixed)
  ground_truth.csv        EXAMPLE_ID, IS_VALID, RULE  (for scoring)

known_bad.csv columns:
  SEQUENCE_ID, FAMILY, TIER, STRATEGY, FIRST_RULE, ALL_RULES, N_STEPS, SEQUENCE

Usage
-----
  python bad_data_generator.py                      # default: bad_data/, 5 per combo
  python bad_data_generator.py --per-combo 8 --seed 1
  python bad_data_generator.py --audit              # also cross-check our two
                                                    # detectors against the set
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

_REPO = Path(__file__).parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "data"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Benign fillers used to push a qualifying event out of its window. They are
# pure metrology (no preconditions, no effects), so inserting them only changes
# step distances — exactly what window-edge violations need.
_FILLERS = ["MEASURE THICKNESS", "MEASURE SURFACE PARTICLES", "MEASURE FILM THICKNESS"]


# When set to an rng (by build), trigger-step selection picks a RANDOM matching
# occurrence instead of the first — so across many samples every trigger step
# type and position gets exercised (comprehensive coverage). For ordering rules
# whose trigger is unique (one SHIP LOT, one CURE), random == first.
_DIVERSIFY_RNG = None


def _first(seq, pred):
    matches = [i for i, s in enumerate(seq) if pred(s)]
    if not matches:
        return None
    if _DIVERSIFY_RNG is not None:
        return _DIVERSIFY_RNG.choice(matches)
    return matches[0]


def _insert_fillers(seq, at, k, rng):
    pad = [rng.choice(_FILLERS) for _ in range(k)]
    return seq[:at] + pad + seq[at:]


# ---------------------------------------------------------------------------
# Injection strategies — each returns a mutated copy or None (not applicable).
# Correctness of the LABEL never depends on the strategy: every result is
# validated by the reference checker afterwards.
# ---------------------------------------------------------------------------

# ── RULE_SHIP_BEFORE_TEST ───────────────────────────────────────────────────

def s_ship_swap(seq, S, rng):
    s = list(seq)
    si = _first(s, lambda x: x == "SHIP LOT")
    ti = _first(s, lambda x: x == "WAFER SORT TEST")
    if si is None or ti is None or si < ti:
        return None
    s.pop(si)
    ti = _first(s, lambda x: x == "WAFER SORT TEST")
    s.insert(ti, "SHIP LOT")
    return s


def s_ship_delete_sort(seq, S, rng):
    s = [x for x in seq if x != "WAFER SORT TEST"]
    return s if any(x == "SHIP LOT" for x in s) and len(s) < len(seq) else None


# ── RULE_DEP_NO_CLEAN ───────────────────────────────────────────────────────

def s_dep_del_window_clean(seq, S, rng):
    s = list(seq)
    d = _first(s, lambda x: x in S["DEPOSITION_STEPS"])
    if d is None:
        return None
    out = [x for j, x in enumerate(s) if not (j < d and (d - j) <= 12 and x in S["CLEAN_STEPS"])]
    return out if len(out) < len(s) else None


def s_dep_push_clean(seq, S, rng):
    s = list(seq)
    d = _first(s, lambda x: x in S["DEPOSITION_STEPS"])
    if d is None:
        return None
    return _insert_fillers(s, d, 13, rng)  # push every prior clean past the 12-window


# ── RULE_ETCH_NO_MASK ───────────────────────────────────────────────────────

def s_etch_del_develop(seq, S, rng):
    s = list(seq)
    ei = _first(s, lambda x: x in S["ETCH_STEPS"] and x not in S["METAL_ETCH_STEPS"])
    if ei is None:
        return None
    for j in range(ei - 1, max(0, ei - 12) - 1, -1):
        if s[j] in ("DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"):
            s.pop(j)
            return s
    return None


def s_etch_push_develop(seq, S, rng):
    s = list(seq)
    ei = _first(s, lambda x: x in S["ETCH_STEPS"] and x not in S["METAL_ETCH_STEPS"])
    if ei is None:
        return None
    di = None
    for j in range(ei - 1, max(0, ei - 12) - 1, -1):
        if s[j] in ("DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"):
            di = j
            break
    if di is None:
        return None
    return _insert_fillers(s, di + 1, 13, rng)


# ── RULE_METAL_ETCH_NO_LITHO ────────────────────────────────────────────────

def s_metal_del_expose(seq, S, rng):
    s = list(seq)
    mi = _first(s, lambda x: x in S["METAL_ETCH_STEPS"])
    if mi is None:
        return None
    for j in range(mi - 1, max(0, mi - 15) - 1, -1):
        if s[j].startswith("EXPOSE LITHO LEVEL"):
            s.pop(j)
            return s
    return None


def s_metal_del_develop(seq, S, rng):
    s = list(seq)
    mi = _first(s, lambda x: x in S["METAL_ETCH_STEPS"])
    if mi is None:
        return None
    for j in range(mi - 1, max(0, mi - 15) - 1, -1):
        if s[j] in ("DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"):
            s.pop(j)
            return s
    return None


# ── RULE_LITHO_LEVEL_SKIP ───────────────────────────────────────────────────

def s_litho_skip(seq, S, rng):
    s = list(seq)
    for i, x in enumerate(s):
        if x == "ALIGN MASK LEVEL 2":
            s[i] = "ALIGN MASK LEVEL 4"
            return s
    return None


def s_litho_decrease(seq, S, rng):
    s = list(seq)
    for i, x in enumerate(s):
        if x == "ALIGN MASK LEVEL 3":
            s[i] = "ALIGN MASK LEVEL 1"
            return s
    return None


# ── RULE_IMPLANT_NO_MASK ────────────────────────────────────────────────────

def s_implant_del_openers(seq, S, rng):
    s = list(seq)
    ii = _first(s, lambda x: x in S["IMPLANT_STEPS"])
    if ii is None:
        return None
    out = [x for j, x in enumerate(s)
           if not (j < ii and (ii - j) <= 15 and x in S["IMPLANT_OPENER_STEPS"])]
    return out if len(out) < len(s) else None


def s_implant_push_opener(seq, S, rng):
    s = list(seq)
    ii = _first(s, lambda x: x in S["IMPLANT_STEPS"])
    if ii is None:
        return None
    oi = None
    for j in range(ii - 1, max(0, ii - 15) - 1, -1):
        if s[j] in S["IMPLANT_OPENER_STEPS"]:
            oi = j
            break
    if oi is None:
        return None
    return _insert_fillers(s, oi + 1, 16, rng)


# ── RULE_CMP_NO_DEP ─────────────────────────────────────────────────────────

def s_cmp_del_dep(seq, S, rng):
    s = list(seq)
    ci = _first(s, lambda x: x in S["CMP_STEPS"])
    if ci is None:
        return None
    out = [x for j, x in enumerate(s)
           if not (j < ci and (ci - j) <= 6 and x in S["FILL_STEPS"])]
    return out if len(out) < len(s) else None


def s_cmp_push_dep(seq, S, rng):
    s = list(seq)
    ci = _first(s, lambda x: x in S["CMP_STEPS"])
    if ci is None:
        return None
    return _insert_fillers(s, ci, 7, rng)


# ── RULE_PAD_OPEN_BEFORE_DEP ────────────────────────────────────────────────

def s_pad_del_passivation(seq, S, rng):
    s = [x for x in seq if x not in ("DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER")]
    return s if any(x in S["PAD_WINDOW_STEPS"] for x in s) and len(s) < len(seq) else None


def s_pad_move_before_pass(seq, S, rng):
    s = list(seq)
    pi = _first(s, lambda x: x in S["PAD_WINDOW_STEPS"])
    di = _first(s, lambda x: x in ("DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"))
    if pi is None or di is None or pi < di:
        return None
    step = s.pop(pi)
    di = _first(s, lambda x: x in ("DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"))
    s.insert(di, step)
    return s


# ── RULE_TEST_BEFORE_PASSIVATION ────────────────────────────────────────────

def s_test_del_cure_and_pads(seq, S, rng):
    s = [x for x in seq if x != "CURE PASSIVATION" and x not in S["PAD_WINDOW_STEPS"]]
    return s if len(s) < len(seq) else None


def s_test_move_before_cure(seq, S, rng):
    s = list(seq)
    ti = _first(s, lambda x: x in S["ELECTRICAL_TEST_STEPS"])
    ci = _first(s, lambda x: x == "CURE PASSIVATION")
    if ti is None or ci is None or ti < ci:
        return None
    step = s.pop(ti)
    ci = _first(s, lambda x: x == "CURE PASSIVATION")
    s.insert(ci, step)
    return s


# ── RULE_BACKSIDE_BEFORE_PASSIVATION ────────────────────────────────────────

def s_backside_before_cure(seq, S, rng):
    s = list(seq)
    bi = _first(s, lambda x: x == "DEPOSIT BACKSIDE METAL")
    ci = _first(s, lambda x: x == "CURE PASSIVATION")
    if bi is None or ci is None or bi < ci:
        return None
    s.pop(bi)
    ci = _first(s, lambda x: x == "CURE PASSIVATION")
    s.insert(ci, "DEPOSIT BACKSIDE METAL")
    return s


# Registry: target rule -> [(strategy_name, fn), ...]
_STRATEGIES = {
    "RULE_SHIP_BEFORE_TEST":          [("ship_swap", s_ship_swap),
                                       ("delete_sort", s_ship_delete_sort)],
    "RULE_DEP_NO_CLEAN":              [("delete_window_clean", s_dep_del_window_clean),
                                       ("push_clean_out_of_window", s_dep_push_clean)],
    "RULE_ETCH_NO_MASK":             [("delete_develop", s_etch_del_develop),
                                      ("push_develop_out_of_window", s_etch_push_develop)],
    "RULE_METAL_ETCH_NO_LITHO":      [("delete_expose", s_metal_del_expose),
                                      ("delete_develop", s_metal_del_develop)],
    "RULE_LITHO_LEVEL_SKIP":         [("skip_level", s_litho_skip),
                                      ("decrease_level", s_litho_decrease)],
    "RULE_IMPLANT_NO_MASK":          [("delete_openers", s_implant_del_openers),
                                      ("push_opener_out_of_window", s_implant_push_opener)],
    "RULE_CMP_NO_DEP":               [("delete_deposit", s_cmp_del_dep),
                                      ("push_deposit_out_of_window", s_cmp_push_dep)],
    "RULE_PAD_OPEN_BEFORE_DEP":      [("delete_passivation", s_pad_del_passivation),
                                      ("move_pad_before_passivation", s_pad_move_before_pass)],
    "RULE_TEST_BEFORE_PASSIVATION":  [("delete_cure_and_pads", s_test_del_cure_and_pads),
                                      ("move_test_before_cure", s_test_move_before_cure)],
    "RULE_BACKSIDE_BEFORE_PASSIVATION": [("move_backside_before_cure", s_backside_before_cure)],
}


# ---------------------------------------------------------------------------
# Hard negatives — valid sequences that LOOK suspicious (false-positive traps)
# ---------------------------------------------------------------------------

def _hard_negative_traps(seq, S, rng):
    """
    Return a list of (description, sequence) that are STILL valid but exercise
    the boundary cases a naive detector tends to flag. The caller verifies each
    is genuinely valid before keeping it.
    """
    from generate_sequences import DEPOSITION_STEPS, CLEAN_STEPS
    traps = []
    # 1. the unmodified valid sequence (consecutive deposits + implants already
    #    occur naturally in via-fill and IGBT cycle 3)
    traps.append(("baseline_valid", list(seq)))
    # 2. consecutive deposition: duplicate a deposition in place. The clean is
    #    still inside its window, so this MUST stay valid — it traps a detector
    #    that naively flags back-to-back depositions.
    for i, s in enumerate(seq):
        if s in DEPOSITION_STEPS:
            traps.append(("consecutive_deposit", seq[:i + 1] + [s] + seq[i + 1:]))
            break
    # 3. redundant clean immediately before a deposition (extra clean, still
    #    valid) — traps a detector that flags "too many cleans".
    for i, s in enumerate(seq):
        if s in CLEAN_STEPS:
            traps.append(("redundant_clean", seq[:i] + [s] + seq[i:]))
            break
    # The caller verifies each candidate is genuinely valid before keeping it,
    # so any mutation that accidentally becomes invalid is simply dropped.
    return traps


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build(per_combo: int = 5, seed: int = 7, target_count: int | None = None):
    """
    Build the known-bad dataset.

    per_combo    : examples per (rule x strategy x family) in the exhaustive pass.
    target_count : if set, after the exhaustive pass keep round-robin generating
                   deduped, verified bad examples until this many exist — for
                   gigantic training/testing sets.
    """
    global _DIVERSIFY_RNG
    from generate_sequences import (
        generate_sequence, validate_sequence,
        DEPOSITION_STEPS, CLEAN_STEPS, ETCH_STEPS, METAL_ETCH_STEPS,
        IMPLANT_STEPS, IMPLANT_OPENER_STEPS, CMP_STEPS, FILL_STEPS,
        PAD_WINDOW_STEPS, ELECTRICAL_TEST_STEPS, BACKSIDE_METAL_STEPS,
    )
    S = dict(
        DEPOSITION_STEPS=DEPOSITION_STEPS, CLEAN_STEPS=CLEAN_STEPS,
        ETCH_STEPS=ETCH_STEPS, METAL_ETCH_STEPS=METAL_ETCH_STEPS,
        IMPLANT_STEPS=IMPLANT_STEPS, IMPLANT_OPENER_STEPS=IMPLANT_OPENER_STEPS,
        CMP_STEPS=CMP_STEPS, FILL_STEPS=FILL_STEPS,
        PAD_WINDOW_STEPS=PAD_WINDOW_STEPS, ELECTRICAL_TEST_STEPS=ELECTRICAL_TEST_STEPS,
        BACKSIDE_METAL_STEPS=BACKSIDE_METAL_STEPS,
    )
    rng = random.Random(seed)
    _DIVERSIFY_RNG = rng    # random trigger occurrence -> every trigger covered
    families = ("mosfet", "igbt", "ic")

    def fresh_valid(fam):
        for _ in range(60):
            s = generate_sequence(fam, rng)
            if not validate_sequence(s):
                return s
        return s

    bad_records = []
    neg_records = []
    seen = set()
    counter = {"ex": 0}

    combos = [(rule, sname, fn, fam)
              for rule, strategies in _STRATEGIES.items()
              for (sname, fn) in strategies
              for fam in families]

    def make_one(rule, sname, fn, fam):
        """Try once to produce a verified, deduped bad record for this combo."""
        base = fresh_valid(fam)
        mut = fn(base, S, rng)
        if mut is None:
            return None
        viol = validate_sequence(mut)
        if not viol:
            return None
        rules = sorted({v.rule for v in viol})
        if rule not in rules:
            return None  # strategy didn't trip its intended rule
        key = tuple(mut)
        if key in seen:
            return None
        seen.add(key)
        counter["ex"] += 1
        return dict(
            sequence_id=f"bad_{counter['ex']:06d}",
            family=fam.upper(),
            tier=1 if len(rules) == 1 else 2,
            strategy=f"{rule}:{sname}",
            first_rule=viol[0].rule,
            trigger_step=viol[0].step_name,   # the exact offending step
            all_rules=";".join(rules),
            steps=mut,
        )

    # ── Exhaustive pass: every rule x strategy x family ───────────────────────
    for rule, sname, fn, fam in combos:
        made = attempts = 0
        while made < per_combo and attempts < per_combo * 25:
            attempts += 1
            rec = make_one(rule, sname, fn, fam)
            if rec is not None:
                bad_records.append(rec)
                made += 1

    # ── Scale-up: round-robin to reach target_count (gigantic datasets) ───────
    if target_count and len(bad_records) < target_count:
        idx = 0
        stalls = 0
        while len(bad_records) < target_count and stalls < len(combos) * 50:
            rec = make_one(*combos[idx % len(combos)])
            idx += 1
            if rec is not None:
                bad_records.append(rec)
                stalls = 0
            else:
                stalls += 1

    # ── Hard negatives: suspicious-but-valid ──────────────────────────────────
    for fam in families:
        for _ in range(per_combo * 3):
            base = fresh_valid(fam)
            for desc, cand in _hard_negative_traps(base, S, rng):
                if validate_sequence(cand):
                    continue  # must be valid to be a hard negative
                key = tuple(cand)
                if key in seen:
                    continue
                seen.add(key)
                counter["ex"] += 1
                neg_records.append(dict(
                    sequence_id=f"neg_{counter['ex']:06d}",
                    family=fam.upper(),
                    tier=0,
                    strategy=f"hard_negative:{desc}",
                    first_rule="",
                    trigger_step="",
                    all_rules="",
                    steps=cand,
                ))

    return bad_records, neg_records


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _write_records(path: Path, records):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["SEQUENCE_ID", "FAMILY", "TIER", "STRATEGY",
                    "FIRST_RULE", "TRIGGER_STEP", "ALL_RULES", "N_STEPS", "SEQUENCE"])
        for r in records:
            w.writerow([r["sequence_id"], r["family"], r["tier"], r["strategy"],
                        r["first_rule"], r.get("trigger_step", ""), r["all_rules"],
                        len(r["steps"]), "|".join(r["steps"])])


def _write_eval_and_truth(eval_path: Path, truth_path: Path, bad, neg, rng):
    rows = [(r, 0, r["first_rule"]) for r in bad] + [(r, 1, "") for r in neg]
    rng.shuffle(rows)
    with eval_path.open("w", newline="", encoding="utf-8") as fe, \
         truth_path.open("w", newline="", encoding="utf-8") as ft:
        we, wt = csv.writer(fe), csv.writer(ft)
        we.writerow(["EXAMPLE_ID", "FAMILY", "SEQUENCE"])
        wt.writerow(["EXAMPLE_ID", "IS_VALID", "RULE"])
        for r, is_valid, rule in rows:
            we.writerow([r["sequence_id"], r["family"], "|".join(r["steps"])])
            wt.writerow([r["sequence_id"], is_valid, rule])


# ---------------------------------------------------------------------------
# Audit: cross-check our two detectors against the labelled bad set
# ---------------------------------------------------------------------------

def _audit(bad, neg):
    from generate_sequences import validate_sequence as ref
    from physics.state_machine import validate_by_state_machine as sm

    print("\nAUDIT — detector agreement on the generated set")
    print("-" * 55)

    # Reference: should flag every bad, pass every hard negative (by construction)
    ref_bad_hit = sum(1 for r in bad if ref(r["steps"]))
    ref_neg_fp = sum(1 for r in neg if ref(r["steps"]))
    print(f"  reference checker : {ref_bad_hit}/{len(bad)} bad flagged, "
          f"{ref_neg_fp}/{len(neg)} false positives on hard-negs")

    # Our physics state machine (the OOD generaliser): how close to reference?
    sm_bad_hit = sum(1 for r in bad if sm(r["steps"]))
    sm_neg_fp = sum(1 for r in neg if sm(r["steps"]))
    # rule-attribution agreement on bad (first rule)
    attr = 0
    for r in bad:
        v = sm(r["steps"])
        if v and v[0].rule == r["first_rule"]:
            attr += 1
    print(f"  physics machine   : {sm_bad_hit}/{len(bad)} bad flagged, "
          f"{sm_neg_fp}/{len(neg)} false positives on hard-negs")
    print(f"  physics attribution match (1st rule): {attr}/{len(bad)}")
    if sm_neg_fp:
        print("  [!] physics machine raised a FALSE POSITIVE on a valid sequence.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Generate a comprehensive known-BAD dataset for testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    p.add_argument("--out-dir", default="bad_data", metavar="DIR")
    p.add_argument("--per-combo", type=int, default=5,
                   help="examples per (rule x strategy x family) (default 5).")
    p.add_argument("--count", type=int, default=None,
                   help="total bad sequences to reach (scales up via round-robin "
                        "for gigantic training/testing sets).")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--audit", action="store_true",
                   help="cross-check our detectors against the labelled set.")
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating known-bad dataset (per_combo={args.per_combo}, "
          f"count={args.count}, seed={args.seed}) …")
    bad, neg = build(per_combo=args.per_combo, seed=args.seed,
                     target_count=args.count)

    _write_records(out / "known_bad.csv", bad)
    _write_records(out / "hard_negatives.csv", neg)
    _write_eval_and_truth(out / "eval_input_anomaly.csv", out / "ground_truth.csv",
                          bad, neg, random.Random(args.seed))

    # Coverage report
    from collections import Counter
    by_rule = Counter(r["first_rule"] for r in bad)
    by_tier = Counter(r["tier"] for r in bad)
    print(f"\n  known_bad.csv        {len(bad)} invalid sequences")
    print(f"  hard_negatives.csv   {len(neg)} valid-but-suspicious sequences")
    print(f"  eval_input_anomaly.csv + ground_truth.csv  ({len(bad)+len(neg)} rows)")
    print(f"  -> {out}/")
    print("\n  coverage by first rule (distinct trigger steps exercised):")
    by_trigger = {}
    for r in bad:
        by_trigger.setdefault(r["first_rule"], set()).add(r.get("trigger_step", ""))
    for rule in sorted(by_rule):
        ntrig = len(by_trigger.get(rule, set()))
        print(f"    {rule:<34} {by_rule[rule]:>5} examples, {ntrig} distinct trigger step(s)")
    print(f"  tier 1 (single-rule): {by_tier.get(1,0)}   "
          f"tier 2 (compound): {by_tier.get(2,0)}")

    missing = set(_STRATEGIES) - set(by_rule)
    if missing:
        print(f"  [WARN] rules with no examples: {sorted(missing)}")

    if args.audit:
        _audit(bad, neg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
