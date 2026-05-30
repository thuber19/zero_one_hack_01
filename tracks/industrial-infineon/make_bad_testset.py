#!/usr/bin/env python3
"""
make_bad_testset.py — generate a FULL-SIZE known-BAD testing dataset (comparable
to the ~3000 known-good provided sequences) and prove it foolproof:

  A) every one of the 10 rules is produced at least once (full rule x strategy x
     family coverage), and
  B) every generated sequence is INDEPENDENTLY confirmed invalid by the reference
     checker AND its labelled first_rule matches the reference's first violation
     (so labels are correct, not assumed).

Then it runs the physics HARNESS (validate_sequence_combined) over the whole set
and reports detection rate + rule-attribution per rule.

Writes (testset/):  eval_input_anomaly.csv (EXAMPLE_ID, FAMILY, SEQUENCE)
                    eval_set_forbidden.csv (EXAMPLE_ID, VIOLATION_RULE)  [official GT]
Usage:  python make_bad_testset.py --count 3000 --seed 11 --out-dir bad_testset
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (str(_ROOT), str(_ROOT / "data")):
    sys.path.insert(0, _p)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import bad_data_generator as BDG
from generate_sequences import validate_sequence as REFERENCE
from physics.state_machine import validate_sequence_combined as HARNESS

TEN_RULES = {
    "RULE_DEP_NO_CLEAN", "RULE_METAL_ETCH_NO_LITHO", "RULE_ETCH_NO_MASK",
    "RULE_LITHO_LEVEL_SKIP", "RULE_IMPLANT_NO_MASK", "RULE_CMP_NO_DEP",
    "RULE_PAD_OPEN_BEFORE_DEP", "RULE_TEST_BEFORE_PASSIVATION",
    "RULE_SHIP_BEFORE_TEST", "RULE_BACKSIDE_BEFORE_PASSIVATION",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out-dir", default="bad_testset")
    args = ap.parse_args()

    print(f"Generating known-BAD testset (~{args.count}, comparable to the 3000 good)...")
    per_combo = max(5, args.count // 40)
    bad, _neg = BDG.build(per_combo=per_combo, seed=args.seed, target_count=args.count)
    print(f"  generated {len(bad)} bad sequences\n")

    # ── A) rule coverage ──
    by_rule = Counter(r["first_rule"] for r in bad)
    missing = TEN_RULES - set(by_rule)
    print("A) RULE COVERAGE (first_rule):")
    for rule in sorted(TEN_RULES):
        print(f"   {rule:34s} {by_rule.get(rule,0):5d}")
    print(f"   distinct rules covered: {len(set(by_rule) & TEN_RULES)}/10"
          + (f"   MISSING: {missing}" if missing else "   (all 10 present)"))

    # ── B) foolproof correctness: independent reference re-verification ──
    not_invalid = 0
    rule_mismatch = 0
    multi = 0
    for r in bad:
        viol = REFERENCE(r["steps"])
        if not viol:
            not_invalid += 1
        else:
            if viol[0].rule != r["first_rule"]:
                rule_mismatch += 1
            if len({v.rule for v in viol}) > 1:
                multi += 1
    print("\nB) FOOLPROOF CORRECTNESS (independent reference re-check):")
    print(f"   sequences NOT actually invalid (should be 0): {not_invalid}")
    print(f"   label != reference's first violation (should be 0): {rule_mismatch}")
    print(f"   (informational) sequences with >1 rule violated: {multi}")
    ok = (not_invalid == 0 and rule_mismatch == 0 and not missing)
    print(f"   => labels {'VERIFIED correct' if ok else 'HAVE PROBLEMS'}")

    # ── write the testing dataset (official GT format) ──
    out = _ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    in_rows, gt_rows = [], []
    for i, r in enumerate(bad, 1):
        eid = f"forbidden_{i:05d}"
        in_rows.append({"EXAMPLE_ID": eid, "FAMILY": r["family"].upper(),
                        "SEQUENCE": "|".join(r["steps"])})
        gt_rows.append({"EXAMPLE_ID": eid, "VIOLATION_RULE": r["first_rule"]})
    with open(out / "eval_input_anomaly.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["EXAMPLE_ID", "FAMILY", "SEQUENCE"]); w.writeheader(); w.writerows(in_rows)
    with open(out / "eval_set_forbidden.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["EXAMPLE_ID", "VIOLATION_RULE"]); w.writeheader(); w.writerows(gt_rows)
    print(f"\nWrote testset -> {out}/eval_input_anomaly.csv (+ eval_set_forbidden.csv GT)")

    # ── run the HARNESS over the whole set ──
    det = attr = 0
    per_rule_det = Counter()
    per_rule_tot = Counter()
    for r in bad:
        viol = HARNESS(r["steps"])
        per_rule_tot[r["first_rule"]] += 1
        if viol:
            det += 1
            per_rule_det[r["first_rule"]] += 1
            if any(v.rule == r["first_rule"] for v in viol):
                attr += 1
    n = len(bad)
    print("\nHARNESS RESULT (physics validate_sequence_combined over the whole testset):")
    print(f"   detected invalid: {det}/{n} = {det/n:.4f}")
    print(f"   rule attribution: {attr}/{n} = {attr/n:.4f}")
    print("   per-rule detection:")
    for rule in sorted(per_rule_tot):
        d, t = per_rule_det[rule], per_rule_tot[rule]
        print(f"     {rule:34s} {d}/{t} = {d/t:.3f}")


if __name__ == "__main__":
    main()
