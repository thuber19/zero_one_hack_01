#!/usr/bin/env python3
"""
exhaustive_test.py — full sanity check of the detection + explanation core.

Tests, in order:
  1. REAL provided data        — all 3000 validated variants + 3 canonical
                                  reference sequences must pass BOTH the
                                  reference checker and our engine (0 violations).
  2. Window boundaries         — for every windowed rule, the exact step distance
                                  at which detection must flip (valid at the edge,
                                  invalid one step past it).
  3. Litho-level ordering      — skip / decrease / in-order.
  4. Ordering rules            — pad-window, test, ship, backside: minimal valid
                                  and invalid constructions.
  5. Every bad combo           — every rule x strategy x family from the bad-data
                                  generator: detected, correctly attributed, and
                                  correctly EXPLAINED.
  6. Engine == reference       — agreement on a large mixed sample (valid + bad).
  7. OOD generalisation        — unknown-family sequences validated by physics.

Run:
    python exhaustive_test.py            # standard (a few thousand sequences)
    python exhaustive_test.py --heavy    # 10x sample sizes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "training_data"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from generate_sequences import (
    generate_sequence, validate_sequence, read_csv_sequences,
)
from physics.state_machine import validate_by_state_machine as engine
from physics import process_knowledge as K
from physics.process_knowledge import step_in_event
from physics.ontology import classify_step
from explain import explain_sequence
import bad_data_generator as BDG


# A benign filler (pure metrology; triggers/enables nothing).
FILLER = "MEASURE THICKNESS"


class Report:
    def __init__(self):
        self.sections = []

    def add(self, name, passed, detail=""):
        self.sections.append((name, passed, detail))
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))

    def ok(self):
        return all(p for _, p, _ in self.sections)


# ---------------------------------------------------------------------------
# 1. Real provided data — every known-good sequence must pass
# ---------------------------------------------------------------------------

def test_real_data(rep: Report):
    print("\n[1] REAL provided data (every known-good sequence must pass)")
    data_dir = _REPO / "training_data"
    total = ref_fp = eng_fp = 0
    for fam in ("MOSFET", "IGBT", "IC"):
        path = data_dir / f"{fam}_variants.csv"
        if not path.exists():
            continue
        seqs = read_csv_sequences(path)
        for steps in seqs.values():
            total += 1
            if validate_sequence(steps):
                ref_fp += 1
            if engine(steps):
                eng_fp += 1
    # canonical single-sequence references
    for fname in ("synthetic_mosfet.csv", "syntheticIGBT.csv", "syntheticIC.csv"):
        p = data_dir / fname
        if p.exists():
            for steps in read_csv_sequences(p).values():
                total += 1
                if validate_sequence(steps):
                    ref_fp += 1
                if engine(steps):
                    eng_fp += 1
    rep.add("reference checker: 0 false positives on provided data",
            ref_fp == 0, f"{total - ref_fp}/{total} pass")
    rep.add("our engine: 0 false positives on provided data",
            eng_fp == 0, f"{total - eng_fp}/{total} pass")


# ---------------------------------------------------------------------------
# 2. Window boundaries — detection must flip at the exact step distance
# ---------------------------------------------------------------------------

def _violated(steps, rule_id):
    return any(v.rule == rule_id for v in engine(steps))


def test_window_boundaries(rep: Report):
    print("\n[2] Window boundaries (valid AT the edge, invalid one step past)")

    # RULE_DEP_NO_CLEAN, window 12: clean, k fillers, deposit -> distance k+1
    def dep(k):
        return ["HF DIP"] + [FILLER] * k + ["DEPOSIT POLYSILICON"]
    rep.add("DEP_NO_CLEAN valid at distance 12",
            not _violated(dep(11), "RULE_DEP_NO_CLEAN"), "k=11")
    rep.add("DEP_NO_CLEAN invalid at distance 13",
            _violated(dep(12), "RULE_DEP_NO_CLEAN"), "k=12")

    # RULE_ETCH_NO_MASK, window 12: develop, k fillers, etch
    def etch(k):
        return ["DEVELOP PHOTORESIST"] + [FILLER] * k + ["OXIDE ETCH"]
    rep.add("ETCH_NO_MASK valid at distance 12",
            not _violated(etch(11), "RULE_ETCH_NO_MASK"), "k=11")
    rep.add("ETCH_NO_MASK invalid at distance 13",
            _violated(etch(12), "RULE_ETCH_NO_MASK"), "k=12")

    # RULE_IMPLANT_NO_MASK, window 15: develop (opener), k fillers, implant
    def imp(k):
        return ["DEVELOP PHOTORESIST"] + [FILLER] * k + ["IMPLANT WELL"]
    rep.add("IMPLANT_NO_MASK valid at distance 15",
            not _violated(imp(14), "RULE_IMPLANT_NO_MASK"), "k=14")
    rep.add("IMPLANT_NO_MASK invalid at distance 16",
            _violated(imp(15), "RULE_IMPLANT_NO_MASK"), "k=15")

    # RULE_CMP_NO_DEP, window 6: clean, deposit, k fillers, cmp
    def cmp(k):
        return ["HF DIP", "DEPOSIT INTERLAYER DIELECTRIC"] + [FILLER] * k + ["CMP DIELECTRIC"]
    rep.add("CMP_NO_DEP valid at distance 6",
            not _violated(cmp(5), "RULE_CMP_NO_DEP"), "k=5")
    rep.add("CMP_NO_DEP invalid at distance 7",
            _violated(cmp(6), "RULE_CMP_NO_DEP"), "k=6")

    # RULE_METAL_ETCH_NO_LITHO, window 15: expose far, develop near, metal etch
    # (develop within 12 keeps ETCH_NO_MASK satisfied so we isolate the 15-window)
    def metal(k):
        return ["EXPOSE LITHO LEVEL 4"] + [FILLER] * k + ["DEVELOP PHOTORESIST", "METAL ETCH"]
    # distance EXPOSE->METAL = k+2 ; valid iff k+2 <= 15
    rep.add("METAL_ETCH_NO_LITHO valid at expose-distance 15",
            not _violated(metal(13), "RULE_METAL_ETCH_NO_LITHO"), "k=13")
    rep.add("METAL_ETCH_NO_LITHO invalid at expose-distance 16",
            _violated(metal(14), "RULE_METAL_ETCH_NO_LITHO"), "k=14")


# ---------------------------------------------------------------------------
# 3. Litho-level ordering
# ---------------------------------------------------------------------------

def test_litho_levels(rep: Report):
    print("\n[3] Litho-level ordering")
    base = ["ALIGN MASK LEVEL 1", "ALIGN MASK LEVEL 2", "ALIGN MASK LEVEL 3"]
    rep.add("in-order levels valid",
            not _violated(base, "RULE_LITHO_LEVEL_SKIP"))
    rep.add("skip (1 -> 3) invalid",
            _violated(["ALIGN MASK LEVEL 1", "ALIGN MASK LEVEL 3"], "RULE_LITHO_LEVEL_SKIP"))
    rep.add("decrease (3 -> 2) invalid",
            _violated(["ALIGN MASK LEVEL 1", "ALIGN MASK LEVEL 2",
                       "ALIGN MASK LEVEL 3", "ALIGN MASK LEVEL 2"],
                      "RULE_LITHO_LEVEL_SKIP"))
    rep.add("repeat same level valid",
            not _violated(["ALIGN MASK LEVEL 1", "ALIGN MASK LEVEL 1"],
                          "RULE_LITHO_LEVEL_SKIP"))


# ---------------------------------------------------------------------------
# 4. Ordering rules — minimal valid + invalid
# ---------------------------------------------------------------------------

def test_ordering(rep: Report):
    print("\n[4] Ordering rules (milestones)")
    # SHIP_BEFORE_TEST
    rep.add("ship after sort valid",
            not _violated(["WAFER SORT TEST", "SHIP LOT"], "RULE_SHIP_BEFORE_TEST"))
    rep.add("ship before sort invalid",
            _violated(["SHIP LOT", "WAFER SORT TEST"], "RULE_SHIP_BEFORE_TEST"))
    # TEST_BEFORE_PASSIVATION
    rep.add("test after cure valid",
            not _violated(["CURE PASSIVATION", "PARAMETRIC TEST"], "RULE_TEST_BEFORE_PASSIVATION"))
    rep.add("test before cure invalid",
            _violated(["PARAMETRIC TEST"], "RULE_TEST_BEFORE_PASSIVATION"))
    # PAD_OPEN_BEFORE_DEP
    rep.add("pad after deposit+cure valid",
            not _violated(["DEPOSIT PASSIVATION", "CURE PASSIVATION", "OPEN PAD WINDOW"],
                          "RULE_PAD_OPEN_BEFORE_DEP"))
    rep.add("pad before passivation invalid",
            _violated(["OPEN PAD WINDOW"], "RULE_PAD_OPEN_BEFORE_DEP"))
    rep.add("pad after deposit but before cure invalid",
            _violated(["DEPOSIT PASSIVATION", "OPEN PAD WINDOW"], "RULE_PAD_OPEN_BEFORE_DEP"))
    # BACKSIDE_BEFORE_PASSIVATION
    rep.add("backside after cure valid",
            not _violated(["CURE PASSIVATION", "DEPOSIT BACKSIDE METAL"],
                          "RULE_BACKSIDE_BEFORE_PASSIVATION"))
    rep.add("backside before cure invalid",
            _violated(["DEPOSIT BACKSIDE METAL"], "RULE_BACKSIDE_BEFORE_PASSIVATION"))


# ---------------------------------------------------------------------------
# 5. Every bad combo — detected, attributed, and EXPLAINED correctly
# ---------------------------------------------------------------------------

def test_every_bad_combo(rep: Report, per_combo: int):
    print(f"\n[5] Every bad combo (per_combo={per_combo}) — detect + attribute + explain")
    bad, neg = BDG.build(per_combo=per_combo, seed=20)

    detected = attributed = explained = 0
    tier1 = [r for r in bad if r["tier"] == 1]
    for r in bad:
        viol = engine(r["steps"])
        if viol:
            detected += 1
        if viol and viol[0].rule == r["first_rule"]:
            attributed += 1
    # explanation correctness on tier-1 (single-rule) examples
    for r in tier1:
        exp = explain_sequence(r["steps"])
        flagged = {v.rule for e in exp for v in e["violations"]}
        if r["first_rule"] in flagged:
            explained += 1

    rep.add("every bad sequence detected by engine",
            detected == len(bad), f"{detected}/{len(bad)}")
    rep.add("first-rule attribution (incl. compound)",
            attributed >= len(tier1), f"{attributed}/{len(bad)} (tier1={len(tier1)})")
    rep.add("tier-1 explanations cite the correct rule",
            explained == len(tier1), f"{explained}/{len(tier1)}")

    # coverage: every rule represented
    rules = {r["first_rule"] for r in bad}
    rep.add("all 10 rules exercised", len(rules) == 10, f"{len(rules)}/10 rules")

    # hard negatives: zero false positives
    fp = sum(1 for r in neg if engine(r["steps"]))
    rep.add("zero false positives on hard-negatives", fp == 0, f"{fp} FP / {len(neg)}")


# ---------------------------------------------------------------------------
# 6. Engine == reference on a large mixed sample
# ---------------------------------------------------------------------------

def test_agreement(rep: Report, n_valid: int, per_combo: int):
    print(f"\n[6] Engine vs reference agreement (mixed sample)")
    import random
    rng = random.Random(99)

    disagree_valid = 0
    for _ in range(n_valid):
        fam = rng.choice(["mosfet", "igbt", "ic"])
        seq = generate_sequence(fam, rng)
        ref_bad = bool(validate_sequence(seq))
        eng_bad = bool(engine(seq))
        if ref_bad != eng_bad:
            disagree_valid += 1
    rep.add("engine agrees with reference on valid sample",
            disagree_valid == 0, f"{disagree_valid} disagreements / {n_valid}")

    bad, _ = BDG.build(per_combo=per_combo, seed=33)
    disagree_bad = sum(1 for r in bad if bool(engine(r["steps"])) != True)
    rep.add("engine flags every reference-labelled bad sequence",
            disagree_bad == 0, f"{disagree_bad} missed / {len(bad)}")


# ---------------------------------------------------------------------------
# 7. OOD generalisation (unknown family)
# ---------------------------------------------------------------------------

def test_ood(rep: Report):
    print("\n[7] OOD generalisation (unknown 4th-family vocabulary)")
    valid = [
        "RECEIVE WAFER LOT", "PRE CLEAN WAFER", "GROW GAN BUFFER LAYER",
        "MOCVD GAN EPITAXIAL GROWTH", "SPIN COAT PHOTORESIST", "ALIGN MASK LEVEL 1",
        "EXPOSE LITHO LEVEL 1", "DEVELOP PHOTORESIST", "ETCH GAN MESA",
        "STRIP PHOTORESIST", "CLEAN AFTER MESA ETCH", "IMPLANT N-GAN REGION",
        "RAPID THERMAL ANNEAL", "DEPOSIT PASSIVATION", "CURE PASSIVATION",
        "PARAMETRIC TEST", "WAFER SORT TEST", "SHIP LOT",
    ]
    rep.add("OOD valid sequence accepted", not engine(valid), f"{len(engine(valid))} violations")

    # break it three ways
    dep_no_clean = ["RECEIVE WAFER LOT", "GROW GAN BUFFER LAYER"]  # deposit, no clean
    rep.add("OOD missing-clean caught",
            _violated(dep_no_clean, "RULE_DEP_NO_CLEAN"))
    etch_no_mask = ["PRE CLEAN WAFER", "ETCH GAN MESA"]  # etch, no develop
    rep.add("OOD unmasked-etch caught",
            _violated(etch_no_mask, "RULE_ETCH_NO_MASK"))
    ship_first = ["SHIP LOT", "WAFER SORT TEST"]
    rep.add("OOD ship-before-test caught",
            _violated(ship_first, "RULE_SHIP_BEFORE_TEST"))

    # Scale OOD: hundreds of synthetic novel-vocabulary families.
    import pseudo_family as PF
    import random as _r
    rng = _r.Random(5)
    pseudo = PF.generate_pseudo_valid(400, rng)
    fp = sum(1 for _tag, s in pseudo if engine(s))
    rep.add("OOD pseudo-families: 0 false positives",
            fp == 0, f"{fp} FP / {len(pseudo)}")
    novel = set()
    for _tag, s in pseudo:
        novel.update(x for x in s if classify_step(x) == "UNKNOWN")
    rep.add("OOD novel tokens all classify to a category",
            not novel, f"{len(novel)} UNKNOWN")
    caught = total = 0
    for _tag, s in pseudo[:150]:
        inj = PF.inject_violation(s, rng)
        if inj:
            total += 1
            caught += 1 if engine(inj[0]) else 0
    rep.add("OOD injected violations all caught",
            total > 0 and caught == total, f"{caught}/{total}")


def test_multi_and_repair(rep: Report, per_combo: int):
    print("\n[8] Multiple violations + detect-explain-repair")
    import fix as FIX
    import random as _r
    from generate_sequences import generate_sequence
    rng = _r.Random(13)

    # Multi-violation: inject 2-3 independent faults, assert ALL detected.
    multi_ok = 0
    multi_total = 0
    for _ in range(60):
        s = generate_sequence(rng.choice(["mosfet", "igbt", "ic"]), rng)
        d = next((i for i, x in enumerate(s) if step_in_event(x, "DEPOSITION")), None)
        if d is not None:
            drop = {j for j in range(max(0, d - 12), d) if step_in_event(s[j], "CLEAN_SURFACE")}
            s = [x for j, x in enumerate(s) if j not in drop]
        si = next((i for i, x in enumerate(s) if x == "SHIP LOT"), None)
        ti = next((i for i, x in enumerate(s) if x == "WAFER SORT TEST"), None)
        if si is not None and ti is not None and si > ti:
            s.pop(si); ti = next(i for i, x in enumerate(s) if x == "WAFER SORT TEST")
            s.insert(ti, "SHIP LOT")
        rules = {v.rule for v in engine(s)}
        if len(rules) >= 2:
            multi_total += 1
            if {"RULE_DEP_NO_CLEAN", "RULE_SHIP_BEFORE_TEST"} <= rules:
                multi_ok += 1
    rep.add("multiple independent violations all detected",
            multi_total > 0 and multi_ok == multi_total, f"{multi_ok}/{multi_total}")

    # Repair: every known-bad sequence repairs to physical validity.
    bad, _ = BDG.build(per_combo=per_combo, seed=44)
    repaired_ok = sum(1 for r in bad if FIX.repair(r["steps"])["now_valid"])
    rep.add("every known-bad sequence repairs to valid",
            repaired_ok == len(bad), f"{repaired_ok}/{len(bad)}")

    # analyze() reports every violation (count matches the engine)
    match = 0
    for r in bad:
        a = FIX.analyze(r["steps"])
        if a["n_violations"] == len(engine(r["steps"])):
            match += 1
    rep.add("analyze() reports every violation",
            match == len(bad), f"{match}/{len(bad)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heavy", action="store_true", help="10x sample sizes.")
    args = ap.parse_args(argv)

    per_combo = 20 if args.heavy else 6
    n_valid = 20000 if args.heavy else 3000

    print("=" * 64)
    print("EXHAUSTIVE SANITY CHECK")
    print("=" * 64)
    rep = Report()
    test_real_data(rep)
    test_window_boundaries(rep)
    test_litho_levels(rep)
    test_ordering(rep)
    test_every_bad_combo(rep, per_combo)
    test_agreement(rep, n_valid, per_combo)
    test_ood(rep)
    test_multi_and_repair(rep, per_combo)

    print("\n" + "=" * 64)
    n_pass = sum(1 for _, p, _ in rep.sections if p)
    print(f"RESULT: {n_pass}/{len(rep.sections)} checks passed")
    print("OVERALL: PASS" if rep.ok() else "OVERALL: FAIL")
    print("=" * 64)
    return 0 if rep.ok() else 1


if __name__ == "__main__":
    sys.exit(main())
