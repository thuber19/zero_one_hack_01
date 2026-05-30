#!/usr/bin/env python3
"""
fix.py — detect EVERY violation, explain it, suggest a fix, and repair to valid.

Detection alone only tells you a wafer route is wrong. This module closes the
loop the way a process engineer would: it finds *all* violations (not just the
first), explains *why* each is physically impossible (from the knowledge base),
proposes a concrete *minimal repair* for each, and then iteratively applies
repairs until the route is physically valid — verifying the result with the
engine at every step. It handles multiple, compound, and cascading violations.

Because the repairs use UNIVERSAL enabler steps (a clean, a develop, a fill) and
the engine reasons by physical category, repair works on the unknown 4th family
too, not just the three known ones.

stdlib + physics only.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "training_data"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from physics.state_machine import validate_by_state_machine
from physics import process_knowledge as K

# Rough severity (higher = more critical) for ranking / triage.
_SEVERITY = {
    "RULE_SHIP_BEFORE_TEST": 5,
    "RULE_TEST_BEFORE_PASSIVATION": 4,
    "RULE_BACKSIDE_BEFORE_PASSIVATION": 4,
    "RULE_PAD_OPEN_BEFORE_DEP": 4,
    "RULE_DEP_NO_CLEAN": 3,
    "RULE_ETCH_NO_MASK": 3,
    "RULE_METAL_ETCH_NO_LITHO": 3,
    "RULE_IMPLANT_NO_MASK": 3,
    "RULE_CMP_NO_DEP": 2,
    "RULE_LITHO_LEVEL_SKIP": 2,
}

# Universal enabler steps used in repairs (recognised by the engine in any family).
_CLEAN = "CLEAN AFTER ETCH"
_DEVELOP = "DEVELOP PHOTORESIST"
_EXPOSE = "EXPOSE LITHO LEVEL {n}"
_FILL = "FILL VIA METAL"


@dataclass
class Finding:
    rule: str
    step_index: int
    step_name: str
    severity: int
    why: str
    fix_description: str


# ---------------------------------------------------------------------------
# Single-violation fixers — each returns a NEW sequence and a description.
# ---------------------------------------------------------------------------

def _insert_before(steps, idx, items):
    return steps[:idx] + list(items) + steps[idx:]


def _earliest_consumer(steps) -> int:
    """Index of the first step that requires cured passivation (pad window /
    electrical test / backside metal); len(steps) if none."""
    for i, s in enumerate(steps):
        if (K.step_in_event(s, "PAD_WINDOW_OPEN")
                or K.step_in_event(s, "ELECTRICAL_TEST")
                or K.step_in_event(s, "BACKSIDE_METAL")):
            return i
    return len(steps)


def _fix_passivation_order(steps, v) -> tuple[list[str], str]:
    """Unified repair for the three passivation-ordering rules. If passivation
    is not deposited+cured, insert the missing milestone(s) before the EARLIEST
    consumer (resolves every related violation at once); otherwise relocate the
    offending step to after CURE PASSIVATION."""
    has_dep = any(x in ("DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER") for x in steps)
    has_cure = "CURE PASSIVATION" in steps

    _DEP_NAMES = ("DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER")
    if has_dep and has_cure:
        # both milestones present -> ordering problem. Two sub-cases:
        #   (a) a DEPOSIT may itself sit AFTER the CURE (corrupted order) — then
        #       relocating the consumer after CURE still leaves it before the
        #       deposit. Canonicalise first: ensure a deposit precedes the cure.
        #   (b) relocate the offending consumer to just after the LAST cure.
        s = steps[:v.step_index] + steps[v.step_index + 1:]
        first_cure = min(j for j, x in enumerate(s) if x == "CURE PASSIVATION")
        dep_idx = [j for j, x in enumerate(s) if x in _DEP_NAMES]
        if not any(d < first_cure for d in dep_idx):
            # no deposit before the first cure -> move one (the earliest deposit
            # that is after the cure) to immediately before the cure.
            d = min(d for d in dep_idx if d > first_cure)
            dep_step = s.pop(d)               # d > first_cure, so first_cure unchanged
            s.insert(first_cure, dep_step)    # deposit now right before that cure
        ci = max(j for j, x in enumerate(s) if x == "CURE PASSIVATION")
        return s[:ci + 1] + [v.step_name] + s[ci + 1:], \
            f"order DEPOSIT/CURE PASSIVATION then move '{v.step_name}' after CURE PASSIVATION"

    if has_cure and not has_dep:
        # cure exists but the passivation deposit was removed -> add the deposit
        # right before the existing cure (one fix resolves all pad/test/backside).
        ci = min(j for j, x in enumerate(steps) if x == "CURE PASSIVATION")
        return _insert_before(steps, ci, ["DEPOSIT PASSIVATION"]), \
            "passivation never deposited — add DEPOSIT PASSIVATION before CURE PASSIVATION"

    if has_dep and not has_cure:
        di = max(j for j, x in enumerate(steps)
                 if x in ("DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"))
        return _insert_before(steps, di + 1, ["CURE PASSIVATION"]), \
            "passivation never cured — add CURE PASSIVATION after the deposit"

    # neither present -> insert both before the earliest consumer
    ci = _earliest_consumer(steps)
    return _insert_before(steps, ci, ["DEPOSIT PASSIVATION", "CURE PASSIVATION"]), \
        "deposit and cure passivation before the first pad/test/backside step"


def _fix_one(steps: list[str], v) -> tuple[list[str], str]:
    """Propose a repair for ONE violation. Insertion repairs put the missing
    enabler immediately before the offending step; ordering repairs relocate or
    insert the required milestone."""
    i, rule = v.step_index, v.rule

    if rule == "RULE_DEP_NO_CLEAN":
        return _insert_before(steps, i, [_CLEAN]), \
            f"insert a cleaning step ('{_CLEAN}') immediately before '{v.step_name}'"

    if rule == "RULE_ETCH_NO_MASK":
        return _insert_before(steps, i, [_DEVELOP]), \
            f"insert '{_DEVELOP}' before the etch '{v.step_name}' to provide a mask"

    if rule == "RULE_METAL_ETCH_NO_LITHO":
        return _insert_before(steps, i, [_EXPOSE.format(n=4), _DEVELOP]), \
            f"insert a lithography expose+develop before '{v.step_name}'"

    if rule == "RULE_IMPLANT_NO_MASK":
        return _insert_before(steps, i, [_DEVELOP]), \
            f"insert '{_DEVELOP}' before '{v.step_name}' to open the implant window"

    if rule == "RULE_CMP_NO_DEP":
        return _insert_before(steps, i, [_FILL]), \
            f"insert a fill/deposition ('{_FILL}') before '{v.step_name}' to give CMP material"

    if rule == "RULE_LITHO_LEVEL_SKIP":
        # renumber every ALIGN MASK LEVEL to be sequential 1,2,3,…
        out, lvl = [], 0
        for s in steps:
            if s.upper().startswith("ALIGN MASK LEVEL"):
                lvl += 1
                out.append(f"ALIGN MASK LEVEL {lvl}")
            else:
                out.append(s)
        return out, "renumber ALIGN MASK LEVEL steps to be sequential (1,2,3,…)"

    if rule == "RULE_SHIP_BEFORE_TEST":
        s = [x for x in steps if x != "SHIP LOT"]
        ti = next((j for j, x in enumerate(s) if x == "WAFER SORT TEST"), None)
        if ti is None:                      # sort missing entirely → add it then ship
            return s + ["WAFER SORT TEST", "SHIP LOT"], \
                "no WAFER SORT TEST present — append sort test then SHIP LOT"
        return s[:ti + 1] + ["SHIP LOT"] + s[ti + 1:], \
            "move SHIP LOT to after WAFER SORT TEST"

    if rule in ("RULE_BACKSIDE_BEFORE_PASSIVATION", "RULE_TEST_BEFORE_PASSIVATION",
                "RULE_PAD_OPEN_BEFORE_DEP"):
        return _fix_passivation_order(steps, v)

    return steps, "no automatic fix available"


# ---------------------------------------------------------------------------
# Analysis + repair
# ---------------------------------------------------------------------------

def analyze(steps: list[str]) -> dict:
    """Full anomaly report: every violation, explained, with a suggested fix."""
    viols = validate_by_state_machine(steps)
    findings = []
    for v in viols:
        _, desc = _fix_one(steps, v)
        findings.append(Finding(
            rule=v.rule, step_index=v.step_index, step_name=v.step_name,
            severity=_SEVERITY.get(v.rule, 1),
            why=v.physical_reason, fix_description=desc))
    findings.sort(key=lambda f: (-f.severity, f.step_index))
    return {
        "is_valid": len(viols) == 0,
        "n_violations": len(viols),
        "distinct_rules": sorted({v.rule for v in viols}),
        "findings": findings,
    }


def repair(steps: list[str], max_iter: int = 30) -> dict:
    """Iteratively repair the route to physical validity, verifying with the
    engine each round (handles cascades: fixing one may expose another)."""
    cur = list(steps)
    applied = []
    for _ in range(max_iter):
        viols = validate_by_state_machine(cur)
        if not viols:
            break
        # fix the most severe violation first
        v = max(viols, key=lambda x: (_SEVERITY.get(x.rule, 1), -x.step_index))
        fixed, desc = _fix_one(cur, v)
        if fixed == cur:
            applied.append(f"[unresolved] {v.rule}: {desc}")
            break
        applied.append(f"{v.rule} @ step {v.step_index}: {desc}")
        cur = fixed
    final = validate_by_state_machine(cur)
    return {
        "repaired": cur,
        "fixes_applied": applied,
        "now_valid": len(final) == 0,
        "remaining": sorted({v.rule for v in final}),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _format(report: dict) -> str:
    if report["is_valid"]:
        return "VALID — no process-logic violations."
    L = [f"INVALID — {report['n_violations']} violation(s) across "
         f"{len(report['distinct_rules'])} rule type(s): {report['distinct_rules']}", ""]
    for f in report["findings"]:
        L.append(f"  [sev {f.severity}] {f.rule} @ step {f.step_index} ({f.step_name})")
        L.append(f"      why : {f.why}")
        L.append(f"      fix : {f.fix_description}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=["mosfet", "igbt", "ic"], default="mosfet")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--faults", type=int, default=3,
                    help="how many violations to inject for the demo.")
    args = ap.parse_args(argv)

    import random
    from generate_sequences import generate_sequence
    from physics.process_knowledge import step_in_event
    rng = random.Random(args.seed)
    s = generate_sequence(args.family, rng)

    # inject up to N independent violations for the demo
    if args.faults >= 1:
        d = next((i for i, x in enumerate(s) if step_in_event(x, "DEPOSITION")), None)
        if d is not None:
            drop = {j for j in range(max(0, d - 12), d) if step_in_event(s[j], "CLEAN_SURFACE")}
            s = [x for j, x in enumerate(s) if j not in drop]
    if args.faults >= 2:
        e = next((i for i, x in enumerate(s) if step_in_event(x, "PATTERNED_ETCH")), None)
        if e is not None:
            for j in range(e - 1, max(0, e - 12) - 1, -1):
                if step_in_event(s[j], "DEVELOP"):
                    s.pop(j); break
    if args.faults >= 3:
        si = next((i for i, x in enumerate(s) if x == "SHIP LOT"), None)
        ti = next((i for i, x in enumerate(s) if x == "WAFER SORT TEST"), None)
        if si is not None and ti is not None and si > ti:
            s.pop(si); ti = next(i for i, x in enumerate(s) if x == "WAFER SORT TEST")
            s.insert(ti, "SHIP LOT")

    print("=== DETECT + EXPLAIN + SUGGEST FIX ===")
    print(_format(analyze(s)))
    print("\n=== REPAIR TO VALID ===")
    r = repair(s)
    for line in r["fixes_applied"]:
        print(f"  applied: {line}")
    print(f"\n  now valid: {r['now_valid']}   remaining: {r['remaining']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
