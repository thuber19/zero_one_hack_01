#!/usr/bin/env python3
"""
explain.py — make the system's understanding visible.

Given a process sequence, this narrates, step by step, WHY each operation is
allowed (which physical precondition it needed and where that precondition was
satisfied) and, for any violation, WHY it is physically impossible — straight
from the declarative knowledge base in physics/process_knowledge.py.

This is the "context-aware after training" surface: the system can explain its
own reasoning about any sequence, known family or not.

Usage
-----
  python explain.py --family mosfet                 # explain a fresh valid seq
  python explain.py --family igbt --break           # inject a fault and explain it
  python explain.py --file bad_data/known_bad.csv --row 1
  python explain.py --export-doc knowledge/PROCESS_MODEL.md   # dump the KB
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO = Path(__file__).parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "data"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from physics import process_knowledge as K
from physics.state_machine import WaferState, apply_step, _extract_litho_level
from physics.ontology import classify_step


def explain_sequence(steps: list[str]) -> list[dict]:
    """
    Replay the sequence and produce a structured explanation per step:
      {index, step, category, events, checks:[(status, rule_id, detail)],
       violations:[PhysicsViolation]}
    `status` is "ok" (precondition satisfied) or "violation".
    """
    state = WaferState()
    result: list[dict] = []

    for i, step in enumerate(steps):
        checks: list[tuple[str, str, str]] = []

        # Windowed rules triggered by this step
        for rule in K.WINDOWED_RULES:
            if not K.step_in_event(step, rule.trigger):
                continue
            for enabler, window in rule.requires:
                idx = state.last_event_idx.get(enabler)
                if idx is not None and (i - idx) <= window:
                    checks.append(("ok", rule.id,
                                   f"{enabler} satisfied by step {idx} "
                                   f"({i - idx} step(s) ago; window {window})"))
                else:
                    checks.append(("violation", rule.id,
                                   f"no {enabler} within the prior {window} steps"))

        # Litho-level rule
        if step.upper().startswith("ALIGN MASK LEVEL"):
            lvl = _extract_litho_level(step)
            if lvl is not None and state.last_aligned_level > 0:
                if lvl > state.last_aligned_level + 1:
                    checks.append(("violation", K.LITHO_RULE["id"],
                                   f"level jumps {state.last_aligned_level}->{lvl}"))
                elif lvl < state.last_aligned_level:
                    checks.append(("violation", K.LITHO_RULE["id"],
                                   f"level decreases {state.last_aligned_level}->{lvl}"))
                else:
                    checks.append(("ok", K.LITHO_RULE["id"],
                                   f"level advances to {lvl} in order"))

        # Ordering rules triggered by this step
        for rule in K.ORDERING_RULES:
            if not K.step_in_event(step, rule.trigger):
                continue
            for flag, reason in rule.requires:
                if state.milestones.get(flag, False):
                    checks.append(("ok", rule.id, f"milestone '{flag}' already reached"))
                else:
                    checks.append(("violation", rule.id, reason))
                    break

        new_state, violations = apply_step(state, step)
        state = new_state

        events = [name for name in K.EVENT_CLASSES if K.step_in_event(step, name)]
        result.append({
            "index": i, "step": step, "category": classify_step(step),
            "events": events, "checks": checks, "violations": violations,
        })

    return result


def format_narrative(steps: list[str], only_interesting: bool = True) -> str:
    """Pretty-print the explanation. With `only_interesting`, show only steps
    that have a precondition check or a violation (skips pure logistics/metrology)."""
    lines: list[str] = []
    exp = explain_sequence(steps)
    n_viol = sum(len(e["violations"]) for e in exp)
    lines.append(f"Sequence: {len(steps)} steps | "
                 f"{'VALID' if n_viol == 0 else f'INVALID ({n_viol} violation(s))'}")
    lines.append("=" * 70)
    for e in exp:
        if only_interesting and not e["checks"] and not e["violations"]:
            continue
        tag = "  " if not e["violations"] else ">>"
        ev = ("  [" + ", ".join(e["events"]) + "]") if e["events"] else ""
        lines.append(f"{tag} {e['index']:>3} {e['step']}{ev}")
        for status, rule_id, detail in e["checks"]:
            mark = "OK " if status == "ok" else "XX "
            lines.append(f"        {mark} {rule_id}: {detail}")
        for v in e["violations"]:
            lines.append(f"        !! VIOLATION {v.rule}")
            lines.append(f"           what : {v.description}")
            lines.append(f"           why  : {v.physical_reason}")
    return "\n".join(lines)


def _load_row(path: Path, row_index: int) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        norm = {k.strip().strip('"'): k for k in (reader.fieldnames or [])}
        seq_key = norm.get("SEQUENCE") or norm.get("PARTIAL_SEQUENCE")
        rows = list(reader)
    if seq_key is None:
        raise ValueError(f"No SEQUENCE/PARTIAL_SEQUENCE column in {path}")
    row = rows[row_index]
    return [s.strip() for s in row[seq_key].split("|") if s.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=["mosfet", "igbt", "ic"], default="mosfet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--break", dest="do_break", action="store_true",
                    help="inject a fault (delete a clean) and explain the failure.")
    ap.add_argument("--file", metavar="CSV", help="explain a sequence from a CSV.")
    ap.add_argument("--row", type=int, default=0, help="row index within --file.")
    ap.add_argument("--all", action="store_true",
                    help="show every step, not only steps with checks.")
    ap.add_argument("--export-doc", metavar="PATH",
                    help="write the full knowledge base to PATH (Markdown) and exit.")
    args = ap.parse_args(argv)

    if args.export_doc:
        Path(args.export_doc).parent.mkdir(parents=True, exist_ok=True)
        Path(args.export_doc).write_text(K.to_markdown(), encoding="utf-8")
        print(f"Knowledge base exported -> {args.export_doc}")
        return 0

    if args.file:
        steps = _load_row(Path(args.file), args.row)
    else:
        import random
        from generate_sequences import generate_sequence
        steps = generate_sequence(args.family, random.Random(args.seed))
        if args.do_break:
            # delete EVERY clean in the 12-step window before the first
            # deposition, so RULE_DEP_NO_CLEAN genuinely trips.
            from physics.process_knowledge import step_in_event
            d = next((i for i, s in enumerate(steps)
                      if step_in_event(s, "DEPOSITION")), None)
            if d is not None:
                drop = {j for j in range(max(0, d - 12), d)
                        if step_in_event(steps[j], "CLEAN_SURFACE")}
                removed = [steps[j] for j in sorted(drop)]
                steps = [s for j, s in enumerate(steps) if j not in drop]
                print(f"(injected fault: removed cleans {removed} before the "
                      f"first deposition)\n")

    print(format_narrative(steps, only_interesting=not args.all))
    return 0


if __name__ == "__main__":
    sys.exit(main())
