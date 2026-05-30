"""
synonyms.py — the authoritative synonym groups of the step vocabulary.

The generator (data/generate_sequences.py) chooses interchangeable step names with
`rng.choice([...])` — those lists ARE the synonym groups (the positions where the
generator makes a random pick among equivalent steps, i.e. the irreducible
next-token entropy quantified in CEILING_ANALYSIS.md). We parse them straight from
the generator with `ast` so this never drifts from the source of truth, and add a
few documented logistics synonyms from generation_rules.md §1.1.

Used by the optional `--synonym-collapse` training loss: predicting ANY member of
the gold step's synonym group counts as correct, so the gradient stops wasting
signal on coin-flips and focuses on real process structure. The model's vocabulary
and outputs stay EXACT (the grader scores exact tokens) — only the loss is grouped.

Public API:
    SYNONYM_GROUPS : list[frozenset[str]]
    group_of(step) -> frozenset[str]   (the step's group, or {step})
"""
from __future__ import annotations

import ast
from pathlib import Path

_GEN = Path(__file__).resolve().parent.parent / "data" / "generate_sequences.py"

# Documented logistics/inspection synonyms (generation_rules.md §1.1) that the
# generator may select positionally rather than via a single rng.choice list.
_SEED_GROUPS = [
    {"LOT RELEASE", "FINAL LOT RELEASE"},
    {"INITIAL WAFER INSPECTION", "PRE CLEAN INSPECTION"},
    {"MEASURE THICKNESS", "MEASURE WAFER THICKNESS"},
    {"PRE CLEAN WAFER", "WAFER CLEAN PRE PROCESS"},
    {"STRIP PHOTORESIST", "STRIP RESIST"},
]


def _extract_choice_groups(src_path: Path) -> list[set]:
    """Find every rng.choice([<string literals>]) in the generator; each list of
    string literals is a synonym group."""
    groups: list[set] = []
    try:
        tree = ast.parse(src_path.read_text(encoding="utf-8"))
    except Exception:
        return groups
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        is_choice = (isinstance(f, ast.Attribute) and f.attr == "choice") or \
                    (isinstance(f, ast.Name) and f.id == "choice")
        if not is_choice or not node.args:
            continue
        arg = node.args[0]
        if isinstance(arg, (ast.List, ast.Tuple)):
            elems = [e.value for e in arg.elts
                     if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            # only groups of >=2 distinct ALL-CAPS step-like strings
            elems = [e for e in elems if e.isupper() and " " in e]
            if len(set(elems)) >= 2:
                groups.append(set(elems))
    return groups


# Distinct named tests/steps that an rng.choice may list together but which are
# NOT interchangeable (different measurements/tests) — never collapse these.
_NON_SYNONYM = {
    "THRESHOLD VOLTAGE TEST", "BREAKDOWN VOLTAGE TEST", "SWITCHING TEST",
    "LEAKAGE TEST", "MEASURE OXIDE THICKNESS",
}


def _build():
    raw = _extract_choice_groups(_GEN) + [set(g) for g in _SEED_GROUPS]
    # drop non-interchangeable members so they remain singletons
    raw = [g - _NON_SYNONYM for g in raw]
    raw = [g for g in raw if len(g) >= 2]
    # merge overlapping groups (union-find style) so each step has ONE group
    merged: list[set] = []
    for g in raw:
        hit = [m for m in merged if m & g]
        if not hit:
            merged.append(set(g))
        else:
            base = hit[0]
            base |= g
            for other in hit[1:]:
                base |= other
                merged.remove(other)
    return [frozenset(m) for m in merged]


SYNONYM_GROUPS: list = _build()
_STEP_TO_GROUP: dict = {s: g for g in SYNONYM_GROUPS for s in g}


def group_of(step: str) -> frozenset:
    """Return the synonym group containing `step` (or a singleton {step})."""
    return _STEP_TO_GROUP.get(step, frozenset({step}))


if __name__ == "__main__":
    import sys
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print(f"{len(SYNONYM_GROUPS)} synonym groups extracted from {_GEN.name}:")
    for g in sorted(SYNONYM_GROUPS, key=lambda x: sorted(x)[0]):
        print("  {", " | ".join(sorted(g)), "}")
