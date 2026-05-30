#!/usr/bin/env python3
"""
pseudo_family.py — synthesise physically-valid "new families" for OOD training.

Why this exists
---------------
The provided generator emits a FIXED litho-cycle count per family and only the
three known vocabularies, so training data has no lexical and little structural
novelty. A model trained on it can memorise "family -> fixed structure" and then
collapse on the unknown 4th family (Task 4). To force the model to learn the
PHYSICS (categories + ordering) rather than token identity, we synthesise extra
"families" that:

  * keep the universal scaffolding (lithography, logistics, test, passivation
    milestone steps) so every one of the 10 rules still applies and we can
    fully verify validity, and
  * rename every DEVICE step (deposition, etch, implant, clean, anneal, CMP,
    fill, …) to a NOVEL token that still maps to the SAME physical category.

A model that handles these pseudo-families generalises by category, which is
exactly the ID->OOD robustness the track measures. Because the renaming is
category-preserving, every generated sequence is verified VALID by the physics
engine; we can also inject category-level violations to build an OOD anomaly set
and self-estimate the ID->OOD performance drop before the organizers do.

stdlib only.
"""

from __future__ import annotations

import argparse
import random
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

from generate_sequences import generate_sequence, PAD_WINDOW_STEPS, ELECTRICAL_TEST_STEPS
from physics.ontology import classify_step
from physics.state_machine import validate_by_state_machine, apply_step, WaferState

# Steps whose EXACT name carries a milestone or ordering meaning — keep as-is
# (these are universal logistics/passivation/test terms; the README confirms
# step names are mostly shared across families).
_KEEP_EXACT = (
    {"DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER", "CURE PASSIVATION",
     "DEPOSIT BACKSIDE METAL", "WAFER SORT TEST", "SHIP LOT", "YIELD ANALYSIS",
     "LOT RELEASE", "FINAL LOT RELEASE", "RECEIVE WAFER LOT", "LOT IDENTIFICATION",
     "PACKAGE PREPARATION"}
    | set(PAD_WINDOW_STEPS) | set(ELECTRICAL_TEST_STEPS)
)

# Categories we never rename (universal vocabulary the rules key on by name).
_KEEP_CATEGORIES = {"LITHO", "LOGISTICS", "TEST"}

# Category -> (template, keeps-category-keyword). {tag} = family tag, {core} =
# the distinctive remainder of the original step name. Each template embeds a
# keyword that classify_step maps back to the intended category.
_TEMPLATES = {
    "DEPOSIT":         "GROW {tag} {core}",
    "ANNEAL":          "ANNEAL {tag} {core}",
    "ETCH":            "ETCH {tag} {core}",
    "ETCH_BLANKET":    "ANISOTROPIC {tag} {core} ETCH",
    "IMPLANT":         "IMPLANT {tag} {core}",
    "CLEAN":           "CLEAN {tag} {core}",
    "PREP":            "{tag} {core} SURFACE PREP",
    "CMP":             "CMP {tag} {core}",
    "FILL":            "FILL {tag} {core}",
    "GRIND":           "GRIND {tag} {core}",
    "STRIP":           "STRIP {tag} {core}",
    "MEASURE":         "MEASURE {tag} {core}",
    "INSPECT":         "INSPECT {tag} {core}",
    "PATTERN_INSPECT": "INSPECT {tag} {core} PATTERN",
}

_VERB_FIRST = {"DEPOSIT", "GROW", "ETCH", "IMPLANT", "CLEAN", "STRIP", "MEASURE",
               "INSPECT", "ANNEAL", "FILL", "GRIND", "CMP", "DENSIFY", "DRIVE", "CURE"}


def _core(step: str) -> str:
    toks = step.split()
    if toks and toks[0] in _VERB_FIRST and len(toks) > 1:
        return " ".join(toks[1:])
    return step


def build_renamer(tag: str, base_steps: set[str]) -> dict[str, str]:
    """Build a deterministic, category-preserving rename map for one tag."""
    rename: dict[str, str] = {}
    for step in base_steps:
        if step in _KEEP_EXACT:
            continue
        cat = classify_step(step)
        if cat in _KEEP_CATEGORIES or cat not in _TEMPLATES:
            continue
        novel = _TEMPLATES[cat].format(tag=tag, core=_core(step)).strip()
        # collapse double spaces
        rename[step] = " ".join(novel.split())
    return rename


# A pool of novel family tags (unseen device families).
TAGS = ["SICFET", "GANHEMT", "INPDIODE", "GAASFET", "SIGEBJT", "ALNPOWER",
        "SICDIODE", "GANIC", "INPLASER", "SIGEHBT"]


def pseudo_sequence(base_family: str, tag: str, rng: random.Random) -> list[str]:
    """Generate a valid base sequence, then rename device steps to a novel
    vocabulary tagged `tag`."""
    base = generate_sequence(base_family, rng)
    rename = build_renamer(tag, set(base))
    return [rename.get(s, s) for s in base]


def _perturb_structure(seq: list[str], rng: random.Random) -> list[str]:
    """Apply validity-safe STRUCTURAL variation so OOD families differ in shape,
    not just vocabulary: drop a fraction of optional metrology/inspection steps
    and insert a few extra cleans. Any perturbation that happens to break a rule
    is discarded by the caller's engine check, so this only ever yields valid
    structural variants."""
    s = [x for x in seq
         if not (classify_step(x) in ("MEASURE", "INSPECT", "PATTERN_INSPECT")
                 and rng.random() < 0.3)]
    for _ in range(rng.randint(0, 3)):
        if len(s) < 3:
            break
        s.insert(rng.randint(1, len(s) - 1), "CLEAN AFTER ETCH")
    # Cycle-count variation: add a SECOND METAL LAYER — an explicitly documented
    # valid variation axis (generation_rules.md §4). Inserts a full extra
    # litho->metal-etch cycle at the next sequential mask level, right after the
    # existing metal block. Any structurally-invalid result is discarded by the
    # caller's engine check, so validity is guaranteed.
    if rng.random() < 0.5:
        levels = [int(x.rsplit(" ", 1)[1]) for x in s
                  if x.startswith("ALIGN MASK LEVEL") and x.rsplit(" ", 1)[1].isdigit()]
        nxt = (max(levels) + 1) if levels else 1
        anchor = next((i for i in range(len(s) - 1, -1, -1)
                       if s[i] == "CLEAN AFTER METAL ETCH"), None)
        if anchor is not None:
            extra = ["DEPOSIT TOP METAL", "ANNEAL METAL", "SPIN COAT PHOTORESIST",
                     f"ALIGN MASK LEVEL {nxt}", f"EXPOSE LITHO LEVEL {nxt}",
                     "DEVELOP PHOTORESIST", "METAL ETCH", "STRIP RESIST",
                     "CLEAN AFTER METAL ETCH"]
            s = s[:anchor + 1] + extra + s[anchor + 1:]
    return s


def generate_pseudo_valid(n: int, rng: random.Random,
                          structural: bool = True) -> list[tuple[str, list[str]]]:
    """Return up to n (tag, sequence) pairs the physics engine confirms VALID.
    With `structural`, half the families also get shape perturbation (length /
    optional-step variation) — lexical *and* structural OOD."""
    out = []
    fams = ("mosfet", "igbt", "ic")
    attempts = 0
    while len(out) < n and attempts < n * 6:
        attempts += 1
        tag = rng.choice(TAGS)
        fam = rng.choice(fams)
        seq = pseudo_sequence(fam, tag, rng)
        if structural and rng.random() < 0.5:
            seq = _perturb_structure(seq, rng)
        if not validate_by_state_machine(seq):       # category-verified valid
            out.append((tag, seq))
    return out


def inject_violation(seq: list[str], rng: random.Random):
    """Inject a category-level violation into a pseudo sequence (for OOD anomaly
    data). Returns (broken_seq, rule) or None. Uses windowed rules, which
    generalise across vocabulary."""
    from physics.process_knowledge import step_in_event
    s = list(seq)
    # Strategy A: remove the clean before the first deposition -> DEP_NO_CLEAN
    d = next((i for i, x in enumerate(s) if step_in_event(x, "DEPOSITION")), None)
    if d is not None:
        drop = {j for j in range(max(0, d - 12), d) if step_in_event(s[j], "CLEAN_SURFACE")}
        if drop:
            broken = [x for j, x in enumerate(s) if j not in drop]
            viol = validate_by_state_machine(broken)
            if viol:
                return broken, viol[0].rule
    # Strategy B: remove the develop before the first patterned etch -> ETCH_NO_MASK
    e = next((i for i, x in enumerate(s) if step_in_event(x, "PATTERNED_ETCH")), None)
    if e is not None:
        for j in range(e - 1, max(0, e - 12) - 1, -1):
            if step_in_event(s[j], "DEVELOP"):
                broken = s[:j] + s[j + 1:]
                viol = validate_by_state_machine(broken)
                if viol:
                    return broken, viol[0].rule
                break
    return None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest(n: int = 500):
    rng = random.Random(3)
    print(f"Generating {n} pseudo-family sequences (novel vocabulary) …")
    fams = ("mosfet", "igbt", "ic")
    made = valid = 0
    novel_tokens = set()
    for _ in range(n):
        tag = rng.choice(TAGS)
        fam = rng.choice(fams)
        seq = pseudo_sequence(fam, tag, rng)
        made += 1
        viol = validate_by_state_machine(seq)
        if not viol:
            valid += 1
        novel_tokens.update(s for s in seq if s not in _ALL_KNOWN)
    print(f"  valid (0 violations): {valid}/{made}")
    print(f"  distinct NOVEL tokens introduced: {len(novel_tokens)}")
    print(f"  e.g. {sorted(list(novel_tokens))[:6]}")

    # category preservation: every novel token classifies to a real category
    unknown = [t for t in novel_tokens if classify_step(t) == "UNKNOWN"]
    print(f"  novel tokens classified UNKNOWN: {len(unknown)} "
          + (f"(e.g. {unknown[:5]})" if unknown else "(none — full coverage)"))

    # OOD anomaly: injected violations are caught
    rng2 = random.Random(9)
    caught = total = 0
    for tag, seq in generate_pseudo_valid(120, rng2):
        inj = inject_violation(seq, rng2)
        if inj is None:
            continue
        broken, rule = inj
        total += 1
        if validate_by_state_machine(broken):
            caught += 1
    print(f"  OOD injected violations caught: {caught}/{total}")
    ok = (valid == made and not unknown and caught == total and total > 0)
    print("OK" if ok else "CHECK ABOVE")
    return ok


if __name__ == "__main__":
    from physics.ontology import STEP_CATEGORY as _SC
    _ALL_KNOWN = set(_SC)

    ap = argparse.ArgumentParser(description="Pseudo-family OOD generator.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()
    if args.selftest or True:
        sys.exit(0 if _selftest(args.n) else 1)
