"""
physics/process_knowledge.py — THE declarative knowledge base.

This module is the single, visible, human-readable source of truth for *how
the wafer process works* and *why every rule is a physical necessity*. Nothing
here is procedural: it is data — state variables, event classes, and rules —
that a generic engine (physics/state_machine.py) interprets. Add a step, a
family, or a whole new rule by editing data here; you never touch control flow.

The four things this file encodes
---------------------------------
1. STATE_VARIABLES  — the physical state of a wafer: what properties exist, and
   *why each one matters*. This is the vocabulary the process reasons in.
2. EVENT_CLASSES    — physically meaningful classes of operation (a deposition,
   a patterned etch, a clean, …). Membership is HYBRID: exact reference
   vocabulary for the ~120 known steps, physical-category reasoning for unknown
   (4th-family) steps. This is what makes the understanding transfer.
3. RULES            — every forbidden situation, expressed as
   (trigger event) needs (enabling event within a window)  OR
   (trigger event) needs (a milestone already reached),
   each carrying the CAUSAL reason it exists. The 10 challenge rules are
   *derived* from these declarations, not hand-coded.
4. PROCESS_FLOW     — the canonical fabrication narrative (receive → clean →
   … → ship), so the system has an explicit account of the whole process.

Generalisation principle
------------------------
A rule never names a step. It names an *event class*. An unknown step from the
4th family is mapped to an event class by physical category, so the same rule —
and the same causal explanation — applies to chips never seen in training.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

# Make the repo root + training_data importable when run standalone.
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "training_data"))

from physics.ontology import (
    classify_step,
    step_physics_vector,
    STEP_CATEGORY,
    CAT_DEPOSIT, CAT_ETCH, CAT_IMPLANT, CAT_CMP, CAT_FILL,
    CAT_CLEAN, CAT_PREP, CAT_ANNEAL,
)

# Authoritative reference vocabulary for KNOWN steps. Unknown steps fall back to
# physical category. If generate_sequences is unavailable, everything uses
# category reasoning (still correct, slightly more permissive).
try:
    from generate_sequences import (
        DEPOSITION_STEPS as _DEP, CLEAN_STEPS as _CLEAN, ETCH_STEPS as _ETCH,
        IMPLANT_STEPS as _IMPLANT, IMPLANT_OPENER_STEPS as _IMPL_OPEN,
        CMP_STEPS as _CMP, FILL_STEPS as _FILL, PAD_WINDOW_STEPS as _PADS,
        ELECTRICAL_TEST_STEPS as _ETESTS,
    )
    HAVE_REFERENCE = True
except Exception:  # pragma: no cover
    _DEP = _CLEAN = _ETCH = _IMPLANT = _IMPL_OPEN = frozenset()
    _CMP = _FILL = _PADS = _ETESTS = frozenset()
    HAVE_REFERENCE = False


# ===========================================================================
# 1. STATE VARIABLES — the physical state of a wafer, and why each matters
# ===========================================================================

@dataclass(frozen=True)
class StateVariable:
    name: str
    description: str
    why_it_matters: str


STATE_VARIABLES: list[StateVariable] = [
    StateVariable(
        "surface_cleanliness",
        "Whether the top surface is chemically clean and well-defined (set by a "
        "clean/prep/anneal/CMP/thermal-oxidation step; decays as later steps add "
        "contamination).",
        "Thin-film growth nucleates on the existing surface. Contaminants get "
        "buried in the film as defects, so a clean surface must exist shortly "
        "before any deposition.",
    ),
    StateVariable(
        "resist_pattern",
        "The photoresist lifecycle: coated → exposed → developed (a physical "
        "mask with openings).",
        "Etchants and implants act everywhere they can reach. A developed resist "
        "pattern is the physical stencil that decides where material is removed "
        "or doped.",
    ),
    StateVariable(
        "mask_level",
        "The highest lithography mask level aligned so far.",
        "Each level's features register to structures built by the previous "
        "level. Levels must advance in order or the geometry mis-registers.",
    ),
    StateVariable(
        "implant_window",
        "Whether an opening exists (oxide etch or developed resist) through which "
        "ions can reach the substrate.",
        "Oxide and resist block implant ions. Doping only lands where a window "
        "was recently opened.",
    ),
    StateVariable(
        "overburden",
        "Whether excess material was recently deposited or filled above the "
        "target plane.",
        "CMP polishes material down to a plane. With nothing deposited above it, "
        "CMP grinds into the structure instead of levelling overburden.",
    ),
    StateVariable(
        "passivation",
        "The protective top layer's lifecycle: deposited → cured.",
        "The final passivation seals the device. Pad-window opening, electrical "
        "test, and backside work all require it to exist and be cured first.",
    ),
    StateVariable(
        "sort_tested",
        "Whether wafer sort test has screened the dice.",
        "Shipping before sort sends untested, possibly defective product to the "
        "customer.",
    ),
]


# ===========================================================================
# 2. EVENT CLASSES — physically meaningful operations (hybrid membership)
# ===========================================================================

@dataclass(frozen=True)
class EventClass:
    """
    A class of operation. A step belongs to it if:
      * its name starts with one of `prefixes` (checked always — handles dynamic
        steps like 'EXPOSE LITHO LEVEL N'), OR
      * the step is KNOWN (documented) and is in `known_steps`, OR
      * the step is UNKNOWN (4th-family) and its physical category is in
        `unknown_categories`, or a physics-vector flag in `unknown_flags` is
        set, or it matches any class in `unknown_union`.
    """
    name: str
    description: str
    known_steps: frozenset = frozenset()
    prefixes: tuple = ()
    unknown_categories: frozenset = frozenset()
    unknown_flags: tuple = ()
    unknown_union: tuple = ()
    # OOD-only AND-keyword groups: an UNKNOWN (4th-family) step matches if, for
    # any group, ALL words in the group appear as WHOLE WORDS in the step name
    # (whole-word, so 'PAD' does NOT match 'PADDLE'), AND the step's physical
    # category is in `unknown_keyword_cats` (so a measurement/clean/etch that
    # merely shares a word is NOT mistaken for the operation). Used for ops whose
    # category alone is ambiguous (pad-window open, backside metal, probe).
    unknown_keywords: tuple = ()
    unknown_keyword_cats: frozenset = frozenset()


EVENT_CLASSES: dict[str, EventClass] = {
    # ── Triggers (operations that have preconditions) ────────────────────────
    "DEPOSITION": EventClass(
        "DEPOSITION", "Grows or deposits a new material layer.",
        known_steps=_DEP, unknown_categories=frozenset({CAT_DEPOSIT})),
    "PATTERNED_ETCH": EventClass(
        "PATTERNED_ETCH", "Removes material through a resist pattern.",
        known_steps=_ETCH, unknown_categories=frozenset({CAT_ETCH})),
    "METAL_ETCH": EventClass(
        "METAL_ETCH", "Patterns the metal interconnect (needs full lithography).",
        known_steps=frozenset({"METAL ETCH", "METAL ETCH DRY"}),
        unknown_keywords=(("METAL", "ETCH"),),
        unknown_keyword_cats=frozenset({"ETCH"})),
    "IMPLANT": EventClass(
        "IMPLANT", "Drives dopant ions into the substrate.",
        known_steps=_IMPLANT, unknown_categories=frozenset({CAT_IMPLANT})),
    "CMP": EventClass(
        "CMP", "Chemical-mechanical planarisation.",
        known_steps=_CMP, unknown_categories=frozenset({CAT_CMP})),
    "PAD_WINDOW_OPEN": EventClass(
        "PAD_WINDOW_OPEN", "Opens a window to the bond pads through passivation.",
        known_steps=_PADS,
        unknown_keywords=(("PAD", "OPEN"), ("PAD", "WINDOW"), ("BOND", "PAD")),
        unknown_keyword_cats=frozenset({"ETCH", "LITHO", "DEPOSIT", "UNKNOWN"})),
    "ELECTRICAL_TEST": EventClass(
        "ELECTRICAL_TEST", "Probe-based electrical characterisation.",
        known_steps=_ETESTS,
        unknown_categories=frozenset({"TEST"}),
        # PROBE only counts as a test if it isn't clearly a clean/align/etc.
        unknown_keywords=(("PROBE",),),
        unknown_keyword_cats=frozenset({"TEST", "UNKNOWN"})),
    "SHIP": EventClass(
        "SHIP", "Releases the lot to the customer.",
        known_steps=frozenset({"SHIP LOT"}),
        prefixes=("SHIP", "DISPATCH")),   # OOD: no other step starts with these
    "BACKSIDE_METAL": EventClass(
        "BACKSIDE_METAL", "Deposits the backside metal contact.",
        known_steps=frozenset({"DEPOSIT BACKSIDE METAL"}),
        unknown_keywords=(("BACKSIDE", "METAL"), ("REAR", "METAL"),
                          ("BACKSIDE", "CONTACT"), ("REAR", "CONTACT")),
        # only the DEPOSITION of backside metal — not measuring/etching/inspecting it
        unknown_keyword_cats=frozenset({"DEPOSIT"})),

    # ── Enablers (operations that satisfy a precondition) ────────────────────
    "CLEAN_SURFACE": EventClass(
        "CLEAN_SURFACE", "Leaves a deposition-ready surface.",
        known_steps=_CLEAN,
        unknown_categories=frozenset({CAT_CLEAN, CAT_PREP, CAT_ANNEAL, CAT_CMP})),
    "DEVELOP": EventClass(
        "DEVELOP", "Develops the resist, creating the physical mask pattern.",
        known_steps=frozenset({"DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"}),
        unknown_flags=("develops_resist",)),
    "EXPOSE": EventClass(
        "EXPOSE", "UV-exposes the resist, writing the latent image.",
        prefixes=("EXPOSE LITHO LEVEL",), unknown_flags=("exposes_resist",)),
    "IMPLANT_OPENER": EventClass(
        "IMPLANT_OPENER", "Opens an implant window (oxide etch or develop).",
        known_steps=_IMPL_OPEN, unknown_union=("PATTERNED_ETCH", "DEVELOP")),
    "DEPOSIT_OR_FILL": EventClass(
        "DEPOSIT_OR_FILL", "Leaves overburden for CMP to planarise.",
        known_steps=_FILL,
        unknown_categories=frozenset({CAT_DEPOSIT, CAT_FILL})),
}


def step_in_event(step: str, event_name: str) -> bool:
    """True if `step` belongs to the named event class (hybrid membership)."""
    ec = EVENT_CLASSES[event_name]
    up = step.upper()
    if any(up.startswith(p) for p in ec.prefixes):
        return True
    if HAVE_REFERENCE and step in STEP_CATEGORY:           # KNOWN step
        return step in ec.known_steps
    # UNKNOWN (4th-family) step → physical reasoning
    cat = classify_step(step)
    if cat in ec.unknown_categories:
        return True
    if ec.unknown_keywords and (not ec.unknown_keyword_cats or cat in ec.unknown_keyword_cats):
        words = set(up.split())                      # WHOLE-word match (PAD != PADDLE)
        for grp in ec.unknown_keywords:
            if all(k in words for k in grp):
                return True
    if ec.unknown_flags:
        pv = step_physics_vector(step)
        if any(pv.get(f) for f in ec.unknown_flags):
            return True
    return any(step_in_event(step, u) for u in ec.unknown_union)


# ===========================================================================
# 3. RULES — every forbidden situation, with its causal reason
# ===========================================================================

@dataclass(frozen=True)
class WindowedRule:
    """`trigger` needs each (enabler, window) to have occurred within `window`
    prior steps. Fires (once) if any required enabler is missing."""
    id: str
    name: str
    trigger: str
    requires: tuple              # ((enabler_event, window), ...)
    plain: str                   # short, plain-English statement
    physical_reason: str         # the causal "why"


@dataclass(frozen=True)
class OrderingRule:
    """`trigger` may only occur after each named milestone is reached. Fires
    (once) on the first unmet milestone, citing its specific reason."""
    id: str
    name: str
    trigger: str
    requires: tuple              # ((milestone_flag, reason_if_missing), ...)
    plain: str
    physical_reason: str


WINDOWED_RULES: list[WindowedRule] = [
    WindowedRule(
        "RULE_DEP_NO_CLEAN", "Deposition needs a clean surface",
        "DEPOSITION", (("CLEAN_SURFACE", 12),),
        "A deposition has no cleaning step in the prior 12 steps.",
        "Thin-film deposition nucleates on the existing surface; contamination "
        "from a prior etch or handling becomes buried defects. A clean, "
        "well-defined surface must exist shortly before any deposition. This is "
        "universal to all thin-film processes, which is why it transfers to "
        "unseen families."),
    WindowedRule(
        "RULE_METAL_ETCH_NO_LITHO", "Metal etch needs full lithography",
        "METAL_ETCH", (("EXPOSE", 15), ("DEVELOP", 15)),
        "A metal etch is missing its EXPOSE and/or DEVELOP within 15 steps.",
        "Metal patterning needs an exact resist image: exposure writes the "
        "latent image, development turns it into a physical mask. Both must be "
        "present and recent, or the etch clears the whole metal layer."),
    WindowedRule(
        "RULE_ETCH_NO_MASK", "Patterned etch needs a developed mask",
        "PATTERNED_ETCH", (("DEVELOP", 12),),
        "A patterned etch has no DEVELOP in the prior 12 steps.",
        "Etchants attack every exposed surface. Developed resist physically "
        "shields what must survive. Without it the etch removes material "
        "uniformly and no device geometry is defined."),
    WindowedRule(
        "RULE_IMPLANT_NO_MASK", "Implant needs an open window",
        "IMPLANT", (("IMPLANT_OPENER", 15),),
        "An implant has no mask opening (oxide etch or develop) within 15 steps.",
        "Oxide and resist block implant ions; doping only reaches the substrate "
        "through a recently opened window. Without one, doping is misplaced or "
        "absent."),
    WindowedRule(
        "RULE_CMP_NO_DEP", "CMP needs something to planarise",
        "CMP", (("DEPOSIT_OR_FILL", 6),),
        "A CMP step has no deposition or fill in the prior 6 steps.",
        "CMP polishes material down to a target plane. With no overburden it "
        "grinds into the underlying structure."),
]

ORDERING_RULES: list[OrderingRule] = [
    OrderingRule(
        "RULE_PAD_OPEN_BEFORE_DEP", "Pad window needs deposited+cured passivation",
        "PAD_WINDOW_OPEN", (
            ("passivation_deposited",
             "Pad window opened before passivation was deposited — there is no "
             "passivation layer to open a window in."),
            ("passivation_cured",
             "Pad window opened before passivation was cured — uncured "
             "passivation cannot hold the pad-etch pattern."),
        ),
        "A pad window is opened before passivation is deposited and cured.",
        "The pad-window etch opens access to the bond pads through the "
        "passivation. The layer must exist and be cross-linked (cured) first, or "
        "the etch attacks the metal/dielectric beneath."),
    OrderingRule(
        "RULE_TEST_BEFORE_PASSIVATION", "Electrical test needs cured passivation",
        "ELECTRICAL_TEST", (
            ("passivation_cured",
             "Electrical test before passivation was cured — probing unprotected "
             "devices damages them and the result is meaningless."),
        ),
        "An electrical test runs before passivation is cured.",
        "Probe needles contact the pads; without cured passivation sealing the "
        "interconnects and active areas, probing causes contamination, leakage "
        "and irreversible damage."),
    OrderingRule(
        "RULE_SHIP_BEFORE_TEST", "Ship needs sort test",
        "SHIP", (
            ("sort_test_done",
             "SHIP LOT before WAFER SORT TEST — lots must pass sort before "
             "shipping."),
        ),
        "The lot ships before wafer sort test.",
        "Sort test screens defective dice. Shipping first sends untested, "
        "possibly defective product to the customer."),
    OrderingRule(
        "RULE_BACKSIDE_BEFORE_PASSIVATION", "Backside metal needs cured passivation",
        "BACKSIDE_METAL", (
            ("passivation_cured",
             "DEPOSIT BACKSIDE METAL before passivation was cured — the "
             "frontside must be sealed before backside processing."),
        ),
        "Backside metal is deposited before passivation is cured.",
        "Backside metallisation subjects the wafer to reactive sputtering, "
        "thermal stress and handling; without cured passivation the finished "
        "front-side devices delaminate, crack, or get contaminated."),
]

# RULE_LITHO_LEVEL_SKIP is handled specially by the engine (it compares numeric
# levels rather than checking an enabler/milestone), but its knowledge lives
# here so the KB stays the single source of truth.
LITHO_RULE = {
    "id": "RULE_LITHO_LEVEL_SKIP",
    "name": "Mask levels advance in order",
    "plain": "A mask level is skipped or decreases.",
    "physical_reason": (
        "Each lithography level patterns features that register to structures "
        "built by the previous level. Skipping a level means those structures "
        "and alignment marks were never created; decreasing a level would "
        "overwrite completed structures."),
}


# ===========================================================================
# Milestones — one-way state flags, and which steps set them
# ===========================================================================

@dataclass(frozen=True)
class Milestone:
    flag: str
    set_by: frozenset
    description: str


MILESTONES: list[Milestone] = [
    Milestone("passivation_deposited",
              frozenset({"DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"}),
              "Protective passivation layer has been deposited."),
    Milestone("passivation_cured", frozenset({"CURE PASSIVATION"}),
              "Passivation has been cured (cross-linked, mechanically stable)."),
    Milestone("sort_test_done", frozenset({"WAFER SORT TEST"}),
              "Wafer sort test has screened the dice."),
]


# Enabler events whose most-recent index the engine must remember (derived).
ENABLER_EVENTS: tuple = tuple(sorted({
    ev for rule in WINDOWED_RULES for ev, _w in rule.requires
}))


# ===========================================================================
# 4. PROCESS FLOW — the canonical fabrication narrative
# ===========================================================================

PROCESS_FLOW: list[tuple[str, str]] = [
    ("Incoming", "Receive the lot, identify it, inspect and measure the bare "
                 "wafer."),
    ("Pre-clean", "Wet/chemical cleans (RCA, HF dip) remove organics and native "
                  "oxide — the wafer must be clean before anything is grown."),
    ("Substrate prep", "Family-specific: epitaxy (MOSFET/IGBT) or backside "
                       "grind/clean (IC) to set up the starting material."),
    ("First oxidation", "Grow the first thermal oxide; it both protects the "
                        "surface and provides a clean base for what follows."),
    ("Litho–etch–implant cycles", "The heart of the process: for each device "
                                  "region, coat→align→expose→develop resist, "
                                  "etch the opening, strip, clean, implant "
                                  "dopants, anneal. Repeated per mask level in "
                                  "ascending order."),
    ("Interlayer dielectric", "Deposit dielectric, densify, CMP flat — the "
                              "insulating layer between devices and wiring."),
    ("Vias", "Litho + etch contact holes through the dielectric, then fill with "
             "barrier/seed/metal and CMP."),
    ("Metallisation", "Deposit metal, pattern it with full lithography, etch, "
                      "strip, clean — the interconnect wiring."),
    ("Passivation", "Deposit and cure the protective top layer, then open "
                    "windows to the bond pads."),
    ("Backside", "Thin the wafer and form the backside contact — only after the "
                 "front is sealed."),
    ("Test & ship", "Final clean and inspection, electrical + sort test, then "
                    "release and ship — never before sort."),
]


# Convenience: all rule objects in one place (for documentation / iteration).
ALL_RULE_IDS: tuple = tuple(
    [r.id for r in WINDOWED_RULES] + [LITHO_RULE["id"]] + [r.id for r in ORDERING_RULES]
)


# ===========================================================================
# Causal dependency graph — derived from the rules (for the demo / explanations)
# ===========================================================================

def causal_edges() -> list[dict]:
    """Every process-logic dependency, derived from the rules: an enabling event
    (or milestone) that a triggering operation depends on, with the rule and the
    physical reason. This IS the 'process logic' the model must learn."""
    edges = []
    for r in WINDOWED_RULES:
        for enabler, window in r.requires:
            edges.append({"from": enabler, "to": r.trigger, "rule": r.id,
                          "window": window, "kind": "needs-within",
                          "why": r.physical_reason})
    for r in ORDERING_RULES:
        for flag, _reason in r.requires:
            edges.append({"from": flag, "to": r.trigger, "rule": r.id,
                          "window": None, "kind": "needs-before",
                          "why": r.physical_reason})
    return edges


def to_mermaid() -> str:
    """Render the dependency graph as a Mermaid flowchart (renders on GitHub)."""
    lines = ["flowchart LR"]
    for e in causal_edges():
        label = e["rule"].replace("RULE_", "")
        if e["window"]:
            label += f" ≤{e['window']}"
        lines.append(f'  {e["from"]}["{e["from"]}"] -->|{label}| {e["to"]}["{e["to"]}"]')
    return "\n".join(lines)


# ===========================================================================
# Human-readable export — render the whole KB as browsable documentation
# ===========================================================================

def to_markdown() -> str:
    """Render the entire knowledge base as a Markdown document."""
    L: list[str] = []
    L.append("# Process Knowledge Model\n")
    L.append("_Auto-generated from `physics/process_knowledge.py` — the single "
             "source of truth. Edit the data there, regenerate this._\n")

    L.append("## How the process works\n")
    for i, (phase, desc) in enumerate(PROCESS_FLOW, 1):
        L.append(f"{i}. **{phase}** — {desc}")
    L.append("")

    L.append("## Wafer state variables (what the process reasons about)\n")
    for sv in STATE_VARIABLES:
        L.append(f"### `{sv.name}`")
        L.append(f"- **What:** {sv.description}")
        L.append(f"- **Why it matters:** {sv.why_it_matters}\n")

    L.append("## Operation (event) classes\n")
    L.append("Membership is hybrid: exact reference vocabulary for known steps, "
             "physical category for unknown 4th-family steps.\n")
    for ec in EVENT_CLASSES.values():
        bits = []
        if ec.known_steps:
            bits.append(f"{len(ec.known_steps)} known steps")
        if ec.prefixes:
            bits.append("prefixes " + ", ".join(ec.prefixes))
        if ec.unknown_categories:
            bits.append("OOD categories {" + ", ".join(sorted(ec.unknown_categories)) + "}")
        if ec.unknown_flags:
            bits.append("OOD flags " + ", ".join(ec.unknown_flags))
        if ec.unknown_union:
            bits.append("OOD union of " + ", ".join(ec.unknown_union))
        L.append(f"- **{ec.name}** — {ec.description}  \n  _matched by:_ {'; '.join(bits)}")
    L.append("")

    L.append("## Rules (why each violation is impossible)\n")
    L.append("### Windowed rules — a trigger needs an enabler within N steps\n")
    for r in WINDOWED_RULES:
        reqs = ", ".join(f"{e} within {w}" for e, w in r.requires)
        L.append(f"#### {r.id} — {r.name}")
        L.append(f"- **Trigger:** {r.trigger}")
        L.append(f"- **Requires:** {reqs}")
        L.append(f"- **Plain:** {r.plain}")
        L.append(f"- **Why:** {r.physical_reason}\n")

    L.append("### Litho-level rule (numeric ordering)\n")
    L.append(f"#### {LITHO_RULE['id']} — {LITHO_RULE['name']}")
    L.append(f"- **Plain:** {LITHO_RULE['plain']}")
    L.append(f"- **Why:** {LITHO_RULE['physical_reason']}\n")

    L.append("### Ordering rules — a trigger needs a milestone first\n")
    for r in ORDERING_RULES:
        L.append(f"#### {r.id} — {r.name}")
        L.append(f"- **Trigger:** {r.trigger}")
        L.append(f"- **Requires milestones:** {', '.join(f for f, _ in r.requires)}")
        L.append(f"- **Plain:** {r.plain}")
        L.append(f"- **Why:** {r.physical_reason}\n")

    L.append("## Milestones (one-way state flags)\n")
    for ms in MILESTONES:
        L.append(f"- **{ms.flag}** — {ms.description}  (set by: "
                 f"{', '.join(sorted(ms.set_by))})")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Process knowledge base.")
    ap.add_argument("--export", metavar="PATH",
                    help="write the KB as Markdown to PATH (default: print summary).")
    args = ap.parse_args()

    if args.export:
        Path(args.export).parent.mkdir(parents=True, exist_ok=True)
        Path(args.export).write_text(to_markdown(), encoding="utf-8")
        print(f"Knowledge base exported -> {args.export}")
    else:
        print(f"State variables : {len(STATE_VARIABLES)}")
        print(f"Event classes   : {len(EVENT_CLASSES)}")
        print(f"Rules           : {len(ALL_RULE_IDS)} -> {', '.join(ALL_RULE_IDS)}")
        print(f"Enabler events  : {', '.join(ENABLER_EVENTS)}")
        print(f"Milestones      : {', '.join(m.flag for m in MILESTONES)}")
        print("\nRun with --export PATH to write the full Markdown model.")
