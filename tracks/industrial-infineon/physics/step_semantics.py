"""
physics/step_semantics.py — absorb the descriptive + parametric CSVs.

The repo ships three layers of data per family:
  * *_variants.csv                    — step SEQUENCES (we use these for the grammar)
  * *_Longdescr.csv                   — STEP -> physical DESCRIPTION
  * *_longdescription_parameters.csv  — STEP -> DESCRIPTION + REALISTIC FAB PARAMETERS

Until now we only used the sequences. This module absorbs the other two layers —
the actual domain knowledge of *what each operation physically does* and the
*real fab-level numbers* (chemistries, temperatures, doses, energies,
thicknesses). It exposes that knowledge to:
  * the explainer (richer, physical "why"),
  * the training-data exporter (natural-language + parameter-augmented corpora
    so the trained model absorbs the physics, not just the token order),
  * the demo (show the system understands the process, not just the sequence).

A step can carry several descriptions/parameter sets (e.g. THERMAL OXIDATION is
used for both the 50 nm field oxide and the 6 nm gate oxide); we keep all of
them, tagged by family. Coverage gaps fall back to the category description in
the knowledge base, so nothing breaks for unseen step names.

stdlib only.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).parent
_DATA = _HERE.parent / "training_data"
sys.path.insert(0, str(_HERE.parent))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class StepSemantics:
    step: str
    descriptions: list[str] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    families: set[str] = field(default_factory=set)

    def add(self, description: str, parameters: str, family: str) -> None:
        if description and description not in self.descriptions:
            self.descriptions.append(description)
        if parameters and parameters not in self.parameters:
            self.parameters.append(parameters)
        if family:
            self.families.add(family)

    @property
    def description(self) -> str:
        return self.descriptions[0] if self.descriptions else ""

    @property
    def example_parameters(self) -> str:
        return self.parameters[0] if self.parameters else ""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_LONGDESCR = {
    "MOSFET": "MOSFET_Longdescr.csv",
    "IGBT":   "IGBT_Longdescr.csv",
    "IC":     "IC_Longdescr.csv",
}
_PARAMS = {
    "MOSFET": "MOSFET_longdescription_parameters.csv",
    "IGBT":   "IGBT_longdescription_parameters.csv",
    "IC":     "IC_longdescription_parameters.csv",
}


def _norm(name: str) -> str:
    # strip BOM, quotes, whitespace; the param header uses a non-breaking hyphen.
    return name.lstrip("﻿").strip().strip('"').strip().upper().replace("‑", "-")


def _read(path: Path):
    """Yield (step, description, parameters) from a description/param CSV."""
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return
        cols = [_norm(h) for h in header]
        try:
            si = cols.index("STEP")
        except ValueError:
            return
        di = next((i for i, c in enumerate(cols) if c.startswith("DESCRIPTION")), None)
        pi = next((i for i, c in enumerate(cols) if c.startswith("REALISTIC")), None)
        for row in reader:
            if not row or len(row) <= si:
                continue
            step = row[si].strip().strip('"')
            desc = row[di].strip().strip('"') if di is not None and len(row) > di else ""
            par = row[pi].strip().strip('"') if pi is not None and len(row) > pi else ""
            if step:
                yield step, desc, par


def load_step_semantics(data_dir: Path | None = None) -> dict[str, StepSemantics]:
    """Build {STEP -> StepSemantics} from all six descriptive CSVs."""
    data_dir = Path(data_dir) if data_dir else _DATA
    table: dict[str, StepSemantics] = {}

    def ingest(path: Path, family: str):
        for step, desc, par in _read(path):
            table.setdefault(step, StepSemantics(step)).add(desc, par, family)

    for fam, fname in _LONGDESCR.items():
        ingest(data_dir / fname, fam)
    for fam, fname in _PARAMS.items():
        ingest(data_dir / fname, fam)
    return table


# Module-level cache (loaded once).
STEP_SEMANTICS: dict[str, StepSemantics] = load_step_semantics()


# ---------------------------------------------------------------------------
# Public accessors (with graceful category fallback)
# ---------------------------------------------------------------------------

def _best_synonym(step: str) -> str:
    """Find the closest documented step in the SAME category by token overlap
    (resolves grammar synonyms like 'DEPOSIT INTERLEVEL DIELECTRIC' ~
    'DEPOSIT INTERLAYER DIELECTRIC', 'METAL ETCH DRY' ~ 'METAL ETCH')."""
    try:
        from physics.ontology import classify_step
    except Exception:
        return ""
    cat = classify_step(step)
    toks = set(step.split())
    best, best_score = "", 0.0
    for cand, sem in STEP_SEMANTICS.items():
        if not sem.description or classify_step(cand) != cat:
            continue
        ct = set(cand.split())
        inter = len(toks & ct)
        if not inter:
            continue
        jacc = inter / len(toks | ct)
        if jacc > best_score:
            best, best_score = cand, jacc
    return STEP_SEMANTICS[best].description if best_score >= 0.5 else ""


def describe(step: str) -> str:
    """Best available physical description for a step. Tries: exact -> closest
    same-category synonym -> category-level blurb from the knowledge base."""
    sem = STEP_SEMANTICS.get(step)
    if sem and sem.description:
        return sem.description
    syn = _best_synonym(step)
    if syn:
        return syn
    # fallback: category description
    try:
        from physics.ontology import classify_step
        from physics.process_knowledge import EVENT_CLASSES
        cat = classify_step(step)
        # map a few categories to event-class blurbs where available
        for ec in EVENT_CLASSES.values():
            if cat in ec.unknown_categories:
                return f"({cat.lower()}) {ec.description}"
        return f"({cat.lower()} operation)"
    except Exception:
        return ""


def parameters(step: str) -> list[str]:
    """All realistic fab parameter strings recorded for a step (may be empty)."""
    sem = STEP_SEMANTICS.get(step)
    return list(sem.parameters) if sem else []


def render_step(step: str, with_params: bool = True) -> str:
    """Render a step as a natural-language line for training / demo."""
    desc = describe(step)
    line = f"{step}: {desc}" if desc else step
    if with_params:
        ps = parameters(step)
        if ps:
            line += f"  [params: {ps[0]}]"
    return line


def coverage() -> dict:
    """Report how much of the known vocabulary has absorbed semantics."""
    try:
        from physics.ontology import STEP_CATEGORY
        vocab = set(STEP_CATEGORY)
    except Exception:
        vocab = set()
    have = set(STEP_SEMANTICS)
    with_params = {s for s, sem in STEP_SEMANTICS.items() if sem.parameters}
    covered = vocab & have
    return {
        "vocab_size": len(vocab),
        "steps_with_semantics": len(have),
        "vocab_covered": len(covered),
        "vocab_missing": sorted(vocab - have),
        "steps_with_parameters": len(with_params),
    }


# ---------------------------------------------------------------------------
# Self-test / report
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    cov = coverage()
    print("Step-semantics absorption report")
    print("-" * 55)
    print(f"  steps with descriptions/params : {cov['steps_with_semantics']}")
    print(f"  steps with fab parameters       : {cov['steps_with_parameters']}")
    print(f"  known-vocab steps covered       : {cov['vocab_covered']}/{cov['vocab_size']}")
    if cov["vocab_missing"]:
        print(f"  vocab missing direct semantics ({len(cov['vocab_missing'])}): "
              f"{', '.join(cov['vocab_missing'][:12])}"
              + (" …" if len(cov['vocab_missing']) > 12 else ""))
        print("  (these fall back to category-level descriptions)")

    print("\n  Examples of absorbed knowledge:")
    for step in ("THERMAL OXIDATION", "IMPLANT WELL", "VIA ETCH",
                 "DEPOSIT BARRIER METAL", "CURE PASSIVATION"):
        sem = STEP_SEMANTICS.get(step)
        if sem:
            print(f"\n  {step}  (families: {', '.join(sorted(sem.families))})")
            for d in sem.descriptions[:2]:
                print(f"    desc : {d}")
            for p in sem.parameters[:2]:
                print(f"    param: {p}")
