"""
physics/parameters.py — fab-parameter plausibility (an ADDITIVE capability).

IMPORTANT (spec compliance): this is NOT one of the 10 challenge rules and does
NOT change Task 1/2/3 or any submission format. The eval sequences are bare step
names with no parameters, so this never affects scoring. It is a deeper-
understanding / demo capability: having absorbed the real fab parameters from
`*_longdescription_parameters.csv`, the system can also sanity-check NUMERIC
process parameters (a wrong temperature, dose, energy, thickness) — the kind of
fault that is invisible to an order-only checker.

Bounds are deliberately GENEROUS, grounded physical limits (not tuned to the
data), so we only flag clear absurdities (e.g. THERMAL OXIDATION at 5000 °C,
a negative dose). Sources: standard semiconductor-fabrication ranges — see
SOURCES.md (Plummer/Deal/Griffin; Campbell; Sze; ion-implantation refs).

stdlib only.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── unicode normalisation (the CSVs use ×10¹³, °C, µm, cm⁻², etc.) ──────────
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def _normalise(text: str) -> str:
    t = text.translate(_SUP)
    t = t.replace("×10", "e").replace("x10", "e").replace("·", " ")
    t = t.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    t = t.replace("µ", "u").replace("μ", "u")
    return t


# quantity_type -> (regex of units, generous physical [min, max] in the unit's
# canonical form). Values outside -> flagged as implausible.
_NUMBER = r"(-?\d+(?:\.\d+)?(?:e-?\d+)?)"

# (unit pattern, quantity, to_canonical(value)->value, (lo, hi), canonical-unit)
_RULES = [
    (r"(?:deg\s*)?C\b|°\s*C",         "temperature", lambda v: v,            (0.0, 2000.0),   "C"),
    (r"keV\b",                         "implant_energy", lambda v: v,         (0.05, 1000.0),  "keV"),
    (r"MeV\b",                         "implant_energy", lambda v: v * 1000,  (0.05, 1000.0),  "keV"),
    (r"cm-?2\b|cm\^?-?2\b|/cm2\b",     "dose",        lambda v: v,            (1e9, 1e18),     "cm^-2"),
    # Thickness spans films (nm) AND whole wafers (~725 um) — bound generously so
    # we only flag absurdities (e.g. 900000 um). 'mm' is deliberately NOT parsed
    # as thickness: in these CSVs mm denotes wafer DIAMETER (200/300 mm), not a
    # process thickness, so treating it as thickness would false-positive.
    (r"nm\b",                          "thickness",   lambda v: v,            (0.05, 2_000_000.0), "nm"),
    (r"um\b|micron",                   "thickness",   lambda v: v * 1000,     (0.05, 2_000_000.0), "nm"),
    (r"mTorr\b",                       "pressure",    lambda v: v,            (0.01, 1e6),     "mTorr"),
    (r"Torr\b",                        "pressure",    lambda v: v * 1000,     (0.01, 1e6),     "mTorr"),
    (r"sccm\b",                        "gas_flow",    lambda v: v,            (0.1, 1e5),      "sccm"),
    (r"slm\b",                         "gas_flow",    lambda v: v * 1000,     (0.1, 1e5),      "sccm"),
    (r"rpm\b",                         "spin_speed",  lambda v: v,            (50.0, 12000.0), "rpm"),
    (r"W\b",                           "rf_power",    lambda v: v,            (1.0, 50000.0),  "W"),
    (r"mJ/cm2\b|mJ/cm\^?2\b",          "exposure_dose", lambda v: v,          (1.0, 5000.0),   "mJ/cm2"),
    (r"min\b",                         "time",        lambda v: v * 60,       (0.5, 1e5),      "s"),
    (r"\bs\b|sec\b",                   "time",        lambda v: v,            (0.5, 1e5),      "s"),
]


def extract_quantities(param_text: str) -> list[dict]:
    """Pull (value, unit, quantity_type, canonical_value) tuples from a fab
    parameter string. Best-effort; unrecognised tokens are ignored."""
    t = _normalise(param_text)
    out = []
    for unit_pat, qty, to_canon, (lo, hi), canon_unit in _RULES:
        for m in re.finditer(_NUMBER + r"\s*(?:" + unit_pat + r")", t):
            try:
                v = float(m.group(1))
            except ValueError:
                continue
            cv = to_canon(v)
            out.append({"quantity": qty, "raw_value": v, "unit": canon_unit,
                        "canonical_value": cv, "lo": lo, "hi": hi})
    return out


def check_parameters(param_text: str) -> list[dict]:
    """Return a list of IMPLAUSIBLE parameters found (empty = all plausible).
    Each: {quantity, value, unit, reason}."""
    flags = []
    for q in extract_quantities(param_text):
        cv = q["canonical_value"]
        if cv < q["lo"] or cv > q["hi"]:
            flags.append({
                "quantity": q["quantity"],
                "value": q["raw_value"],
                "canonical_value": cv,
                "unit": q["unit"],
                "reason": (f"{q['quantity']} {cv:g} {q['unit']} is outside the "
                           f"plausible range [{q['lo']:g}, {q['hi']:g}] {q['unit']}"),
            })
    return flags


def check_step(step: str) -> list[dict]:
    """Check the absorbed parameters for a known step (uses step_semantics)."""
    from physics import step_semantics as SEM
    flags = []
    for p in SEM.parameters(step):
        flags.extend(check_parameters(p))
    return flags


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    from physics import step_semantics as SEM

    # 1) Sanity: real absorbed parameters should be (almost) all plausible.
    total = flagged = 0
    examples = []
    for step, sem in SEM.STEP_SEMANTICS.items():
        for p in sem.parameters:
            total += 1
            f = check_parameters(p)
            if f:
                flagged += 1
                if len(examples) < 5:
                    examples.append((step, p, f[0]["reason"]))
    print(f"Real fab parameters checked: {total}, flagged implausible: {flagged}")
    for step, p, why in examples:
        print(f"  [check] {step}: {p}  -> {why}")

    # 2) Extraction demo
    print("\nExtraction examples:")
    for s in ["Boron; 150 keV; dose 2×10¹³ cm⁻²; tilt 7°",
              "Dry O₂ furnace; 1000 °C; O₂ 5 slm; thickness 50 nm",
              "LPCVD SiH₄ 200 sccm; 620 °C; 300 mTorr; thickness 200 nm"]:
        qs = extract_quantities(s)
        print(f"  '{s[:48]}...' -> " + ", ".join(f"{q['quantity']}={q['canonical_value']:g}{q['unit']}" for q in qs))

    # 3) Absurd values MUST be flagged
    print("\nAbsurd-value detection (must flag):")
    for bad in ["THERMAL OXIDATION at 5000 °C; thickness 50 nm",
                "IMPLANT; Boron; -40 keV; dose 2×10¹³ cm⁻²",
                "DEPOSIT; thickness 900000 um"]:
        f = check_parameters(bad)
        print(f"  '{bad}' -> {[x['reason'] for x in f] or 'NOT FLAGGED (!)'}")
