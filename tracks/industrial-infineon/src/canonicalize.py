"""
Canonicalize synonym step names to reduce ambiguity.

Many process steps have interchangeable names (e.g. "STRIP PHOTORESIST" vs
"STRIP RESIST"). This module maps them to a single canonical form so the
model doesn't waste capacity guessing which synonym was used.

This is applied BEFORE tokenization and BEFORE evaluation, so both training
data and predictions use the same canonical vocabulary.
"""

# Map: variant -> canonical form
# The canonical form is the first (most descriptive) variant.
SYNONYM_MAP = {
    # Strip
    "STRIP RESIST": "STRIP PHOTORESIST",
    "STRIP RESIST LEVEL 2": "STRIP PHOTORESIST",
    # RCA cleans
    "WET CLEAN RCA1": "RCA CLEAN 1",
    "WET CLEAN RCA2": "RCA CLEAN 2",
    # Inspection
    "PRE CLEAN INSPECTION": "INITIAL WAFER INSPECTION",
    # Measurements
    "MEASURE INITIAL GEOMETRY": "MEASURE INITIAL THICKNESS",
    "MEASURE DIELECTRIC THICKNESS": "MEASURE FILM THICKNESS",
    "MEASURE SURFACE PLANARITY": "MEASURE PLANARITY",
    "MEASURE VIA RESISTANCE": "MEASURE CONTACT RESISTANCE",
    "MEASURE WAFER THICKNESS": "MEASURE THICKNESS",
    "MEASURE PASSIVATION QUALITY": "MEASURE PASSIVATION THICKNESS",
    "MEASURE OXIDE THICKNESS": "MEASURE FILM THICKNESS",
    # Deposition
    "DEPOSIT INTERLEVEL DIELECTRIC": "DEPOSIT INTERLAYER DIELECTRIC",
    "DEPOSIT PASSIVATION LAYER": "DEPOSIT PASSIVATION",
    "DEPOSIT TOP METAL": "DEPOSIT METAL 1",
    "DEPOSIT TUNGSTEN SEED": "DEPOSIT METAL SEED",
    # CMP
    "CMP INTERLAYER DIELECTRIC": "CMP DIELECTRIC",
    "CMP VIA FILL": "CMP METAL",
    # Densify
    "DENSIFY OXIDE": "DENSIFY DIELECTRIC",
    # Anneal
    "ANNEAL METAL 1": "ANNEAL METAL",
    "ANNEAL POLYSILICON": "POLYSILICON ANNEAL",
    # Via
    "VIA ETCH THROUGH DIELECTRIC": "VIA ETCH",
    "DIELECTRIC ETCH VIA": "VIA ETCH",
    "FILL VIA TUNGSTEN": "FILL VIA METAL",
    "VIA OPENING INSPECTION": "VIA INSPECTION",
    # Passivation
    "OPEN BOND PAD WINDOW": "OPEN PAD WINDOW",
    "OPEN PAD WINDOW LITHO": "PAD WINDOW LITHO",
    "DEVELOP PAD WINDOW": "DEVELOP PHOTORESIST",
    "PASSIVATION ETCH": "PASSIVATION ETCH PAD OPENING",
    # Surface
    "MEASURE SURFACE DEFECTS": "MEASURE SURFACE PARTICLES",
    # Lot
    "FINAL LOT RELEASE": "LOT RELEASE",
    # Backside
    "BACKSIDE CLEAN FINAL": "BACKSIDE CLEAN",
    # Electrical
    "ELECTRICAL PARAMETRIC TEST": "PARAMETRIC TEST",
    # Pre-process clean
    "WAFER CLEAN PRE PROCESS": "PRE CLEAN WAFER",
    # Pattern inspection variants (keep level-specific ones, merge generic)
    "PATTERN INSPECTION LEVEL 1": "INSPECT PATTERN LEVEL 1",
    "PATTERN INSPECTION LEVEL 2": "INSPECT PATTERN LEVEL 2",
}


def canonicalize_step(step: str) -> str:
    """Map a step name to its canonical form."""
    return SYNONYM_MAP.get(step, step)


def canonicalize_sequence(steps: list[str]) -> list[str]:
    """Canonicalize all steps in a sequence."""
    return [canonicalize_step(s) for s in steps]
