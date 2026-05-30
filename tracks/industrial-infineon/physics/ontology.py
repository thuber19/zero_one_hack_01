"""
physics/ontology.py

Maps every known semiconductor process step to its physical functional
category, and computes a binary physics feature vector that captures what
each step REQUIRES (preconditions) and what it DOES (effects).

Three-tier classification:
  1. Exact lookup  — all ~120 known steps from MOSFET/IGBT/IC vocabulary
  2. Prefix match  — dynamic steps: "ALIGN MASK LEVEL N", "EXPOSE LITHO LEVEL N", ...
  3. Keyword match — unknown steps from a 4th or unseen family

The keyword fallback is why this generalises: a step named "GROW GAN BUFFER LAYER"
will be classified as DEPOSIT because it contains "GROW", and will therefore
inherit all deposition preconditions regardless of the specific material.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Category constants — the 14 functional categories of thin-film manufacturing.
# These are physically universal; they apply to any process that involves
# surface preparation, deposition, patterning, etching, and testing.
# ---------------------------------------------------------------------------

CAT_LOGISTICS       = "LOGISTICS"       # receive, release, ship — no state change
CAT_INSPECT         = "INSPECT"         # visual/inline quality checks — no state change
CAT_MEASURE         = "MEASURE"         # metrology — no material change
CAT_CLEAN           = "CLEAN"           # removes contamination; makes surface deposition-ready
CAT_PREP            = "PREP"            # surface conditioning / epitaxy prep (also cleans)
CAT_DEPOSIT         = "DEPOSIT"         # grows or deposits a new material layer
CAT_ANNEAL          = "ANNEAL"          # high-temp treatment; activates dopants; also cleans surface
CAT_LITHO           = "LITHO"           # photolithography steps (coat → expose → develop)
CAT_PATTERN_INSPECT = "PATTERN_INSPECT" # post-develop pattern verification
CAT_ETCH            = "ETCH"            # patterned material removal (requires resist mask)
CAT_ETCH_BLANKET    = "ETCH_BLANKET"    # blanket / non-patterned etch (no mask needed)
CAT_GRIND           = "GRIND"           # mechanical thinning / planarization prep
CAT_STRIP           = "STRIP"           # removes photoresist
CAT_IMPLANT         = "IMPLANT"         # ion implantation (requires open window)
CAT_CMP             = "CMP"             # chemical-mechanical planarization (requires deposit)
CAT_FILL            = "FILL"            # via fill (consumed by CMP)
CAT_TEST            = "TEST"            # electrical characterisation (requires passivation cured)

# ---------------------------------------------------------------------------
# Explicit step → category mapping.
# Every step that appears in generate_sequences.py's grammar is listed here.
# ---------------------------------------------------------------------------

STEP_CATEGORY: dict[str, str] = {

    # ── Logistics ────────────────────────────────────────────────────────────
    "RECEIVE WAFER LOT":           CAT_LOGISTICS,
    "LOT IDENTIFICATION":          CAT_LOGISTICS,
    "LOT RELEASE":                 CAT_LOGISTICS,
    "FINAL LOT RELEASE":           CAT_LOGISTICS,
    "SHIP LOT":                    CAT_LOGISTICS,
    "PACKAGE PREPARATION":         CAT_LOGISTICS,

    # ── Incoming / inline inspection ─────────────────────────────────────────
    "INITIAL WAFER INSPECTION":    CAT_INSPECT,
    "PRE CLEAN INSPECTION":        CAT_INSPECT,
    "SUBSTRATE CHECK":             CAT_INSPECT,
    "EPITAXIAL WAFER CHECK":       CAT_INSPECT,
    "EPITAXIAL REWORK CHECK":      CAT_INSPECT,
    "PRE ANNEAL CHECK":            CAT_INSPECT,
    "BACKSIDE THINNING CHECK":     CAT_INSPECT,
    "BACKSIDE METALLIZATION PREP": CAT_PREP,     # surface prep before backside metal

    # ── Metrology (non-destructive, no state change) ─────────────────────────
    "MEASURE THICKNESS":                CAT_MEASURE,
    "MEASURE INITIAL THICKNESS":        CAT_MEASURE,
    "MEASURE INITIAL GEOMETRY":         CAT_MEASURE,
    "MEASURE GEOMETRY":                 CAT_MEASURE,
    "MEASURE SURFACE PARTICLES":        CAT_MEASURE,
    "MEASURE SURFACE DEFECTS":          CAT_MEASURE,
    "MEASURE BACKSIDE ROUGHNESS":       CAT_MEASURE,
    "MEASURE EPITAXY THICKNESS":        CAT_MEASURE,
    "MEASURE RESISTIVITY":              CAT_MEASURE,
    "MEASURE OXIDE THICKNESS":          CAT_MEASURE,
    "MEASURE GATE OXIDE THICKNESS":     CAT_MEASURE,
    "MEASURE FILM THICKNESS":           CAT_MEASURE,
    "MEASURE DIELECTRIC THICKNESS":     CAT_MEASURE,
    "MEASURE POLY THICKNESS":           CAT_MEASURE,
    "MEASURE GATE CD":                  CAT_MEASURE,
    "MEASURE PLANARITY":                CAT_MEASURE,
    "MEASURE SURFACE PLANARITY":        CAT_MEASURE,
    "MEASURE OPENING CD":               CAT_MEASURE,
    "MEASURE WINDOW CD":                CAT_MEASURE,
    "MEASURE SURFACE UNIFORMITY":       CAT_MEASURE,
    "MEASURE LINE WIDTH":               CAT_MEASURE,
    "MEASURE VIA CD":                   CAT_MEASURE,
    "MEASURE VIA RESISTANCE":           CAT_MEASURE,
    "MEASURE CONTACT RESISTANCE":       CAT_MEASURE,
    "MEASURE BACKSIDE CONTACT":         CAT_MEASURE,
    "MEASURE SHEET RESISTANCE":         CAT_MEASURE,
    "MEASURE JUNCTION DEPTH":           CAT_MEASURE,
    "MEASURE JUNCTION PROFILE":         CAT_MEASURE,
    "MEASURE PASSIVATION THICKNESS":    CAT_MEASURE,
    "MEASURE PASSIVATION QUALITY":      CAT_MEASURE,
    "MEASURE PAD OPENING":              CAT_MEASURE,
    "MEASURE SPACER WIDTH":             CAT_MEASURE,
    "MEASURE DEVICE PARAMETER":         CAT_MEASURE,
    "MEASURE WAFER THICKNESS":          CAT_MEASURE,
    "MEASURE OXIDE QUALITY":            CAT_MEASURE,
    "MEASURE METAL THICKNESS":          CAT_MEASURE,
    # Final inspection steps
    "FINAL CLEAN":                      CAT_CLEAN,   # it IS a clean step
    "FINAL THICKNESS MEASURE":          CAT_MEASURE,
    "FINAL GEOMETRY CHECK":             CAT_MEASURE,
    "FINAL OXIDE CHECK":                CAT_MEASURE,
    "FINAL CD INSPECTION":              CAT_MEASURE,
    "FINAL PARTICLE INSPECTION":        CAT_MEASURE,
    "FINAL ELECTRICAL TEST PREP":       CAT_INSPECT,
    # YIELD ANALYSIS is a data-analysis / reporting step, not a probe-based
    # electrical test. It changes no wafer state and must NOT inherit the
    # "passivation must be cured" precondition of CAT_TEST, so it is INSPECT.
    "YIELD ANALYSIS":                   CAT_INSPECT,

    # ── Cleaning steps ───────────────────────────────────────────────────────
    # These create a clean, well-defined surface — mandatory before deposition.
    "PRE CLEAN WAFER":              CAT_CLEAN,
    "WAFER CLEAN PRE PROCESS":      CAT_CLEAN,
    "WAFER CLEAN PRE-GRIND":        CAT_CLEAN,
    "WAFER SURFACE CLEAN":          CAT_CLEAN,
    "BACKSIDE CLEAN":               CAT_CLEAN,
    "FRONTSIDE CLEAN":              CAT_CLEAN,
    "BACKSIDE CLEAN FINAL":         CAT_CLEAN,
    "FRONTSIDE CLEAN FINAL":        CAT_CLEAN,
    "WET CLEAN RCA1":               CAT_CLEAN,
    "WET CLEAN RCA2":               CAT_CLEAN,
    "RCA CLEAN 1":                  CAT_CLEAN,
    "RCA CLEAN 2":                  CAT_CLEAN,
    "HF DIP":                       CAT_CLEAN,   # strips native oxide → H-terminated
    "DRY WAFER":                    CAT_CLEAN,
    "DRY WAFER BACKSIDE":           CAT_CLEAN,
    "CLEAN AFTER ETCH":             CAT_CLEAN,   # post-etch clean — removes etch byproducts
    "CLEAN AFTER OXIDE ETCH":       CAT_CLEAN,
    "CLEAN AFTER POLY ETCH":        CAT_CLEAN,
    "CLEAN AFTER VIA ETCH":         CAT_CLEAN,
    "CLEAN AFTER METAL ETCH":       CAT_CLEAN,
    "CLEAN AFTER WINDOW ETCH":      CAT_CLEAN,
    "CLEAN AFTER FIELD ETCH":       CAT_CLEAN,
    "CLEAN PAD OPENING":            CAT_CLEAN,
    "BACKSIDE ETCH CLEAN":          CAT_CLEAN,
    "BACKSIDE RINSE":               CAT_CLEAN,
    "BACKSIDE DRY":                 CAT_CLEAN,
    "OXIDE STRIP":                  CAT_CLEAN,
    "RINSE WET WAFER_EDGE":         CAT_CLEAN,
    "SURFACE PREP FOR DEPOSITION":  CAT_CLEAN,  # IC-specific surface conditioning

    # ── Surface preparation (also cleans, but more specialised) ──────────────
    "EPITAXY PREP":                 CAT_PREP,
    "EPITAXIAL LAYER PREP":         CAT_PREP,
    "GATE OXIDE PREP":              CAT_PREP,   # conditions gate area; functionally a clean

    # ── Deposition steps ─────────────────────────────────────────────────────
    # REQUIRE a clean surface. CREATE a new material layer.
    "THERMAL OXIDATION":            CAT_DEPOSIT,  # grows SiO2; also leaves a clean surface
    "GATE OXIDE GROWTH":            CAT_DEPOSIT,
    "DEPOSIT PAD OXIDE":            CAT_DEPOSIT,
    "EPITAXIAL DEPOSITION":         CAT_DEPOSIT,
    "DEPOSIT POLYSILICON":          CAT_DEPOSIT,
    "DEPOSIT SPACER DIELECTRIC":    CAT_DEPOSIT,
    "DEPOSIT FIELD OXIDE":          CAT_DEPOSIT,
    "DEPOSIT GATE OXIDE OR DIELECTRIC": CAT_DEPOSIT,
    "DEPOSIT INTERLAYER DIELECTRIC":    CAT_DEPOSIT,
    "DEPOSIT INTERLEVEL DIELECTRIC":    CAT_DEPOSIT,
    "DEPOSIT BARRIER METAL":        CAT_DEPOSIT,
    "DEPOSIT METAL SEED":           CAT_DEPOSIT,
    "DEPOSIT TUNGSTEN SEED":        CAT_DEPOSIT,
    "DEPOSIT METAL 1":              CAT_DEPOSIT,
    "DEPOSIT TOP METAL":            CAT_DEPOSIT,
    "DEPOSIT BACKSIDE METAL":       CAT_DEPOSIT,  # additionally requires passivation cured
    "DEPOSIT PASSIVATION":          CAT_DEPOSIT,  # additionally marks passivation_deposited
    "DEPOSIT PASSIVATION LAYER":    CAT_DEPOSIT,
    "DEPOSIT BACKSIDE PROTECTION":  CAT_DEPOSIT,

    # ── Anneal / thermal treatment ───────────────────────────────────────────
    # High-temperature steps. Activate dopants. Leave a thermally passivated
    # surface that is also suitable for the next deposition.
    "ANNEAL OXIDE":                 CAT_ANNEAL,
    "POLYSILICON ANNEAL":           CAT_ANNEAL,
    "ANNEAL POLYSILICON":           CAT_ANNEAL,
    "ANNEAL DIELECTRIC":            CAT_ANNEAL,
    "ANNEAL METAL":                 CAT_ANNEAL,
    "ANNEAL METAL 1":               CAT_ANNEAL,
    "RAPID THERMAL ANNEAL":         CAT_ANNEAL,
    "LIGHT ANNEAL":                 CAT_ANNEAL,
    "DRIVE IN DIFFUSION":           CAT_ANNEAL,
    "EPITAXY ANNEAL":               CAT_ANNEAL,
    "BACKSIDE ANNEAL":              CAT_ANNEAL,
    "DENSIFY DIELECTRIC":           CAT_ANNEAL,
    "DENSIFY OXIDE":                CAT_ANNEAL,
    "CURE PASSIVATION":             CAT_ANNEAL,  # special: marks passivation_cured

    # ── Lithography ──────────────────────────────────────────────────────────
    # SPIN COAT → SOFT BAKE → ALIGN → EXPOSE → [POST EXPOSE BAKE] →
    # DEVELOP → [INSPECT] → [HARD BAKE]
    "SPIN COAT PHOTORESIST":        CAT_LITHO,
    "SOFT BAKE":                    CAT_LITHO,
    "POST EXPOSE BAKE":             CAT_LITHO,
    "HARD BAKE":                    CAT_LITHO,
    "DEVELOP PHOTORESIST":          CAT_LITHO,
    "DEVELOP PAD WINDOW":           CAT_LITHO,
    # Pad window opening is a litho sub-step in the passivation block
    "OPEN PAD WINDOW":              CAT_LITHO,
    "OPEN BOND PAD WINDOW":         CAT_LITHO,
    "PAD WINDOW LITHO":             CAT_LITHO,
    "OPEN PAD WINDOW LITHO":        CAT_LITHO,
    # Pattern inspection steps (post-develop, before etch)
    "POLY PATTERN INSPECTION":      CAT_PATTERN_INSPECT,
    "VIA INSPECTION":               CAT_PATTERN_INSPECT,
    "VIA OPENING INSPECTION":       CAT_PATTERN_INSPECT,
    "METAL PATTERN INSPECTION":     CAT_PATTERN_INSPECT,
    "P BODY WINDOW INSPECTION":     CAT_PATTERN_INSPECT,
    "FIELD PATTERN INSPECTION":     CAT_PATTERN_INSPECT,

    # ── Patterned etch (REQUIRE developed resist mask) ────────────────────────
    "OXIDE ETCH":                   CAT_ETCH,
    "OXIDE ETCH DRY":               CAT_ETCH,
    "POLYSILICON ETCH":             CAT_ETCH,
    "POLYSILICON ETCH DRY":         CAT_ETCH,
    "ETCH SILICON OR OXIDE WINDOW": CAT_ETCH,
    "FIELD OXIDE ETCH":             CAT_ETCH,
    "VIA ETCH":                     CAT_ETCH,
    "VIA ETCH THROUGH DIELECTRIC":  CAT_ETCH,
    "DIELECTRIC ETCH VIA":          CAT_ETCH,
    "METAL ETCH":                   CAT_ETCH,
    "METAL ETCH DRY":               CAT_ETCH,
    "PASSIVATION ETCH PAD OPENING": CAT_ETCH,
    "PASSIVATION ETCH":             CAT_ETCH,

    # ── Blanket etch (no resist mask needed) ─────────────────────────────────
    # ANISOTROPIC ETCH SPACER etches everything directionally — the spacer
    # geometry is defined by the deposition thickness, not a photomask.
    "ANISOTROPIC ETCH SPACER":      CAT_ETCH_BLANKET,
    "ETCH WET BACKSIDE":            CAT_ETCH_BLANKET,

    # ── Mechanical grinding / thinning ───────────────────────────────────────
    "GRINDING WAFER BACKSIDE":      CAT_GRIND,
    "BACKSIDE GRIND":               CAT_GRIND,

    # ── Photoresist strip ─────────────────────────────────────────────────────
    "STRIP PHOTORESIST":            CAT_STRIP,
    "STRIP RESIST":                 CAT_STRIP,

    # ── Ion implantation ─────────────────────────────────────────────────────
    # REQUIRE an open oxide/implant window.
    "IMPLANT WELL":                 CAT_IMPLANT,
    "IMPLANT SOURCE DRAIN":         CAT_IMPLANT,
    "IMPLANT SOURCE REGION":        CAT_IMPLANT,
    "IMPLANT LDD":                  CAT_IMPLANT,
    "IMPLANT P BODY":               CAT_IMPLANT,
    "IMPLANT N BUFFER":             CAT_IMPLANT,
    "IMPLANT CHANNEL STOP":         CAT_IMPLANT,
    "IMPLANT DRAIN / CATHODE REGION": CAT_IMPLANT,
    "IMPLANT N-TYPE":               CAT_IMPLANT,

    # ── Chemical-mechanical planarization ────────────────────────────────────
    # REQUIRE a recently deposited/filled layer.
    "CMP DIELECTRIC":               CAT_CMP,
    "CMP INTERLAYER DIELECTRIC":    CAT_CMP,
    "CMP METAL":                    CAT_CMP,
    "CMP VIA FILL":                 CAT_CMP,

    # ── Via fill ─────────────────────────────────────────────────────────────
    "FILL VIA METAL":               CAT_FILL,
    "FILL VIA TUNGSTEN":            CAT_FILL,

    # ── Electrical tests ─────────────────────────────────────────────────────
    # REQUIRE passivation cured (device must be protected before probing).
    "PARAMETRIC TEST":              CAT_TEST,
    "ELECTRICAL PARAMETRIC TEST":   CAT_TEST,
    "THRESHOLD VOLTAGE TEST":       CAT_TEST,
    "LEAKAGE TEST":                 CAT_TEST,
    "BREAKDOWN VOLTAGE TEST":       CAT_TEST,
    "SWITCHING TEST":               CAT_TEST,
    "WAFER SORT TEST":              CAT_TEST,  # special: marks sort_test_done
}


# ---------------------------------------------------------------------------
# Dynamic step prefix → category.
# These step names contain a numeric level suffix (N = 1, 2, 3, …).
# ---------------------------------------------------------------------------

_DYNAMIC_PREFIXES: list[tuple[str, str]] = [
    ("ALIGN MASK LEVEL",           CAT_LITHO),
    ("EXPOSE LITHO LEVEL",         CAT_LITHO),
    ("INSPECT PATTERN LEVEL",      CAT_PATTERN_INSPECT),
    ("PATTERN INSPECTION LEVEL",   CAT_PATTERN_INSPECT),
    ("STRIP RESIST LEVEL",         CAT_STRIP),
    ("MEASURE CD LEVEL",           CAT_MEASURE),
]


# ---------------------------------------------------------------------------
# Keyword → category fallback for steps from unseen families (Task 4 OOD).
# Order matters: more specific strings must come before generic ones.
# ---------------------------------------------------------------------------

#
# ORDERING IS CORRECTNESS. The list is scanned top-to-bottom; the first keyword
# that appears as a substring wins. The order below resolves genuine ambiguities
# in real (and hypothetical OOD) step names:
#
#   * "CLEAN" must beat "ETCH"  so "CLEAN AFTER PLASMA ETCH" → CLEAN, not ETCH.
#   * "ETCH"  must beat "DRY"   so "PLASMA DRY ETCH"        → ETCH,  not CLEAN.
#   * "ANNEAL" must beat "EPITAX"/deposit so "EPITAXY ANNEAL" → ANNEAL.
#   * bare "ION" is deliberately ABSENT: it is a substring of ISOLATION,
#     INSPECTION, PASSIVATION, OXIDATION, … and would wreck OOD classification.
#     Implants are detected by "IMPLANT" / "DOPE" instead.
#
_KEYWORD_FALLBACK: list[tuple[str, str]] = [
    # ── A. Compound / highly specific overrides ──────────────────────────────
    ("CURE PASSIV",      CAT_ANNEAL),    # cure of passivation = thermal, not deposit
    ("SURFACE PREP",     CAT_CLEAN),
    ("SPIN COAT",        CAT_LITHO),

    # ── B. Clean-dominant (the word CLEAN/RINSE/WASH overrides everything) ────
    # so "CLEAN AFTER <anything> ETCH" is correctly a clean step.
    ("CLEAN",            CAT_CLEAN),
    ("RINSE",            CAT_CLEAN),
    ("WASH",             CAT_CLEAN),
    ("RCA",              CAT_CLEAN),
    ("HF DIP",           CAT_CLEAN),
    ("DESCUM",           CAT_CLEAN),   # plasma descum = residue removal (OOD-safe; no known step uses it)

    # ── C. Resist strip ──────────────────────────────────────────────────────
    ("STRIP",            CAT_STRIP),

    # ── D. Lithography verbs ─────────────────────────────────────────────────
    ("PHOTORESIST",      CAT_LITHO),
    ("DEVELOP",          CAT_LITHO),
    ("EXPOSE",           CAT_LITHO),
    ("BAKE",             CAT_LITHO),
    ("LITHO",            CAT_LITHO),
    ("ALIGN",            CAT_LITHO),

    # ── E. Thermal / anneal (before deposit so "EPITAXY ANNEAL" is ANNEAL) ────
    ("ANNEAL",           CAT_ANNEAL),
    ("DIFFUS",           CAT_ANNEAL),
    ("DENSIF",           CAT_ANNEAL),
    ("DRIVE",            CAT_ANNEAL),
    ("CURE",             CAT_ANNEAL),
    ("SINTER",           CAT_ANNEAL),   # contact sintering
    ("FIRE",             CAT_ANNEAL),   # co-firing of screen-printed contacts (PV)

    # ── F. Etch (BLANKET before generic ETCH; both before DRY) ────────────────
    ("ANISOTROPIC",      CAT_ETCH_BLANKET),
    ("BLANKET",          CAT_ETCH_BLANKET),
    ("ETCH",             CAT_ETCH),

    # ── G. Deposition / growth (PASSIV here so "X PASSIVATION" deposits) ──────
    ("DEPOSIT",          CAT_DEPOSIT),
    ("GROW",             CAT_DEPOSIT),
    ("NUCLEAT",          CAT_DEPOSIT),   # nucleation / seed-layer growth (ALD etc.)
    ("EPITAX",           CAT_DEPOSIT),
    ("OXIDAT",           CAT_DEPOSIT),
    ("OXIDIS",           CAT_DEPOSIT),   # British spelling
    ("MOCVD",            CAT_DEPOSIT),
    ("CVD",              CAT_DEPOSIT),
    ("ALD",              CAT_DEPOSIT),
    ("SPUTTER",          CAT_DEPOSIT),
    ("EVAPORAT",         CAT_DEPOSIT),   # e-beam / thermal evaporation
    ("PLATE",            CAT_DEPOSIT),   # electroplating (e.g. Cu via fill)
    ("PRINT",            CAT_DEPOSIT),   # screen-print metallisation (PV)
    ("PASSIV",           CAT_DEPOSIT),   # DEPOSIT PASSIVATION variants

    # ── H. Implant (NO bare "ION" — see note above) ──────────────────────────
    ("IMPLANT",          CAT_IMPLANT),
    ("DOPE",             CAT_IMPLANT),

    # ── I. Planarisation / grind / fill ──────────────────────────────────────
    ("CMP",              CAT_CMP),
    ("PLANAR",           CAT_CMP),
    ("POLISH",           CAT_CMP),   # chemical-mechanical polish synonym
    ("GRIND",            CAT_GRIND),
    ("FILL",             CAT_FILL),

    # ── J. Dry as a last-resort clean (only if no stronger signal matched) ────
    ("DRY",              CAT_CLEAN),

    # ── K. Test / metrology / inspection ─────────────────────────────────────
    ("SORT",             CAT_TEST),
    ("TEST",             CAT_TEST),
    ("MEASURE",          CAT_MEASURE),
    ("INSPECT",          CAT_INSPECT),
    ("CHECK",            CAT_INSPECT),

    # ── L. Logistics ─────────────────────────────────────────────────────────
    ("SHIP",             CAT_LOGISTICS),
    ("RECEIVE",          CAT_LOGISTICS),
    ("RELEASE",          CAT_LOGISTICS),
    ("PACKAGE",          CAT_LOGISTICS),
]


def classify_step(step: str) -> str:
    """
    Return the physical functional category of a process step.

    Works for:
      - All ~120 known MOSFET / IGBT / IC step names (exact lookup)
      - Dynamic steps with numeric level suffix (prefix match)
      - Unknown steps from a 4th or completely novel family (keyword match)

    Returns "UNKNOWN" only when no keyword in the step name matches any
    known physical function — which should be extremely rare given the
    standardised vocabulary of thin-film manufacturing.

    Parameters
    ----------
    step : str
        Process step name exactly as it appears in the CSV, uppercase.

    Returns
    -------
    str
        One of the CAT_* constants defined in this module.
    """
    # Tier 1: exact lookup
    if step in STEP_CATEGORY:
        return STEP_CATEGORY[step]

    step_upper = step.upper()

    # Tier 2: dynamic prefix match (e.g. "ALIGN MASK LEVEL 3")
    for prefix, category in _DYNAMIC_PREFIXES:
        if step_upper.startswith(prefix):
            return category

    # Tier 3: keyword match for unseen families
    for keyword, category in _KEYWORD_FALLBACK:
        if keyword in step_upper:
            return category

    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Physics feature vector
# ---------------------------------------------------------------------------

def step_physics_vector(step: str) -> dict[str, bool]:
    """
    Return a binary feature vector that encodes what a step REQUIRES
    (preconditions) and what it DOES (effects) in terms of wafer physics.

    This vector is the same for all steps in a given category, plus
    a small set of step-specific overrides (e.g. DEPOSIT BACKSIDE METAL
    additionally requires passivation to be cured; CURE PASSIVATION
    additionally marks the cured flag; WAFER SORT TEST marks sort_done).

    Using this vector as model input means the model reasons about physical
    function rather than step identity — enabling transfer to unseen families.

    Keys
    ----
    Preconditions (what must already be true):
      needs_clean_surface       : surface must be ready for deposition
      needs_resist_present      : photoresist must be coated on wafer
      needs_resist_exposed      : resist must have been UV-exposed
      needs_mask_patterned      : resist must be developed (pattern open)
      needs_passivation_cured   : passivation must be cured before this step
      needs_sort_done           : wafer sort test must precede this step
      needs_implant_window      : an oxide window must be open for implant
      needs_material_to_remove  : a layer must exist to CMP
      needs_passivation_exist   : passivation layer must exist (before cure)

    Effects (what this step causes):
      makes_surface_clean       : leaves surface ready for next deposition
      makes_surface_dirty       : leaves residue requiring post-clean
      deposits_material         : adds a physical layer to the wafer
      removes_material          : removes material (etch / CMP / grind)
      coats_resist              : applies photoresist
      exposes_resist            : UV-exposes the resist
      develops_resist           : develops the pattern (opens the mask)
      removes_resist            : strips photoresist
      opens_implant_window      : creates an oxide-open region for implant
      activates_dopants         : anneals implanted species to active sites
      marks_passivation_exist   : passivation layer is now present
      marks_passivation_cured   : passivation is now cross-linked / sealed
      marks_sort_done           : wafer sort is now complete
      advances_litho_level      : this step belongs to a new mask level
      is_measurement_only       : no material or state change
      is_logistics_only         : no physical action on wafer
    """
    cat = classify_step(step)
    step_upper = step.upper()

    # Default vector — all False
    v: dict[str, bool] = {
        "needs_clean_surface":      False,
        "needs_resist_present":     False,
        "needs_resist_exposed":     False,
        "needs_mask_patterned":     False,
        "needs_passivation_cured":  False,
        "needs_sort_done":          False,
        "needs_implant_window":     False,
        "needs_material_to_remove": False,
        "needs_passivation_exist":  False,
        "makes_surface_clean":      False,
        "makes_surface_dirty":      False,
        "deposits_material":        False,
        "removes_material":         False,
        "coats_resist":             False,
        "exposes_resist":           False,
        "develops_resist":          False,
        "removes_resist":           False,
        "opens_implant_window":     False,
        "activates_dopants":        False,
        "marks_passivation_exist":  False,
        "marks_passivation_cured":  False,
        "marks_sort_done":          False,
        "advances_litho_level":     False,
        "is_measurement_only":      False,
        "is_logistics_only":        False,
    }

    # ── Category-level defaults ───────────────────────────────────────────────

    if cat == CAT_LOGISTICS:
        v["is_logistics_only"] = True

    elif cat in (CAT_INSPECT, CAT_PATTERN_INSPECT):
        v["is_measurement_only"] = True

    elif cat == CAT_MEASURE:
        v["is_measurement_only"] = True

    elif cat == CAT_CLEAN:
        v["makes_surface_clean"] = True

    elif cat == CAT_PREP:
        v["makes_surface_clean"] = True   # prep steps clean the surface

    elif cat == CAT_DEPOSIT:
        v["needs_clean_surface"] = True
        v["deposits_material"] = True
        v["makes_surface_dirty"] = True   # deposition generates particles

    elif cat == CAT_ANNEAL:
        v["makes_surface_clean"] = True   # thermal treatment passivates surface
        v["activates_dopants"] = True

    elif cat == CAT_LITHO:
        if "SPIN COAT" in step_upper or "COAT PHOTORESIST" in step_upper:
            v["coats_resist"] = True
        elif "EXPOSE" in step_upper:
            v["needs_resist_present"] = True
            v["exposes_resist"] = True
            v["advances_litho_level"] = "ALIGN MASK LEVEL" in step_upper or "EXPOSE LITHO LEVEL" in step_upper
        elif "DEVELOP" in step_upper:
            v["needs_resist_exposed"] = True
            v["develops_resist"] = True
            v["opens_implant_window"] = True   # developed pattern opens oxide for implant
        elif "BAKE" in step_upper:
            v["needs_resist_present"] = True   # baking requires resist to exist

    elif cat == CAT_ETCH:
        v["needs_mask_patterned"] = True
        v["removes_material"] = True
        v["makes_surface_dirty"] = True    # etch byproducts require post-etch clean
        v["opens_implant_window"] = True   # etch opens the oxide window

    elif cat == CAT_ETCH_BLANKET:
        # Blanket etch: no mask needed (e.g. spacer etch uses geometry, not photomask)
        v["removes_material"] = True
        v["makes_surface_dirty"] = True

    elif cat == CAT_GRIND:
        v["removes_material"] = True
        v["makes_surface_dirty"] = True

    elif cat == CAT_STRIP:
        v["removes_resist"] = True

    elif cat == CAT_IMPLANT:
        v["needs_implant_window"] = True

    elif cat in (CAT_CMP, CAT_FILL):
        if cat == CAT_CMP:
            v["needs_material_to_remove"] = True
            v["removes_material"] = True
            v["makes_surface_clean"] = True  # CMP leaves flat, clean surface
        else:  # FILL
            v["deposits_material"] = True
            v["makes_surface_dirty"] = True  # fill leaves excess material for CMP

    elif cat == CAT_TEST:
        v["needs_passivation_cured"] = True

    # ── Step-specific overrides ───────────────────────────────────────────────

    # DEPOSIT PASSIVATION / DEPOSIT PASSIVATION LAYER
    if "PASSIVATION" in step_upper and cat == CAT_DEPOSIT:
        v["marks_passivation_exist"] = True

    # DEPOSIT BACKSIDE METAL additionally requires the frontside to be sealed
    if "BACKSIDE METAL" in step_upper and cat == CAT_DEPOSIT:
        v["needs_passivation_cured"] = True

    # CURE PASSIVATION — requires the passivation layer to exist first
    if step_upper == "CURE PASSIVATION":
        v["needs_passivation_exist"] = True
        v["marks_passivation_cured"] = True
        v["activates_dopants"] = False  # cure is not a dopant-activation anneal

    # WAFER SORT TEST — marks the sort as done
    if "WAFER SORT TEST" in step_upper:
        v["marks_sort_done"] = True

    # SHIP LOT — requires sort to be complete
    if "SHIP LOT" in step_upper:
        v["needs_sort_done"] = True
        v["is_logistics_only"] = True

    # THERMAL OXIDATION is in DEPOSIT but also creates a thermally clean SiO2 surface
    if step_upper == "THERMAL OXIDATION":
        v["makes_surface_dirty"] = False   # the grown oxide IS a clean, well-defined surface
        v["makes_surface_clean"] = True

    return v


# ---------------------------------------------------------------------------
# Utility: all unique categories used in this vocabulary
# ---------------------------------------------------------------------------

ALL_CATEGORIES: frozenset[str] = frozenset(STEP_CATEGORY.values()) | {
    CAT_PATTERN_INSPECT, CAT_ETCH_BLANKET, CAT_GRIND, CAT_FILL,
}


if __name__ == "__main__":
    # Quick self-test: classify every known step and print category distribution
    from collections import Counter
    counter: Counter = Counter()
    for step, cat in STEP_CATEGORY.items():
        counter[cat] += 1
    print("Category distribution over known vocabulary:")
    for cat, count in counter.most_common():
        print(f"  {cat:<20} {count:>3}")

    print("\nKeyword-based fallback for hypothetical GaN steps:")
    unknown_steps = [
        "GROW GAN BUFFER LAYER",
        "MOCVD GAN EPITAXIAL GROWTH",
        "DEPOSIT TI AL CONTACT METAL",
        "ETCH GAN MESA",
        "IMPLANT N-GAN REGION",
        "ANNEAL OHMIC CONTACTS",
        "WAFER SORT GAN TEST",
    ]
    for s in unknown_steps:
        print(f"  {s:<40} -> {classify_step(s)}")
