"""Bridge to the team's physics / categorization modules (on main).

Reuses, rather than forks, two things from the team's work:
  * physics.ontology.classify_step  — functional categorization of a step into
    ~17 universal categories, with a keyword fallback that generalizes to unseen
    vocabulary (the OOD engine for a hidden 4th family).
  * src.canonicalize.canonicalize_*  — maps interchangeable synonym step names
    ("STRIP RESIST" -> "STRIP PHOTORESIST", "WET CLEAN RCA1" -> "RCA CLEAN 1", …)
    to a single canonical form, shrinking the vocabulary the model must learn.

Categorization (classify_step) and canonicalization are imported INDEPENDENTLY
so that losing one (e.g. the team deleted src/canonicalize.py on main) does not
disable the other. Canonicalize falls back to our vendored procseq/canon.py.

Canonicalization is gated by PROCSEQ_CANON and defaults OFF: the real grader
scores exact surface forms, so collapsing synonyms risks mismatching the held-out
ground truth. Enable (PROCSEQ_CANON=1) only as a training-efficiency A/B probe.
"""
import os
import sys
from pathlib import Path

_TRACK = Path(__file__).resolve().parents[2]  # tracks/industrial-infineon
for _p in (str(_TRACK), str(_TRACK / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- Categorization (team's physics.ontology; identity fallback) ---
try:
    from physics.ontology import classify_step as _classify_step, STEP_CATEGORY
    CATEGORIZER_AVAILABLE = True
except Exception:  # pragma: no cover - portability fallback
    CATEGORIZER_AVAILABLE = False
    STEP_CATEGORY = {}

    def _classify_step(step):
        return "UNKNOWN"

# --- Canonicalization (team's src/canonicalize if present, else vendored) ---
try:
    import canonicalize as _canon          # team's (deleted on current main)
except Exception:
    from procseq import canon as _canon    # our vendored copy

AVAILABLE = CATEGORIZER_AVAILABLE
CATEGORIES = sorted(set(STEP_CATEGORY.values())) if STEP_CATEGORY else ["UNKNOWN"]

# Default OFF (see module docstring). Set PROCSEQ_CANON=1 to enable.
CANON_ENABLED = os.environ.get("PROCSEQ_CANON", "0") in ("1", "true", "True")


def canon_sequence(steps):
    """Canonicalize a list of steps if enabled, else return a copy unchanged."""
    return _canon.canonicalize_sequence(steps) if CANON_ENABLED else list(steps)


def canon_step(step):
    """Canonicalize a single step if enabled, else return it unchanged."""
    return _canon.canonicalize_step(step) if CANON_ENABLED else step


def category_of(step):
    """Functional category of a step (exact -> prefix -> keyword OOD fallback)."""
    return _classify_step(step)
