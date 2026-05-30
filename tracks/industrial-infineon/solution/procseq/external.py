"""Bridge to the team's physics / categorization modules (on main).

Reuses, rather than forks, two things from the team's work:
  * physics.ontology.classify_step  — functional categorization of a step into
    ~17 universal categories, with a keyword fallback that generalizes to unseen
    vocabulary (the OOD engine for a hidden 4th family).
  * src.canonicalize.canonicalize_*  — maps interchangeable synonym step names
    ("STRIP RESIST" -> "STRIP PHOTORESIST", "WET CLEAN RCA1" -> "RCA CLEAN 1", …)
    to a single canonical form, shrinking the vocabulary the model must learn.

Importing keeps the team's code authoritative (no copy/paste drift). The track
modules live at ../.. (tracks/industrial-infineon): `physics` is a package,
`canonicalize` is a top-level module under src/.

Canonicalization is gated by the PROCSEQ_CANON env var so we can A/B it:
  PROCSEQ_CANON=0  -> identity (raw synonyms preserved)
  (unset / 1)      -> canonicalize (default)
"""
import os
import sys
from pathlib import Path

_TRACK = Path(__file__).resolve().parents[2]  # tracks/industrial-infineon
for _p in (str(_TRACK), str(_TRACK / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Degrade gracefully if the team's physics/canonicalize modules aren't present
# (keeps procseq runnable from a detached copy of solution/).
try:
    from physics.ontology import classify_step as _classify_step, STEP_CATEGORY
    import canonicalize as _canon
    AVAILABLE = True
except Exception:  # pragma: no cover - portability fallback
    AVAILABLE = False
    STEP_CATEGORY = {}

    def _classify_step(step):
        return "UNKNOWN"

    class _canon:  # type: ignore
        @staticmethod
        def canonicalize_sequence(steps):
            return list(steps)

        @staticmethod
        def canonicalize_step(step):
            return step

CATEGORIES = sorted(set(STEP_CATEGORY.values())) if STEP_CATEGORY else ["UNKNOWN"]

CANON_ENABLED = (
    AVAILABLE
    and os.environ.get("PROCSEQ_CANON", "1") not in ("0", "false", "False", "")
)


def canon_sequence(steps):
    """Canonicalize a list of steps if enabled, else return a copy unchanged."""
    return _canon.canonicalize_sequence(steps) if CANON_ENABLED else list(steps)


def canon_step(step):
    """Canonicalize a single step if enabled, else return it unchanged."""
    return _canon.canonicalize_step(step) if CANON_ENABLED else step


def category_of(step):
    """Functional category of a step (exact -> prefix -> keyword OOD fallback)."""
    return _classify_step(step)
