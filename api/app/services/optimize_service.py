from __future__ import annotations


def predict_yield_for_sequence(sequence: list[dict]) -> dict:
    """Estimate yield via live physics rule-check; heuristic fallback if unavailable."""
    if not sequence:
        return {"predicted_yield": 0.5, "confidence": 0.6, "cached": False}

    steps = [s.get("step_name") or s.get("step_id") or "" for s in sequence]
    steps = [s for s in steps if s]

    # Try live physics anomaly check first
    try:
        from app.services.procseq_service import live_anomaly
        # Guess family from sequence (default MOSFET)
        family = _guess_family(sequence)
        result = live_anomaly(steps, family)
        if result["is_valid"]:
            return {"predicted_yield": 0.92, "confidence": 0.91, "cached": False}
        else:
            return {"predicted_yield": 0.35, "confidence": 0.95, "cached": False}
    except Exception:
        pass

    # Heuristic fallback
    categories = [s.get("category", "") for s in sequence]
    transitions = sum(1 for i in range(1, len(categories)) if categories[i] != categories[i - 1])
    max_transitions = len(categories) - 1
    ordering_score = 1.0 - (transitions / max_transitions) if max_transitions > 0 else 1.0
    predicted_yield = round(0.55 + ordering_score * 0.40, 3)
    confidence = round(0.65 + ordering_score * 0.25, 3)
    return {"predicted_yield": predicted_yield, "confidence": confidence, "cached": False}


def _guess_family(sequence: list[dict]) -> str:
    """Infer product family from sequence metadata; default MOSFET."""
    for s in sequence:
        fam = s.get("family", "")
        if fam in ("MOSFET", "IGBT", "IC"):
            return fam
    return "MOSFET"
