from __future__ import annotations


def predict_yield_for_sequence(sequence: list[dict]) -> dict:
    """
    Estimate yield for a reordered sequence. Currently uses a simple heuristic
    (ordering score based on step category consistency). Replace with real model
    inference when MODEL_PATH is available.
    """
    if not sequence:
        return {"predicted_yield": 0.5, "confidence": 0.6, "cached": False}

    # Heuristic: penalize sequences where process categories are non-contiguous
    categories = [s.get("category", "") for s in sequence]
    transitions = sum(1 for i in range(1, len(categories)) if categories[i] != categories[i - 1])
    max_transitions = len(categories) - 1
    ordering_score = 1.0 - (transitions / max_transitions) if max_transitions > 0 else 1.0

    # Base yield in [0.55, 0.95], better ordering → higher yield
    predicted_yield = round(0.55 + ordering_score * 0.40, 3)
    confidence = round(0.65 + ordering_score * 0.25, 3)

    return {"predicted_yield": predicted_yield, "confidence": confidence, "cached": False}
