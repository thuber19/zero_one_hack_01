from __future__ import annotations
import json
import os
from pathlib import Path

_predictor = None


def load_model_if_available():
    global _predictor
    model_dir = os.environ.get("MODEL_DIR", "")
    if not model_dir or not Path(model_dir).exists():
        print(f"INFO: MODEL_DIR not set or not found ({model_dir}), using fixture fallback")
        return None

    try:
        import sys
        src_dir = Path(__file__).parents[3] / "tracks" / "industrial-infineon" / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        from inference import ProcessPredictor

        model_size = os.environ.get("MODEL_SIZE", "tiny")
        device = "mps" if _mps_available() else "cpu"
        _predictor = ProcessPredictor.load(Path(model_dir), model_size=model_size, device=device)
        print(f"ProcessPredictor loaded from {model_dir} (size={model_size}, device={device})")
        return _predictor
    except Exception as e:
        print(f"Warning: failed to load ProcessPredictor from {model_dir}: {e}")
        return None


def _mps_available() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


def get_predictor():
    return _predictor


def _load_fixture(name: str = "bad_batch"):
    fixture_path = Path(__file__).parent.parent / "fixtures" / f"{name}.json"
    if fixture_path.exists():
        return json.loads(fixture_path.read_text())
    return None


def predict_sequence(batch_id: str | None, sequence: list | None) -> dict:
    """Run anomaly detection via ProcessPredictor, fall back to fixture."""
    if _predictor is not None and sequence:
        try:
            steps = [s.get("step_name", s.get("step_id", "")) for s in sequence]
            family = os.environ.get("DEFAULT_FAMILY", "IC")
            result = _predictor.detect_anomaly(steps, family=family, use_physics=True)

            is_valid = result.get("is_valid", True)
            score = float(result.get("score", 0.5))
            predicted_rule = result.get("predicted_rule", "") or ""
            explanation = result.get("explanation", "") or ""
            avg_loss = float(result.get("avg_loss", 0.0))

            # Per-step: no per-step scores from the new API — use avg_loss normalized
            per_step = []
            for i, s in enumerate(sequence):
                # Uniform risk from the sequence-level signal
                risk = min(avg_loss / 2.0, 1.0) if not is_valid else avg_loss * 0.3
                per_step.append({
                    "step_id": s.get("step_id", f"step_{i:03d}"),
                    "step_name": s.get("step_name", f"step_{i:03d}"),
                    "category": s.get("category", "process"),
                    "risk_score": round(risk, 4),
                    "confidence_lo": round(max(0.0, risk - 0.08), 4),
                    "confidence_hi": round(min(1.0, risk + 0.08), 4),
                    "shap": [],
                })

            anomalies = []
            if not is_valid and predicted_rule:
                anomalies.append({
                    "step_id": "sequence",
                    "rule": predicted_rule,
                    "description": explanation,
                })

            return {
                "batch_id": batch_id or "inferred",
                "predicted_yield": round(score, 4),
                "confidence": round(score, 4),
                "risk_steps_detected": len(result.get("rf_violations", [])),
                "anomalous_batches": 0 if is_valid else 1,
                "per_step": per_step,
                "anomalies": anomalies,
            }
        except Exception as e:
            print(f"Warning: inference failed: {e}, falling back to fixture")

    return _load_fixture("bad_batch") or {
        "batch_id": batch_id or "stub",
        "predicted_yield": 0.614,
        "confidence": 0.73,
        "risk_steps_detected": 17,
        "anomalous_batches": 3,
        "per_step": [],
        "anomalies": [],
    }
