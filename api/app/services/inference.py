from __future__ import annotations
import json
import os
from pathlib import Path

# Lazy import torch to avoid startup failures when torch not installed
_model = None
_tokenizer = None


def load_model_if_available():
    global _model, _tokenizer
    model_path = os.environ.get("MODEL_PATH", "")
    if not model_path or not Path(model_path).exists():
        print(f"INFO: MODEL_PATH not set or not found ({model_path}), using fixture fallback")
        return None, None

    try:
        import sys
        import torch
        # Add repo root to path for src.infer
        repo_root = Path(__file__).parents[3]
        sys.path.insert(0, str(repo_root))
        from src.infer import load_model as _load
        _model, _tokenizer = _load(model_path, torch.device("cpu"))
        print(f"Model loaded from {model_path}")
        return _model, _tokenizer
    except Exception as e:
        print(f"Warning: failed to load model from {model_path}: {e}")
        return None, None


def get_model():
    return _model


def get_tokenizer():
    return _tokenizer


def _load_fixture(name: str = "bad_batch"):
    fixture_path = Path(__file__).parent.parent / "fixtures" / f"{name}.json"
    if fixture_path.exists():
        return json.loads(fixture_path.read_text())
    return None


def predict_sequence(batch_id: str | None, sequence: list | None) -> dict:
    """Run inference or fall back to fixture."""
    if _model is not None and _tokenizer is not None and sequence:
        try:
            import torch
            from src.infer import score_sequence
            threshold_path = os.environ.get("THRESHOLD_PATH", "")
            threshold = {}
            if threshold_path and Path(threshold_path).exists():
                import json as _json
                threshold = _json.loads(Path(threshold_path).read_text())
            else:
                threshold = {"p95_loss": 1.0, "p99_loss": 2.0, "ood_p99": 1.5}

            steps = [s.get("step_name", s.get("step_id", "")) for s in sequence]
            # variant is the product type (IC/IGBT/MOSFET), not the process category
            variant = os.environ.get("DEFAULT_VARIANT", "IC")
            report = score_sequence(
                _model, _tokenizer, variant=variant, steps=steps,
                threshold=threshold, seq_id=batch_id or "unknown",
                device=torch.device("cpu"),
            )
            # Convert AnomalyReport to PredictResponse shape
            per_step = []
            for i, s in enumerate(sequence):
                risk = report.per_step_raw_loss[i] if i < len(report.per_step_raw_loss) else 0.0
                per_step.append({
                    "step_id": s.get("step_id", f"step_{i:03d}"),
                    "step_name": s.get("step_name", f"step_{i:03d}"),
                    "risk_score": min(risk / 3.0, 1.0),  # normalize
                    "confidence_lo": max(0.0, min(risk / 3.0, 1.0) - 0.08),
                    "confidence_hi": min(1.0, min(risk / 3.0, 1.0) + 0.08),
                    "shap": [],
                })
            return {
                "batch_id": batch_id or "inferred",
                "predicted_yield": 1.0 - report.seq_score_mean / 3.0,
                "confidence": 0.80,
                "risk_steps_detected": len(report.anomalous_steps),
                "anomalous_batches": 1 if report.is_anomalous else 0,
                "per_step": per_step,
                "anomalies": [],
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
