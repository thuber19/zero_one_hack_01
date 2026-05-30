from __future__ import annotations
import numpy as np


_explainer = None


def init_explainer(model, background: np.ndarray | None):
    global _explainer
    if background is None or model is None:
        _explainer = None
        return None

    try:
        import shap
        _explainer = shap.DeepExplainer(model, background)
        print("SHAP: using DeepExplainer")
    except Exception as e:
        print(f"SHAP: DeepExplainer failed ({e}), falling back to KernelExplainer")
        try:
            import shap
            _explainer = shap.KernelExplainer(model, background[:50])
            print("SHAP: using KernelExplainer")
        except Exception as e2:
            print(f"SHAP: KernelExplainer also failed ({e2}), SHAP disabled")
            _explainer = None

    return _explainer


def get_explainer():
    return _explainer


def compute_shap(step_features: list[float]) -> list[float] | None:
    if _explainer is None:
        return None
    try:
        values = _explainer.shap_values(np.array([step_features]))
        if isinstance(values, list):
            values = values[0]
        return values[0].tolist() if hasattr(values[0], 'tolist') else list(values[0])
    except Exception:
        return None
