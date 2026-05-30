from __future__ import annotations
import os
from pathlib import Path
import numpy as np


_background: np.ndarray | None = None


def load_background(path: str | None = None) -> np.ndarray | None:
    global _background
    if _background is not None:
        return _background

    bg_path = path or os.environ.get("SHAP_BACKGROUND_PATH", "")
    if not bg_path or not Path(bg_path).exists():
        return None

    try:
        _background = np.load(bg_path)
        return _background
    except Exception as e:
        print(f"Warning: could not load SHAP background from {bg_path}: {e}")
        return None


def get_background() -> np.ndarray | None:
    return _background
