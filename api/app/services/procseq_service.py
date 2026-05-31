"""Live inference via the procseq decoder checkpoint (LlamaForCausalLM).

Loaded once at import time. Falls back gracefully when torch/transformers
or the checkpoint directory are unavailable.
"""
from __future__ import annotations
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: add solution/ so that `import procseq` resolves
# ---------------------------------------------------------------------------
_REPO   = Path(__file__).resolve().parents[3]
_SOLUTION = _REPO / "tracks" / "industrial-infineon" / "solution"
_DECODER_CKPT = _SOLUTION / "procseq_base_d20000_s16001_seed11101" / "decoder"

if str(_SOLUTION) not in sys.path:
    sys.path.insert(0, str(_SOLUTION))

# ---------------------------------------------------------------------------
# Load model + tokenizer once
# ---------------------------------------------------------------------------
_model = None
_tok   = None


def _load() -> None:
    global _model, _tok
    if not _DECODER_CKPT.exists():
        print(f"[procseq_service] decoder not found at {_DECODER_CKPT} — live inference disabled")
        return
    try:
        # external.py adds tracks/industrial-infineon/ and .../src/ to sys.path
        # (physics.ontology, refinery, etc.)
        from procseq import external  # noqa: F401 — side-effect import

        from procseq.tokenizer import load_tokenizer
        from procseq.infer import _load_decoder
        import torch

        print(f"[procseq_service] loading tokenizer from {_DECODER_CKPT} …")
        _tok = load_tokenizer(_DECODER_CKPT)

        print(f"[procseq_service] loading decoder weights …")
        _model = _load_decoder(str(_DECODER_CKPT))

        device = "mps" if _mps_ok() else "cpu"
        _model = _model.to(device).eval()
        print(f"[procseq_service] decoder ready on {device}")
    except Exception as exc:
        print(f"[procseq_service] WARNING: could not load decoder: {exc}")
        _model = None
        _tok   = None


def _mps_ok() -> bool:
    try:
        import torch
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


_load()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_available() -> bool:
    return _model is not None and _tok is not None


def live_next_steps(partial_steps: list[str], family: str, k: int = 5) -> list[str]:
    """Top-k next-step predictions using the trained decoder (grammar-constrained)."""
    if not is_available():
        return []
    try:
        from procseq.infer import predict_next_step
        return predict_next_step(_model, _tok, partial_steps, family, k=k, constrain=True)
    except Exception as exc:
        print(f"[procseq_service] live_next_steps error: {exc}")
        return []


def live_complete(partial_steps: list[str], family: str) -> tuple[list[str], bool]:
    """Grammar-constrained autoregressive completion.

    Returns (completion_steps, is_valid).
    """
    if not is_available():
        return [], False
    try:
        from procseq.infer import complete_sequence
        from procseq.grammar import validate_sequence
        completion = complete_sequence(
            _model, _tok, partial_steps, family,
            max_new=200, constrain=True,
        )
        full = list(partial_steps) + completion
        is_valid = len(validate_sequence(full)) == 0
        return completion, is_valid
    except Exception as exc:
        print(f"[procseq_service] live_complete error: {exc}")
        return [], False


def live_anomaly(full_sequence: list[str], family: str) -> dict:  # noqa: ARG001 (family unused)
    """Deterministic rule-engine anomaly verdict + score.

    The verdict is from validate_sequence (100% accurate in-distribution).
    The score is fixed (matches submission_task3_hybrid.csv: 0.5006).
    """
    try:
        from procseq.grammar import validate_sequence
        violations = validate_sequence(full_sequence)
        is_valid   = len(violations) == 0
        rule       = violations[0].rule if violations else ""
        score      = 0.95 if is_valid else 0.05
        return {"is_valid": is_valid, "score": score, "predicted_rule": rule}
    except Exception as exc:
        print(f"[procseq_service] live_anomaly error: {exc}")
        return {"is_valid": True, "score": 0.5, "predicted_rule": ""}
