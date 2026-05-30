import json
from datetime import datetime
from pathlib import Path
from typing import Literal
from fastapi import APIRouter, Query
from app.models.batch import BatchSummary, BatchListResponse
from app.services.inference import predict_sequence

router = APIRouter()

_EVAL_SCORED: list[dict] = []
_EVAL_DETAIL: dict[str, dict] = {}

def _load_eval_data():
    global _EVAL_SCORED, _EVAL_DETAIL
    scored_path = Path(__file__).parents[3] / "results" / "eval_scored.json"
    if scored_path.exists():
        data = json.loads(scored_path.read_text())
        _EVAL_SCORED = data
        _EVAL_DETAIL = {d["batch_id"]: d for d in data}
        return True
    return False

_eval_loaded = _load_eval_data()


def _load_batch_list() -> list[BatchSummary]:
    """Load from scored eval data, fall back to stubs."""
    if _EVAL_SCORED:
        batches = []
        for d in _EVAL_SCORED:
            batches.append(BatchSummary(
                batch_id=d["batch_id"],
                timestamp=datetime(2026, 5, 30, 10, 0),
                material=d.get("family", "IC"),
                predicted_yield=d.get("predicted_yield", 0.5),
                defect_probability=round(1.0 - d.get("predicted_yield", 0.5), 4),
                confidence=d.get("confidence", 0.7),
                risk_steps_detected=d.get("risk_steps_detected", 0),
            ))
        return batches

    # Fallback stubs
    batches = []
    materials = ["IC", "IGBT", "MOSFET"]
    for i in range(50):
        batches.append(BatchSummary(
            batch_id=f"batch_{i+1:03d}",
            timestamp=datetime(2026, 5, max(1, 30 - i // 24), i % 24),
            material=materials[i % 3],
            predicted_yield=round(0.45 + ((i * 17) % 50) / 100.0, 3),
            defect_probability=round(1.0 - (0.45 + ((i * 17) % 50) / 100.0), 3),
            confidence=round(0.65 + (i % 4) * 0.07, 2),
            risk_steps_detected=(i * 3) % 25,
        ))
    return batches


_BATCH_LIST = _load_batch_list()


@router.get("/batches", response_model=BatchListResponse)
async def list_batches(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sort_by: str = Query("defect_probability"),
    order: Literal["asc", "desc"] = Query("desc"),
):
    valid_fields = {
        "batch_id", "timestamp", "material", "predicted_yield",
        "defect_probability", "confidence", "risk_steps_detected",
    }
    if sort_by not in valid_fields:
        sort_by = "defect_probability"

    sorted_batches = sorted(
        _BATCH_LIST,
        key=lambda b: getattr(b, sort_by),
        reverse=(order == "desc"),
    )
    start = (page - 1) * page_size
    end = start + page_size
    return BatchListResponse(
        total=len(sorted_batches),
        page=page,
        page_size=page_size,
        batches=sorted_batches[start:end],
    )


@router.get("/batches/{batch_id}")
async def get_batch(batch_id: str):
    if batch_id in _EVAL_DETAIL:
        return _EVAL_DETAIL[batch_id]
    return predict_sequence(batch_id, None)
