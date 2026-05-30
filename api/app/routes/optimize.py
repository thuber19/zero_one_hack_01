from fastapi import APIRouter
from app.models.optimize import OptimizeRequest, OptimizeResponse
from app.services.optimize_service import predict_yield_for_sequence
from app.shap.cache import cache_get, cache_set, make_sequence_key

router = APIRouter()


@router.post("/optimize", response_model=OptimizeResponse)
async def optimize(req: OptimizeRequest):
    seq_dicts = [s.model_dump() for s in req.sequence]
    seq_key = make_sequence_key(seq_dicts)

    cached = cache_get(seq_key)
    if cached:
        return OptimizeResponse(**{**cached, "cached": True})

    result = predict_yield_for_sequence(seq_dicts)
    cache_set(seq_key, result)
    return OptimizeResponse(**result)
