from fastapi import APIRouter, Request
from app.models.predict import PredictRequest, PredictResponse
from app.services.inference import predict_sequence
from app.shap.cache import cache_get, cache_set

router = APIRouter()


@router.post("/predict")
async def predict(req: PredictRequest, request: Request):
    batch_id = req.batch_id

    # Check LRU cache
    if batch_id:
        cached = cache_get(batch_id)
        if cached:
            return cached

    sequence = [s.model_dump() for s in req.sequence] if req.sequence else None
    result = predict_sequence(batch_id, sequence)

    # Cache by batch_id
    if batch_id:
        cache_set(batch_id, result)

    return result
