from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Literal

router = APIRouter()


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    shap_background_loaded: bool
    cache_size: int


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    app_state = request.app.state
    model_loaded = getattr(app_state, 'model', None) is not None
    shap_loaded = getattr(app_state, 'shap_explainer', None) is not None
    cache = getattr(app_state, 'lru_cache', None)
    cache_size = len(cache) if cache is not None else 0
    status = "ok"  # always ok for now; degrade if model fails to load
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        shap_background_loaded=shap_loaded,
        cache_size=cache_size,
    )
