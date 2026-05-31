"""Live inference endpoints — real-time predictions from the procseq decoder."""
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()


class InferRequest(BaseModel):
    steps: list[str]
    family: str = "MOSFET"


@router.post("/infer/nextstep")
async def infer_nextstep(req: InferRequest):
    from app.services.procseq_service import live_next_steps, is_available
    predictions = live_next_steps(req.steps, req.family, k=5)
    return JSONResponse({"predictions": predictions, "available": is_available()})


@router.post("/infer/complete")
async def infer_complete(req: InferRequest):
    from app.services.procseq_service import live_complete, is_available
    completion, is_valid = live_complete(req.steps, req.family)
    return JSONResponse({
        "completion": completion,
        "is_valid": is_valid,
        "available": is_available(),
    })


@router.post("/infer/anomaly")
async def infer_anomaly(req: InferRequest):
    from app.services.procseq_service import live_anomaly
    result = live_anomaly(req.steps, req.family)
    return JSONResponse(result)
