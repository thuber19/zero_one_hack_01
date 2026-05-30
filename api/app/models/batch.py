from datetime import datetime
from pydantic import BaseModel


class BatchSummary(BaseModel):
    batch_id: str
    timestamp: datetime
    material: str
    predicted_yield: float
    defect_probability: float
    confidence: float
    risk_steps_detected: int


class BatchListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    batches: list[BatchSummary]
