from pydantic import BaseModel


class OptimizeStepInput(BaseModel):
    step_id: str
    step_name: str
    category: str


class OptimizeRequest(BaseModel):
    sequence: list[OptimizeStepInput]


class OptimizeResponse(BaseModel):
    predicted_yield: float
    confidence: float
    cached: bool
