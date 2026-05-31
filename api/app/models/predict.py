from __future__ import annotations
from pydantic import BaseModel, ConfigDict


class SensorReadings(BaseModel):
    model_config = ConfigDict(extra="allow")


class StepInput(BaseModel):
    step_id: str
    step_name: str
    category: str
    duration_min: float | None = None
    sensors: SensorReadings | None = None


class PredictRequest(BaseModel):
    batch_id: str | None = None
    sequence: list[StepInput] | None = None


class ShapContribution(BaseModel):
    feature: str
    value: float
    contribution: float


class Anomaly(BaseModel):
    step_id: str
    anomaly_score: float
    type: str


class StepResult(BaseModel):
    step_id: str
    step_name: str
    risk_score: float
    confidence_lo: float
    confidence_hi: float
    shap: list[ShapContribution]


class PredictResponse(BaseModel):
    batch_id: str
    predicted_yield: float
    confidence: float
    risk_steps_detected: int
    anomalous_batches: int
    per_step: list[StepResult]
    anomalies: list[Anomaly]
