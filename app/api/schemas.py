from typing import Any
from pydantic import BaseModel, Field

class PatientSummary(BaseModel):
    patient_id: str
    age: int
    sex: str
    latest_timestamp: str
    deterioration_risk: float
    risk_tier: str

class Driver(BaseModel):
    feature: str
    contribution: float
    direction: str
    display_value: str

class RiskResponse(BaseModel):
    patient_id: str
    horizon_hours: int
    risk_probability: float = Field(ge=0, le=1)
    risk_tier: str
    confidence: str
    quality_flags: list[str]
    drivers: list[Driver]
    current_measurements: dict[str, Any]
    generated_at: str

class ModelResponse(BaseModel):
    model_name: str
    version: str
    horizon_hours: int
    features: list[str]
    metrics: dict[str, float]
    training_rows: int
    patients: int
    trained_at: str
