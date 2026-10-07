from __future__ import annotations
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import STATIC_DIR, MODEL_PATH, MODEL_DIR, DATA_DIR
from app.api.schemas import RiskResponse, ModelResponse, PatientSummary
from app.services.model_service import ModelService

@asynccontextmanager
async def lifespan(app):
    global service
    if not MODEL_PATH.exists() or not (DATA_DIR / "synthetic_ehr.csv").exists():
        from app.ml.synthetic import save_demo
        from app.ml.train import train_and_save
        save_demo(DATA_DIR / "synthetic_ehr.csv")
        train_and_save(MODEL_DIR)
    service = ModelService()
    yield

app = FastAPI(title="Clinical Deterioration Early-Warning & Explainability System", version="1.0.0",
              description="Educational clinical decision-support prototype using synthetic data.", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(STATIC_DIR / "dashboard.html")

@app.get("/health")
def health():
    return {"status":"ok", "model_loaded": service is not None, "synthetic_data": True}

@app.get("/model", response_model=ModelResponse)
def model_info():
    return service.metadata

@app.get("/patients", response_model=list[PatientSummary])
def patients():
    return service.patients()

@app.get("/patients/{patient_id}/risk", response_model=RiskResponse)
def patient_risk(patient_id: str):
    try: return service.risk(patient_id)
    except KeyError: raise HTTPException(404, f"Unknown patient_id: {patient_id}")

@app.get("/audit")
def audit():
    return service.audit.recent()
