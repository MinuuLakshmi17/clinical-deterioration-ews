from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "app" / "web"
DB_PATH = BASE_DIR / "data" / "audit.db"
ENV = os.getenv("CLINICAL_EWS_ENV", "demo")
MODEL_PATH = MODEL_DIR / "deterioration_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
