from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import pandas as pd
from app.core.config import MODEL_PATH, METADATA_PATH, DATA_DIR
from app.ml.features import prepare_matrix
from app.ml.explain import local_surrogate
from app.services.audit import AuditStore

class ModelService:
    def __init__(self):
        self.artifact = joblib.load(MODEL_PATH)
        self.metadata = json.loads(METADATA_PATH.read_text())
        self.data = pd.read_csv(DATA_DIR / "synthetic_ehr.csv", parse_dates=["timestamp"])
        self.audit = AuditStore(Path(DATA_DIR) / "audit.db")

    @staticmethod
    def tier(p):
        if p >= 0.75: return "HIGH"
        if p >= 0.45: return "MODERATE"
        return "LOW"

    def patients(self):
        # Feature engineering needs recent history: deltas and rolling means
        # are computed within each patient's timeline, so score the latest
        # row of a short trailing window rather than an isolated row.
        recent = self.data.sort_values("timestamp").groupby("patient_id").tail(4)
        X = prepare_matrix(recent)
        last = recent.groupby("patient_id").tail(1)
        X_last = X.loc[last.index]
        probs = self.artifact["model"].predict_proba(X_last)[:, 1]
        out = []
        for (_, row), p in zip(last.iterrows(), probs):
            out.append({"patient_id": row.patient_id, "age": int(row.age), "sex": row.sex,
                        "latest_timestamp": row.timestamp.isoformat(), "deterioration_risk": float(p), "risk_tier": self.tier(p)})
        return sorted(out, key=lambda x: x["deterioration_risk"], reverse=True)

    def risk(self, patient_id: str):
        rows = self.data[self.data.patient_id == patient_id].sort_values("timestamp")
        if rows.empty: raise KeyError(patient_id)
        latest = rows.tail(1)
        X = prepare_matrix(rows.tail(4)).iloc[[-1]]
        p = float(self.artifact["model"].predict_proba(X)[0, 1])
        flags = []
        raw_vitals = ["heart_rate","sbp","dbp","resp_rate","spo2","temperature","lactate","wbc","creatinine","oxygen_flow"]
        missing = [v for v in raw_vitals if pd.isna(latest.iloc[0][v])]
        if missing: flags.append(f"Missing latest measurements: {', '.join(missing)}")
        if p >= 0.75: flags.append("High-risk prediction requires clinician review; this is not a diagnosis.")
        confidence = "high" if len(flags) == 0 else ("moderate" if len(missing) <= 2 else "low")
        display = {}
        for f in self.artifact["features"]:
            if f in X.columns: display[f] = f"{float(X.iloc[0][f]):.2f}"
        drivers = local_surrogate(self.artifact["model"], X, display)
        current = {k: (None if pd.isna(v) else float(v) if isinstance(v, (int,float)) else v)
                   for k,v in latest.iloc[0].to_dict().items() if k not in ["patient_id","timestamp","deterioration_6h"]}
        result = {"patient_id": patient_id, "horizon_hours": 6, "risk_probability": p,
                  "risk_tier": self.tier(p), "confidence": confidence, "quality_flags": flags,
                  "drivers": drivers, "current_measurements": current,
                  "generated_at": datetime.now(timezone.utc).isoformat()}
        self.audit.record(patient_id, p, self.tier(p), drivers)
        return result
