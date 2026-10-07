from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd

RNG_SEED = 20261006

def generate_synthetic_ehr(n_patients: int = 240, hours: int = 36, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for p in range(n_patients):
        pid = f"P{p+1:05d}"
        age = int(rng.integers(18, 91))
        sex = rng.choice(["F", "M"])
        comorb = int(rng.poisson(1.4))
        comorb = min(comorb, 6)
        baseline = {
            "heart_rate": rng.normal(78, 8), "sbp": rng.normal(124, 10),
            "dbp": rng.normal(74, 7), "resp_rate": rng.normal(16, 2.2),
            "spo2": rng.normal(97.2, 1.0), "temperature": rng.normal(36.8, 0.25),
            "lactate": rng.normal(1.3, 0.25), "wbc": rng.normal(8.0, 1.7),
            "creatinine": rng.normal(0.95, 0.18), "oxygen_flow": max(0, rng.normal(1.0, 1.0)),
        }
        # Latent event determines a future deterioration trajectory.
        event = rng.random() < (0.22 + 0.035 * comorb + 0.002 * max(age - 60, 0))
        event_start = int(rng.integers(10, max(11, hours - 5))) if event else 999
        severity = rng.uniform(0.7, 1.5)
        history = []
        for h in range(hours):
            t = start + timedelta(hours=h)
            phase = max(0, h - event_start)
            ramp = min(phase / 5.0, 1.0) * severity
            vals = {
                "heart_rate": baseline["heart_rate"] + 18*ramp + rng.normal(0, 3),
                "sbp": baseline["sbp"] - 22*ramp + rng.normal(0, 4),
                "dbp": baseline["dbp"] - 10*ramp + rng.normal(0, 2.5),
                "resp_rate": baseline["resp_rate"] + 7*ramp + rng.normal(0, 1),
                "spo2": baseline["spo2"] - 5*ramp + rng.normal(0, 0.6),
                "temperature": baseline["temperature"] + 0.7*ramp + rng.normal(0, 0.08),
                "lactate": baseline["lactate"] + 1.8*ramp + rng.normal(0, 0.12),
                "wbc": baseline["wbc"] + 4*ramp + rng.normal(0, 0.35),
                "creatinine": baseline["creatinine"] + 0.55*ramp + rng.normal(0, 0.04),
                "oxygen_flow": baseline["oxygen_flow"] + 4*ramp + rng.normal(0, 0.3),
            }
            for k in vals:
                vals[k] = max(vals[k], 0.01)
            # Missingness is modest but increases slightly in unstable periods.
            miss_p = 0.025 + 0.035*ramp
            for k in list(vals):
                if rng.random() < miss_p:
                    vals[k] = np.nan
            # Positive label means deterioration event starts in next 6h.
            future_event = event and (event_start > h) and (event_start <= h + 6)
            history.append({"patient_id": pid, "timestamp": t, "age": age, "sex": sex,
                            "comorbidity_burden": comorb, **vals, "deterioration_6h": int(future_event)})
        rows.extend(history)
    return pd.DataFrame(rows)


def save_demo(path: str | Path) -> pd.DataFrame:
    df = generate_synthetic_ehr()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
