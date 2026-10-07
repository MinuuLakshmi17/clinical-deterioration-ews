from __future__ import annotations
import numpy as np
import pandas as pd

RAW_VITALS = [
    "heart_rate", "sbp", "dbp", "resp_rate", "spo2", "temperature",
    "lactate", "wbc", "creatinine", "oxygen_flow"
]

BASE_FEATURES = [
    "age", "comorbidity_burden", *RAW_VITALS,
    "map", "shock_index", "oxygenation_gap", "missing_count"
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["patient_id", "timestamp"])
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    grouped = out.groupby("patient_id", group_keys=False)
    for col in RAW_VITALS:
        out[f"{col}_delta1h"] = grouped[col].diff(1)
        out[f"{col}_delta3h"] = grouped[col].diff(3)
        out[f"{col}_roll3h_mean"] = grouped[col].transform(lambda s: s.rolling(3, min_periods=1).mean())
    out["map"] = (out["sbp"] + 2 * out["dbp"]) / 3.0
    out["shock_index"] = out["heart_rate"] / out["sbp"].clip(lower=50)
    out["oxygenation_gap"] = 100 - out["spo2"]
    out["missing_count"] = out[RAW_VITALS].isna().sum(axis=1)
    delta_cols = [c for c in out.columns if "delta" in c]
    roll_cols = [c for c in out.columns if "roll3h" in c]
    out[delta_cols + roll_cols] = out[delta_cols + roll_cols].fillna(0)
    return out


def model_feature_names() -> list[str]:
    names = BASE_FEATURES.copy()
    for col in RAW_VITALS:
        names += [f"{col}_delta1h", f"{col}_delta3h", f"{col}_roll3h_mean"]
    return names


def prepare_matrix(df: pd.DataFrame) -> pd.DataFrame:
    feat = engineer_features(df)
    cols = model_feature_names()
    return feat[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
