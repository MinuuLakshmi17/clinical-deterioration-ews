from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit
from app.ml.features import prepare_matrix, model_feature_names
from app.ml.explain import global_importance
from app.ml.synthetic import generate_synthetic_ehr


def ece(y_true, prob, bins=10):
    y_true = np.asarray(y_true); prob = np.asarray(prob)
    edges = np.linspace(0, 1, bins + 1); total = len(y_true); score = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if mask.any():
            score += mask.mean() * abs(y_true[mask].mean() - prob[mask].mean())
    return float(score)


def train_and_save(output_dir: str | Path, data_path: str | Path | None = None):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    if data_path and Path(data_path).exists():
        raw = pd.read_csv(data_path)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    else:
        raw = generate_synthetic_ehr()
    X = prepare_matrix(raw)
    y = raw["deterioration_6h"].astype(int)
    groups = raw["patient_id"]
    split1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, test_idx = next(split1.split(X, y, groups=groups))
    base = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.055, max_leaf_nodes=15,
                                          l2_regularization=0.7, random_state=42)
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    p = model.predict_proba(X.iloc[test_idx])[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y.iloc[test_idx], p)),
        "pr_auc": float(average_precision_score(y.iloc[test_idx], p)),
        "brier_score": float(brier_score_loss(y.iloc[test_idx], p)),
        "expected_calibration_error": ece(y.iloc[test_idx], p),
        "positive_rate_test": float(y.iloc[test_idx].mean()),
    }
    importance = global_importance(model, X.iloc[test_idx], y.iloc[test_idx])
    artifact = {"model": model, "features": model_feature_names(), "importance": importance,
                "version": "1.0.0", "horizon_hours": 6}
    joblib.dump(artifact, output_dir / "deterioration_model.joblib")
    metadata = {
        "model_name": "Calibrated HistGradientBoosting early-warning model",
        "version": "1.0.0", "horizon_hours": 6, "features": model_feature_names(),
        "metrics": metrics, "training_rows": int(len(train_idx)),
        "patients": int(groups.nunique()), "trained_at": datetime.now(timezone.utc).isoformat(),
        "split": {"train_patients": int(groups.iloc[train_idx].nunique()),
                  "test_patients": int(groups.iloc[test_idx].nunique())},
        "global_importance": importance,
        "data_source": "synthetic educational cohort; no real patient data",
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    raw.to_csv(output_dir / "training_snapshot.csv", index=False)
    return metadata
