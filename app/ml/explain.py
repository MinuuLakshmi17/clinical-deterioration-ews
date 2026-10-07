from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def global_importance(model, X: pd.DataFrame, y: pd.Series, seed: int = 42) -> list[dict]:
    result = permutation_importance(model, X, y, n_repeats=5, random_state=seed, scoring="roc_auc")
    order = np.argsort(result.importances_mean)[::-1]
    return [{"feature": X.columns[i], "importance": float(max(result.importances_mean[i], 0))} for i in order[:15]]


def local_surrogate(model, x: pd.DataFrame, feature_display: dict[str, str], seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    base = x.iloc[0].to_numpy(dtype=float)
    X_local = np.repeat(base[None, :], 160, axis=0)
    scales = np.maximum(np.abs(base) * 0.05, 0.1)
    noise = rng.normal(0, scales, size=X_local.shape)
    X_local += noise
    # Keep clinical ranges sane for perturbations.
    X_local = pd.DataFrame(X_local, columns=x.columns)
    p_local = model.predict_proba(X_local)[:, 1]
    centered = X_local - x.iloc[0]
    denom = (centered.pow(2).sum(axis=0) + 1e-8)
    effects = ((centered.T @ (p_local - p_local.mean())) / denom).sort_values()
    rows = []
    for feature, coef in effects.items():
        rows.append({"feature": feature, "contribution": float(coef),
                     "direction": "increases risk" if coef > 0 else "decreases risk",
                     "display_value": feature_display.get(feature, str(round(float(x.iloc[0][feature]), 3)))})
    rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    return rows[:8]
