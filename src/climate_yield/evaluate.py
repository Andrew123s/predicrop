"""Forecast metrics. MAE / RMSE / bias are in the target unit (hg/ha)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mase_scale(y_train: np.ndarray, naive_train: np.ndarray) -> float:
    """MAE of the naive (previous-year) forecast on the training period."""
    return float(np.mean(np.abs(np.asarray(y_train) - np.asarray(naive_train))))


def regression_metrics(y_true, y_pred, naive_scale: float | None = None, near_zero: float = 1.0) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if not np.isfinite(y_pred).all():
        raise ValueError("predictions contain non-finite values")
    out = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "Bias": float(np.mean(y_pred - y_true)),   # >0 means overprediction
    }
    # MAPE is only reported when no target is near zero (a single tiny yield
    # would dominate it); the median APE is robust and needs only y > 0.
    ape = np.abs(y_pred - y_true) / np.where(y_true > 0, np.abs(y_true), np.nan)
    out["MAPE_pct"] = float(100 * np.mean(ape)) if np.min(np.abs(y_true)) > near_zero else float("nan")
    out["MedAPE_pct"] = float(100 * np.median(ape)) if (y_true > 0).all() else float("nan")
    # Error of log yield: scale-free, comparable across crops.
    if (y_true > 0).all() and (y_pred > 0).all():
        out["MAE_log"] = float(np.mean(np.abs(np.log(y_pred) - np.log(y_true))))
    out["MASE"] = float(out["MAE"] / naive_scale) if naive_scale else float("nan")
    return out


def grouped_metrics(df: pd.DataFrame, by: str | list[str], y="y_true", p="y_pred") -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by):
        m = regression_metrics(g[y], g[p])
        m.update({"n": len(g)})
        if isinstance(by, str):
            m[by] = key
        else:
            m.update(dict(zip(by, key)))
        rows.append(m)
    return pd.DataFrame(rows)
