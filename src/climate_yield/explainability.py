"""Permutation importance and SHAP explanations for fitted pipelines.

Importance describes what the *model* relies on for prediction. Predictors are
correlated (e.g. lagged yields with rolling means, rainfall with its anomaly)
and the data are observational, so importance is not evidence of causal effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .feature_engineering import target_to_yield


def permutation_table(pipe: Pipeline, X: pd.DataFrame, y_true_hg_ha: np.ndarray,
                      n_repeats: int, seed: int) -> pd.DataFrame:
    """Increase in MAE (hg/ha) when one raw input column is shuffled.

    The back-transformation to hg/ha always uses the *unshuffled* ``log_y_lag1``,
    so shuffling a column only perturbs its role as a model input. Shuffling
    ``log_y_lag1`` destroys the anchor of every lag feature at once and is
    reported like any other column.
    """
    rng = np.random.default_rng(seed)
    anchor = X["log_y_lag1"].to_numpy()
    y_true_hg_ha = np.asarray(y_true_hg_ha, dtype=float)

    def mae(frame):
        return float(np.mean(np.abs(target_to_yield(pipe.predict(frame), anchor) - y_true_hg_ha)))

    base = mae(X)
    rows = []
    for col in X.columns:
        scores = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            scores.append(mae(Xp) - base)
        rows.append({"feature": col, "mae_increase_hg_ha": float(np.mean(scores)), "std": float(np.std(scores))})
    return pd.DataFrame(rows).sort_values("mae_increase_hg_ha", ascending=False).reset_index(drop=True)


def shap_values(pipe: Pipeline, X: pd.DataFrame):
    """SHAP values (log-growth units) on the transformed design matrix."""
    import shap

    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    Z = prep.transform(X)
    names = list(prep.get_feature_names_out())
    Zdf = pd.DataFrame(Z, columns=names, index=X.index)
    kind = type(model).__name__
    if kind in ("LinearRegression",):
        explainer = shap.LinearExplainer(model, Zdf)
    else:
        explainer = shap.TreeExplainer(model)
    sv = explainer(Zdf, check_additivity=False) if kind not in ("LinearRegression",) else explainer(Zdf)
    return sv, Zdf


def shap_global_table(sv, Zdf: pd.DataFrame) -> pd.DataFrame:
    vals = np.abs(np.asarray(sv.values)).mean(axis=0)
    out = pd.DataFrame({"feature": Zdf.columns, "mean_abs_shap": vals})
    # aggregate one-hot crop columns and missing indicators back to their source
    out["source"] = out["feature"].str.replace(r"^crop_.*", "crop", regex=True).str.replace(
        r"^missingindicator_", "", regex=True)
    return out.groupby("source", as_index=False)["mean_abs_shap"].sum().sort_values(
        "mean_abs_shap", ascending=False).reset_index(drop=True)
