"""Model definitions, time-aware tuning and fitting.

All models predict ``target_log_growth = log(y_t) - log(y_{t-1})``; forecasts in
hg/ha are ``y_{t-1} * exp(prediction)``. Tuning is scored on MAE in hg/ha after
this back-transformation, i.e. on the same quantity that is reported.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .feature_engineering import CATEGORICAL_FEATURES, MODEL_TARGET, target_to_yield


class NaiveLastValue(BaseEstimator, RegressorMixin):
    """Persistence forecast: y_t = y_{t-1}, i.e. zero predicted log growth."""

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return np.zeros(len(X))


def make_preprocessor(features: list[str], scale: bool) -> ColumnTransformer:
    numeric = [f for f in features if f not in CATEGORICAL_FEATURES]
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), numeric)]
    cats = [f for f in features if f in CATEGORICAL_FEATURES]
    if cats:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def model_specs(settings: dict, seed: int) -> dict[str, dict[str, Any]]:
    """Estimator, whether to scale inputs, and the tuning search space."""
    n_rf = int(settings["rf_n_estimators"])
    n_boost = int(settings["boosting_n_estimators"])
    return {
        "Naive (last value)": {"estimator": NaiveLastValue(), "scale": False, "space": None},
        "DummyRegressor (mean growth)": {"estimator": DummyRegressor(strategy="mean"), "scale": False, "space": None},
        "LinearRegression": {"estimator": LinearRegression(), "scale": True, "space": None},
        "RandomForest": {
            "estimator": RandomForestRegressor(n_estimators=n_rf, random_state=seed, n_jobs=-1),
            "scale": False,
            "space": {
                "max_depth": [8, 12, 16, None],
                "min_samples_leaf": [5, 10, 20, 40],
                "max_features": [0.3, 0.5, 0.8],
                "max_samples": [0.5, 0.8],
            },
        },
        "XGBoost": {
            "estimator": xgb.XGBRegressor(
                n_estimators=n_boost, random_state=seed, n_jobs=-1, tree_method="hist", verbosity=0
            ),
            "scale": False,
            "space": {
                "learning_rate": [0.01, 0.02, 0.05],
                "max_depth": [3, 4, 6],
                "min_child_weight": [5, 20, 50],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "reg_lambda": [1.0, 5.0, 20.0],
                "objective": ["reg:squarederror", "reg:pseudohubererror"],
            },
        },
        "LightGBM": {
            "estimator": lgb.LGBMRegressor(n_estimators=n_boost, random_state=seed, n_jobs=-1, verbose=-1),
            "scale": False,
            "space": {
                "learning_rate": [0.01, 0.02, 0.05],
                "num_leaves": [15, 31, 63],
                "min_child_samples": [20, 50, 100],
                "subsample": [0.7, 0.85, 1.0],
                "subsample_freq": [1],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "reg_lambda": [0.0, 5.0, 20.0],
                "objective": ["regression", "huber"],
            },
        },
    }


def build_pipeline(estimator, features: list[str], scale: bool, params: dict | None = None) -> Pipeline:
    est = clone(estimator)
    if params:
        est.set_params(**params)
    return Pipeline([("prep", make_preprocessor(features, scale)), ("model", est)])


def predict_yield(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return target_to_yield(pipe.predict(X), X["log_y_lag1"].to_numpy())


def _fold_mae(pipe: Pipeline, df: pd.DataFrame, features: list[str], folds) -> list[float]:
    scores = []
    for tr, va in folds:
        p = clone(pipe).fit(df.iloc[tr][features], df.iloc[tr][MODEL_TARGET])
        pred = predict_yield(p, df.iloc[va][features])
        scores.append(float(np.mean(np.abs(pred - df.iloc[va]["yield_hg_ha"].to_numpy()))))
    return scores


@dataclass
class TuningResult:
    params: dict
    cv_mae: float
    cv_fold_mae: list[float]
    tuning_seconds: float
    trials: list[dict] = field(default_factory=list)


def tune(spec: dict, df: pd.DataFrame, features: list[str], folds, n_iter: int, seed: int) -> TuningResult:
    """Random search scored by mean expanding-window CV MAE (hg/ha).

    ``df`` must contain development years only; the test period is never seen.
    """
    t0 = time.perf_counter()
    candidates = [{}] if spec["space"] is None else list(ParameterSampler(spec["space"], n_iter=n_iter, random_state=seed))
    trials = []
    for params in candidates:
        pipe = build_pipeline(spec["estimator"], features, spec["scale"], params)
        fold = _fold_mae(pipe, df, features, folds)
        trials.append({"params": params, "cv_mae": float(np.mean(fold)), "fold_mae": fold})
    best = min(trials, key=lambda t: t["cv_mae"])
    return TuningResult(best["params"], best["cv_mae"], best["fold_mae"], time.perf_counter() - t0, trials)


def fit_timed(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, float]:
    t0 = time.perf_counter()
    pipe.fit(X, y)
    return pipe, time.perf_counter() - t0


def predict_timed(pipe: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    t0 = time.perf_counter()
    g = pipe.predict(X)
    secs = time.perf_counter() - t0
    return g, target_to_yield(g, X["log_y_lag1"].to_numpy()), secs
