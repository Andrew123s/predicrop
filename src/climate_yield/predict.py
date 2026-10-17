"""Load saved model bundles and produce validated forecasts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data_validation import SchemaError, validate_feature_frame
from .feature_engineering import CATEGORICAL_FEATURES, target_to_yield

REQUIRED_NON_NULL = ["log_y_lag1"]


class ForecastInputError(ValueError):
    """Raised when a forecast cannot be produced from the supplied inputs."""


@dataclass
class Forecaster:
    bundle: dict
    path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "Forecaster":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"model artifact not found: {path}. Run scripts/train_models.py first.")
        bundle = joblib.load(path)
        for key in ("pipeline", "features", "feature_set"):
            if key not in bundle:
                raise ValueError(f"{path} is not a valid model bundle (missing {key!r})")
        return cls(bundle, path)

    @property
    def features(self) -> list[str]:
        return list(self.bundle["features"])

    @property
    def known_crops(self) -> list[str]:
        prep = self.bundle["pipeline"].named_steps["prep"]
        for name, trans, cols in prep.transformers_:
            if name == "cat":
                return list(trans.categories_[0])
        return []

    def validate(self, X: pd.DataFrame) -> list[str]:
        """Raise on unusable input; return human-readable warnings otherwise."""
        try:
            validate_feature_frame(X, self.features)
        except SchemaError as exc:
            raise ForecastInputError(str(exc)) from exc
        for c in REQUIRED_NON_NULL:
            if X[c].isna().any():
                raise ForecastInputError(f"{c} (previous-year yield) is required for a one-year-ahead forecast")
        warnings = []
        if "crop" in X:
            unknown = sorted(set(X["crop"]) - set(self.known_crops))
            if unknown:
                raise ForecastInputError(f"crop(s) not seen in training: {unknown}")
        num = [c for c in self.features if c not in CATEGORICAL_FEATURES]
        miss = X[num].isna().any()
        if miss.any():
            warnings.append("missing inputs imputed with training medians: " + ", ".join(miss[miss].index))
        ranges = self.bundle.get("feature_ranges") or {}
        out = [c for c, (lo, hi) in ranges.items() if c in X and ((X[c] < lo) | (X[c] > hi)).any()]
        if out:
            warnings.append("inputs outside the training range (model extrapolates): " + ", ".join(out))
        return warnings

    def predict_frame(self, X: pd.DataFrame, levels: list[float] | None = None) -> tuple[pd.DataFrame, list[str]]:
        warnings = self.validate(X)
        pipe = self.bundle["pipeline"]
        g = pipe.predict(X[self.features])
        out = pd.DataFrame(index=X.index)
        out["pred_log_growth"] = g
        out["yield_pred_hg_ha"] = target_to_yield(g, X["log_y_lag1"].to_numpy())
        cal = self.bundle.get("calibrator")
        if cal is not None:
            for lv in levels or cal.levels:
                lo, hi = cal.interval(g, X["log_y_lag1"].to_numpy(), lv, X["crop"].to_numpy() if "crop" in X else None)
                out[f"lower_{int(round(lv * 100))}"] = lo
                out[f"upper_{int(round(lv * 100))}"] = hi
        return out, warnings


def forecast_rows(features: pd.DataFrame, area: str, crop: str, target_year: int) -> pd.DataFrame:
    """Select the leakage-safe feature row(s) for one series and target year."""
    if not isinstance(target_year, (int, np.integer)):
        raise ForecastInputError("target_year must be an integer")
    sel = features[(features["area"] == area) & (features["crop"] == crop)]
    if sel.empty:
        raise ForecastInputError(f"no yield history for {crop} in {area}")
    row = sel[sel["year"] == target_year]
    if row.empty:
        lo, hi = int(sel["year"].min()), int(sel["year"].max())
        raise ForecastInputError(f"target year {target_year} outside supported range {lo + 1}-{hi} for this series")
    if row["log_y_lag1"].isna().all():
        raise ForecastInputError(f"no observed yield for {target_year - 1}; a one-year-ahead forecast needs it")
    return row


def history_warnings(features: pd.DataFrame, area: str, crop: str, target_year: int) -> list[str]:
    h = features[(features["area"] == area) & (features["crop"] == crop) & (features["year"] < target_year)]
    h = h.dropna(subset=["yield_hg_ha"]).sort_values("year")
    w = []
    if len(h) < 10:
        w.append(f"short history: only {len(h)} observed years")
    last5 = h["yield_hg_ha"].tail(5)
    if len(last5) == 5 and last5.nunique() == 1:
        w.append("last 5 observed yields are identical (likely FAO estimates, not measurements)")
    return w
