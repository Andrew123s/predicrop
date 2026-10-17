"""Climate sensitivity (what-if) analysis with the conditional model.

This is a *sensitivity analysis* of a statistical model, not climate-impact
attribution: same-year annual rainfall and temperature features are perturbed
while every other input is held at its observed value, and the change in the
model prediction is reported. Annual country means are a coarse proxy for
growing-season conditions, and many perturbations push inputs outside the
range seen for a given country, where tree models extrapolate flatly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .feature_engineering import target_to_yield

MAX_RAIN_CHANGE_PCT = 50.0
MAX_TEMP_CHANGE_C = 5.0
RAIN_COLS = ("rain_total_mm", "rain_max_month_mm", "rain_min_month_mm")
TEMP_COLS = ("temp_mean_c", "temp_max_month_c", "temp_min_month_c")


def validate_scenario(rain_pct, temp_c) -> None:
    for name, v, lim in (("rainfall_change_pct", rain_pct, MAX_RAIN_CHANGE_PCT),
                         ("temperature_change_c", temp_c, MAX_TEMP_CHANGE_C)):
        if v is None or isinstance(v, bool) or not isinstance(v, (int, float, np.number)):
            raise ValueError(f"{name} must be a number")
        if not np.isfinite(v):
            raise ValueError(f"{name} must be finite")
        if abs(v) > lim:
            raise ValueError(f"{name}={v} outside supported range +/-{lim}")


@dataclass(frozen=True)
class Scenario:
    name: str
    rainfall_change_pct: float = 0.0
    temperature_change_c: float = 0.0

    def __post_init__(self) -> None:
        validate_scenario(self.rainfall_change_pct, self.temperature_change_c)


def scenarios_from_config(cfg_scen: dict) -> list[Scenario]:
    out = [Scenario("baseline")]
    out += [Scenario(f"rain_{p:+g}pct", rainfall_change_pct=float(p)) for p in cfg_scen["rainfall_change_pct"]]
    out += [Scenario(f"temp_{t:+g}C", temperature_change_c=float(t)) for t in cfg_scen["temperature_change_c"]]
    out += [Scenario(c["name"], float(c["rainfall_change_pct"]), float(c["temperature_change_c"]))
            for c in cfg_scen["combined"]]
    return out


def apply_scenario(X: pd.DataFrame, sc: Scenario) -> pd.DataFrame:
    """Perturb same-year climate features consistently (anomalies recomputed).

    Rainfall is scaled multiplicatively (accumulated quantity, stays >= 0);
    temperatures are shifted additively.
    """
    out = X.copy()
    f = 1.0 + sc.rainfall_change_pct / 100.0
    for col in RAIN_COLS:
        out[col] = out[col] * f
    for col in TEMP_COLS:
        out[col] = out[col] + sc.temperature_change_c
    out["rain_anom_pct"] = 100.0 * (out["rain_total_mm"] / out["rain_total_base_mm"] - 1.0)
    out["temp_anom_c"] = out["temp_mean_c"] - out["temp_mean_base_c"]
    out["rain_x_temp_anom"] = out["rain_anom_pct"] * out["temp_anom_c"]
    return out


def country_climate_ranges(climate_annual: pd.DataFrame) -> pd.DataFrame:
    """Observed 1901-2016 range of annual rainfall / mean temperature per area."""
    return climate_annual.groupby("area").agg(
        rain_min=("rain_total_mm", "min"), rain_max=("rain_total_mm", "max"),
        temp_min=("temp_mean_c", "min"), temp_max=("temp_mean_c", "max"),
    ).reset_index()


def extrapolation_flags(Xs: pd.DataFrame, ranges: pd.DataFrame) -> pd.DataFrame:
    """Flag scenario inputs outside the country's observed historical range."""
    r = Xs[["area"]].merge(ranges, on="area", how="left")
    rain = Xs["rain_total_mm"].to_numpy()
    temp = Xs["temp_mean_c"].to_numpy()
    return pd.DataFrame({
        "rain_outside_history": (rain < r["rain_min"].to_numpy()) | (rain > r["rain_max"].to_numpy()),
        "temp_outside_history": (temp < r["temp_min"].to_numpy()) | (temp > r["temp_max"].to_numpy()),
    }, index=Xs.index)


def run_scenarios(pipe, X_base: pd.DataFrame, scenarios: list[Scenario], ranges: pd.DataFrame,
                  features: list[str], climate_annual: pd.DataFrame | None = None) -> pd.DataFrame:
    """Predictions for each scenario relative to the baseline prediction."""
    base_pred = target_to_yield(pipe.predict(X_base[features]), X_base["log_y_lag1"])
    frames = []
    for sc in scenarios:
        Xs = apply_scenario(X_base, sc)
        pred = target_to_yield(pipe.predict(Xs[features]), Xs["log_y_lag1"])
        flags = extrapolation_flags(Xs, ranges)
        analogue = (has_historical_analogue(Xs, climate_annual) if climate_annual is not None
                    else np.full(len(Xs), np.nan))
        frames.append(pd.DataFrame({
            "scenario": sc.name,
            "rainfall_change_pct": sc.rainfall_change_pct,
            "temperature_change_c": sc.temperature_change_c,
            "area": X_base["area"].to_numpy(),
            "crop": X_base["crop"].to_numpy(),
            "year": X_base["year"].to_numpy(),
            "baseline_pred": base_pred,
            "scenario_pred": pred,
            "pct_change": 100.0 * (pred / base_pred - 1.0),
            "rain_outside_history": flags["rain_outside_history"].to_numpy(),
            "temp_outside_history": flags["temp_outside_history"].to_numpy(),
            "historical_analogue": analogue,
        }))
    return pd.concat(frames, ignore_index=True)


def has_historical_analogue(Xs: pd.DataFrame, climate_annual: pd.DataFrame,
                            rain_tol_pct: float = 10.0, temp_tol_c: float = 0.5) -> np.ndarray:
    """True where the country experienced at least one year (1901-2016) whose
    annual rainfall is within ``rain_tol_pct`` % and mean temperature within
    ``temp_tol_c`` deg C of the scenario *combination*."""
    hist = {a: (g["rain_total_mm"].to_numpy(), g["temp_mean_c"].to_numpy())
            for a, g in climate_annual.groupby("area")}
    out = np.zeros(len(Xs), dtype=bool)
    for i, (area, r, t) in enumerate(Xs[["area", "rain_total_mm", "temp_mean_c"]].itertuples(index=False)):
        if area in hist:
            hr, ht = hist[area]
            out[i] = bool(np.any((np.abs(hr - r) <= rain_tol_pct / 100.0 * r) & (np.abs(ht - t) <= temp_tol_c)))
    return out
