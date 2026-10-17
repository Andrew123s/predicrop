"""Leakage-safe feature construction for one-year-ahead yield forecasting.

Forecast issue time: the start of target year ``t``. Every *forecast* feature
uses only information up to the end of year ``t-1``:

* lagged yields of the **same** (area, crop) series, aligned on calendar year
  (a missing year yields NaN, never the previous available row);
* trailing (right-aligned, non-centred) rolling statistics over ``t-k..t-1``;
* calendar-year climate of ``t-1`` and anomalies against a fixed 1901-1960
  climatology (which precedes every target year);
* national pesticide use of ``t-1``.

Same-year (``t``) climate is *not* known at forecast issue time. It is exposed
only through the separate ``conditional`` feature set, used for climate-scenario
(what-if) analysis and clearly labelled as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_integration import CLIMATE_FEATURES

GROUP = ["area", "crop"]

HISTORY_FEATURES = [
    "log_y_lag1",
    "log_y_lag2",
    "log_y_lag3",
    "growth_lag1",
    "roll3_mean_log",
    "roll5_mean_log",
    "roll5_std_log",
    "dev_from_roll5",
    "trend_4y",
    "n_prior_obs",
]
LAGGED_CLIMATE_FEATURES = [f"{c}_lag1" for c in CLIMATE_FEATURES] + [
    "rain_anom_pct_lag1",
    "temp_anom_c_lag1",
    "rain_total_base_mm",
    "temp_mean_base_c",
]
PESTICIDE_FEATURES = ["log_pesticides_lag1", "pesticides_growth_lag1"]
SAME_YEAR_CLIMATE_FEATURES = CLIMATE_FEATURES + ["rain_anom_pct", "temp_anom_c", "rain_x_temp_anom"]
CATEGORICAL_FEATURES = ["crop"]

FEATURE_SETS = {
    "history": HISTORY_FEATURES + CATEGORICAL_FEATURES,
    "forecast": HISTORY_FEATURES + LAGGED_CLIMATE_FEATURES + PESTICIDE_FEATURES + CATEGORICAL_FEATURES,
    "conditional": HISTORY_FEATURES + LAGGED_CLIMATE_FEATURES + PESTICIDE_FEATURES
    + SAME_YEAR_CLIMATE_FEATURES + CATEGORICAL_FEATURES,
}

TARGET = "yield_hg_ha"
MODEL_TARGET = "target_log_growth"   # log(y_t) - log(y_{t-1})

FEATURE_DOCS = {
    "log_y_lag1": "log yield in t-1 (same area & crop)",
    "log_y_lag2": "log yield in t-2",
    "log_y_lag3": "log yield in t-3",
    "growth_lag1": "log growth from t-2 to t-1",
    "roll3_mean_log": "mean log yield over t-3..t-1 (>=2 obs)",
    "roll5_mean_log": "mean log yield over t-5..t-1 (>=2 obs)",
    "roll5_std_log": "std of log yield over t-5..t-1 (>=3 obs)",
    "dev_from_roll5": "log_y_lag1 minus roll5_mean_log",
    "trend_4y": "average annual log growth from t-5 to t-1",
    "n_prior_obs": "number of yield observations before t",
    "rain_total_mm_lag1": "annual precipitation total in t-1 (sum of 12 monthly mm values)",
    "rain_max_month_mm_lag1": "wettest month precipitation in t-1 (mm)",
    "rain_min_month_mm_lag1": "driest month precipitation in t-1 (mm)",
    "temp_mean_c_lag1": "mean of 12 monthly mean temperatures in t-1 (deg C)",
    "temp_max_month_c_lag1": "warmest monthly mean temperature in t-1 (deg C)",
    "temp_min_month_c_lag1": "coldest monthly mean temperature in t-1 (deg C)",
    "rain_anom_pct_lag1": "t-1 rainfall as % deviation from 1901-1960 country mean",
    "temp_anom_c_lag1": "t-1 mean temperature minus 1901-1960 country mean (deg C)",
    "rain_total_base_mm": "1901-1960 mean annual rainfall of the country (static)",
    "temp_mean_base_c": "1901-1960 mean annual temperature of the country (static)",
    "log_pesticides_lag1": "log(1 + national pesticide use, tonnes) in t-1 (all crops, not per ha)",
    "pesticides_growth_lag1": "log change in national pesticide use from t-2 to t-1",
    "rain_total_mm": "SAME-YEAR annual rainfall (conditional/scenario model only)",
    "rain_max_month_mm": "SAME-YEAR wettest month rainfall (conditional only)",
    "rain_min_month_mm": "SAME-YEAR driest month rainfall (conditional only)",
    "temp_mean_c": "SAME-YEAR mean temperature (conditional only)",
    "temp_max_month_c": "SAME-YEAR warmest-month mean temperature (conditional only)",
    "temp_min_month_c": "SAME-YEAR coldest-month mean temperature (conditional only)",
    "rain_anom_pct": "SAME-YEAR rainfall anomaly vs 1901-1960, % (conditional only)",
    "temp_anom_c": "SAME-YEAR temperature anomaly vs 1901-1960, deg C (conditional only)",
    "rain_x_temp_anom": "rain_anom_pct x temp_anom_c interaction (conditional only)",
    "crop": "crop name (one-hot encoded inside the model pipeline)",
}


def _complete_grid(yield_panel: pd.DataFrame, extend_to: int | None) -> pd.DataFrame:
    """Reindex every (area, crop) series onto consecutive calendar years.

    Grid rows added for missing years carry NaN yield, so ``shift(k)`` on the
    grid is an exact calendar lag of ``k`` years within the series.
    """
    frames = []
    for (area, crop), g in yield_panel.groupby(GROUP, sort=True):
        last = max(int(g["year"].max()), extend_to or 0)
        years = np.arange(int(g["year"].min()), last + 1)
        idx = pd.DataFrame({"area": area, "crop": crop, "year": years})
        frames.append(idx.merge(g[["year", TARGET]], on="year", how="left"))
    return pd.concat(frames, ignore_index=True)


def add_yield_history(grid: pd.DataFrame) -> pd.DataFrame:
    df = grid.sort_values(GROUP + ["year"]).reset_index(drop=True)
    df["log_y"] = np.log(df[TARGET])
    g = df.groupby(GROUP, sort=False)["log_y"]
    for k in (1, 2, 3, 5):
        df[f"log_y_lag{k}"] = g.shift(k)
    past = g.shift(1)  # strictly prior values only
    pg = past.groupby([df["area"], df["crop"]], sort=False)
    df["roll3_mean_log"] = pg.transform(lambda s: s.rolling(3, min_periods=2).mean())
    df["roll5_mean_log"] = pg.transform(lambda s: s.rolling(5, min_periods=2).mean())
    df["roll5_std_log"] = pg.transform(lambda s: s.rolling(5, min_periods=3).std())
    df["n_prior_obs"] = pg.transform(lambda s: s.notna().cumsum())
    df["growth_lag1"] = df["log_y_lag1"] - df["log_y_lag2"]
    df["dev_from_roll5"] = df["log_y_lag1"] - df["roll5_mean_log"]
    df["trend_4y"] = (df["log_y_lag1"] - df["log_y_lag5"]) / 4.0
    return df.drop(columns=["log_y_lag5"])


def climate_feature_table(climate_annual: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Area-year climate features: same-year values and t-1 lags, plus anomalies."""
    c = climate_annual.merge(baseline, on="area", how="left", validate="m:1")
    c["rain_anom_pct"] = 100.0 * (c["rain_total_mm"] / c["rain_total_base_mm"] - 1.0)
    c["temp_anom_c"] = c["temp_mean_c"] - c["temp_mean_base_c"]
    c["rain_x_temp_anom"] = c["rain_anom_pct"] * c["temp_anom_c"]
    lag_cols = CLIMATE_FEATURES + ["rain_anom_pct", "temp_anom_c"]
    lagged = c[["area", "year"] + lag_cols].copy()
    lagged["year"] = lagged["year"] + 1   # value of t-1 attached to row t
    lagged = lagged.rename(columns={col: f"{col}_lag1" for col in lag_cols})
    same = c[["area", "year", "rain_total_base_mm", "temp_mean_base_c"] + SAME_YEAR_CLIMATE_FEATURES]
    # Outer join so that a target year one past the last climate year still
    # receives its t-1 climate (needed for real forecasts).
    out = same.merge(lagged, on=["area", "year"], how="outer", validate="1:1")
    static = baseline.set_index("area")
    for col in ("rain_total_base_mm", "temp_mean_base_c"):
        out[col] = out["area"].map(static[col])
    return out


def pesticide_feature_table(pesticides: pd.DataFrame) -> pd.DataFrame:
    p = pesticides[["area", "year", "pesticides_tonnes"]].copy()
    p["log_p"] = np.log1p(p["pesticides_tonnes"])
    lag1 = p[["area", "year", "log_p"]].assign(year=p["year"] + 1).rename(columns={"log_p": "log_pesticides_lag1"})
    lag2 = p[["area", "year", "log_p"]].assign(year=p["year"] + 2).rename(columns={"log_p": "_lp2"})
    out = lag1.merge(lag2, on=["area", "year"], how="outer", validate="1:1")
    out["pesticides_growth_lag1"] = out["log_pesticides_lag1"] - out["_lp2"]
    return out.drop(columns="_lp2")


def build_feature_table(
    yield_panel: pd.DataFrame,
    climate_annual: pd.DataFrame,
    baseline: pd.DataFrame,
    pesticides: pd.DataFrame,
    extend_to: int | None = None,
) -> pd.DataFrame:
    """Full feature table on the calendar grid of every series.

    ``extend_to`` appends future target years (yield NaN) so that forecasts can
    be generated for years after the last observation.
    """
    grid = _complete_grid(yield_panel[GROUP + ["year", TARGET]], extend_to)
    df = add_yield_history(grid)
    areas = set(df["area"])
    clim = climate_feature_table(climate_annual[climate_annual["area"].isin(areas)], baseline)
    df = df.merge(clim, on=["area", "year"], how="left", validate="m:1")
    df = df.merge(pesticide_feature_table(pesticides), on=["area", "year"], how="left", validate="m:1")
    df[MODEL_TARGET] = df["log_y"] - df["log_y_lag1"]
    return df


def modelling_rows(features: pd.DataFrame, first_year: int, last_year: int) -> pd.DataFrame:
    """Rows eligible for the benchmark: observed target and lag-1, plus climate for
    both t-1 and t, so that every feature set is evaluated on identical rows."""
    m = (
        features[TARGET].notna()
        & features["log_y_lag1"].notna()
        & features["year"].between(first_year, last_year)
        & features["rain_total_mm"].notna()
        & features["rain_total_mm_lag1"].notna()
    )
    return features.loc[m].reset_index(drop=True)


def target_to_yield(pred_log_growth: np.ndarray, log_y_lag1: np.ndarray) -> np.ndarray:
    """Back-transform predicted log growth to yield in hg/ha."""
    return np.exp(np.asarray(log_y_lag1, dtype=float) + np.asarray(pred_log_growth, dtype=float))
