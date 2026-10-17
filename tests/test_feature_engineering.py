import numpy as np
import pandas as pd
import pytest

from climate_yield import feature_engineering as fe
from climate_yield.data_integration import integrate


def _series(F, area, crop):
    return F[(F.area == area) & (F.crop == crop)].set_index("year")


def test_lags_are_calendar_aligned_within_series(synth_bundle):
    F = synth_bundle["features"]
    s = _series(F, "Alpha", "Wheat")
    assert s.loc[1980, "log_y_lag1"] == pytest.approx(np.log(s.loc[1979, "yield_hg_ha"]))
    assert s.loc[1980, "log_y_lag3"] == pytest.approx(np.log(s.loc[1977, "yield_hg_ha"]))
    # Beta/Wheat 1990 is missing: the 1991 lag must be NaN, not the 1989 value
    b = _series(F, "Beta", "Wheat")
    assert np.isnan(b.loc[1991, "log_y_lag1"])
    assert b.loc[1991, "log_y_lag2"] == pytest.approx(np.log(b.loc[1989, "yield_hg_ha"]))
    # first year of every series has no lag (never borrows another series)
    first = F.groupby(["area", "crop"]).head(1)
    assert first["log_y_lag1"].isna().all()


def test_rolling_features_use_only_past_values(synth_bundle):
    s = _series(synth_bundle["features"], "Gamma", "Maize")
    logs = np.log(s["yield_hg_ha"])
    assert s.loc[1990, "roll3_mean_log"] == pytest.approx(logs.loc[1987:1989].mean())
    assert s.loc[1990, "roll5_std_log"] == pytest.approx(logs.loc[1985:1989].std())
    assert s.loc[1990, "trend_4y"] == pytest.approx((logs.loc[1989] - logs.loc[1985]) / 4)
    assert s.loc[1990, "n_prior_obs"] == 30


def test_changing_current_or_future_yield_does_not_change_features(synth_yield, synth_bundle, synth_pesticides):
    annual, base = synth_bundle["annual"], synth_bundle["baseline"]
    feats = fe.FEATURE_SETS["conditional"]
    y2 = synth_yield.copy()
    mask = (y2.area == "Alpha") & (y2.crop == "Wheat") & (y2.year >= 1990)
    y2.loc[mask, "yield_hg_ha"] *= 10  # tamper with target year and all later years
    panel2, _ = integrate(y2, synth_pesticides, annual)
    F1 = synth_bundle["features"]
    F2 = fe.build_feature_table(panel2, annual, base, synth_pesticides, extend_to=2001)
    r1 = _series(F1, "Alpha", "Wheat").loc[1990, feats]
    r2 = _series(F2, "Alpha", "Wheat").loc[1990, feats]
    pd.testing.assert_series_equal(r1, r2)
    # ...while the 1991 lag does see the 1990 change
    assert _series(F2, "Alpha", "Wheat").loc[1991, "log_y_lag1"] != _series(F1, "Alpha", "Wheat").loc[1991, "log_y_lag1"]


def test_target_never_in_feature_sets():
    for feats in fe.FEATURE_SETS.values():
        assert fe.TARGET not in feats and fe.MODEL_TARGET not in feats and "log_y" not in feats


def test_forecast_set_excludes_same_year_climate():
    for c in fe.SAME_YEAR_CLIMATE_FEATURES:
        assert c not in fe.FEATURE_SETS["forecast"] and c not in fe.FEATURE_SETS["history"]
        assert c in fe.FEATURE_SETS["conditional"]


def test_climate_and_pesticide_lags(synth_bundle, synth_pesticides):
    F, annual = synth_bundle["features"], synth_bundle["annual"]
    s = _series(F, "Beta", "Maize")
    prev = annual[(annual.area == "Beta") & (annual.year == 1994)].iloc[0]
    same = annual[(annual.area == "Beta") & (annual.year == 1995)].iloc[0]
    assert s.loc[1995, "rain_total_mm_lag1"] == pytest.approx(prev.rain_total_mm)
    assert s.loc[1995, "rain_total_mm"] == pytest.approx(same.rain_total_mm)
    p = synth_pesticides.set_index(["area", "year"])["pesticides_tonnes"]
    assert s.loc[1995, "log_pesticides_lag1"] == pytest.approx(np.log1p(p[("Beta", 1994)]))
    assert np.isnan(s.loc[1990, "log_pesticides_lag1"])  # pesticides start 1990 -> no 1989 value


def test_target_definition_and_back_transform(synth_bundle):
    s = _series(synth_bundle["features"], "Alpha", "Maize")
    g = s.loc[1985, fe.MODEL_TARGET]
    assert g == pytest.approx(np.log(s.loc[1985, "yield_hg_ha"] / s.loc[1984, "yield_hg_ha"]))
    assert fe.target_to_yield([g], [s.loc[1985, "log_y_lag1"]])[0] == pytest.approx(s.loc[1985, "yield_hg_ha"])


def test_extend_to_creates_future_rows(synth_bundle):
    s = _series(synth_bundle["features"], "Alpha", "Wheat")
    assert 2001 in s.index and np.isnan(s.loc[2001, "yield_hg_ha"])
    assert s.loc[2001, "log_y_lag1"] == pytest.approx(np.log(s.loc[2000, "yield_hg_ha"]))


def test_modelling_rows_require_target_lag_and_climate(synth_bundle):
    M = fe.modelling_rows(synth_bundle["features"], 1991, 2000)
    assert M["year"].between(1991, 2000).all()
    assert M[["yield_hg_ha", "log_y_lag1", "rain_total_mm", "rain_total_mm_lag1"]].notna().all().all()
    assert not ((M.area == "Beta") & (M.crop == "Wheat") & (M.year.isin([1990, 1991]))).any()
