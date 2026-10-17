import joblib
import numpy as np
import pandas as pd
import pytest

from climate_yield import feature_engineering as fe
from climate_yield.predict import Forecaster, ForecastInputError, forecast_rows, history_warnings
from climate_yield.scenarios import (Scenario, apply_scenario, country_climate_ranges, extrapolation_flags,
                                     has_historical_analogue, run_scenarios, validate_scenario)
from climate_yield.train import build_pipeline, model_specs
from climate_yield.uncertainty import ConformalCalibrator


@pytest.fixture
def forecaster(synth_bundle, tmp_path):
    M = fe.modelling_rows(synth_bundle["features"], 1991, 2000)
    feats = fe.FEATURE_SETS["conditional"]
    spec = model_specs({"rf_n_estimators": 10, "boosting_n_estimators": 30}, 0)["LightGBM"]
    pipe = build_pipeline(spec["estimator"], feats, False).fit(M[feats], M[fe.MODEL_TARGET])
    cal = ConformalCalibrator([0.8, 0.9], 5).fit(M[fe.MODEL_TARGET] - pipe.predict(M[feats]), M["crop"])
    bundle = {"pipeline": pipe, "features": feats, "feature_set": "conditional", "calibrator": cal,
              "algorithm": "LightGBM",
              "feature_ranges": {c: [float(M[c].min()), float(M[c].max())] for c in feats if c != "crop"}}
    path = tmp_path / "bundle.joblib"
    joblib.dump(bundle, path)
    return Forecaster.load(path), M


def test_missing_artifact_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        Forecaster.load(tmp_path / "nope.joblib")
    joblib.dump({"x": 1}, tmp_path / "bad.joblib")
    with pytest.raises(ValueError):
        Forecaster.load(tmp_path / "bad.joblib")


def test_prediction_is_consistent_and_intervals_ordered(forecaster):
    f, M = forecaster
    X = M[M.year == 2000]
    p1, _ = f.predict_frame(X)
    p2, _ = f.predict_frame(X.copy())
    pd.testing.assert_frame_equal(p1, p2)
    direct = np.exp(X["log_y_lag1"] + f.bundle["pipeline"].predict(X[f.features]))
    np.testing.assert_allclose(p1["yield_pred_hg_ha"], direct)
    assert (p1["lower_90"] <= p1["lower_80"]).all() and (p1["lower_80"] <= p1["yield_pred_hg_ha"]).all()
    assert (p1["yield_pred_hg_ha"] <= p1["upper_80"]).all() and (p1["upper_80"] <= p1["upper_90"]).all()


def test_input_validation(forecaster):
    f, M = forecaster
    X = M[M.year == 2000].copy()
    with pytest.raises(ForecastInputError):
        f.predict_frame(X.drop(columns="rain_total_mm"))
    bad = X.copy()
    bad["log_y_lag1"] = np.nan
    with pytest.raises(ForecastInputError):
        f.predict_frame(bad)
    bad = X.copy()
    bad["crop"] = "Quinoa"
    with pytest.raises(ForecastInputError):
        f.predict_frame(bad)
    bad = X.copy()
    bad["temp_mean_c"] = 99.0
    _, warn = f.predict_frame(bad)
    assert any("outside the training range" in w for w in warn)
    bad = X.copy()
    bad.loc[bad.index[0], "log_pesticides_lag1"] = np.nan
    _, warn = f.predict_frame(bad)
    assert any("imputed" in w for w in warn)


def test_forecast_generation_for_future_year(synth_bundle):
    F = synth_bundle["features"]
    row = forecast_rows(F, "Alpha", "Wheat", 2001)
    assert len(row) == 1 and np.isnan(row["yield_hg_ha"].iloc[0])
    with pytest.raises(ForecastInputError):
        forecast_rows(F, "Alpha", "Rice", 2001)
    with pytest.raises(ForecastInputError):
        forecast_rows(F, "Alpha", "Wheat", 2010)
    with pytest.raises(ForecastInputError):
        forecast_rows(F, "Beta", "Wheat", 1991)  # 1990 missing -> no lag-1
    with pytest.raises(ForecastInputError):
        forecast_rows(F, "Alpha", "Wheat", 2001.5)
    assert history_warnings(F, "Alpha", "Wheat", 1965)  # short history flagged


def test_scenario_input_validation():
    validate_scenario(-20, 2.0)
    for bad in [(-80, 0), (0, 9), (float("nan"), 0), ("10", 0), (None, 1), (True, 0)]:
        with pytest.raises(ValueError):
            validate_scenario(*bad)
    with pytest.raises(ValueError):
        Scenario("x", rainfall_change_pct=60)


def test_apply_scenario_math(synth_bundle):
    M = fe.modelling_rows(synth_bundle["features"], 1991, 2000)
    X = M.head(5)
    S = apply_scenario(X, Scenario("s", -10, 2.0))
    np.testing.assert_allclose(S["rain_total_mm"], 0.9 * X["rain_total_mm"])
    np.testing.assert_allclose(S["temp_max_month_c"], X["temp_max_month_c"] + 2)
    np.testing.assert_allclose(S["temp_anom_c"], X["temp_anom_c"] + 2)
    np.testing.assert_allclose(S["rain_anom_pct"], 100 * (S["rain_total_mm"] / X["rain_total_base_mm"] - 1))
    pd.testing.assert_series_equal(S["log_y_lag1"], X["log_y_lag1"])        # history untouched
    pd.testing.assert_series_equal(S["temp_mean_c_lag1"], X["temp_mean_c_lag1"])  # lagged climate untouched


def test_scenarios_and_extrapolation(forecaster, synth_bundle):
    f, M = forecaster
    X = M[M.year == 2000].reset_index(drop=True)
    ranges = country_climate_ranges(synth_bundle["annual"])
    res = run_scenarios(f.bundle["pipeline"], X, [Scenario("baseline"), Scenario("hot", 0, 5.0)], ranges,
                        f.features, synth_bundle["annual"])
    base = res[res.scenario == "baseline"]
    np.testing.assert_allclose(base["pct_change"], 0, atol=1e-9)
    assert base["historical_analogue"].all()
    hot = res[res.scenario == "hot"]
    assert hot["temp_outside_history"].all()  # +5 C beyond the synthetic history
    flags = extrapolation_flags(apply_scenario(X, Scenario("w", 50, 0)), ranges)
    assert flags["rain_outside_history"].any()
    assert not has_historical_analogue(apply_scenario(X, Scenario("hot", 0, 5.0)), synth_bundle["annual"]).any()
