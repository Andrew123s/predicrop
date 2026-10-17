import numpy as np
import pytest

from climate_yield.evaluate import mase_scale, regression_metrics
from climate_yield.uncertainty import ConformalCalibrator, conformal_quantile, coverage_summary


def test_metrics_known_values():
    m = regression_metrics([100, 200, 300], [110, 190, 330], naive_scale=20)
    assert m["MAE"] == pytest.approx(50 / 3)
    assert m["RMSE"] == pytest.approx(np.sqrt((100 + 100 + 900) / 3))
    assert m["Bias"] == pytest.approx(10.0)
    assert m["MAPE_pct"] == pytest.approx(100 * np.mean([0.1, 0.05, 0.1]))
    assert m["MASE"] == pytest.approx(m["MAE"] / 20)
    assert m["R2"] == pytest.approx(1 - 1100 / 20000)


def test_percentage_metrics_disabled_near_zero():
    m = regression_metrics([0.0, 10.0], [1.0, 9.0])
    assert np.isnan(m["MAPE_pct"]) and np.isnan(m["MedAPE_pct"])
    m = regression_metrics([0.5, 10.0, 20.0], [1.0, 9.0, 20.0])
    assert np.isnan(m["MAPE_pct"]) and m["MedAPE_pct"] == pytest.approx(10.0)


def test_metric_input_checks():
    with pytest.raises(ValueError):
        regression_metrics([1, 2], [1])
    with pytest.raises(ValueError):
        regression_metrics([1, 2], [1, np.nan])


def test_mase_scale():
    assert mase_scale([10, 12, 15], [9, 10, 12]) == pytest.approx(2.0)


def test_conformal_quantile_finite_sample():
    r = np.arange(1, 11, dtype=float)       # n = 10
    assert conformal_quantile(r, 0.8) == 9.0  # ceil(11 * 0.8) = 9th smallest
    assert conformal_quantile(r, 0.95) == np.inf
    with pytest.raises(ValueError):
        conformal_quantile(r, 1.5)


def test_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    cal = rng.normal(0, 0.2, 5000)
    calib = ConformalCalibrator([0.8, 0.9]).fit(cal)
    pred = np.zeros(20000)
    base = np.log(np.full(20000, 1000.0))
    truth = np.exp(base + rng.normal(0, 0.2, 20000))
    for lv in (0.8, 0.9):
        lo, hi = calib.interval(pred, base, lv)
        assert (lo < hi).all()
        assert coverage_summary(truth, lo, hi, lv)["observed_coverage"] == pytest.approx(lv, abs=0.02)


def test_mondrian_groups_and_fallback():
    res = np.r_[np.full(50, 0.1), np.full(50, 1.0), np.full(5, 5.0)]
    groups = np.r_[["a"] * 50, ["b"] * 50, ["c"] * 5]
    cal = ConformalCalibrator([0.9], min_group_size=30).fit(res, groups)
    hw = cal.half_width(0.9, np.array(["a", "b", "c"]), 3)
    assert hw[0] == pytest.approx(0.1) and hw[1] == pytest.approx(1.0)
    assert hw[2] == pytest.approx(cal.global_q[0.9])  # too few points -> pooled
    with pytest.raises(ValueError):
        cal.half_width(0.5, None, 1)
