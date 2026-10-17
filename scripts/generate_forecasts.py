"""Phase E: forecasts from the saved models.

* ``reports/forecasts_<year>.csv`` - genuine forecasts for the year after the
  last observed yield, from the history-only deployment model (climate and
  pesticide records end before that year, so they cannot be used).
* ``reports/backtest_2017.csv`` - the benchmark climate+pesticide model
  (trained on 1991-2012) forecasting 2017, a year never used for training,
  tuning, selection or calibration.

    python scripts/generate_forecasts.py
"""

from __future__ import annotations

import json
import sys

import numpy as np

from climate_yield.config import load_config
from climate_yield.evaluate import regression_metrics
from climate_yield.feature_engineering import TARGET
from climate_yield.pipeline import prepare_data
from climate_yield.predict import Forecaster
from climate_yield.uncertainty import coverage_summary


def main() -> int:
    cfg = load_config()
    models = cfg.path("models_dir")
    reports = cfg.path("reports_dir")
    data = prepare_data(cfg, save=False)
    F = data.features

    hist = Forecaster.load(models / "deploy" / "history_forecaster.joblib")
    year = int(hist.bundle["forecast_year"])
    rows = F[(F["year"] == year) & F["log_y_lag1"].notna()].copy()
    pred, warn = hist.predict_frame(rows)
    out = rows[["area", "crop", "year"]].join(pred)
    out["last_observed_hg_ha"] = np.exp(rows["log_y_lag1"]).round(0)
    out["model"] = f"{hist.bundle['algorithm']} (history-only)"
    out.to_csv(reports / f"forecasts_{year}.csv", index=False)
    print(f"{len(out)} forecasts for {year}; warnings: {warn}")

    best = Forecaster.load(models / "benchmark" / "best_model.joblib")
    bt = F[(F["year"] == 2017) & F[TARGET].notna() & F["log_y_lag1"].notna() & F["rain_total_mm_lag1"].notna()].copy()
    p, warn2 = best.predict_frame(bt)
    bt_out = bt[["area", "crop", "year", TARGET]].join(p)
    bt_out.to_csv(reports / "backtest_2017.csv", index=False)
    scale = json.loads((reports / "training_summary.json").read_text())["mase_scale_hg_ha"]
    m = regression_metrics(bt[TARGET], p["yield_pred_hg_ha"], naive_scale=scale)
    naive = regression_metrics(bt[TARGET], np.exp(bt["log_y_lag1"]), naive_scale=scale)
    cov = {lv: coverage_summary(bt[TARGET], p[f"lower_{lv}"], p[f"upper_{lv}"], lv / 100) for lv in (80, 90)}
    summary = {"forecast_year": year, "n_forecasts": len(out), "forecast_warnings": warn,
               "backtest_2017": {"n": len(bt), "model": best.bundle["algorithm"], "metrics": m,
                                 "naive_metrics": naive, "coverage": cov, "warnings": warn2}}
    with open(reports / "forecast_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(json.dumps(summary["backtest_2017"], indent=1, default=str)[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
