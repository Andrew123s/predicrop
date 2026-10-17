"""Phases E-F: climate-sensitivity scenarios and the Markdown reports.

Writes reports/scenario_results.csv, reports/scenario_report.md and
reports/forecasting_report.md from the saved models and training outputs.

    python scripts/generate_reports.py
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from climate_yield.config import load_config  # noqa: E402
from climate_yield.pipeline import prepare_data  # noqa: E402
from climate_yield.predict import Forecaster  # noqa: E402
from climate_yield.reporting import df_to_markdown, save  # noqa: E402
from climate_yield.scenarios import country_climate_ranges, run_scenarios, scenarios_from_config  # noqa: E402


def scenario_analysis(cfg, data, figs, reports) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    model = Forecaster.load(cfg.path("models_dir") / "deploy" / "conditional_scenario.joblib")
    M = data.modelling
    base_year = int(M["year"].max())
    X = M[M["year"] == base_year].reset_index(drop=True)
    scen = scenarios_from_config(cfg["scenarios"])
    res = run_scenarios(model.bundle["pipeline"], X, scen, country_climate_ranges(data.climate_annual),
                        model.features, data.climate_annual)
    res.to_csv(reports / "scenario_results.csv", index=False)
    s = res[res["scenario"] != "baseline"]
    summ = s.groupby(["scenario", "rainfall_change_pct", "temperature_change_c"]).agg(
        median_pct_change=("pct_change", "median"),
        p10_pct_change=("pct_change", lambda v: v.quantile(0.1)),
        p90_pct_change=("pct_change", lambda v: v.quantile(0.9)),
        share_rain_outside_history=("rain_outside_history", "mean"),
        share_temp_outside_history=("temp_outside_history", "mean"),
        share_with_historical_analogue=("historical_analogue", "mean"),
        n=("pct_change", "size"),
    ).reset_index()
    by_crop = s.pivot_table(index="crop", columns="scenario", values="pct_change", aggfunc="median")
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(by_crop, annot=True, fmt=".1f", cmap="RdBu", center=0, ax=ax,
                cbar_kws={"label": "median % change vs baseline prediction"})
    ax.set_title(f"Model sensitivity to hypothetical same-year climate changes (baseline year {base_year})")
    save(fig, figs / "scenario_sensitivity_by_crop.png")
    # response curves for temperature and rainfall (grid), median across series
    grid = []
    from climate_yield.scenarios import Scenario
    for dr in (-30, -20, -10, 0, 10, 20, 30):
        for dt in (0.0, 1.0, 2.0, 3.0):
            r = run_scenarios(model.bundle["pipeline"], X, [Scenario("g", dr, dt)],
                              country_climate_ranges(data.climate_annual), model.features)
            grid.append({"rain_pct": dr, "temp_c": dt, "median_pct_change": r["pct_change"].median()})
    grid = pd.DataFrame(grid)
    grid.to_csv(reports / "scenario_grid.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for dt, g in grid.groupby("temp_c"):
        ax.plot(g["rain_pct"], g["median_pct_change"], marker="o", label=f"+{dt:g} deg C")
    ax.axhline(0, color="grey", lw=1)
    ax.set_xlabel("rainfall change (%)")
    ax.set_ylabel("median % change in predicted yield")
    ax.set_title("Scenario response surface (median across series)")
    ax.legend()
    save(fig, figs / "scenario_response.png")
    meta = {"base_year": base_year, "n_series": len(X), "algorithm": model.bundle["algorithm"],
            "train_years": model.bundle["train_years"]}
    return summ, grid, meta


def scenario_report(summ, grid, meta, reports) -> None:
    L = ["# Climate-scenario (sensitivity) report", "",
         "**What this is:** a sensitivity analysis of a statistical model. Each scenario changes the *same-year* annual "
         "rainfall and temperature inputs of the conditional model while every other input (yield history, lagged "
         "climate, pesticides) is held at its observed value. The reported number is the change in the model's "
         "predicted yield relative to its own prediction for observed conditions.", "",
         "**What this is not:** a climate projection or an attribution of yield changes to climate. Real climate change "
         "alters seasonality, extremes, CO2, pests and farmer adaptation, none of which this model represents.", "",
         "## Set-up", "",
         f"- Model: {meta['algorithm']} trained on the `conditional` feature set, target years "
         f"{meta['train_years'][0]}-{meta['train_years'][1]}.",
         f"- Baseline: observed conditions of {meta['base_year']} (the last year with climate data) for {meta['n_series']:,} "
         "(country, crop) series.",
         "- Rainfall changes are multiplicative on annual total, wettest- and driest-month rainfall; temperature changes "
         "are additive on annual mean, warmest- and coldest-month means; anomalies vs 1901-1960 are recomputed. "
         "Scenario magnitudes are set in `config/config.yaml` and validated (|rain| <= 50%, |temp| <= 5 deg C).",
         "- Extrapolation checks: (i) is the scenario value outside the country's observed 1901-2016 range? "
         "(ii) did the country ever experience a year within +/-10% rainfall *and* +/-0.5 deg C of the scenario "
         "combination (a historical analogue)?", "",
         "## Results", "", df_to_markdown(summ, "{:,.3f}"), "",
         "Response surface (median % change across series):", "",
         df_to_markdown(grid.pivot(index="rain_pct", columns="temp_c", values="median_pct_change").reset_index(), "{:.2f}"), "",
         "Figures: `figures/scenario_sensitivity_by_crop.png`, `figures/scenario_response.png`. "
         "Per-series results: `scenario_results.csv`.", "",
         "## Interpretation", "",
         f"- Median responses are below 2% in every configured scenario (largest: "
         f"{summ.loc[summ['median_pct_change'].abs().idxmax(), 'scenario']} at "
         f"{summ['median_pct_change'].abs().max():.2f}%). Drier scenarios lower predictions slightly; warming scenarios "
         f"barely move them because {100 * summ.set_index('scenario').loc['temp_+1C', 'share_temp_outside_history']:.0f}% "
         f"of series are pushed beyond their warmest observed year by just +1 deg C ({meta['base_year']} was among the warmest observed "
         "years in most countries), where a tree ensemble cannot respond. The near-zero warming response is therefore an "
         "extrapolation artefact, not evidence that warming is harmless.",
         "- The model's sensitivity to same-year climate is small compared with year-to-year yield variability. This agrees "
         "with the EDA (weak associations) and the feature-set comparison in `forecasting_report.md`.",
         "- Tree ensembles are piecewise-constant: beyond the range of training data the response flattens, so "
         "large perturbations understate (or arbitrarily cap) any true response. Rows flagged as outside the "
         "country's history or without a historical analogue should be treated as extrapolation.",
         "- Annual national climate means mix growing and non-growing seasons and irrigated and rain-fed land, which "
         "dilutes real weather signals.", ""]
    (reports / "scenario_report.md").write_text("\n".join(L), encoding="utf-8")


def forecasting_report(reports) -> None:
    comp = pd.read_csv(reports / "model_comparison.csv")
    summ = json.loads((reports / "training_summary.json").read_text())
    abl = pd.read_csv(reports / "feature_set_comparison.csv")
    cov = pd.read_csv(reports / "interval_coverage.csv")
    covc = pd.read_csv(reports / "interval_coverage_by_crop.csv")
    covy = pd.read_csv(reports / "interval_coverage_by_year.csv")
    byc = pd.read_csv(reports / "metrics_by_crop.csv")
    perm = pd.read_csv(reports / "permutation_importance.csv")
    fs = json.loads((reports / "forecast_summary.json").read_text()) if (reports / "forecast_summary.json").exists() else None
    shap_path = reports / "shap_importance.csv"
    best = summ["best_model"]
    cols = ["Model", "Validation MAE", "Holdout Validation MAE (2009-2012)", "Test MAE", "Test RMSE", "Test R2",
            "Prediction Bias", "Test MedAPE %", "Test MAE log", "Test MASE", "Training Time (s)", "Prediction Time (s)"]
    naive = comp.set_index("Model").loc["Naive (last value)"]
    b = comp.set_index("Model").loc[best]
    L = ["# Forecasting report", "",
         f"Training mode: **{summ['mode']}**. Every number in this report was produced by `scripts/train_models.py` "
         "and `scripts/generate_forecasts.py` on the supplied data.", "",
         "## Task", "",
         "- **Target:** FAO crop yield in hg/ha for a (country, crop) series in year *t*.",
         "- **Horizon:** one year ahead. Forecasts are issued at the start of year *t* using yields up to *t-1*, "
         "calendar-year climate of *t-1* and national pesticide use of *t-1*.",
         "- **Primary task:** forecasting future yields of *known* (country, crop) series. Unseen countries or crops "
         "are out of scope (every feature is anchored on the series' own history).",
         "- Models predict log growth log(y_t / y_{t-1}); forecasts are y_{t-1} x exp(prediction).", "",
         "## Validation design", "",
         f"- Chronological partitions of target years: train {summ['split']['train']}, validation {summ['split']['validation']}, "
         f"test {summ['split']['test']} (rows: {summ['rows']['train']:,} / {summ['rows']['validation']:,} / {summ['rows']['test']:,}).",
         f"- Hyper-parameters tuned by random search scored on expanding-window CV over the development years with "
         f"validation folds {summ['cv_folds']}; each fold trains only on earlier years.",
         "- `Validation MAE` = mean CV MAE of the selected configuration; the model with the lowest value is selected. "
         "The test period was used once, after selection.",
         "- All models are evaluated on identical rows.", "",
         "## Model comparison (test period)", "", df_to_markdown(comp[cols], "{:,.3f}"), "",
         f"**Selected model (lowest validation MAE): {best}.** On the test years it has MAE {b['Test MAE']:,.0f} hg/ha vs "
         f"{naive['Test MAE']:,.0f} for the naive last-value forecast ({100 * (1 - b['Test MAE'] / naive['Test MAE']):.1f}% lower), "
         f"RMSE {b['Test RMSE']:,.0f} vs {naive['Test RMSE']:,.0f}, R2 {b['Test R2']:.3f} vs {naive['Test R2']:.3f}, and "
         f"median absolute percentage error {b['Test MedAPE %']:.2f}% vs {naive['Test MedAPE %']:.2f}%.", "",
         *([("The naive forecast has the lower *median* percentage error. The model's gains come from the tail: it "
             "partly anticipates mean reversion after large shocks (lag-1 autocorrelation of growth is negative), which "
             "lowers MAE, RMSE and mean log error, while for the many stable series - including long runs of identical "
             "FAO values, where persistence is exactly right - it adds small deviations. Persistence is therefore a "
             "strong baseline that the models beat on average error but not uniformly."), ""]
           if b["Test MedAPE %"] > naive["Test MedAPE %"] else []),
         "Metric notes: MAE/RMSE/bias are in hg/ha (10,000 hg/ha = 1 t/ha) and are dominated by high-mass crops "
         "(potatoes, carrots, cauliflowers). R2 on yield levels is high for every model because between-series level "
         "differences are huge; it says little about forecast skill, so MASE (MAE relative to the naive forecast's "
         "training-period MAE) and MAE of log yield are also reported. MAPE is reported but is inflated by a few "
         "near-zero / erratic yields; median APE is more robust.", "",
         "## Does climate or pesticide information help?", "",
         f"Same algorithm and hyper-parameters ({best}), same rows:", "", df_to_markdown(abl, "{:,.3f}"), "",
         "- `history`: lagged yields only. `forecast`: + lagged climate & pesticides (the deployable benchmark model). "
         "`conditional`: + **same-year** climate, which is *not* known at forecast time; it is an upper bound on what "
         "perfect seasonal-climate knowledge could add and is used only for scenarios.", "",
         "## Bias", "",
         f"- Mean prediction error of {best} on the test years: {b['Prediction Bias']:,.0f} hg/ha "
         f"(naive: {naive['Prediction Bias']:,.0f}). Positive = overprediction.",
         "- By crop (test):", "", df_to_markdown(byc[["crop", "model", "n", "MAE", "Bias", "MedAPE_pct"]], "{:,.1f}"), "",
         "## Uncertainty (split conformal)", "",
         "- Residuals of the model fitted on 1991-2008 and evaluated on 2009-2012 calibrate the intervals; test labels "
         "are never used for calibration. Per-crop (Mondrian) quantiles when a crop has >= 30 calibration points.",
         "", df_to_markdown(cov, "{:,.3f}"), "",
         "Observed coverage by test year (per-crop method):", "",
         df_to_markdown(covy[["year", "nominal_coverage", "observed_coverage", "median_relative_width", "n"]], "{:.3f}"), "",
         "Lowest observed coverage by crop:", "",
         df_to_markdown(covc.sort_values("observed_coverage").head(8)[["crop", "nominal_coverage", "observed_coverage", "n"]], "{:.3f}"), "",
         "Assumption check: split conformal requires calibration and test residuals to be exchangeable. "
         "`residual_spread_by_year.csv` compares the median absolute residual per year in the calibration and test windows; "
         "differences there translate directly into under- or over-coverage.", "",
         "## Explainability", "",
         f"- Model explained: {summ['explainability']['model']} (best tree model). SHAP: {summ['explainability'].get('shap')}.",
         "- Permutation importance (MAE increase in hg/ha when a column is shuffled, test sample):", "",
         df_to_markdown(perm.head(12), "{:,.1f}"), ""]
    if shap_path.exists():
        L += ["- Mean |SHAP| (log-growth units, grouped by source column):", "",
              df_to_markdown(pd.read_csv(shap_path).head(12), "{:.4f}"), ""]
    L += ["Importance is predictive reliance, not causal effect; correlated predictors share or swap importance.", ""]
    if fs:
        bt = fs["backtest_2017"]
        L += ["## Extra out-of-sample check: 2017", "",
              f"- The benchmark model (trained 1991-2012) forecasting {bt['n']:,} series in 2017: MAE "
              f"{bt['metrics']['MAE']:,.0f} hg/ha (naive {bt['naive_metrics']['MAE']:,.0f}), median APE "
              f"{bt['metrics']['MedAPE_pct']:.2f}% (naive {bt['naive_metrics']['MedAPE_pct']:.2f}%), 90% interval coverage "
              f"{bt['coverage']['90']['observed_coverage']:.3f}, 80% coverage {bt['coverage']['80']['observed_coverage']:.3f}.",
              f"- Forecasts for {fs['forecast_year']} ({fs['n_forecasts']:,} series) from the history-only deployment model are in "
              f"`forecasts_{fs['forecast_year']}.csv`; they cannot be evaluated because no {fs['forecast_year']} yields are available.", ""]
    L += ["## Figures", "",
          "`figures/model_comparison.png`, `actual_vs_predicted.png`, `forecast_examples.png`, `residual_distribution.png`, "
          "`residuals_vs_predicted.png`, `error_over_time.png`, `medape_by_crop.png`, `interval_coverage_by_crop.png`, "
          "`permutation_importance.png`, `shap_importance.png`, `shap_beeswarm.png`, `shap_waterfall_example.png`.", ""]
    (reports / "forecasting_report.md").write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    cfg = load_config()
    reports, figs = cfg.path("reports_dir"), cfg.path("figures_dir")
    data = prepare_data(cfg, save=False)
    summ, grid, meta = scenario_analysis(cfg, data, figs, reports)
    print(summ.to_string())
    scenario_report(summ, grid, meta, reports)
    forecasting_report(reports)
    print("wrote reports/scenario_report.md and reports/forecasting_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
