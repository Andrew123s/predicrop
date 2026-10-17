"""Interactive front-end for the climate-yield forecasting platform.

    streamlit run app/streamlit_app.py

Uses saved model artifacts (run scripts/train_models.py first); nothing is
retrained when the app starts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from climate_yield.config import load_config  # noqa: E402
from climate_yield.feature_engineering import FEATURE_DOCS  # noqa: E402
from climate_yield.pipeline import prepare_data  # noqa: E402
from climate_yield.predict import Forecaster, ForecastInputError, forecast_rows, history_warnings  # noqa: E402
from climate_yield.scenarios import (MAX_RAIN_CHANGE_PCT, MAX_TEMP_CHANGE_C, Scenario,  # noqa: E402
                                     country_climate_ranges, run_scenarios)

st.set_page_config(page_title="Crop-yield forecasting", page_icon=":ear_of_rice:", layout="wide")
CFG = load_config()
REPORTS = CFG.path("reports_dir")
FIGS = CFG.path("figures_dir")
MODELS = CFG.path("models_dir")
UNIT = CFG["task"]["target_unit"]


# ---------------------------------------------------------------- cached loaders
@st.cache_resource(show_spinner="Loading and integrating the four datasets...")
def get_data():
    return prepare_data(CFG, save=False)


@st.cache_resource(show_spinner="Loading model...")
def get_model(rel: str):
    return Forecaster.load(MODELS / rel)


@st.cache_data
def read_csv(name: str) -> pd.DataFrame | None:
    p = REPORTS / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def read_json(name: str) -> dict | None:
    p = REPORTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def show_figure(name: str, caption: str | None = None) -> None:
    p = FIGS / name
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.info(f"Figure `{name}` not found - run the scripts listed in the README.")


def try_model(rel: str) -> Forecaster | None:
    try:
        return get_model(rel)
    except FileNotFoundError:
        st.error(f"Model artifact `models/{rel}` is missing. Run `python scripts/train_models.py` first.")
    except Exception as exc:  # corrupt / incompatible artifact
        st.error(f"Could not load `models/{rel}`: {exc}")
    return None


def fmt(v: float) -> str:
    return f"{v:,.0f} {UNIT} ({v / 10000:,.2f} t/ha)"


# ---------------------------------------------------------------- pages
def page_overview(data) -> None:
    st.title("Climate & agricultural yield forecasting")
    st.markdown(
        "One-year-ahead forecasts of national crop yields from historical FAO yield and pesticide data and "
        "World Bank-style monthly climate data. The platform separates **forecasting** (using only information "
        "available before the target year) from **relationship / sensitivity analysis** (how the model responds "
        "to hypothetical same-year climate)."
    )
    t = data.tidy
    summary = pd.DataFrame([
        {"file": "yield.csv", "rows": len(t["yield"]), "years": f"{t['yield'].year.min()}-{t['yield'].year.max()}",
         "content": "FAO yield by country x crop", "unit": "hg/ha"},
        {"file": "pesticides.csv", "rows": len(t["pesticides"]),
         "years": f"{t['pesticides'].year.min()}-{t['pesticides'].year.max()}",
         "content": "national pesticide use, all crops", "unit": "tonnes"},
        {"file": "rainfall.csv", "rows": len(t["rainfall"]), "years": f"{t['rainfall'].year.min()}-{t['rainfall'].year.max()}",
         "content": "monthly country precipitation", "unit": "mm"},
        {"file": "temp.csv", "rows": len(t["temperature"]),
         "years": f"{t['temperature'].year.min()}-{t['temperature'].year.max()}",
         "content": "monthly country mean temperature", "unit": "deg C"},
    ])
    st.subheader("Datasets")
    st.dataframe(summary, hide_index=True, use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Integrated yield records", f"{len(data.panel):,}")
    c2.metric("Countries", data.panel["area"].nunique())
    c3.metric("Crops", data.panel["crop"].nunique())
    c4.metric("Benchmark rows (1991-2016)", f"{len(data.modelling):,}")
    st.markdown(f"**Crops:** {', '.join(sorted(data.panel['crop'].unique()))}")
    st.markdown(
        f"**Target:** crop yield in `{UNIT}` (10,000 hg/ha = 1 t/ha). **Horizon:** 1 year ahead. "
        "**Primary task:** forecasting known (country, crop) series."
    )
    summ = read_json("training_summary.json")
    if summ:
        st.markdown(f"**Temporal split (target years):** train {summ['split']['train']}, validation "
                    f"{summ['split']['validation']}, test {summ['split']['test']}. **Selected model:** {summ['best_model']} "
                    f"(training mode: {summ['mode']}).")
    avail = sorted(str(p.relative_to(MODELS)).replace("\\", "/") for p in MODELS.rglob("*.joblib")) if MODELS.exists() else []
    st.markdown("**Saved model artifacts:** " + (", ".join(f"`{a}`" for a in avail) or "none - run training"))
    with st.expander("Feature dictionary"):
        st.dataframe(pd.DataFrame({"feature": list(FEATURE_DOCS), "description": list(FEATURE_DOCS.values())}),
                     hide_index=True, use_container_width=True)


def page_explorer(data) -> None:
    st.title("Data explorer")
    panel, ann, F = data.panel, data.climate_annual, data.features
    crops = sorted(panel["crop"].unique())
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Yield trends", "Climate", "Pesticides", "Relationships", "Missing data"])
    with tab1:
        crop = st.selectbox("Crop", crops, index=crops.index("Wheat"), key="ex_crop")
        areas = sorted(panel.loc[panel.crop == crop, "area"].unique())
        default = [a for a in ("India", "United States of America", "France", "Kenya") if a in areas]
        sel = st.multiselect("Countries", areas, default=default)
        if sel:
            d = panel[(panel.crop == crop) & panel.area.isin(sel)].pivot(index="year", columns="area", values="yield_hg_ha")
            st.line_chart(d, y_label=f"yield ({UNIT})")
        st.markdown("**Crop comparison** - median yield across countries by year")
        comp = panel[panel.crop.isin(st.multiselect("Crops to compare", crops, default=["Wheat", "Maize", "Potatoes"]))]
        if not comp.empty:
            st.line_chart(comp.groupby(["year", "crop"])["yield_hg_ha"].median().unstack(), y_label=f"median {UNIT}")
    with tab2:
        area = st.selectbox("Country", sorted(ann["area"].unique()),
                            index=sorted(ann["area"].unique()).index("India"), key="ex_clim")
        c = ann[ann.area == area].set_index("year")
        st.markdown("Annual rainfall total (sum of monthly mm) and mean temperature (mean of monthly means), 1901-2016.")
        col1, col2 = st.columns(2)
        col1.line_chart(c[["rain_total_mm"]], y_label="mm")
        col2.line_chart(c[["temp_mean_c", "temp_max_month_c", "temp_min_month_c"]], y_label="deg C")
    with tab3:
        p = data.tidy["pesticides"]
        sel = st.multiselect("Countries", sorted(p.area.unique()), default=["India", "Brazil", "France"], key="ex_pest")
        if sel:
            st.line_chart(p[p.area.isin(sel)].pivot(index="year", columns="area", values="pesticides_tonnes"),
                          y_label="tonnes (national, all crops)")
        st.caption("National totals for all crops - not per hectare and not crop-specific.")
    with tab4:
        crop = st.selectbox("Crop", crops, index=crops.index("Maize"), key="rel_crop")
        xvar = st.selectbox("Predictor", ["rain_anom_pct", "temp_anom_c", "rain_anom_pct_lag1", "temp_anom_c_lag1",
                                          "log_pesticides_lag1", "pesticides_growth_lag1"])
        d = F[(F.crop == crop) & (F.year <= 2016)].dropna(subset=[xvar, "target_log_growth"])
        d = d[d["target_log_growth"].abs() < 1.5]
        st.scatter_chart(d, x=xvar, y="target_log_growth", height=420)
        if len(d) > 10:
            rho = d[[xvar, "target_log_growth"]].corr(method="spearman").iloc[0, 1]
            st.write(f"Spearman rho = {rho:.3f} (n = {len(d):,}). Association only - not a causal effect. "
                     f"`{xvar}`: {FEATURE_DOCS.get(xvar, '')}")
    with tab5:
        miss = data.modelling.isna().mean().mul(100).round(2)
        st.dataframe(miss[miss > 0].rename("% missing (benchmark rows)").to_frame(), use_container_width=True)
        rep = data.integration_report
        st.json({k: rep[k] for k in ("areas_without_climate", "yield_areas_without_pesticides",
                                     "rows_missing_same_year_pesticides", "zero_yield_rows_removed")})
        show_figure("eda_missing_data.png")


def page_benchmark() -> None:
    st.title("Model benchmarking")
    comp = read_csv("model_comparison.csv")
    if comp is None:
        st.warning("No results found. Run `python scripts/train_models.py`.")
        return
    st.markdown("Test years 2013-2016, identical rows for all models. MAE/RMSE/bias in hg/ha. "
                "`Validation MAE` = mean expanding-window CV MAE used for model selection.")
    st.dataframe(comp.drop(columns=["Best Params"]).style.format(precision=3), hide_index=True, use_container_width=True)
    show_figure("model_comparison.png")
    abl = read_csv("feature_set_comparison.csv")
    if abl is not None:
        st.subheader("Information sets")
        st.dataframe(abl, hide_index=True, use_container_width=True)
        st.caption("`conditional` uses same-year climate that is not known at forecast time (scenario model only).")
    preds = read_csv("test_predictions_all_models.csv")
    if preds is not None:
        model = st.selectbox("Model", comp["Model"].tolist(),
                             index=int(comp["Validation MAE"].idxmin()))
        d = preds[preds.model == model].copy()
        d["log_residual"] = np.log(d["y_pred"]) - np.log(d["y_true"])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Actual vs predicted (log10 hg/ha)**")
            st.scatter_chart(pd.DataFrame({"observed": np.log10(d["y_true"]), "forecast": np.log10(d["y_pred"])}),
                             x="observed", y="forecast", height=380)
        with c2:
            st.markdown("**Residuals vs forecast** (log residual, clipped to +/-2)")
            st.scatter_chart(pd.DataFrame({"log10 forecast": np.log10(d["y_pred"]),
                                           "log residual": d["log_residual"].clip(-2, 2)}),
                             x="log10 forecast", y="log residual", height=380)
        st.markdown("**Residual distribution** (log residual)")
        hist, edges = np.histogram(d["log_residual"].clip(-1, 1), bins=60)
        st.bar_chart(pd.DataFrame({"count": hist}, index=np.round((edges[:-1] + edges[1:]) / 2, 3)))
    yr = read_csv("metrics_by_year.csv")
    if yr is not None:
        st.markdown("**Forecast error over time** (MAE by target year; 2009-2012 from 1991-2008 fits, 2013-2016 from 1991-2012 fits)")
        st.line_chart(yr.pivot(index="year", columns="model", values="MAE"), y_label="MAE (hg/ha)")
    byc = read_csv("metrics_by_crop.csv")
    if byc is not None:
        st.markdown("**Performance by crop (test)**")
        st.dataframe(byc[["crop", "model", "n", "MAE", "RMSE", "Bias", "MedAPE_pct", "MAE_log"]],
                     hide_index=True, use_container_width=True)
    show_figure("forecast_examples.png", "Example series: observed vs one-year-ahead forecasts")


def page_forecast(data) -> None:
    st.title("Yield forecasting")
    F = data.features
    mode = st.radio("Forecast type", ["Future forecast (history-only model)",
                                      "Backtest with climate & pesticide model (trained 1991-2012)"])
    if mode.startswith("Future"):
        fc = try_model("deploy/history_forecaster.joblib")
        if fc is None:
            return
        years = [int(fc.bundle["forecast_year"])]
        st.info(f"Climate data end in 2016 and pesticide data in 2018, so a genuine forecast for {years[0]} can only use "
                "past yields. No future climate values are requested: they are unknown at forecast time.")
    else:
        fc = try_model("benchmark/best_model.joblib")
        if fc is None:
            return
        years = list(range(2013, 2018))
        st.info("This model uses yields, calendar-year climate and national pesticide use from the year *before* the target "
                "year, all observed. 2013-2016 are the test years; 2017 is an additional never-used year.")
    crops = sorted(F["crop"].unique())
    c1, c2, c3 = st.columns(3)
    crop = c1.selectbox("Crop", crops, index=crops.index("Wheat"))
    areas = sorted(F.loc[(F.crop == crop) & F.yield_hg_ha.notna(), "area"].unique())
    area = c2.selectbox("Country", areas, index=areas.index("India") if "India" in areas else 0)
    year = c3.selectbox("Target year", years, index=len(years) - 1)
    try:
        row = forecast_rows(F, area, crop, int(year))
        pred, warns = fc.predict_frame(row)
    except ForecastInputError as exc:
        st.error(f"Cannot forecast: {exc}")
        return
    warns += history_warnings(F, area, crop, int(year))
    if fc.bundle["feature_set"] != "history" and row["rain_total_mm_lag1"].isna().any():
        warns.append("previous-year climate missing")
    p = pred.iloc[0]
    k1, k2, k3 = st.columns(3)
    k1.metric("Forecast", f"{p['yield_pred_hg_ha']:,.0f} {UNIT}", f"{100 * (np.exp(p['pred_log_growth']) - 1):+.1f}% vs previous year")
    if "lower_80" in pred:
        k2.metric("80% interval", f"{p['lower_80']:,.0f} - {p['upper_80']:,.0f}")
        k3.metric("90% interval", f"{p['lower_90']:,.0f} - {p['upper_90']:,.0f}")
    obs = row["yield_hg_ha"].iloc[0]
    if pd.notna(obs):
        st.write(f"Observed {year}: **{fmt(obs)}** - forecast error {p['yield_pred_hg_ha'] - obs:+,.0f} {UNIT}.")
    st.caption(f"Model: {fc.bundle['algorithm']} ({fc.bundle['feature_set']} features), trained on target years "
               f"{fc.bundle['train_years'][0]}-{fc.bundle['train_years'][1]}; conformal intervals calibrated on "
               f"{fc.bundle.get('calibration_years')}. Intervals assume future errors resemble calibration-period errors.")
    for w in warns:
        st.warning(w)
    hist = F[(F.area == area) & (F.crop == crop) & F.yield_hg_ha.notna()].set_index("year")["yield_hg_ha"]
    chart = pd.DataFrame({"observed": hist})
    chart.loc[int(year), "forecast"] = p["yield_pred_hg_ha"]
    if "lower_90" in pred:
        chart.loc[int(year), "lower 90%"] = p["lower_90"]
        chart.loc[int(year), "upper 90%"] = p["upper_90"]
    st.line_chart(chart.sort_index().loc[1990:], y_label=UNIT)
    with st.expander("Inputs used (all from before the target year)"):
        shown = row[fc.features].T.rename(columns=lambda _: "value")
        shown["description"] = [FEATURE_DOCS.get(i, "") for i in shown.index]
        st.dataframe(shown, use_container_width=True)


def page_scenarios(data) -> None:
    st.title("Climate scenarios (hypothetical)")
    st.warning("Sensitivity analysis of a statistical model, not a climate projection or causal attribution. "
               "Same-year rainfall/temperature are changed while everything else stays at observed values.")
    fc = try_model("deploy/conditional_scenario.joblib")
    if fc is None:
        return
    M = data.modelling
    base_year = int(M["year"].max())
    X0 = M[M.year == base_year]
    crops = sorted(X0["crop"].unique())
    c1, c2 = st.columns(2)
    crop = c1.selectbox("Crop", crops, index=crops.index("Maize"))
    areas = sorted(X0.loc[X0.crop == crop, "area"].unique())
    area = c2.selectbox("Country", areas, index=areas.index("Kenya") if "Kenya" in areas else 0)
    c3, c4 = st.columns(2)
    dr = c3.slider("Rainfall change (%)", -MAX_RAIN_CHANGE_PCT, MAX_RAIN_CHANGE_PCT, -10.0, 5.0)
    dt = c4.slider("Temperature change (deg C)", -MAX_TEMP_CHANGE_C, MAX_TEMP_CHANGE_C, 2.0, 0.25)
    try:
        sc = Scenario("user", float(dr), float(dt))
    except ValueError as exc:
        st.error(str(exc))
        return
    X = X0[(X0.area == area) & (X0.crop == crop)].reset_index(drop=True)
    ranges = country_climate_ranges(data.climate_annual)
    res = run_scenarios(fc.bundle["pipeline"], X, [sc], ranges, fc.features, data.climate_annual).iloc[0]
    b = X.iloc[0]
    st.markdown(f"**Baseline:** observed {base_year} climate in {area}: rainfall {b.rain_total_mm:,.0f} mm "
                f"({b.rain_anom_pct:+.1f}% vs 1901-60), mean temperature {b.temp_mean_c:.2f} deg C "
                f"({b.temp_anom_c:+.2f} vs 1901-60). **Scenario:** {b.rain_total_mm * (1 + dr / 100):,.0f} mm, "
                f"{b.temp_mean_c + dt:.2f} deg C.")
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Model prediction, observed {base_year} climate", f"{res['baseline_pred']:,.0f} {UNIT}")
    k2.metric("Model prediction, scenario", f"{res['scenario_pred']:,.0f} {UNIT}", f"{res['pct_change']:+.2f}%")
    obs = b.yield_hg_ha
    k3.metric(f"Observed {base_year} yield", f"{obs:,.0f} {UNIT}")
    cal = fc.bundle.get("calibrator")
    if cal is not None:
        q = cal.half_width(0.9, np.array([crop]), 1)[0]
        st.caption(f"90% conformal half-width for this crop: x/ {np.exp(q):.2f} around a prediction (calibrated on "
                   f"2009-2012 residuals of a 1991-2008 fit). The scenario effect is small relative to this uncertainty "
                   f"if |change| << {100 * (np.exp(q) - 1):.0f}%.")
    if res["rain_outside_history"] or res["temp_outside_history"]:
        st.error("Extrapolation: scenario rainfall or temperature lies outside this country's observed 1901-2016 range.")
    if not res["historical_analogue"]:
        st.warning("No historical analogue: the country never had a year within +/-10% rainfall and +/-0.5 deg C of "
                   "this combination. Treat the prediction as extrapolation.")
    rng = ranges.set_index("area").loc[area]
    st.caption(f"Observed 1901-2016 range in {area}: rainfall {rng.rain_min:,.0f}-{rng.rain_max:,.0f} mm, "
               f"temperature {rng.temp_min:.2f}-{rng.temp_max:.2f} deg C.")
    st.subheader("Pre-computed scenarios across all series")
    s = read_csv("scenario_results.csv")
    if s is not None:
        st.dataframe(s[s.scenario != "baseline"].groupby("scenario").agg(
            median_pct_change=("pct_change", "median"), share_outside_history=("temp_outside_history", "mean"),
            share_with_analogue=("historical_analogue", "mean")), use_container_width=True)
    show_figure("scenario_sensitivity_by_crop.png")


def page_explain(data) -> None:
    st.title("Explainability")
    summ = read_json("training_summary.json") or {}
    model_name = summ.get("best_tree_model", "LightGBM")
    st.markdown(f"Explanations are for the **{model_name}** benchmark model (forecast feature set, test years).")
    perm = read_csv("permutation_importance.csv")
    if perm is not None:
        st.subheader("Permutation importance")
        st.bar_chart(perm.set_index("feature")["mae_increase_hg_ha"].head(15), horizontal=True,
                     x_label="MAE increase (hg/ha) when shuffled")
    c1, c2 = st.columns(2)
    with c1:
        show_figure("shap_importance.png", "Mean |SHAP| by source feature")
    with c2:
        show_figure("shap_beeswarm.png", "SHAP beeswarm (one-hot crop columns hidden)")
    st.subheader("Local explanation")
    slug = model_name.lower()
    fc = try_model(f"benchmark/{slug}.joblib")
    if fc is not None:
        T = data.modelling[data.modelling.year.between(2013, 2016)]
        crops = sorted(T.crop.unique())
        a, b, c = st.columns(3)
        crop = a.selectbox("Crop", crops, index=crops.index("Wheat"), key="sh_crop")
        areas = sorted(T.loc[T.crop == crop, "area"].unique())
        area = b.selectbox("Country", areas, index=areas.index("India") if "India" in areas else 0, key="sh_area")
        yrs = sorted(T.loc[(T.crop == crop) & (T.area == area), "year"].unique())
        year = c.selectbox("Year", yrs, index=len(yrs) - 1)
        row = T[(T.crop == crop) & (T.area == area) & (T.year == year)]
        try:
            from climate_yield.explainability import shap_values
            sv, Z = shap_values(fc.bundle["pipeline"], row[fc.features])
            contrib = pd.Series(sv.values[0], index=Z.columns)
            contrib = contrib[contrib.abs() > 0].sort_values(key=np.abs, ascending=False).head(12)
            st.bar_chart(contrib, horizontal=True, x_label="SHAP contribution to predicted log growth")
            st.caption(f"Base value {float(np.ravel(sv.base_values)[0]):+.4f}; predicted log growth "
                       f"{float(np.ravel(sv.base_values)[0] + sv.values[0].sum()):+.4f}.")
        except Exception as exc:
            st.info(f"SHAP local explanation unavailable: {exc}")
    st.subheader("Limitations")
    st.markdown("- Importance measures what the model uses to predict, not what causes yields.\n"
                "- Lagged yields, rolling means and trends are highly correlated; credit is split among them arbitrarily.\n"
                "- Climate variables are national calendar-year averages; weak importance does not mean weather is "
                "irrelevant to crops, only that these aggregates add little predictive information here.\n"
                "- Pesticide tonnes are national totals for all crops; their importance cannot be read as an effect of "
                "pesticide application.")


PAGES = {
    "Overview": page_overview,
    "Data explorer": page_explorer,
    "Model benchmarking": page_benchmark,
    "Yield forecasting": page_forecast,
    "Climate scenarios": page_scenarios,
    "Explainability": page_explain,
}


def main() -> None:
    choice = st.sidebar.radio("Page", list(PAGES))
    st.sidebar.caption("Data: FAO (yield, pesticides) and World Bank-style monthly climate. Models: see README.")
    try:
        data = get_data()
    except FileNotFoundError as exc:
        st.error(f"Raw data file missing: {exc}. Place the four CSVs at the repository root.")
        return
    fn = PAGES[choice]
    if fn is page_benchmark:
        fn()
    else:
        fn(data)


main()
