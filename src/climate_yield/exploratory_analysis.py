"""Exploratory analysis: figures and the numbers quoted in reports/eda_report.md.

Relationships are estimated *descriptively*. The first-difference panel
regression (``climate_association``) removes time-invariant series effects and
common year shocks, but it is still an observational association between
country-average annual climate and national yields, not a causal effect.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402

from .feature_engineering import FEATURE_SETS  # noqa: E402
from .reporting import save  # noqa: E402


def yield_distributions(panel: pd.DataFrame, figs: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.histplot(np.log10(panel["yield_hg_ha"]), bins=80, ax=axes[0])
    axes[0].set_xlabel("log10 yield (hg/ha)")
    axes[0].set_title("Distribution of yields (all crops, all years)")
    order = panel.groupby("crop")["yield_hg_ha"].median().sort_values().index
    sns.boxplot(data=panel, y="crop", x="yield_hg_ha", order=order, ax=axes[1], fliersize=1, log_scale=True)
    axes[1].set_xlabel("yield (hg/ha, log scale)")
    axes[1].set_title("Yield by crop")
    save(fig, figs / "eda_yield_distribution.png")
    med = panel.groupby("crop")["yield_hg_ha"].median()
    return {"median_by_crop": med.round(0).to_dict(),
            "skew_raw": float(stats.skew(panel["yield_hg_ha"])),
            "skew_log": float(stats.skew(np.log(panel["yield_hg_ha"])))}


def yield_trends(panel: pd.DataFrame, figs: Path) -> pd.DataFrame:
    """Median across countries of each series' yield indexed to its 1961-1970 mean."""
    p = panel.copy()
    base = p[p["year"].between(1961, 1970)].groupby(["area", "crop"])["yield_hg_ha"].mean().rename("base")
    p = p.join(base, on=["area", "crop"])
    p["index"] = 100 * p["yield_hg_ha"] / p["base"]
    trend = p.dropna(subset=["index"]).groupby(["crop", "year"])["index"].median().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for crop, g in trend.groupby("crop"):
        ax.plot(g["year"], g["index"], lw=1.3, label=crop)
    ax.set_ylabel("median yield index (1961-70 mean = 100)")
    ax.set_title("Yield trends: median across countries with 1961-70 data")
    ax.legend(fontsize=7, ncol=2)
    save(fig, figs / "eda_yield_trends.png")
    # average annual log growth per series over its whole record
    growth = (panel.sort_values("year").groupby(["area", "crop"])
              .apply(lambda g: np.polyfit(g["year"], np.log(g["yield_hg_ha"]), 1)[0] if len(g) >= 10 else np.nan,
                     include_groups=False)).dropna()
    return pd.DataFrame({"crop": [k[1] for k in growth.index], "trend": growth.to_numpy()})


def yield_by_country(panel: pd.DataFrame, figs: Path, crop: str = "Wheat", n: int = 20) -> None:
    d = panel[(panel["crop"] == crop) & panel["year"].between(2007, 2016)]
    top = d.groupby("area")["yield_hg_ha"].median().sort_values()
    keep = list(top.index[:n // 2]) + list(top.index[-n // 2:])
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.boxplot(data=d[d["area"].isin(keep)], y="area", x="yield_hg_ha", order=keep, ax=ax, fliersize=1)
    ax.set_title(f"{crop} yield 2007-2016: 10 lowest and 10 highest median countries")
    ax.set_xlabel("hg/ha")
    save(fig, figs / "eda_wheat_by_country.png")


def climate_trends(climate_annual: pd.DataFrame, baseline: pd.DataFrame, figs: Path) -> dict:
    c = climate_annual.merge(baseline, on="area")
    c["temp_anom"] = c["temp_mean_c"] - c["temp_mean_base_c"]
    c["rain_anom_pct"] = 100 * (c["rain_total_mm"] / c["rain_total_base_mm"] - 1)
    g = c.groupby("year")[["temp_anom", "rain_anom_pct"]].median()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(g.index, g["temp_anom"], color="tab:red")
    axes[0].plot(g.index, g["temp_anom"].rolling(10).mean(), color="black", lw=1.5, label="10-yr trailing mean")
    axes[0].set_ylabel("deg C vs 1901-60")
    axes[0].set_title("Median country temperature anomaly")
    axes[0].legend()
    axes[1].plot(g.index, g["rain_anom_pct"], color="tab:blue")
    axes[1].plot(g.index, g["rain_anom_pct"].rolling(10).mean(), color="black", lw=1.5)
    axes[1].set_ylabel("% vs 1901-60")
    axes[1].set_title("Median country annual rainfall anomaly")
    save(fig, figs / "eda_climate_trends.png")
    lin_t = stats.linregress(g.loc[1961:2016].index, g.loc[1961:2016, "temp_anom"])
    lin_r = stats.linregress(g.loc[1961:2016].index, g.loc[1961:2016, "rain_anom_pct"])
    return {"temp_anom_2007_2016": float(g.loc[2007:2016, "temp_anom"].mean()),
            "temp_trend_c_per_decade_1961_2016": float(10 * lin_t.slope), "temp_trend_p": float(lin_t.pvalue),
            "rain_trend_pct_per_decade_1961_2016": float(10 * lin_r.slope), "rain_trend_p": float(lin_r.pvalue)}


def climate_yield_scatter(F: pd.DataFrame, figs: Path) -> dict:
    """Same-year climate anomaly vs log growth (yield change) and vs yield level."""
    d = F.dropna(subset=["target_log_growth", "rain_anom_pct", "temp_anom_c"])
    d = d[d["target_log_growth"].abs() < 1.5]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, x, lab in ((axes[0, 0], "rain_anom_pct", "rainfall anomaly (% vs 1901-60)"),
                       (axes[0, 1], "temp_anom_c", "temperature anomaly (deg C vs 1901-60)")):
        ax.hexbin(d[x].clip(-80, 150), d["target_log_growth"], gridsize=50, bins="log", cmap="Blues", mincnt=1)
        ax.set_xlabel(lab)
        ax.set_ylabel("log growth of yield vs previous year")
    axes[0, 0].set_title("Rainfall anomaly vs yield change (same year)")
    axes[0, 1].set_title("Temperature anomaly vs yield change (same year)")
    lvl = F.dropna(subset=["yield_hg_ha", "rain_total_mm"])
    lvl = lvl[lvl["crop"] == "Wheat"]
    axes[1, 0].scatter(lvl["rain_total_mm"], lvl["yield_hg_ha"], s=3, alpha=0.3)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("annual rainfall (mm)")
    axes[1, 0].set_ylabel("wheat yield (hg/ha)")
    axes[1, 0].set_title("Wheat: yield level vs rainfall (between-country, confounded)")
    axes[1, 1].scatter(lvl["temp_mean_c"], lvl["yield_hg_ha"], s=3, alpha=0.3, color="tab:red")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("annual mean temperature (deg C)")
    axes[1, 1].set_title("Wheat: yield level vs temperature (between-country, confounded)")
    save(fig, figs / "eda_climate_yield_relationships.png")
    out = {}
    for x in ("rain_anom_pct", "temp_anom_c"):
        r, p = stats.spearmanr(d[x], d["target_log_growth"])
        out[f"spearman_growth_vs_{x}"] = (float(r), float(p))
    for x in ("rain_total_mm", "temp_mean_c"):
        r, p = stats.spearmanr(lvl[x], lvl["yield_hg_ha"])
        out[f"spearman_wheat_level_vs_{x}"] = (float(r), float(p))
    return out


def pesticide_relationships(F: pd.DataFrame, panel: pd.DataFrame, figs: Path) -> dict:
    lv = panel.dropna(subset=["pesticides_tonnes"])
    lv = lv[lv["crop"] == "Maize"]
    d = F.dropna(subset=["pesticides_growth_lag1", "target_log_growth"])
    d = d[d["target_log_growth"].abs() < 1.5]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].scatter(np.log10(lv["pesticides_tonnes"] + 1), lv["yield_hg_ha"], s=3, alpha=0.3)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("log10(1 + national pesticide use, tonnes)")
    axes[0].set_ylabel("maize yield (hg/ha)")
    axes[0].set_title("Maize yield vs national pesticide use (same year)")
    axes[1].hexbin(d["pesticides_growth_lag1"].clip(-2, 2), d["target_log_growth"], gridsize=50, bins="log", cmap="Greens", mincnt=1)
    axes[1].set_xlabel("log change in pesticide use, t-2 -> t-1")
    axes[1].set_ylabel("log growth of yield, t-1 -> t")
    axes[1].set_title("Lagged pesticide change vs yield change")
    save(fig, figs / "eda_pesticide_relationships.png")
    r1, p1 = stats.spearmanr(np.log1p(lv["pesticides_tonnes"]), np.log(lv["yield_hg_ha"]))
    r2, p2 = stats.spearmanr(d["pesticides_growth_lag1"], d["target_log_growth"])
    return {"spearman_maize_level": (float(r1), float(p1)), "spearman_growth": (float(r2), float(p2)),
            "n_growth": int(len(d))}


def correlation_heatmap(M: pd.DataFrame, figs: Path) -> pd.DataFrame:
    cols = [c for c in FEATURE_SETS["conditional"] if c != "crop"] + ["target_log_growth"]
    corr = M[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Spearman rho"})
    ax.set_title("Predictor correlations (modelling rows 1991-2016)")
    save(fig, figs / "eda_correlation_heatmap.png")
    return corr


def missing_data(panel: pd.DataFrame, M: pd.DataFrame, figs: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    by_year = panel.groupby("year").agg(rows=("yield_hg_ha", "size"),
                                        pest=("pesticides_tonnes", lambda s: s.notna().mean()),
                                        clim=("rain_total_mm", lambda s: s.notna().mean()))
    axes[0].plot(by_year.index, by_year["rows"], color="black")
    axes[0].set_ylabel("yield records")
    axes[0].set_title("Yield records per year")
    ax2 = axes[1]
    ax2.plot(by_year.index, 100 * by_year["pest"], label="pesticides available")
    ax2.plot(by_year.index, 100 * by_year["clim"], label="same-year climate available")
    ax2.set_ylabel("% of yield records")
    ax2.set_title("Covariate availability by year")
    ax2.legend()
    save(fig, figs / "eda_missing_data.png")
    miss = M[[c for c in FEATURE_SETS["conditional"] if c != "crop"]].isna().mean().sort_values(ascending=False)
    return {"modelling_missing_share": miss[miss > 0].round(4).to_dict()}


def outliers(F: pd.DataFrame, figs: Path) -> dict:
    g = F["target_log_growth"].dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(g.clip(-2, 2), bins=120, ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("year-on-year log growth of yield (clipped to +/-2)")
    ax.set_title("Year-on-year yield changes (heavy tails = possible data errors)")
    save(fig, figs / "eda_growth_outliers.png")
    return {"n": int(len(g)), "share_abs_gt_0.5": float((g.abs() > 0.5).mean()),
            "share_abs_gt_1": float((g.abs() > 1).mean()), "kurtosis": float(stats.kurtosis(g))}


def autocorrelation(F: pd.DataFrame) -> dict:
    d = F.dropna(subset=["target_log_growth", "growth_lag1"])
    r, p = stats.pearsonr(d["target_log_growth"], d["growth_lag1"])
    lens = F.dropna(subset=["yield_hg_ha"]).groupby(["area", "crop"]).size()
    return {"lag1_acf_of_growth": float(r), "p": float(p), "series_len_median": float(lens.median()),
            "series_ge_30": int((lens >= 30).sum()), "n_series": int(len(lens))}


def climate_association(F: pd.DataFrame, figs: Path) -> pd.DataFrame:
    """First-difference panel regression of yield log growth on changes in
    same-year rainfall / temperature anomalies, with crop and year fixed
    effects and country-clustered standard errors; pooled and per crop."""
    d = F.dropna(subset=["target_log_growth", "rain_anom_pct", "rain_anom_pct_lag1", "temp_anom_c", "temp_anom_c_lag1"]).copy()
    d = d[d["target_log_growth"].abs() < 1.5]
    d["d_rain10"] = (d["rain_anom_pct"] - d["rain_anom_pct_lag1"]) / 10.0
    d["d_temp"] = d["temp_anom_c"] - d["temp_anom_c_lag1"]
    rows = []

    def fit(sub, label, formula):
        m = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(sub["area"])[0]})
        for v in ("d_rain10", "d_temp"):
            lo, hi = m.conf_int().loc[v]
            rows.append({"sample": label, "term": v, "coef": m.params[v], "ci_low": lo, "ci_high": hi,
                         "p_value": m.pvalues[v], "n": int(m.nobs), "countries": int(sub["area"].nunique())})

    fit(d, "all crops", "target_log_growth ~ d_rain10 + d_temp + C(crop) + C(year)")
    for crop, sub in d.groupby("crop"):
        if sub["area"].nunique() >= 15:
            fit(sub, crop, "target_log_growth ~ d_rain10 + d_temp + C(year)")
    res = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, term, lab in ((axes[0], "d_rain10", "per +10 pp rainfall anomaly"), (axes[1], "d_temp", "per +1 deg C")):
        r = res[res["term"] == term].iloc[::-1]
        ax.errorbar(100 * r["coef"], r["sample"], xerr=[100 * (r["coef"] - r["ci_low"]), 100 * (r["ci_high"] - r["coef"])],
                    fmt="o", capsize=3)
        ax.axvline(0, color="grey", lw=1)
        ax.set_xlabel(f"approx. % change in yield {lab} (95% CI)")
    axes[0].set_title("Rainfall association")
    axes[1].set_title("Temperature association")
    fig.suptitle("First-difference association (crop & year FE, country-clustered SE) - not causal")
    save(fig, figs / "eda_climate_association.png")
    return res
