"""Orchestrates tuning, benchmarking, uncertainty, explainability and the
deployment models. Called by ``scripts/train_models.py``."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import reporting as rp
from .config import Config
from .evaluate import grouped_metrics, mase_scale, regression_metrics
from .explainability import permutation_table, shap_global_table, shap_values
from .feature_engineering import FEATURE_SETS, MODEL_TARGET, TARGET
from .pipeline import PreparedData, prepare_data
from .temporal_validation import TemporalSplit, assert_no_temporal_overlap, expanding_window_folds
from .train import build_pipeline, fit_timed, model_specs, predict_timed, tune
from .uncertainty import ConformalCalibrator, coverage_summary

TREE_MODELS = ("RandomForest", "XGBoost", "LightGBM")
EXAMPLE_SERIES = [
    ("India", "Wheat"),
    ("United States of America", "Maize"),
    ("Germany", "Potatoes"),
    ("Australia", "Wheat"),
    ("Kenya", "Maize"),
    ("Russian Federation", "Barley"),
]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _bundle(pipe, algorithm, params, feature_set, train_years, calibrator=None, extra=None) -> dict:
    return {
        "pipeline": pipe,
        "algorithm": algorithm,
        "params": params,
        "feature_set": feature_set,
        "features": FEATURE_SETS[feature_set],
        "train_years": train_years,
        "calibrator": calibrator,
        "target_transform": "log_growth",
        "target_unit": "hg/ha",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        **(extra or {}),
    }


def run_benchmark(cfg: Config, mode: str | None = None, data: PreparedData | None = None, log=print) -> dict:
    settings = cfg.mode_settings(mode)
    seed = cfg.seed
    np.random.seed(seed)
    cfg.ensure_dirs()
    figs = cfg.path("figures_dir")
    reports = cfg.path("reports_dir")
    bench_dir = cfg.path("models_dir") / "benchmark"
    deploy_dir = cfg.path("models_dir") / "deploy"
    bench_dir.mkdir(parents=True, exist_ok=True)
    deploy_dir.mkdir(parents=True, exist_ok=True)

    data = data or prepare_data(cfg)
    M = data.modelling
    split = TemporalSplit.from_config(cfg["task"])
    dev = M[split.mask(M["year"], "development")].reset_index(drop=True)
    train = M[split.mask(M["year"], "train")].reset_index(drop=True)
    val = M[split.mask(M["year"], "validation")].reset_index(drop=True)
    test = M[split.mask(M["year"], "test")].reset_index(drop=True)
    assert_no_temporal_overlap(dev["year"], test["year"])
    assert_no_temporal_overlap(train["year"], val["year"])
    folds = expanding_window_folds(dev["year"], cfg["task"]["cv_folds"])
    features = FEATURE_SETS["forecast"]
    scale_train = mase_scale(train[TARGET], np.exp(train["log_y_lag1"]))
    log(f"rows: train={len(train)} val={len(val)} test={len(test)}; mode={settings['mode']}")

    specs = model_specs(settings, seed)
    rows, test_preds, val_preds, tuned = [], [], [], {}
    for name, spec in specs.items():
        log(f"[{name}] tuning ({settings['n_iter'] if spec['space'] else 0} candidates x {len(folds)} folds)")
        tr = tune(spec, dev, features, folds, settings["n_iter"], seed)
        tuned[name] = tr
        # holdout validation: fit on 1991-2008, forecast 2009-2012 (used for conformal calibration)
        p_val = build_pipeline(spec["estimator"], features, spec["scale"], tr.params).fit(train[features], train[MODEL_TARGET])
        g_val, y_val, _ = predict_timed(p_val, val[features])
        val_preds.append(val[["area", "crop", "year", TARGET]].assign(model=name, y_pred=y_val, g_pred=g_val))
        # final benchmark fit: all development years, forecast the untouched test years
        pipe, fit_s = fit_timed(build_pipeline(spec["estimator"], features, spec["scale"], tr.params),
                                dev[features], dev[MODEL_TARGET])
        g_test, y_test, pred_s = predict_timed(pipe, test[features])
        m = regression_metrics(test[TARGET], y_test, naive_scale=scale_train)
        mv = regression_metrics(val[TARGET], y_val, naive_scale=scale_train)
        rows.append({
            "Model": name,
            "Validation MAE": tr.cv_mae,
            "Holdout Validation MAE (2009-2012)": mv["MAE"],
            "Test MAE": m["MAE"],
            "Test RMSE": m["RMSE"],
            "Test R2": m["R2"],
            "Prediction Bias": m["Bias"],
            "Test MAPE %": m["MAPE_pct"],
            "Test MedAPE %": m["MedAPE_pct"],
            "Test MAE log": m["MAE_log"],
            "Test MASE": m["MASE"],
            "Training Time (s)": fit_s,
            "Prediction Time (s)": pred_s,
            "Tuning Time (s)": tr.tuning_seconds,
            "Best Params": json.dumps(tr.params, default=str),
        })
        test_preds.append(test[["area", "crop", "year", TARGET]].assign(model=name, y_pred=y_test, g_pred=g_test))
        joblib.dump(_bundle(pipe, name, tr.params, "forecast", [int(dev.year.min()), int(dev.year.max())]),
                    bench_dir / f"{slug(name)}.joblib", compress=3)
        log(f"   cv MAE={tr.cv_mae:,.0f}  test MAE={m['MAE']:,.0f}  R2={m['R2']:.3f}")

    comp = pd.DataFrame(rows)
    comp.to_csv(reports / "model_comparison.csv", index=False)
    test_pred = pd.concat(test_preds, ignore_index=True).rename(columns={TARGET: "y_true"})
    val_pred = pd.concat(val_preds, ignore_index=True).rename(columns={TARGET: "y_true"})
    tuning_log = {n: {"best_params": t.params, "cv_mae": t.cv_mae, "fold_mae": t.cv_fold_mae,
                      "tuning_seconds": t.tuning_seconds, "trials": t.trials} for n, t in tuned.items()}
    with open(reports / "tuning_results.json", "w", encoding="utf-8") as fh:
        json.dump(tuning_log, fh, indent=2, default=str)

    best_name = comp.sort_values("Validation MAE").iloc[0]["Model"]
    best_tree = comp[comp["Model"].isin(TREE_MODELS)].sort_values("Validation MAE").iloc[0]["Model"]
    log(f"best model by validation MAE: {best_name}; best tree model: {best_tree}")
    best_spec = specs[best_name]
    best_params = tuned[best_name].params

    # ---------------- uncertainty (conformal, calibrated on 2009-2012) ----------------
    levels = [float(x) for x in cfg["uncertainty"]["levels"]]
    mg = int(cfg["uncertainty"]["mondrian_min_group_size"])
    bv = val_pred[val_pred["model"] == best_name]
    resid = val[MODEL_TARGET].to_numpy() - bv["g_pred"].to_numpy()
    cal_crop = ConformalCalibrator(levels, mg).fit(resid, val["crop"].to_numpy())
    cal_pool = ConformalCalibrator(levels, mg).fit(resid)
    bt = test_pred[test_pred["model"] == best_name].reset_index(drop=True)
    cov_rows, cov_crop, cov_year = [], [], []
    for lv in levels:
        for label, cal, grp in (("per-crop (Mondrian)", cal_crop, test["crop"].to_numpy()), ("pooled", cal_pool, None)):
            lo, hi = cal.interval(bt["g_pred"], test["log_y_lag1"], lv, grp)
            cov_rows.append({"method": label, **coverage_summary(test[TARGET], lo, hi, lv)})
            if label.startswith("per-crop"):
                bt[f"lower_{int(lv * 100)}"], bt[f"upper_{int(lv * 100)}"] = lo, hi
                tmp = test[["crop", "year", TARGET]].assign(lo=lo, hi=hi)
                for key, col, store in (("crop", "crop", cov_crop), ("year", "year", cov_year)):
                    for k, g in tmp.groupby(col):
                        store.append({key: k, **coverage_summary(g[TARGET], g["lo"], g["hi"], lv)})
    cov = pd.DataFrame(cov_rows)
    cov_crop, cov_year = pd.DataFrame(cov_crop), pd.DataFrame(cov_year)
    cov.to_csv(reports / "interval_coverage.csv", index=False)
    cov_crop.to_csv(reports / "interval_coverage_by_crop.csv", index=False)
    cov_year.to_csv(reports / "interval_coverage_by_year.csv", index=False)
    # exchangeability check: residual spread by calibration year vs test year
    spread = pd.concat([
        val[["year"]].assign(abs_resid=np.abs(resid), partition="calibration"),
        test[["year"]].assign(abs_resid=np.abs(test[MODEL_TARGET].to_numpy() - bt["g_pred"].to_numpy()), partition="test"),
    ]).groupby(["partition", "year"])["abs_resid"].median().reset_index()
    spread.to_csv(reports / "residual_spread_by_year.csv", index=False)
    rp.plot_coverage(cov_crop, "crop", figs / "interval_coverage_by_crop.png",
                     f"{best_name}: observed test coverage by crop (per-crop conformal)")

    best_bundle = _bundle(joblib.load(bench_dir / f"{slug(best_name)}.joblib")["pipeline"], best_name, best_params,
                          "forecast", [int(dev.year.min()), int(dev.year.max())], cal_crop,
                          {"calibration_years": list(split.validation_years)})
    joblib.dump(best_bundle, bench_dir / "best_model.joblib", compress=3)

    # ---------------- evaluation detail & figures ----------------
    bt.to_csv(reports / "test_predictions_best.csv", index=False)
    test_pred.drop(columns="g_pred").to_csv(reports / "test_predictions_all_models.csv", index=False)
    by_crop = pd.concat([grouped_metrics(test_pred[test_pred.model == n], "crop").assign(model=n)
                         for n in (best_name, "Naive (last value)")])
    by_crop.to_csv(reports / "metrics_by_crop.csv", index=False)
    err_year = pd.concat([
        grouped_metrics(pd.concat([val_pred[val_pred.model == n], test_pred[test_pred.model == n]]), "year").assign(model=n)
        for n in comp["Model"]
    ])
    err_year.to_csv(reports / "metrics_by_year.csv", index=False)
    bias_crop = by_crop[["crop", "model", "Bias", "MedAPE_pct", "n"]]

    rp.plot_model_comparison(comp, figs / "model_comparison.png")
    rp.plot_actual_vs_predicted(bt["y_true"], bt["y_pred"], f"{best_name}: test {split.describe()['test']}",
                                figs / "actual_vs_predicted.png")
    rp.plot_residuals(bt, f"{best_name} (test)", figs / "residual_distribution.png", figs / "residuals_vs_predicted.png")
    rp.plot_error_over_time(err_year[err_year.model != "DummyRegressor (mean growth)"], figs / "error_over_time.png",
                            split.test_years[0])
    rp.plot_metric_by_group(by_crop, "crop", "MedAPE_pct", "model", figs / "medape_by_crop.png",
                            "Median absolute % error by crop (test)")
    series_preds = pd.concat([
        val_pred[val_pred.model == best_name].assign(partition="validation"),
        bt.assign(partition="test", lower=bt["lower_90"], upper=bt["upper_90"]),
    ])
    rp.plot_series_examples(data.panel, series_preds, EXAMPLE_SERIES, figs / "forecast_examples.png")

    # ---------------- feature-set ablation (same rows, same algorithm & params) ----------------
    abl = []
    for fs in ("history", "forecast", "conditional"):
        f = FEATURE_SETS[fs]
        cv = float(np.mean([
            regression_metrics(dev.iloc[va][TARGET], predict_timed(
                build_pipeline(best_spec["estimator"], f, best_spec["scale"], best_params).fit(
                    dev.iloc[tr][f], dev.iloc[tr][MODEL_TARGET]), dev.iloc[va][f])[1])["MAE"]
            for tr, va in folds]))
        p = build_pipeline(best_spec["estimator"], f, best_spec["scale"], best_params).fit(dev[f], dev[MODEL_TARGET])
        m = regression_metrics(test[TARGET], predict_timed(p, test[f])[1], naive_scale=scale_train)
        abl.append({"feature_set": fs, "n_features": len(f), "uses_same_year_climate": fs == "conditional",
                    "valid_forecast": fs != "conditional", "cv_MAE": cv, "test_MAE": m["MAE"],
                    "test_RMSE": m["RMSE"], "test_R2": m["R2"], "test_MedAPE_pct": m["MedAPE_pct"],
                    "test_Bias": m["Bias"]})
    abl = pd.DataFrame(abl)
    abl.to_csv(reports / "feature_set_comparison.csv", index=False)
    log(abl.to_string())

    # ---------------- explainability ----------------
    expl = {"model": best_tree}
    tree_pipe = joblib.load(bench_dir / f"{slug(best_tree)}.joblib")["pipeline"]
    rng = np.random.default_rng(seed)
    n_s = min(int(cfg["training"]["shap_sample_size"]), len(test))
    sample = test.iloc[np.sort(rng.choice(len(test), n_s, replace=False))]
    t0 = time.perf_counter()
    perm = permutation_table(tree_pipe, sample[features], sample[TARGET].to_numpy(),
                             int(cfg["training"]["permutation_repeats"]), seed)
    perm.to_csv(reports / "permutation_importance.csv", index=False)
    rp.plot_importance(perm, "mae_increase_hg_ha", figs / "permutation_importance.png",
                       f"{best_tree}: permutation importance (MAE increase, test sample)")
    expl["permutation_seconds"] = time.perf_counter() - t0
    try:
        import shap
        import matplotlib.pyplot as plt

        t0 = time.perf_counter()
        sv, Z = shap_values(tree_pipe, sample[features])
        glob = shap_global_table(sv, Z)
        glob.to_csv(reports / "shap_importance.csv", index=False)
        rp.plot_importance(glob, "mean_abs_shap", figs / "shap_importance.png",
                           f"{best_tree}: mean |SHAP| (log-growth units, test sample)")
        keep = [c for c in Z.columns if not c.startswith("crop_") and not c.startswith("missingindicator_")]
        idx = [list(Z.columns).index(c) for c in keep]
        plt.figure()
        shap.summary_plot(sv.values[:, idx], Z[keep], show=False, max_display=18)
        rp.save(plt.gcf(), figs / "shap_beeswarm.png")
        # local explanation: largest absolute forecast error in the sample
        err = np.abs(predict_timed(tree_pipe, sample[features])[1] - sample[TARGET].to_numpy())
        i = int(np.argmax(err))
        plt.figure()
        shap.plots.waterfall(sv[i], max_display=14, show=False)
        rp.save(plt.gcf(), figs / "shap_waterfall_example.png")
        r = sample.iloc[i]
        expl.update({"shap": "ok", "shap_seconds": time.perf_counter() - t0,
                     "local_example": {"area": r["area"], "crop": r["crop"], "year": int(r["year"]),
                                       "observed": float(r[TARGET]), "abs_error": float(err[i])}})
    except Exception as exc:  # documented, not hidden
        expl.update({"shap": f"failed: {exc!r}"})
        log(f"SHAP failed: {exc!r}")

    # ---------------- deployment models ----------------
    F = data.features
    last_obs = int(data.panel["year"].max())
    H = F[F[TARGET].notna() & F["log_y_lag1"].notna() & F["year"].between(cfg["task"]["first_target_year"], last_obs)]
    H = H.reset_index(drop=True)
    hf = FEATURE_SETS["history"]
    cal_years = (last_obs - 4, last_obs)
    h_tr, h_ca = H[H.year < cal_years[0]], H[H.year >= cal_years[0]]
    p_cal = build_pipeline(best_spec["estimator"], hf, best_spec["scale"], best_params).fit(h_tr[hf], h_tr[MODEL_TARGET])
    h_res = h_ca[MODEL_TARGET].to_numpy() - p_cal.predict(h_ca[hf])
    h_cal = ConformalCalibrator(levels, mg).fit(h_res, h_ca["crop"].to_numpy())
    p_hist = build_pipeline(best_spec["estimator"], hf, best_spec["scale"], best_params).fit(H[hf], H[MODEL_TARGET])
    joblib.dump(_bundle(p_hist, best_name, best_params, "history", [int(H.year.min()), last_obs], h_cal,
                        {"calibration_years": list(cal_years), "forecast_year": last_obs + 1}),
                deploy_dir / "history_forecaster.joblib", compress=3)

    cf = FEATURE_SETS["conditional"]
    p_cv = build_pipeline(best_spec["estimator"], cf, best_spec["scale"], best_params).fit(train[cf], train[MODEL_TARGET])
    c_cal = ConformalCalibrator(levels, mg).fit(val[MODEL_TARGET].to_numpy() - p_cv.predict(val[cf]), val["crop"].to_numpy())
    p_cond = build_pipeline(best_spec["estimator"], cf, best_spec["scale"], best_params).fit(M[cf], M[MODEL_TARGET])
    train_ranges = {c: [float(M[c].min()), float(M[c].max())] for c in cf if c != "crop"}
    joblib.dump(_bundle(p_cond, best_name, best_params, "conditional", [int(M.year.min()), int(M.year.max())], c_cal,
                        {"calibration_years": list(split.validation_years), "feature_ranges": train_ranges}),
                deploy_dir / "conditional_scenario.joblib", compress=3)

    summary = {
        "mode": settings["mode"],
        "split": split.describe(),
        "cv_folds": cfg["task"]["cv_folds"],
        "rows": {"train": len(train), "validation": len(val), "test": len(test), "development": len(dev)},
        "best_model": best_name,
        "best_tree_model": best_tree,
        "best_params": best_params,
        "mase_scale_hg_ha": scale_train,
        "coverage": cov.to_dict(orient="records"),
        "conformal_quantiles_log": {str(k): v for k, v in cal_crop.global_q.items()},
        "explainability": expl,
        "deploy": {"history_forecaster": {"train_years": [int(H.year.min()), last_obs], "calibration_years": list(cal_years),
                                          "forecast_year": last_obs + 1},
                   "conditional_scenario": {"train_years": [int(M.year.min()), int(M.year.max())]}},
        "bias_by_crop": bias_crop.to_dict(orient="records"),
    }
    with open(reports / "training_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return summary
