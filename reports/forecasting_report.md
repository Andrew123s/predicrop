# Forecasting report

Training mode: **full**. Every number in this report was produced by `scripts/train_models.py` and `scripts/generate_forecasts.py` on the supplied data.

## Task

- **Target:** FAO crop yield in hg/ha for a (country, crop) series in year *t*.
- **Horizon:** one year ahead. Forecasts are issued at the start of year *t* using yields up to *t-1*, calendar-year climate of *t-1* and national pesticide use of *t-1*.
- **Primary task:** forecasting future yields of *known* (country, crop) series. Unseen countries or crops are out of scope (every feature is anchored on the series' own history).
- Models predict log growth log(y_t / y_{t-1}); forecasts are y_{t-1} x exp(prediction).

## Validation design

- Chronological partitions of target years: train 1991-2008, validation 2009-2012, test 2013-2016 (rows: 20,408 / 4,857 / 4,905).
- Hyper-parameters tuned by random search scored on expanding-window CV over the development years with validation folds [[2005, 2006], [2007, 2008], [2009, 2010], [2011, 2012]]; each fold trains only on earlier years.
- `Validation MAE` = mean CV MAE of the selected configuration; the model with the lowest value is selected. The test period was used once, after selection.
- All models are evaluated on identical rows.

## Model comparison (test period)

| Model | Validation MAE | Holdout Validation MAE (2009-2012) | Test MAE | Test RMSE | Test R2 | Prediction Bias | Test MedAPE % | Test MAE log | Test MASE | Training Time (s) | Prediction Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Naive (last value) | 9,569.901 | 10,018.307 | 10,660.110 | 29,785.503 | 0.935 | -817.867 | 5.260 | 0.118 | 1.216 | 0.151 | 0.022 |
| DummyRegressor (mean growth) | 9,662.018 | 10,102.210 | 10,716.602 | 29,997.214 | 0.935 | 390.558 | 5.178 | 0.118 | 1.222 | 0.177 | 0.020 |
| LinearRegression | 9,465.330 | 9,880.832 | 10,471.209 | 25,914.698 | 0.951 | -553.055 | 6.143 | 0.114 | 1.194 | 0.296 | 0.028 |
| RandomForest | 9,155.404 | 9,523.557 | 10,008.504 | 26,050.304 | 0.951 | -522.374 | 5.533 | 0.109 | 1.141 | 12.916 | 0.158 |
| XGBoost | 9,232.213 | 9,566.139 | 10,097.827 | 26,605.322 | 0.949 | -141.604 | 5.496 | 0.109 | 1.151 | 2.967 | 0.034 |
| LightGBM | 9,241.199 | 9,555.538 | 10,035.548 | 26,107.262 | 0.950 | -318.179 | 5.441 | 0.109 | 1.144 | 2.158 | 0.077 |

**Selected model (lowest validation MAE): RandomForest.** On the test years it has MAE 10,009 hg/ha vs 10,660 for the naive last-value forecast (6.1% lower), RMSE 26,050 vs 29,786, R2 0.951 vs 0.935, and median absolute percentage error 5.53% vs 5.26%.

The naive forecast has the lower *median* percentage error. The model's gains come from the tail: it partly anticipates mean reversion after large shocks (lag-1 autocorrelation of growth is negative), which lowers MAE, RMSE and mean log error, while for the many stable series - including long runs of identical FAO values, where persistence is exactly right - it adds small deviations. Persistence is therefore a strong baseline that the models beat on average error but not uniformly.

Metric notes: MAE/RMSE/bias are in hg/ha (10,000 hg/ha = 1 t/ha) and are dominated by high-mass crops (potatoes, carrots, cauliflowers). R2 on yield levels is high for every model because between-series level differences are huge; it says little about forecast skill, so MASE (MAE relative to the naive forecast's training-period MAE) and MAE of log yield are also reported. MAPE is reported but is inflated by a few near-zero / erratic yields; median APE is more robust.

## Does climate or pesticide information help?

Same algorithm and hyper-parameters (RandomForest), same rows:

| feature_set | n_features | uses_same_year_climate | valid_forecast | cv_MAE | test_MAE | test_RMSE | test_R2 | test_MedAPE_pct | test_Bias |
|---|---|---|---|---|---|---|---|---|---|
| history | 11 | False | True | 9,193.962 | 10,055.463 | 25,946.526 | 0.951 | 5.519 | -996.800 |
| forecast | 23 | False | True | 9,155.404 | 10,008.504 | 26,050.304 | 0.951 | 5.533 | -522.374 |
| conditional | 32 | True | False | 9,197.381 | 10,013.167 | 26,230.830 | 0.950 | 5.553 | -587.062 |

- `history`: lagged yields only. `forecast`: + lagged climate & pesticides (the deployable benchmark model). `conditional`: + **same-year** climate, which is *not* known at forecast time; it is an upper bound on what perfect seasonal-climate knowledge could add and is used only for scenarios.

## Bias

- Mean prediction error of RandomForest on the test years: -522 hg/ha (naive: -818). Positive = overprediction.
- By crop (test):

| crop | model | n | MAE | Bias | MedAPE_pct |
|---|---|---|---|---|---|
| Barley | RandomForest | 401 | 3,326.9 | -803.7 | 7.8 |
| Buckwheat | RandomForest | 110 | 1,760.1 | -219.3 | 9.1 |
| Carrots and turnips | RandomForest | 482 | 24,287.7 | 115.5 | 4.7 |
| Cauliflowers and broccoli | RandomForest | 368 | 15,783.4 | 5,390.9 | 4.8 |
| Garlic | RandomForest | 384 | 10,299.9 | -2,531.6 | 4.7 |
| Ginger | RandomForest | 148 | 7,595.7 | -802.9 | 3.9 |
| Hops | RandomForest | 112 | 1,354.6 | -2.2 | 3.8 |
| Maize | RandomForest | 652 | 6,627.5 | -828.4 | 6.4 |
| Oats | RandomForest | 296 | 2,417.5 | -416.4 | 6.4 |
| Onions, shallots, green | RandomForest | 200 | 11,637.0 | -91.8 | 2.7 |
| Potatoes | RandomForest | 606 | 17,934.1 | -2,286.1 | 4.9 |
| Rye | RandomForest | 248 | 3,495.9 | -114.6 | 7.7 |
| Sweet potatoes | RandomForest | 418 | 9,803.4 | -2,093.3 | 3.2 |
| Wheat | RandomForest | 480 | 3,377.1 | -403.6 | 7.8 |
| Barley | Naive (last value) | 401 | 3,570.4 | -614.1 | 7.2 |
| Buckwheat | Naive (last value) | 110 | 1,880.7 | -289.8 | 6.3 |
| Carrots and turnips | Naive (last value) | 482 | 25,984.5 | 134.6 | 4.3 |
| Cauliflowers and broccoli | Naive (last value) | 368 | 17,393.6 | 3,903.8 | 4.7 |
| Garlic | Naive (last value) | 384 | 10,787.2 | -2,915.5 | 4.9 |
| Ginger | Naive (last value) | 148 | 6,901.7 | -1,506.0 | 3.0 |
| Hops | Naive (last value) | 112 | 1,759.0 | -221.9 | 6.9 |
| Maize | Naive (last value) | 652 | 7,114.0 | -1,398.6 | 5.6 |
| Oats | Naive (last value) | 296 | 2,813.7 | -275.3 | 6.7 |
| Onions, shallots, green | Naive (last value) | 200 | 12,066.2 | -701.7 | 2.6 |
| Potatoes | Naive (last value) | 606 | 19,140.4 | -3,016.9 | 4.6 |
| Rye | Naive (last value) | 248 | 3,883.0 | -48.6 | 8.4 |
| Sweet potatoes | Naive (last value) | 418 | 9,845.5 | -1,723.8 | 3.0 |
| Wheat | Naive (last value) | 480 | 3,752.4 | -360.7 | 7.5 |

## Uncertainty (split conformal)

- Residuals of the model fitted on 1991-2008 and evaluated on 2009-2012 calibrate the intervals; test labels are never used for calibration. Per-crop (Mondrian) quantiles when a crop has >= 30 calibration points.

| method | nominal_coverage | observed_coverage | mean_width_hg_ha | median_width_hg_ha | median_relative_width | n |
|---|---|---|---|---|---|---|
| per-crop (Mondrian) | 0.800 | 0.818 | 32,422.179 | 20,677.018 | 0.347 | 4905 |
| pooled | 0.800 | 0.813 | 36,423.473 | 21,031.704 | 0.347 | 4905 |
| per-crop (Mondrian) | 0.900 | 0.903 | 52,888.952 | 33,453.891 | 0.549 | 4905 |
| pooled | 0.900 | 0.907 | 59,138.932 | 34,148.103 | 0.564 | 4905 |

Observed coverage by test year (per-crop method):

| year | nominal_coverage | observed_coverage | median_relative_width | n |
|---|---|---|---|---|
| 2013 | 0.800 | 0.836 | 0.346 | 1228 |
| 2014 | 0.800 | 0.805 | 0.340 | 1227 |
| 2015 | 0.800 | 0.811 | 0.351 | 1224 |
| 2016 | 0.800 | 0.821 | 0.350 | 1226 |
| 2013 | 0.900 | 0.917 | 0.550 | 1228 |
| 2014 | 0.900 | 0.896 | 0.543 | 1227 |
| 2015 | 0.900 | 0.891 | 0.552 | 1224 |
| 2016 | 0.900 | 0.907 | 0.548 | 1226 |

Lowest observed coverage by crop:

| crop | nominal_coverage | observed_coverage | n |
|---|---|---|---|
| Potatoes | 0.800 | 0.785 | 606 |
| Maize | 0.800 | 0.793 | 652 |
| Sweet potatoes | 0.800 | 0.797 | 418 |
| Ginger | 0.800 | 0.804 | 148 |
| Garlic | 0.800 | 0.807 | 384 |
| Carrots and turnips | 0.800 | 0.824 | 482 |
| Barley | 0.800 | 0.825 | 401 |
| Cauliflowers and broccoli | 0.800 | 0.826 | 368 |

Assumption check: split conformal requires calibration and test residuals to be exchangeable. `residual_spread_by_year.csv` compares the median absolute residual per year in the calibration and test windows; differences there translate directly into under- or over-coverage.

## Explainability

- Model explained: RandomForest (best tree model). SHAP: ok.
- Permutation importance (MAE increase in hg/ha when a column is shuffled, test sample):

| feature | mae_increase_hg_ha | std |
|---|---|---|
| dev_from_roll5 | 1,114.4 | 128.3 |
| growth_lag1 | 705.3 | 46.3 |
| log_y_lag1 | 134.5 | 51.9 |
| trend_4y | 93.2 | 32.1 |
| roll5_std_log | 41.3 | 11.0 |
| log_pesticides_lag1 | 37.1 | 6.2 |
| temp_anom_c_lag1 | 31.1 | 8.2 |
| temp_min_month_c_lag1 | 22.4 | 5.4 |
| rain_min_month_mm_lag1 | 19.4 | 7.7 |
| rain_max_month_mm_lag1 | 17.6 | 7.3 |
| temp_mean_base_c | 13.8 | 2.4 |
| n_prior_obs | 9.7 | 5.6 |

- Mean |SHAP| (log-growth units, grouped by source column):

| source | mean_abs_shap |
|---|---|
| dev_from_roll5 | 0.0269 |
| growth_lag1 | 0.0179 |
| trend_4y | 0.0061 |
| temp_anom_c_lag1 | 0.0048 |
| log_y_lag1 | 0.0047 |
| temp_max_month_c_lag1 | 0.0017 |
| roll5_std_log | 0.0017 |
| temp_mean_c_lag1 | 0.0015 |
| log_pesticides_lag1 | 0.0015 |
| rain_min_month_mm_lag1 | 0.0015 |
| log_y_lag3 | 0.0014 |
| rain_total_base_mm | 0.0013 |

Importance is predictive reliance, not causal effect; correlated predictors share or swap importance.

## Extra out-of-sample check: 2017

- The benchmark model (trained 1991-2012) forecasting 1,227 series in 2017: MAE 8,881 hg/ha (naive 9,188), median APE 4.62% (naive 4.20%), 90% interval coverage 0.905, 80% coverage 0.828.
- Forecasts for 2020 (1,204 series) from the history-only deployment model are in `forecasts_2020.csv`; they cannot be evaluated because no 2020 yields are available.

## Figures

`figures/model_comparison.png`, `actual_vs_predicted.png`, `forecast_examples.png`, `residual_distribution.png`, `residuals_vs_predicted.png`, `error_over_time.png`, `medape_by_crop.png`, `interval_coverage_by_crop.png`, `permutation_importance.png`, `shap_importance.png`, `shap_beeswarm.png`, `shap_waterfall_example.png`.
