# Climate-scenario (sensitivity) report

**What this is:** a sensitivity analysis of a statistical model. Each scenario changes the *same-year* annual rainfall and temperature inputs of the conditional model while every other input (yield history, lagged climate, pesticides) is held at its observed value. The reported number is the change in the model's predicted yield relative to its own prediction for observed conditions.

**What this is not:** a climate projection or an attribution of yield changes to climate. Real climate change alters seasonality, extremes, CO2, pests and farmer adaptation, none of which this model represents.

## Set-up

- Model: RandomForest trained on the `conditional` feature set, target years 1991-2016.
- Baseline: observed conditions of 2016 (the last year with climate data) for 1,226 (country, crop) series.
- Rainfall changes are multiplicative on annual total, wettest- and driest-month rainfall; temperature changes are additive on annual mean, warmest- and coldest-month means; anomalies vs 1901-1960 are recomputed. Scenario magnitudes are set in `config/config.yaml` and validated (|rain| <= 50%, |temp| <= 5 deg C).
- Extrapolation checks: (i) is the scenario value outside the country's observed 1901-2016 range? (ii) did the country ever experience a year within +/-10% rainfall *and* +/-0.5 deg C of the scenario combination (a historical analogue)?

## Results

| scenario | rainfall_change_pct | temperature_change_c | median_pct_change | p10_pct_change | p90_pct_change | share_rain_outside_history | share_temp_outside_history | share_with_historical_analogue | n |
|---|---|---|---|---|---|---|---|---|---|
| hotter_drier | -10.000 | 2.000 | -0.281 | -2.371 | 0.763 | 0.037 | 1.000 | 0.000 | 1226 |
| hotter_wetter | 10.000 | 1.000 | 0.167 | -0.467 | 1.024 | 0.098 | 0.933 | 0.197 | 1226 |
| rain_+10pct | 10.000 | 0.000 | 0.000 | -0.365 | 0.612 | 0.098 | 0.000 | 1.000 | 1226 |
| rain_+20pct | 20.000 | 0.000 | -0.033 | -0.796 | 1.013 | 0.294 | 0.000 | 0.716 | 1226 |
| rain_-10pct | -10.000 | 0.000 | -0.153 | -1.285 | 0.360 | 0.037 | 0.000 | 0.942 | 1226 |
| rain_-20pct | -20.000 | 0.000 | -0.693 | -3.902 | 0.310 | 0.131 | 0.000 | 0.865 | 1226 |
| temp_+1C | 0.000 | 1.000 | 0.080 | -0.499 | 0.749 | 0.000 | 0.933 | 0.249 | 1226 |
| temp_+2C | 0.000 | 2.000 | 0.079 | -0.665 | 0.790 | 0.000 | 1.000 | 0.007 | 1226 |

Response surface (median % change across series):

| rain_pct | 0.0 | 1.0 | 2.0 | 3.0 |
|---|---|---|---|---|
| -30 | -1.62 | -1.87 | -1.83 | -1.78 |
| -20 | -0.69 | -0.99 | -1.14 | -1.18 |
| -10 | -0.15 | -0.24 | -0.28 | -0.28 |
| 0 | 0.00 | 0.08 | 0.08 | 0.08 |
| 10 | 0.00 | 0.17 | 0.20 | 0.20 |
| 20 | -0.03 | 0.13 | 0.18 | 0.16 |
| 30 | -0.12 | 0.03 | 0.06 | 0.07 |

Figures: `figures/scenario_sensitivity_by_crop.png`, `figures/scenario_response.png`. Per-series results: `scenario_results.csv`.

## Interpretation

- Median responses are below 2% in every configured scenario (largest: rain_-20pct at 0.69%). Drier scenarios lower predictions slightly; warming scenarios barely move them because 93% of series are pushed beyond their warmest observed year by just +1 deg C (2016 was among the warmest observed years in most countries), where a tree ensemble cannot respond. The near-zero warming response is therefore an extrapolation artefact, not evidence that warming is harmless.
- The model's sensitivity to same-year climate is small compared with year-to-year yield variability. This agrees with the EDA (weak associations) and the feature-set comparison in `forecasting_report.md`.
- Tree ensembles are piecewise-constant: beyond the range of training data the response flattens, so large perturbations understate (or arbitrarily cap) any true response. Rows flagged as outside the country's history or without a historical analogue should be treated as extrapolation.
- Annual national climate means mix growing and non-growing seasons and irrigated and rain-fed land, which dilutes real weather signals.
