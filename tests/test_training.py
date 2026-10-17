import joblib
import numpy as np
import pytest

from climate_yield import feature_engineering as fe
from climate_yield.temporal_validation import expanding_window_folds
from climate_yield.train import NaiveLastValue, build_pipeline, model_specs, predict_yield, tune

SETTINGS = {"rf_n_estimators": 10, "boosting_n_estimators": 20}


@pytest.fixture
def modelling(synth_bundle):
    return fe.modelling_rows(synth_bundle["features"], 1991, 2000)


@pytest.mark.parametrize("name", list(model_specs(SETTINGS, 0)))
def test_every_model_trains_and_predicts(name, modelling):
    spec = model_specs(SETTINGS, 0)[name]
    feats = fe.FEATURE_SETS["forecast"]
    tr, te = modelling[modelling.year <= 1997], modelling[modelling.year > 1997]
    pipe = build_pipeline(spec["estimator"], feats, spec["scale"]).fit(tr[feats], tr[fe.MODEL_TARGET])
    y = predict_yield(pipe, te[feats])
    assert y.shape == (len(te),) and np.isfinite(y).all() and (y > 0).all()


def test_naive_reproduces_previous_year(modelling):
    feats = fe.FEATURE_SETS["history"]
    pipe = build_pipeline(NaiveLastValue(), feats, False).fit(modelling[feats], modelling[fe.MODEL_TARGET])
    np.testing.assert_allclose(predict_yield(pipe, modelling[feats]), np.exp(modelling["log_y_lag1"]))


def test_tuning_uses_only_supplied_folds(modelling):
    spec = model_specs(SETTINGS, 0)["LightGBM"]
    folds = expanding_window_folds(modelling["year"], [[1995, 1996], [1997, 1998]])
    res = tune(spec, modelling, fe.FEATURE_SETS["forecast"], folds, n_iter=2, seed=0)
    assert len(res.trials) == 2 and len(res.cv_fold_mae) == 2
    assert res.cv_mae == pytest.approx(min(t["cv_mae"] for t in res.trials))


def test_reproducible_with_fixed_seed(modelling):
    feats = fe.FEATURE_SETS["forecast"]
    preds = []
    for _ in range(2):
        spec = model_specs(SETTINGS, 7)["RandomForest"]
        p = build_pipeline(spec["estimator"], feats, False).fit(modelling[feats], modelling[fe.MODEL_TARGET])
        preds.append(p.predict(modelling[feats]))
    np.testing.assert_allclose(preds[0], preds[1])


def test_save_and_load_roundtrip(modelling, tmp_path):
    spec = model_specs(SETTINGS, 0)["XGBoost"]
    feats = fe.FEATURE_SETS["forecast"]
    pipe = build_pipeline(spec["estimator"], feats, False).fit(modelling[feats], modelling[fe.MODEL_TARGET])
    path = tmp_path / "m.joblib"
    joblib.dump({"pipeline": pipe}, path)
    loaded = joblib.load(path)["pipeline"]
    np.testing.assert_allclose(loaded.predict(modelling[feats]), pipe.predict(modelling[feats]))


def test_imputer_fitted_on_training_rows_only(modelling):
    feats = fe.FEATURE_SETS["forecast"]
    tr = modelling[modelling.year <= 1995]
    pipe = build_pipeline(model_specs(SETTINGS, 0)["LinearRegression"]["estimator"], feats, True)
    pipe.fit(tr[feats], tr[fe.MODEL_TARGET])
    imputer = pipe.named_steps["prep"].named_transformers_["num"].named_steps["impute"]
    num = [f for f in feats if f != "crop"]
    np.testing.assert_allclose(imputer.statistics_, tr[num].median().to_numpy(), equal_nan=True)
