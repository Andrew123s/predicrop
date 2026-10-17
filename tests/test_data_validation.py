import pandas as pd
import pytest

from climate_yield import data_loader as dl
from climate_yield import data_validation as dv


def test_load_four_real_datasets(cfg, real_data_available):
    if not real_data_available:
        pytest.skip("raw CSVs not present")
    raw = dl.load_raw_tables(cfg)
    assert set(raw) == {"yield", "pesticides", "rainfall", "temperature"}
    for name, df in raw.items():
        assert list(df.columns) == dl.RAW_SCHEMAS[name]
        rep = dv.validate_raw(name, df)
        assert rep.ok, rep.errors
    tidy = dl.load_tidy_tables(cfg, raw)
    assert tidy["yield"]["yield_hg_ha"].ge(0).all()
    assert set(tidy["rainfall"]["month"]) == set(range(1, 13))
    # the two monthly climate files are row-aligned
    assert (raw["rainfall"]["Country"].to_numpy() == raw["temperature"]["Country"].to_numpy()).all()


def _raw_yield():
    return pd.DataFrame({"Domain": ["Crops"] * 3, "Area": ["A", "A", "B"], "Element": ["Yield"] * 3,
                         "Item": ["Wheat"] * 3, "Year": [2000, 2001, 2000], "Unit": ["hg/ha"] * 3,
                         "Value": [100, 0, 200]})


def test_schema_missing_column_is_error():
    assert not dv.validate_raw("yield", _raw_yield().drop(columns="Unit")).ok


def test_unexpected_unit_is_error():
    df = _raw_yield()
    df.loc[0, "Unit"] = "t/ha"
    assert any("Unit" in e for e in dv.validate_raw("yield", df).errors)
    with pytest.raises(dv.SchemaError):
        dv.require_valid("yield", df)


def test_duplicate_keys_and_missing_values_detected():
    df = pd.concat([_raw_yield(), _raw_yield().iloc[[0]]])
    assert any("duplicated" in e for e in dv.validate_raw("yield", df).errors)
    df = _raw_yield().astype({"Value": float})
    df.loc[1, "Value"] = None
    assert any("missing" in e for e in dv.validate_raw("yield", df).errors)


def test_zero_yield_is_warning_not_error():
    rep = dv.validate_raw("yield", _raw_yield())
    assert rep.ok and any("zero" in w for w in rep.warnings)


def test_constant_runs_and_jumps():
    y = pd.DataFrame({"area": "A", "crop": "W", "year": range(2000, 2010),
                      "yield_hg_ha": [5, 5, 5, 5, 5, 5, 7, 70, 8, 9]})
    runs = dv.constant_runs(y, 5)
    assert len(runs) == 1 and runs.iloc[0]["length"] == 6
    assert list(dv.extreme_jumps(y, 5)["year"]) == [2007, 2008]


def test_validate_feature_frame():
    with pytest.raises(dv.SchemaError):
        dv.validate_feature_frame(pd.DataFrame({"a": [1]}), ["a", "b"])
    with pytest.raises(dv.SchemaError):
        dv.validate_feature_frame(pd.DataFrame({"a": [float("inf")]}), ["a"])
    with pytest.raises(dv.SchemaError):
        dv.validate_feature_frame(pd.DataFrame({"a": ["x"]}), ["a"])
