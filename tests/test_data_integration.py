import numpy as np
import pandas as pd
import pytest

from climate_yield import data_integration as di
from climate_yield import data_loader as dl


def test_annual_rain_is_summed_and_temperature_averaged(synth_climate):
    rain, temp = synth_climate
    ann = di.aggregate_climate_annual(rain, temp)
    a = ann[(ann.area == "Alpha") & (ann.year == 1950)].iloc[0]
    r = rain[(rain.country == "Alpha") & (rain.year == 1950)]["rain_mm"]
    t = temp[(temp.country == "Alpha") & (temp.year == 1950)]["temp_c"]
    assert a.rain_total_mm == pytest.approx(r.sum())
    assert a.temp_mean_c == pytest.approx(t.mean())
    assert a.temp_max_month_c == pytest.approx(t.max())
    assert a.rain_min_month_mm == pytest.approx(r.min())


def test_incomplete_climate_years_are_dropped(synth_climate):
    rain, temp = synth_climate
    rain = rain[~((rain.country == "Beta") & (rain.year == 1999) & (rain.month == 7))]
    ann = di.aggregate_climate_annual(rain, temp)
    assert ann[(ann.area == "Beta") & (ann.year == 1999)].empty
    assert not ann[(ann.area == "Beta") & (ann.year == 1998)].empty


def test_name_harmonisation_applies_map():
    clim = pd.DataFrame({"country": ["Russia", "Alpha"], "iso3": ["RUS", "ALP"], "year": 2000, "month": 1, "rain_mm": 1.0})
    assert list(di.harmonise_climate_names(clim).area) == ["Russian Federation", "Alpha"]


def test_integration_does_not_inflate_rows(synth_yield, synth_climate, synth_pesticides):
    rain, temp = synth_climate
    ann = di.aggregate_climate_annual(rain, temp)
    panel, rep = di.integrate(synth_yield, synth_pesticides, ann)
    assert len(panel) == len(synth_yield)
    assert rep.lines["final_panel_rows"] == len(synth_yield)
    assert not panel.duplicated(["area", "crop", "year"]).any()
    # pesticides are national: identical for both crops of a country-year
    p = panel[(panel.area == "Alpha") & (panel.year == 1995)]
    assert p["pesticides_tonnes"].nunique() == 1 and len(p) == 2


def test_duplicate_join_keys_raise(synth_yield, synth_climate, synth_pesticides):
    rain, temp = synth_climate
    ann = di.aggregate_climate_annual(rain, temp)
    with pytest.raises(ValueError):
        di.integrate(synth_yield, pd.concat([synth_pesticides, synth_pesticides.iloc[[0]]]), ann)
    with pytest.raises(ValueError):
        di.integrate(pd.concat([synth_yield, synth_yield.iloc[[0]]]), synth_pesticides, ann)
    with pytest.raises(Exception):  # pandas MergeError from validate="m:1"
        di.integrate(synth_yield, synth_pesticides, pd.concat([ann, ann.iloc[[0]]]))


def test_exclusions_zero_yields_and_unmatched_areas(synth_yield, synth_climate, synth_pesticides):
    rain, temp = synth_climate
    ann = di.aggregate_climate_annual(rain, temp)
    y = synth_yield.copy()
    y.loc[0, "yield_hg_ha"] = 0.0
    extra = pd.DataFrame({"area": ["Nowhere"], "crop": ["Wheat"], "year": [1990], "yield_hg_ha": [1.0]})
    panel, rep = di.integrate(pd.concat([y, extra]), synth_pesticides, ann, exclude_areas=["Gamma"])
    assert rep.lines["zero_yield_rows_removed"] == 1
    assert "Nowhere" in rep.lines["areas_without_climate"]
    assert set(panel.area) == {"Alpha", "Beta"}


def test_real_name_map_targets_exist(cfg, real_data_available):
    if not real_data_available:
        pytest.skip("raw CSVs not present")
    raw = dl.load_raw_tables(cfg)
    fao = set(raw["yield"]["Area"])
    wb = set(raw["rainfall"]["Country"])
    for src, dst in di.COUNTRY_NAME_MAP.items():
        assert src in wb, src
        assert dst in fao, dst
    assert len({di.COUNTRY_NAME_MAP.get(c, c) for c in wb}) == len(wb)  # one-to-one


def test_climate_baseline_window(synth_climate):
    rain, temp = synth_climate
    ann = di.aggregate_climate_annual(rain, temp)
    base = di.climate_baseline(ann, (1901, 1950))
    exp = ann[(ann.area == "Alpha") & ann.year.between(1901, 1950)].rain_total_mm.mean()
    assert base.set_index("area").loc["Alpha", "rain_total_base_mm"] == pytest.approx(exp)
    assert np.isfinite(base.temp_mean_base_c).all()
