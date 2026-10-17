"""Shared fixtures. Synthetic data here exist only to test software behaviour;
they are never mixed with the real datasets or reported as model performance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from climate_yield.config import load_config
from climate_yield.data_integration import aggregate_climate_annual, climate_baseline, integrate
from climate_yield.feature_engineering import build_feature_table

AREAS = ["Alpha", "Beta", "Gamma"]
CROPS = ["Wheat", "Maize"]
YEARS = list(range(1960, 2001))


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def real_data_available(cfg):
    return all(cfg.raw_file(n).exists() for n in cfg["raw_files"])


@pytest.fixture
def synth_yield():
    rng = np.random.default_rng(0)
    rows = []
    for ai, a in enumerate(AREAS):
        for ci, c in enumerate(CROPS):
            level = 20000 * (1 + ai) * (1 + ci)
            for y in YEARS:
                rows.append({"area": a, "crop": c, "year": y,
                             "yield_hg_ha": float(level * 1.01 ** (y - 1960) * np.exp(rng.normal(0, 0.05)))})
    df = pd.DataFrame(rows)
    # a gap so that calendar-aligned lags can be tested
    return df[~((df.area == "Beta") & (df.crop == "Wheat") & (df.year == 1990))].reset_index(drop=True)


@pytest.fixture
def synth_climate():
    rng = np.random.default_rng(1)
    rain, temp = [], []
    for a in AREAS:
        for y in range(1901, 2001):
            for m in range(1, 13):
                rain.append({"country": a, "iso3": a[:3].upper(), "year": y, "month": m,
                             "rain_mm": float(max(0, rng.normal(80, 20)))})
                temp.append({"country": a, "iso3": a[:3].upper(), "year": y, "month": m,
                             "temp_c": float(15 + 8 * np.sin(m / 12 * 2 * np.pi) + rng.normal(0, 1))})
    return pd.DataFrame(rain), pd.DataFrame(temp)


@pytest.fixture
def synth_pesticides():
    return pd.DataFrame([{"area": a, "year": y, "pesticides_tonnes": float(100 * (i + 1) + y - 1990)}
                         for i, a in enumerate(AREAS) for y in range(1990, 2001)])


@pytest.fixture
def synth_bundle(synth_yield, synth_climate, synth_pesticides):
    rain, temp = synth_climate
    annual = aggregate_climate_annual(rain, temp)
    base = climate_baseline(annual, (1901, 1950))
    panel, _ = integrate(synth_yield, synth_pesticides, annual)
    F = build_feature_table(panel, annual, base, synth_pesticides, extend_to=2001)
    return {"features": F, "annual": annual, "baseline": base, "panel": panel}
