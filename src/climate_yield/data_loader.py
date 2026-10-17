"""Load the four original CSV files into tidy, consistently named frames.

The raw files are read-only inputs. Column names are standardised here; values
and units are preserved exactly as supplied.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Exact raw headers observed during inspection (after stripping whitespace).
RAW_SCHEMAS = {
    "yield": ["Domain", "Area", "Element", "Item", "Year", "Unit", "Value"],
    "pesticides": ["Domain", "Area", "Element", "Item", "Year", "Unit", "Value"],
    "rainfall": ["Rainfall - (MM)", "Year", "Statistics", "Country", "ISO3"],
    "temperature": ["Temperature - (Celsius)", "Year", "Statistics", "Country", "ISO3"],
}


def read_raw(path: str | Path) -> pd.DataFrame:
    """Read a raw CSV, stripping the UTF-8 BOM and padding around headers/values."""
    df = pd.read_csv(path, encoding="utf-8-sig", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


def tidy_yield(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.rename(columns={"Area": "area", "Item": "crop", "Year": "year", "Value": "yield_hg_ha"})
    return out[["area", "crop", "year", "yield_hg_ha"]].astype({"year": int, "yield_hg_ha": float})


def tidy_pesticides(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.rename(columns={"Area": "area", "Year": "year", "Value": "pesticides_tonnes"})
    return out[["area", "year", "pesticides_tonnes"]].astype({"year": int, "pesticides_tonnes": float})


def tidy_climate(raw: pd.DataFrame, value_col: str, name: str) -> pd.DataFrame:
    """Monthly climate records -> (country, iso3, year, month, <name>)."""
    month_abbr = raw["Statistics"].str.replace(" Average", "", regex=False)
    unknown = set(month_abbr) - set(MONTHS)
    if unknown:
        raise ValueError(f"Unexpected Statistics labels: {sorted(unknown)}")
    out = pd.DataFrame(
        {
            "country": raw["Country"],
            "iso3": raw["ISO3"],
            "year": raw["Year"].astype(int),
            "month": month_abbr.map({m: i + 1 for i, m in enumerate(MONTHS)}).astype(int),
            name: raw[value_col].astype(float),
        }
    )
    return out


def load_raw_tables(cfg: Config) -> dict[str, pd.DataFrame]:
    """Raw tables exactly as on disk (headers/whitespace normalised only)."""
    return {name: read_raw(cfg.raw_file(name)) for name in RAW_SCHEMAS}


def load_tidy_tables(cfg: Config, raw: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    raw = raw or load_raw_tables(cfg)
    return {
        "yield": tidy_yield(raw["yield"]),
        "pesticides": tidy_pesticides(raw["pesticides"]),
        "rainfall": tidy_climate(raw["rainfall"], "Rainfall - (MM)", "rain_mm"),
        "temperature": tidy_climate(raw["temperature"], "Temperature - (Celsius)", "temp_c"),
    }
