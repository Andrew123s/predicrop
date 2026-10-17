"""Schema checks and data-quality profiling for the raw and tidy tables."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data_loader import RAW_SCHEMAS


class SchemaError(ValueError):
    """Raised when a table does not match its expected structure."""


@dataclass
class ValidationReport:
    name: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


# Constant metadata columns: a change here would mean a different variable or unit.
EXPECTED_CONSTANTS = {
    "yield": {"Domain": "Crops", "Element": "Yield", "Unit": "hg/ha"},
    "pesticides": {
        "Domain": "Pesticides Use",
        "Element": "Agricultural Use",
        "Item": "Pesticides (total)",
        "Unit": "tonnes",
    },
}
KEYS = {
    "yield": ["Area", "Item", "Year"],
    "pesticides": ["Area", "Year"],
    "rainfall": ["Country", "Year", "Statistics"],
    "temperature": ["Country", "Year", "Statistics"],
}
VALUE_COLS = {
    "yield": "Value",
    "pesticides": "Value",
    "rainfall": "Rainfall - (MM)",
    "temperature": "Temperature - (Celsius)",
}
# Physically plausible bounds used only to flag (not drop) records.
PLAUSIBLE_RANGES = {
    "rainfall": (0.0, 3000.0),       # mm per month
    "temperature": (-60.0, 50.0),    # monthly mean deg C
}


def validate_raw(name: str, df: pd.DataFrame) -> ValidationReport:
    """Validate one raw table against the structure observed during inspection."""
    rep = ValidationReport(name)
    expected = RAW_SCHEMAS[name]
    if list(df.columns) != expected:
        rep.errors.append(f"columns {list(df.columns)} != expected {expected}")
        return rep
    nulls = df.isna().sum()
    for col, n in nulls[nulls > 0].items():
        rep.errors.append(f"{n} missing values in {col}")
    for col, value in EXPECTED_CONSTANTS.get(name, {}).items():
        observed = set(df[col].unique())
        if observed != {value}:
            rep.errors.append(f"{col} expected only {value!r}, found {sorted(observed)}")
    dup = df.duplicated(KEYS[name]).sum()
    if dup:
        rep.errors.append(f"{dup} duplicated keys on {KEYS[name]}")
    val = df[VALUE_COLS[name]]
    if not np.issubdtype(val.dtype, np.number):
        rep.errors.append(f"{VALUE_COLS[name]} is not numeric")
        return rep
    if name in ("yield", "pesticides", "rainfall") and (val < 0).any():
        rep.errors.append(f"{(val < 0).sum()} negative values in {VALUE_COLS[name]}")
    if name in PLAUSIBLE_RANGES:
        lo, hi = PLAUSIBLE_RANGES[name]
        out = ((val < lo) | (val > hi)).sum()
        if out:
            rep.warnings.append(f"{out} values outside plausible range [{lo}, {hi}]")
    zeros = int((val == 0).sum())
    if zeros and name == "yield":
        rep.warnings.append(f"{zeros} zero yields (treated as missing: log undefined, implausible)")
    return rep


def require_valid(name: str, df: pd.DataFrame) -> None:
    rep = validate_raw(name, df)
    if not rep.ok:
        raise SchemaError(f"{name}: " + "; ".join(rep.errors))


def profile(df: pd.DataFrame) -> dict:
    """Generic profile used by the data-quality report."""
    num = df.select_dtypes(include="number")
    cat = df.select_dtypes(exclude="number")
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_summary": num.describe().T if not num.empty else pd.DataFrame(),
        "categorical_cardinality": {c: int(df[c].nunique()) for c in cat.columns},
    }


def constant_runs(tidy_yield: pd.DataFrame, min_run: int = 5) -> pd.DataFrame:
    """Series where the same yield value repeats for >= ``min_run`` consecutive years.

    Long flat runs usually indicate FAO estimates/imputations rather than
    measurements; they are flagged, not removed.
    """
    rows = []
    for (area, crop), g in tidy_yield.sort_values("year").groupby(["area", "crop"]):
        v = g["yield_hg_ha"].to_numpy()
        yrs = g["year"].to_numpy()
        start = 0
        for i in range(1, len(v) + 1):
            if i == len(v) or v[i] != v[start] or yrs[i] != yrs[i - 1] + 1:
                if i - start >= min_run:
                    rows.append({"area": area, "crop": crop, "start": int(yrs[start]),
                                 "end": int(yrs[i - 1]), "length": i - start, "value": v[start]})
                start = i
    return pd.DataFrame(rows, columns=["area", "crop", "start", "end", "length", "value"])


def extreme_jumps(tidy_yield: pd.DataFrame, factor: float = 5.0) -> pd.DataFrame:
    """Year-on-year changes larger than ``factor``x in either direction (possible entry errors)."""
    y = tidy_yield[tidy_yield["yield_hg_ha"] > 0].sort_values(["area", "crop", "year"]).copy()
    prev = y.groupby(["area", "crop"])["yield_hg_ha"].shift(1)
    prev_year = y.groupby(["area", "crop"])["year"].shift(1)
    ratio = y["yield_hg_ha"] / prev
    mask = (prev_year == y["year"] - 1) & ((ratio > factor) | (ratio < 1 / factor))
    out = y.loc[mask, ["area", "crop", "year", "yield_hg_ha"]].copy()
    out["previous"] = prev[mask]
    out["ratio"] = ratio[mask]
    return out.reset_index(drop=True)


def validate_feature_frame(df: pd.DataFrame, required: list[str]) -> None:
    """Used by prediction code: required columns present and numeric."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"missing feature columns: {missing}")
    for c in required:
        if c == "crop":
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise SchemaError(f"feature {c} must be numeric")
        if np.isinf(df[c].to_numpy(dtype=float)).any():
            raise SchemaError(f"feature {c} contains infinite values")
