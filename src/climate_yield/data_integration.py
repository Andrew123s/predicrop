"""Harmonise identifiers and integrate yield, climate and pesticide data.

Join design (decided after inspecting the actual values):

* yield.csv (FAO) is keyed by ``(Area, Item, Year)``; it is the target table.
* pesticides.csv (FAO) is keyed by ``(Area, Year)`` and uses the same FAO
  country names, so it joins many-to-one on ``(area, year)``. It holds national
  totals for all crops, so the same value is attached to every crop of a country.
* rainfall.csv / temp.csv (World Bank style, monthly, 1901-2016) are keyed by
  ``(Country, Year, Statistics=month)``. Country names differ from FAO names, so
  an explicit, reviewed name map is applied. Monthly values are aggregated to
  calendar-year statistics before joining many-to-one on ``(area, year)``.

No crop calendar is available, so climate cannot be aligned to crop-specific
growing seasons; calendar-year aggregates are the defensible common resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# World Bank climate country name -> FAO area name. Built by comparing the two
# name sets; every entry was checked by hand. Names identical in both sources
# need no entry.
COUNTRY_NAME_MAP = {
    "The Bahamas": "Bahamas",
    "Bolivia": "Bolivia (Plurinational State of)",
    "Brunei": "Brunei Darussalam",
    "Cape Verde": "Cabo Verde",
    "China": "China, mainland",
    "Congo (Republic of the)": "Congo",
    "Congo (Democratic Republic of the)": "Democratic Republic of the Congo",
    "Czech Republic": "Czechia",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Swaziland": "Eswatini",
    "Iran": "Iran (Islamic Republic of)",
    "Laos": "Lao People's Democratic Republic",
    "Federated States of Micronesia": "Micronesia (Federated States of)",
    "Republic of Montenegro": "Montenegro",
    "Myanmar (Burma)": "Myanmar",
    "Macedonia": "North Macedonia",
    "Moldova": "Republic of Moldova",
    "Russia": "Russian Federation",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Lucia": "Saint Lucia",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "Republic of Serbia": "Serbia",
    "Syria": "Syrian Arab Republic",
    "Timor Leste": "Timor-Leste",
    "United Kingdom": "United Kingdom of Great Britain and Northern Ireland",
    "United States": "United States of America",
    "Venezuela": "Venezuela (Bolivarian Republic of)",
    "Vietnam": "Viet Nam",
}

CLIMATE_FEATURES = [
    "rain_total_mm",
    "rain_max_month_mm",
    "rain_min_month_mm",
    "temp_mean_c",
    "temp_max_month_c",
    "temp_min_month_c",
]


@dataclass
class IntegrationReport:
    lines: dict = field(default_factory=dict)

    def add(self, key: str, value) -> None:
        self.lines[key] = value


def harmonise_climate_names(climate: pd.DataFrame) -> pd.DataFrame:
    out = climate.copy()
    out["area"] = out["country"].replace(COUNTRY_NAME_MAP)
    if out[["area", "year", "month"]].duplicated().any():
        raise ValueError("name harmonisation created duplicate (area, year, month) keys")
    return out


def aggregate_climate_annual(rain: pd.DataFrame, temp: pd.DataFrame) -> pd.DataFrame:
    """Monthly -> calendar-year statistics per area.

    Rainfall is accumulated precipitation, so the annual total is the *sum* of the
    12 monthly values; wettest/driest month capture intra-annual distribution.
    Temperature is a monthly mean, so it is *averaged* (never summed); the
    warmest/coldest monthly means are reported. These are monthly means, not
    daily extremes. Years without all 12 months are dropped.
    """
    rain = harmonise_climate_names(rain)
    temp = harmonise_climate_names(temp)
    r = rain.groupby(["area", "year"]).agg(
        n_rain=("rain_mm", "size"),
        rain_total_mm=("rain_mm", "sum"),
        rain_max_month_mm=("rain_mm", "max"),
        rain_min_month_mm=("rain_mm", "min"),
    )
    t = temp.groupby(["area", "year"]).agg(
        n_temp=("temp_c", "size"),
        temp_mean_c=("temp_c", "mean"),
        temp_max_month_c=("temp_c", "max"),
        temp_min_month_c=("temp_c", "min"),
    )
    annual = r.join(t, how="inner").reset_index()
    annual = annual[(annual["n_rain"] == 12) & (annual["n_temp"] == 12)]
    return annual.drop(columns=["n_rain", "n_temp"]).reset_index(drop=True)


def climate_baseline(annual: pd.DataFrame, years: tuple[int, int]) -> pd.DataFrame:
    """Per-area climatology over a fixed historical window (default 1901-1960)."""
    lo, hi = years
    base = annual[(annual["year"] >= lo) & (annual["year"] <= hi)]
    return base.groupby("area").agg(
        rain_total_base_mm=("rain_total_mm", "mean"),
        temp_mean_base_c=("temp_mean_c", "mean"),
    ).reset_index()


def integrate(
    yield_df: pd.DataFrame,
    pesticides: pd.DataFrame,
    climate_annual: pd.DataFrame,
    exclude_areas: list[str] | None = None,
) -> tuple[pd.DataFrame, IntegrationReport]:
    """Return the yield panel with annual climate and pesticides attached.

    Climate and pesticides are attached for the same calendar year as the yield
    record; *lagging* (to respect prediction-time availability) happens in
    feature engineering, not here. Rows whose area has no climate record at all
    are excluded from the panel and listed in the report.
    """
    rep = IntegrationReport()
    rep.add("yield_rows_raw", len(yield_df))
    rep.add("pesticide_rows_raw", len(pesticides))
    rep.add("climate_area_years", len(climate_annual))

    y = yield_df.copy()
    exclude_areas = exclude_areas or []
    n_excl = int(y["area"].isin(exclude_areas).sum())
    y = y[~y["area"].isin(exclude_areas)]
    rep.add("excluded_aggregate_areas", {"areas": exclude_areas, "rows": n_excl})

    zero = y["yield_hg_ha"] <= 0
    rep.add("zero_yield_rows_removed", int(zero.sum()))
    y = y[~zero]

    if y.duplicated(["area", "crop", "year"]).any():
        raise ValueError("duplicate (area, crop, year) keys in yield data")
    if pesticides.duplicated(["area", "year"]).any():
        raise ValueError("duplicate (area, year) keys in pesticide data")

    climate_areas = set(climate_annual["area"])
    no_climate = sorted(set(y["area"]) - climate_areas)
    rows_no_climate = int(y["area"].isin(no_climate).sum())
    rep.add("areas_without_climate", no_climate)
    rep.add("rows_without_climate_area", rows_no_climate)
    rep.add("climate_areas_without_yield", sorted(climate_areas - set(y["area"])))
    y = y[~y["area"].isin(no_climate)]

    n_before = len(y)
    merged = y.merge(climate_annual, on=["area", "year"], how="left", validate="m:1")
    merged = merged.merge(pesticides, on=["area", "year"], how="left", validate="m:1")
    if len(merged) != n_before:
        raise AssertionError("join changed the row count (duplicate inflation)")

    rep.add("pesticide_areas_without_yield", sorted(set(pesticides["area"]) - set(yield_df["area"])))
    rep.add("yield_areas_without_pesticides", sorted(set(y["area"]) - set(pesticides["area"])))
    rep.add("rows_missing_same_year_climate", int(merged["rain_total_mm"].isna().sum()))
    rep.add("rows_missing_same_year_pesticides", int(merged["pesticides_tonnes"].isna().sum()))
    rep.add("final_panel_rows", len(merged))
    rep.add("final_panel_series", int(merged.groupby(["area", "crop"]).ngroups))
    rep.add("final_panel_areas", int(merged["area"].nunique()))
    return merged.sort_values(["area", "crop", "year"]).reset_index(drop=True), rep
