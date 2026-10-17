# Data

## Raw inputs (repository root, never modified)

| file | rows | key | content | unit | coverage |
|---|---|---|---|---|---|
| `yield.csv` | 63,691 | Area, Item, Year | FAO crop yield (Element = *Yield*) | hg/ha | 1961-2019, 203 areas, 14 crops |
| `pesticides.csv` | 4,789 | Area, Year | FAO *Pesticides (total)*, Element = *Agricultural Use* | tonnes | 1990-2018, 172 areas |
| `rainfall.csv` | 272,832 | Country, Year, Statistics | monthly precipitation, label `<Mon> Average` | mm | 1901-2016, 196 countries |
| `temp.csv` | 272,832 | Country, Year, Statistics | monthly mean temperature, label `<Mon> Average` | deg C | 1901-2016, 196 countries |

The yield and pesticide files have the layout of FAOSTAT exports; the climate files have the layout of the
World Bank Climate Change Knowledge Portal country series. None of the files carries its own
licence or citation, so the provenance is inferred from structure and naming (see the main README).

## Data dictionary

### yield.csv

| column | type | role | description |
|---|---|---|---|
| Domain | text | constant | always `Crops` |
| Area | text | **join key** (country) | FAO country / territory name; includes the aggregate `China` and historical entities (USSR, Yugoslav SFR, ...) |
| Element | text | constant | always `Yield` - the value is yield per harvested area, not production |
| Item | text | **grouping key** (crop) | 14 crops: Barley, Buckwheat, Carrots and turnips, Cauliflowers and broccoli, Garlic, Ginger, Hops, Maize, Oats, Onions shallots green, Potatoes, Rye, Sweet potatoes, Wheat |
| Year | int | **time key** | calendar year of the harvest statistic |
| Unit | text | constant | always `hg/ha` (hectograms per hectare; 10,000 hg/ha = 1 t/ha) |
| Value | int | **target** | yield in hg/ha; 5 zeros (removed) and one value of 1 hg/ha |

### pesticides.csv

| column | type | role | description |
|---|---|---|---|
| Domain | text | constant | `Pesticides Use` |
| Area | text | join key | FAO country name (same naming as yield.csv) |
| Element | text | constant | `Agricultural Use` |
| Item | text | constant | `Pesticides (total)` - all pesticide classes combined |
| Year | int | time key | calendar year |
| Unit | text | constant | `tonnes` |
| Value | int | predictor (lagged) | national total for **all crops**; not per hectare, not crop-specific |

### rainfall.csv / temp.csv

| column | type | role | description |
|---|---|---|---|
| `Rainfall - (MM)` / `Temperature - (Celsius)` | float | predictor | monthly country value |
| Year | int | time key | calendar year |
| Statistics | text | month | `Jan Average` ... `Dec Average` |
| Country | text | join key | World Bank country name - 28 names differ from FAO names and are mapped explicitly |
| ISO3 | text | identifier | ISO 3166 alpha-3 code (not present in the FAO files, so not usable as a join key) |

## Derived files (`data/processed/`, regenerated, not committed)

Created by `scripts/run_eda.py` / `scripts/inspect_data.py` via `climate_yield.pipeline.prepare_data`:

| file | content |
|---|---|
| `integrated_panel.csv` | yield records with same-year annual climate and pesticides attached (country x crop x year) |
| `climate_annual.csv` | annual climate statistics per country-year (rain total, wettest/driest month, mean/warmest/coldest-month temperature) |
| `modelling_dataset.csv` | the 30,170 benchmark rows (target years 1991-2016) with all engineered features |
| `integration_report.json` | row counts, unmatched areas, exclusions |

Feature definitions and their availability at prediction time are listed in
`src/climate_yield/feature_engineering.py` (`FEATURE_DOCS`) and in the README.
