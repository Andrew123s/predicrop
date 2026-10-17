"""End-to-end data preparation shared by scripts, notebooks and the app."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import data_integration as di
from . import data_loader as dl
from . import data_validation as dv
from . import feature_engineering as fe
from .config import Config


@dataclass
class PreparedData:
    tidy: dict[str, pd.DataFrame]
    climate_annual: pd.DataFrame
    baseline: pd.DataFrame
    panel: pd.DataFrame
    features: pd.DataFrame          # full calendar grid incl. future target years
    modelling: pd.DataFrame         # benchmark rows (identical for every model)
    integration_report: dict


def prepare_data(cfg: Config, extend_to: int | None = None, save: bool = True) -> PreparedData:
    raw = dl.load_raw_tables(cfg)
    for name, df in raw.items():
        dv.require_valid(name, df)
    tidy = dl.load_tidy_tables(cfg, raw)
    annual = di.aggregate_climate_annual(tidy["rainfall"], tidy["temperature"])
    baseline = di.climate_baseline(annual, tuple(cfg["integration"]["climate_baseline_years"]))
    panel, rep = di.integrate(tidy["yield"], tidy["pesticides"], annual, cfg["integration"]["exclude_fao_areas"])
    last_obs = int(panel["year"].max())
    features = fe.build_feature_table(panel, annual, baseline, tidy["pesticides"], extend_to=extend_to or last_obs + 1)
    task = cfg["task"]
    modelling = fe.modelling_rows(features, task["first_target_year"], task["last_target_year"])
    rep = dict(rep.lines)
    rep["modelling_rows"] = len(modelling)
    rep["modelling_series"] = int(modelling.groupby(["area", "crop"]).ngroups)
    rep["modelling_years"] = [int(modelling["year"].min()), int(modelling["year"].max())]
    if save:
        out = cfg.path("processed_dir")
        out.mkdir(parents=True, exist_ok=True)
        panel.to_csv(out / "integrated_panel.csv", index=False)
        annual.to_csv(out / "climate_annual.csv", index=False)
        modelling.to_csv(out / "modelling_dataset.csv", index=False)
        with open(out / "integration_report.json", "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=2, ensure_ascii=False)
    return PreparedData(tidy, annual, baseline, panel, features, modelling, rep)


def fingerprint(cfg: Config, extra: dict | None = None) -> str:
    """Hash of raw data, configuration and source code; used to skip retraining
    when nothing that affects the models has changed."""
    h = hashlib.sha256()
    for name in cfg["raw_files"]:
        h.update(Path(cfg.raw_file(name)).read_bytes())
    h.update(json.dumps(cfg.raw, sort_keys=True).encode())
    for p in sorted((cfg.root / "src" / "climate_yield").glob("*.py")):
        h.update(p.read_bytes())
    if extra:
        h.update(json.dumps(extra, sort_keys=True).encode())
    return h.hexdigest()
