"""Configuration loading and repository paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


@dataclass
class Config:
    raw: dict[str, Any]
    root: Path = REPO_ROOT

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def path(self, key: str) -> Path:
        """Absolute path for an entry under ``paths``."""
        return self.root / self.raw["paths"][key]

    def raw_file(self, name: str) -> Path:
        return self.path("raw_dir") / self.raw["raw_files"][name]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def mode_settings(self, mode: str | None = None) -> dict[str, Any]:
        mode = mode or self.raw["training"]["mode"]
        if mode not in self.raw["training"]["modes"]:
            raise ValueError(f"Unknown training mode {mode!r}")
        return {"mode": mode, **self.raw["training"]["modes"][mode]}

    def ensure_dirs(self) -> None:
        for key in ("processed_dir", "models_dir", "reports_dir", "figures_dir"):
            self.path(key).mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw)
