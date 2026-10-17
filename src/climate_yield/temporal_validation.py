"""Chronological partitions and expanding-window cross-validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    train_years: tuple[int, int]
    validation_years: tuple[int, int]
    test_years: tuple[int, int]

    def __post_init__(self) -> None:
        tr, va, te = self.train_years, self.validation_years, self.test_years
        for lo, hi in (tr, va, te):
            if lo > hi:
                raise ValueError(f"invalid year range {lo}-{hi}")
        if not (tr[1] < va[0] and va[1] < te[0]):
            raise ValueError("partitions must be chronological and non-overlapping")

    @classmethod
    def from_config(cls, task: dict) -> "TemporalSplit":
        return cls(
            train_years=(int(task["first_target_year"]), int(task["train_end_year"])),
            validation_years=tuple(int(y) for y in task["validation_years"]),
            test_years=tuple(int(y) for y in task["test_years"]),
        )

    def mask(self, years: pd.Series, part: str) -> pd.Series:
        lo, hi = {"train": self.train_years, "validation": self.validation_years,
                  "test": self.test_years, "development": (self.train_years[0], self.validation_years[1])}[part]
        return years.between(lo, hi)

    def describe(self) -> dict:
        return {
            "train": f"{self.train_years[0]}-{self.train_years[1]}",
            "validation": f"{self.validation_years[0]}-{self.validation_years[1]}",
            "test": f"{self.test_years[0]}-{self.test_years[1]}",
        }


def expanding_window_folds(years: pd.Series | np.ndarray, fold_years: list[list[int]]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Index pairs for rolling-origin CV.

    For each fold ``[lo, hi]`` the training indices are every row with
    ``year < lo`` and the validation indices every row with ``lo <= year <= hi``.
    Training data therefore always precede validation data.
    """
    years = np.asarray(years)
    folds = []
    for lo, hi in fold_years:
        tr = np.flatnonzero(years < lo)
        va = np.flatnonzero((years >= lo) & (years <= hi))
        if len(tr) == 0 or len(va) == 0:
            raise ValueError(f"fold {lo}-{hi} has an empty train or validation set")
        folds.append((tr, va))
    return folds


def assert_no_temporal_overlap(train_years, eval_years) -> None:
    if np.max(train_years) >= np.min(eval_years):
        raise AssertionError("training data overlap or follow evaluation data")
