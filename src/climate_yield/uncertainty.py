"""Split-conformal prediction intervals in log-growth space.

Residuals ``r = g_true - g_pred`` (log growth) come from a model fitted on
earlier years and evaluated on a later calibration window that precedes the
test period. For level ``1 - alpha`` the half-width is the finite-sample
conformal quantile of ``|r|``. Intervals are symmetric in log space, hence
multiplicative (asymmetric) in hg/ha: ``y_{t-1} * exp(g_pred +/- q)``.

Volatility differs strongly between crops, so a Mondrian (per-crop) quantile is
used when a crop has enough calibration points, with the pooled quantile as
fallback. Validity assumes calibration and future residuals are exchangeable;
with temporal drift this holds only approximately, which is why coverage is
checked on the untouched test years.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def conformal_quantile(abs_residuals: np.ndarray, level: float) -> float:
    r = np.sort(np.asarray(abs_residuals, dtype=float))
    n = len(r)
    if n == 0:
        raise ValueError("no calibration residuals")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    k = int(np.ceil((n + 1) * level))
    if k > n:
        return float("inf")
    return float(r[k - 1])


@dataclass
class ConformalCalibrator:
    levels: list[float]
    min_group_size: int = 30
    global_q: dict = field(default_factory=dict)
    group_q: dict = field(default_factory=dict)
    n_calibration: int = 0

    def fit(self, residuals, groups=None) -> "ConformalCalibrator":
        residuals = np.abs(np.asarray(residuals, dtype=float))
        if not np.isfinite(residuals).all():
            raise ValueError("calibration residuals must be finite")
        self.n_calibration = len(residuals)
        self.global_q = {lv: conformal_quantile(residuals, lv) for lv in self.levels}
        self.group_q = {}
        if groups is not None:
            s = pd.Series(residuals)
            for gname, idx in s.groupby(np.asarray(groups)).groups.items():
                if len(idx) >= self.min_group_size:
                    self.group_q[gname] = {lv: conformal_quantile(s.loc[idx].to_numpy(), lv) for lv in self.levels}
        return self

    def half_width(self, level: float, groups, n: int) -> np.ndarray:
        if level not in self.global_q:
            raise ValueError(f"level {level} was not calibrated; available {list(self.global_q)}")
        if groups is None:
            return np.full(n, self.global_q[level])
        return np.array([self.group_q.get(g, self.global_q)[level] for g in groups])

    def interval(self, pred_log_growth, log_y_lag1, level: float, groups=None):
        g = np.asarray(pred_log_growth, dtype=float)
        base = np.asarray(log_y_lag1, dtype=float)
        q = self.half_width(level, None if groups is None else np.asarray(groups), len(g))
        return np.exp(base + g - q), np.exp(base + g + q)


def coverage_summary(y_true, lower, upper, level: float) -> dict:
    y_true, lower, upper = (np.asarray(a, dtype=float) for a in (y_true, lower, upper))
    inside = (y_true >= lower) & (y_true <= upper)
    return {
        "nominal_coverage": level,
        "observed_coverage": float(inside.mean()),
        "mean_width_hg_ha": float(np.mean(upper - lower)),
        "median_width_hg_ha": float(np.median(upper - lower)),
        "median_relative_width": float(np.median((upper - lower) / y_true)),
        "n": int(len(y_true)),
    }
