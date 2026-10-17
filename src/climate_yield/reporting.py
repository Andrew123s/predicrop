"""Figures and Markdown helpers for the evaluation reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def df_to_markdown(df: pd.DataFrame, floatfmt: str = "{:,.3f}") -> str:
    """Minimal Markdown table writer (avoids an optional ``tabulate`` dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_actual_vs_predicted(y, p, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 6))
    hb = ax.hexbin(y, p, gridsize=60, bins="log", xscale="log", yscale="log", cmap="viridis", mincnt=1)
    lo, hi = np.nanmin([y.min(), p.min()]), np.nanmax([y.max(), p.max()])
    ax.plot([lo, hi], [lo, hi], color="crimson", lw=1, label="perfect forecast")
    ax.set_xlabel("Observed yield (hg/ha, log scale)")
    ax.set_ylabel("Forecast yield (hg/ha, log scale)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.colorbar(hb, ax=ax, label="count (log)")
    return save(fig, path)


def plot_residuals(df: pd.DataFrame, title: str, path_hist: Path, path_scatter: Path) -> tuple[Path, Path]:
    log_res = np.log(df["y_pred"]) - np.log(df["y_true"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(log_res.clip(-1.5, 1.5), bins=80, ax=axes[0])
    axes[0].axvline(0, color="crimson", lw=1)
    axes[0].set_xlabel("log(forecast) - log(observed)  [clipped to +/-1.5]")
    axes[0].set_title("Residual distribution (log scale)")
    sns.histplot((df["y_pred"] - df["y_true"]).clip(-1e5, 1e5), bins=80, ax=axes[1])
    axes[1].axvline(0, color="crimson", lw=1)
    axes[1].set_xlabel("forecast - observed (hg/ha) [clipped to +/-100,000]")
    axes[1].set_title("Residual distribution (hg/ha)")
    fig.suptitle(title)
    p1 = save(fig, path_hist)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(df["y_pred"], log_res, s=4, alpha=0.25)
    ax.axhline(0, color="crimson", lw=1)
    ax.set_xscale("log")
    ax.set_ylim(-2, 2)
    ax.set_xlabel("Forecast yield (hg/ha, log scale)")
    ax.set_ylabel("log residual (forecast - observed)")
    ax.set_title(title + ": residuals vs forecast")
    p2 = save(fig, path_scatter)
    return p1, p2


def plot_error_over_time(err: pd.DataFrame, path: Path, test_start: int) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for model, g in err.groupby("model"):
        ax.plot(g["year"], g["MAE"], marker="o", ms=3, label=model)
    ax.axvline(test_start - 0.5, color="grey", ls="--", lw=1)
    ax.text(test_start - 0.4, ax.get_ylim()[1] * 0.95, "test period", va="top", fontsize=9)
    ax.set_xlabel("Target year")
    ax.set_ylabel("MAE (hg/ha)")
    ax.set_title("Forecast error by year (validation fits on 1991-2008, test fits on 1991-2012)")
    ax.legend(fontsize=8, ncol=2)
    return save(fig, path)


def plot_series_examples(history: pd.DataFrame, preds: pd.DataFrame, series: list[tuple[str, str]], path: Path) -> Path:
    n = len(series)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.3 * rows), squeeze=False)
    for ax, (area, crop) in zip(axes.ravel(), series):
        h = history[(history["area"] == area) & (history["crop"] == crop)]
        p = preds[(preds["area"] == area) & (preds["crop"] == crop)].sort_values("year")
        ax.plot(h["year"], h["yield_hg_ha"], color="black", lw=1.4, label="observed")
        if "lower" in p and p["lower"].notna().any():
            ax.fill_between(p["year"], p["lower"], p["upper"], color="tab:blue", alpha=0.2, label="90% conformal")
        for part, g in p.groupby("partition"):
            ax.plot(g["year"], g["y_pred"], marker="o", ms=3, lw=1, label=f"forecast ({part})")
        ax.set_title(f"{crop} - {area}", fontsize=10)
        ax.set_ylabel("hg/ha")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("One-year-ahead forecasts vs observed yields")
    return save(fig, path)


def plot_metric_by_group(df: pd.DataFrame, group: str, metric: str, hue: str, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    order = df[df[hue] == df[hue].iloc[0]].sort_values(metric)[group]
    sns.barplot(data=df, y=group, x=metric, hue=hue, order=order, ax=ax)
    ax.set_title(title)
    return save(fig, path)


def plot_model_comparison(comp: pd.DataFrame, path: Path) -> Path:
    d = comp.melt(id_vars="Model", value_vars=["Validation MAE", "Test MAE"], var_name="split", value_name="MAE")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=d, y="Model", x="MAE", hue="split", ax=ax)
    ax.set_xlabel("MAE (hg/ha)")
    ax.set_title("Model comparison (validation = mean expanding-window CV MAE)")
    return save(fig, path)


def plot_importance(df: pd.DataFrame, value: str, path: Path, title: str, top: int = 20) -> Path:
    d = df.head(top)
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(d) + 1.5))
    sns.barplot(data=d, y=d.columns[0], x=value, ax=ax, color="tab:blue")
    ax.set_title(title)
    return save(fig, path)


def plot_coverage(cov: pd.DataFrame, group: str, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=cov, x=group, y="observed_coverage", hue="nominal_coverage", ax=ax)
    for lv in sorted(cov["nominal_coverage"].unique()):
        ax.axhline(lv, ls="--", lw=1, color="grey")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=60)
    return save(fig, path)
