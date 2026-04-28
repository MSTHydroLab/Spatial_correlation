#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fit_line(x: np.ndarray, y: np.ndarray):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    coef = np.polyfit(x[mask], y[mask], 1)
    return coef


def plot_scatter_with_trend(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    out_png: Path,
    title: str,
    ylabel: str,
):
    fig, ax = plt.subplots(figsize=(11, 7))

    methods = ["IDW", "OK"]
    colors = {"IDW": "tab:blue", "OK": "tab:orange"}

    for method in methods:
        sub = df[df["method"] == method].copy()
        if sub.empty:
            continue

        x = pd.to_numeric(sub[xcol], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub[ycol], errors="coerce").to_numpy(dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        ax.scatter(
            x,
            y,
            s=18,
            alpha=0.18,
            label=method,
            color=colors[method],
        )

        coef = fit_line(x, y)
        if coef is not None:
            xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
            yy = coef[0] * xx + coef[1]
            ax.plot(
                xx,
                yy,
                linestyle="--",
                linewidth=2.5,
                color=colors[method],
                label=f"{method} trend",
            )

    ax.set_title(title, fontsize=20, weight="bold")
    ax.set_xlabel("Average distance to target station (km)", fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot distance vs interpolation metric from combo-level LOOCV results.")
    ap.add_argument("--csv", required=True, type=Path, help="Event_*_loo_metrics_by_combo.csv")
    ap.add_argument("--out-dir", required=True, type=Path, help="Output directory for plots")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    required = ["method", "avg_distance_km", "mae", "rmse", "kge"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_scatter_with_trend(
        df=df,
        xcol="avg_distance_km",
        ycol="mae",
        out_png=args.out_dir / "distance_vs_mae_scatter.png",
        title="Distance vs MAE (mm) (scatter)",
        ylabel="MAE (mm)",
    )

    plot_scatter_with_trend(
        df=df,
        xcol="avg_distance_km",
        ycol="rmse",
        out_png=args.out_dir / "distance_vs_rmse_scatter.png",
        title="Distance vs RMSE (mm) (scatter)",
        ylabel="RMSE (mm)",
    )

    plot_scatter_with_trend(
        df=df,
        xcol="avg_distance_km",
        ycol="kge",
        out_png=args.out_dir / "distance_vs_kge_scatter.png",
        title="Distance vs KGE (scatter)",
        ylabel="KGE",
    )


if __name__ == "__main__":
    main()