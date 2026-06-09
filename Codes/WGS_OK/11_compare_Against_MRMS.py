#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
python 11_compare_Against_MRMS.py \
  --event 7 \
  --out-dir "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/11_Compare_Against_MRMS" \
  --make-figure \
  --fig-width 11 \
  --fig-height 8.5 \
  --label-size 15 \
  --tick-size 12 \
  --legend-size 13 \
  --value-size 11

'''
BASE_OK = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK")
BASE_IDW = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_IDW")

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

REFERENCE_PATH = (
    BASE_OK
    / "Radar_Event_TimeSeries"
    / "MRMS_Event_TimeSeries"
    / "Event_{event}"
    / "Event_{event}_grid_rain_hourly_mm_MRMS.csv"
)

PRODUCT_PATHS = {
    "OK": BASE_OK / "03_Interpolated_Rain/Event_{event}_grid_rain_hourly_mm.csv",
    "IDW": BASE_IDW / "03_Interpolated_Rain/Event_{event}_grid_rain_hourly_mm.csv",
    "R(A)": BASE_OK / "Radar_Event_TimeSeries/RA/Event_{event}/Event_{event}_grid_rain_hourly_mm_RA.csv",
    "R(Z)": BASE_OK / "Radar_Event_TimeSeries/RZ/Event_{event}/Event_{event}_grid_rain_hourly_mm_RZ.csv",
    "R(KDP)": BASE_OK / "Radar_Event_TimeSeries/RKDP/Event_{event}/Event_{event}_grid_rain_hourly_mm_RKDP.csv",
    "Composite_2": BASE_OK / "Radar_Event_TimeSeries/Composite_2/Event_{event}/Event_{event}_grid_rain_hourly_mm_Composite_2.csv",
}

OUT_DIR = BASE_OK / "11_Compare_Against_MRMS"

PLOT_ORDER = ["OK", "IDW", "R(A)", "R(Z)", "R(KDP)", "Composite_2"]


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------

def normalize_grid_col(col) -> str:
    s = str(col).strip()
    try:
        f = float(s)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
    except Exception:
        pass
    return s


def load_grid_rain_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    time_col = "time_local" if "time_local" in df.columns else df.columns[0]

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).set_index(time_col)

    df.columns = [normalize_grid_col(c) for c in df.columns]
    df = df.loc[:, ~df.columns.astype(str).str.lower().str.startswith("unnamed")]
    df = df.apply(pd.to_numeric, errors="coerce")

    return df.sort_index().sort_index(axis=1)


def align_to_reference(ref: pd.DataFrame, sim: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_time = ref.index.intersection(sim.index)
    common_cols = ref.columns.intersection(sim.columns)

    if len(common_time) == 0:
        raise ValueError("No common timestamps")
    if len(common_cols) == 0:
        raise ValueError("No common grid cells")

    ref_a = ref.loc[common_time, common_cols].sort_index().sort_index(axis=1)
    sim_a = sim.loc[common_time, common_cols].sort_index().sort_index(axis=1)

    return ref_a, sim_a


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def compute_metrics(ref: np.ndarray, sim: np.ndarray) -> dict[str, float]:
    """
    ref = MRMS
    sim = comparison product

    Bias is mean error:
        mean(Product / MRMS)

    MAE:
        mean(abs(Product - MRMS))

    RMSE:
        sqrt(mean((Product - MRMS)^2))
    """
    ref = np.asarray(ref, dtype=float)
    sim = np.asarray(sim, dtype=float)

    mask = np.isfinite(ref) & np.isfinite(sim)
    n = int(mask.sum())

    if n == 0:
        return {
            "n": 0,
            "mrms_sum_mm": np.nan,
            "product_sum_mm": np.nan,
            "cumulative_diff_mm": np.nan,
            "cumulative_diff_pct": np.nan,
            "bias_mm": np.nan,
            "mae_mm": np.nan,
            "mse_mm2": np.nan,
            "rmse_mm": np.nan,
            "correlation": np.nan,
            "bias_ratio": np.nan,
        }

    r = ref[mask]
    s = sim[mask]
    err = s - r

    mrms_sum = float(np.sum(r))
    product_sum = float(np.sum(s))

    return {
        "n": n,
        "mrms_sum_mm": mrms_sum,
        "product_sum_mm": product_sum,
        "cumulative_diff_mm": product_sum - mrms_sum,
        "cumulative_diff_pct": 100.0 * (product_sum - mrms_sum) / mrms_sum if mrms_sum != 0 else np.nan,
        "bias_mm": float(np.mean(err)),
        "mae_mm": float(np.mean(np.abs(err))),
        "mse_mm2": float(np.mean(err ** 2)),
        "rmse_mm": float(np.sqrt(np.mean(err ** 2))),
        "correlation": float(np.corrcoef(r, s)[0, 1]) if n >= 2 else np.nan,
        "bias_ratio": product_sum / mrms_sum if mrms_sum != 0 else np.nan,
    }


def compare_event(event: int, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref_path = Path(str(REFERENCE_PATH).format(event=event))
    ref = load_grid_rain_csv(ref_path)

    event_rows = []
    cell_rows = []

    for product in PLOT_ORDER:
        template = PRODUCT_PATHS[product]
        sim_path = Path(str(template).format(event=event))

        if not sim_path.exists():
            event_rows.append({
                "event": event,
                "product": product,
                "status": "missing_file",
                "file": str(sim_path),
            })
            continue

        try:
            sim = load_grid_rain_csv(sim_path)
            ref_a, sim_a = align_to_reference(ref, sim)

            overall = compute_metrics(
                ref_a.to_numpy(dtype=float).ravel(),
                sim_a.to_numpy(dtype=float).ravel(),
            )

            event_rows.append({
                "event": event,
                "product": product,
                "status": "ok",
                "reference": "MRMS",
                "reference_file": str(ref_path),
                "file": str(sim_path),
                "n_common_hours": len(ref_a.index),
                "n_common_cells": len(ref_a.columns),
                **overall,
            })

            for gid in ref_a.columns:
                m = compute_metrics(
                    ref_a[gid].to_numpy(dtype=float),
                    sim_a[gid].to_numpy(dtype=float),
                )
                cell_rows.append({
                    "event": event,
                    "product": product,
                    "grid_id": gid,
                    **m,
                })

        except Exception as e:
            event_rows.append({
                "event": event,
                "product": product,
                "status": "error",
                "file": str(sim_path),
                "error": str(e),
            })

    event_df = pd.DataFrame(event_rows)
    cell_df = pd.DataFrame(cell_rows)

    out_dir.mkdir(parents=True, exist_ok=True)

    event_df.to_csv(out_dir / f"Event_{event}_metrics_against_MRMS.csv", index=False)
    cell_df.to_csv(out_dir / f"Event_{event}_cell_metrics_against_MRMS.csv", index=False)

    return event_df, cell_df


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def autolabel_bars(
    ax,
    bars,
    fmt: str = "{:.1f}",
    fontsize: int = 11,
    offset_frac: float = 0.015,
) -> None:
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    offset = offset_frac * yrange

    for bar in bars:
        h = bar.get_height()
        if not np.isfinite(h):
            continue

        x = bar.get_x() + bar.get_width() / 2.0

        if h >= 0:
            y = h + offset
            va = "bottom"
        else:
            y = h - offset
            va = "top"

        ax.text(
            x,
            y,
            fmt.format(h),
            ha="center",
            va=va,
            fontsize=fontsize,
            color="black",
        )


def plot_event_metrics(
    event_df: pd.DataFrame,
    event: int,
    out_png: Path,
    *,
    fig_width: float = 11.0,
    fig_height: float = 8.5,
    dpi: int = 300,
    label_size: int = 15,
    tick_size: int = 12,
    legend_size: int = 13,
    value_size: int = 11,
    title: str | None = None,
) -> None:
    df = event_df[event_df["status"] == "ok"].copy()

    if df.empty:
        raise ValueError(f"No successful product comparisons found for Event {event}")

    df["product"] = pd.Categorical(df["product"], categories=PLOT_ORDER, ordered=True)
    df = df.sort_values("product")

    products = df["product"].astype(str).tolist()
    x = np.arange(len(products))

    bias = df["bias_ratio"].to_numpy(dtype=float)
    mae = df["mae_mm"].to_numpy(dtype=float)
    rmse = df["rmse_mm"].to_numpy(dtype=float)
    cumdiff = df["cumulative_diff_mm"].to_numpy(dtype=float)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": [1.0, 1.05]},
    )

    if title:
        fig.suptitle(title, fontsize=24, fontweight="bold", x=0.02, ha="left")

    # -------------------------
    # Top: Bias, MAE, RMSE
    # -------------------------
    width = 0.23

    b1 = ax1.bar(x - width, bias, width, label="Bias ratio", color="#ed7d31")
    b2 = ax1.bar(x, mae, width, label="MAE", color="#a5a5a5")
    b3 = ax1.bar(x + width, rmse, width, label="RMSE", color="#ffc000")

    ax1.set_ylabel("Metric value", fontsize=label_size, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(products, fontsize=tick_size, fontweight="bold")
    ax1.tick_params(axis="y", labelsize=tick_size)
    ax1.grid(axis="y", alpha=0.35, linewidth=0.8)
    ax1.set_axisbelow(True)

    ax1.legend(
        loc="upper right",
        frameon=False,
        ncol=3,
        fontsize=legend_size,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=1.3,
    )

    top_max = np.nanmax([np.nanmax(np.abs(bias)), np.nanmax(mae), np.nanmax(rmse)])
    if not np.isfinite(top_max) or top_max <= 0:
        top_max = 1.0
    ax1.set_ylim(0, top_max * 1.25)
    autolabel_bars(ax1, b1, fmt="{:.2f}", fontsize=value_size)
    autolabel_bars(ax1, b2, fmt="{:.1f}", fontsize=value_size)
    autolabel_bars(ax1, b3, fmt="{:.1f}", fontsize=value_size)

    # -------------------------
    # Bottom: cumulative difference
    # -------------------------
    b4 = ax2.bar(x, cumdiff, width=0.35, color="#4472c4", label="ΣProduct − ΣMRMS")

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Cumulative Rain Difference (mm)", fontsize=label_size, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(products, fontsize=tick_size, fontweight="bold")
    ax2.tick_params(axis="y", labelsize=tick_size)
    ax2.grid(axis="y", alpha=0.35, linewidth=0.8)
    ax2.set_axisbelow(True)

    max_abs = np.nanmax(np.abs(cumdiff))
    if not np.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0
    ax2.set_ylim(-max_abs * 1.30, max_abs * 1.30)

    ax2.legend(
        loc="lower right",
        frameon=False,
        fontsize=legend_size,
        handlelength=1.0,
        handletextpad=0.4,
    )

    autolabel_bars(ax2, b4, fmt="{:.2f}", fontsize=value_size)

    # Clean boxes
    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96] if title else None)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved figure] {out_png}")


def plot_all_events_metrics(
    all_event_df: pd.DataFrame,
    out_dir: Path,
    *,
    dpi: int,
    fig_width: float,
    fig_height: float,
    label_size: int,
    tick_size: int,
    legend_size: int,
    value_size: int,
    no_title: bool,
) -> None:
    for event, event_df in all_event_df.groupby("event"):
        title = None if no_title else f"Results: Metrics ({event} event)"

        out_png = out_dir / f"Event_{int(event)}_metrics_against_MRMS.png"

        plot_event_metrics(
            event_df=event_df,
            event=int(event),
            out_png=out_png,
            fig_width=fig_width,
            fig_height=fig_height,
            dpi=dpi,
            label_size=label_size,
            tick_size=tick_size,
            legend_size=legend_size,
            value_size=value_size,
            title=title,
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare OK, IDW, and radar rainfall products against MRMS reference."
    )

    parser.add_argument(
        "--event",
        type=int,
        nargs="+",
        required=True,
        help="One or more events, e.g. --event 4 or --event 4 7",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)

    parser.add_argument("--make-figure", action="store_true")
    parser.add_argument("--no-title", action="store_true")

    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--fig-width", type=float, default=11.0)
    parser.add_argument("--fig-height", type=float, default=8.5)
    parser.add_argument("--label-size", type=int, default=15)
    parser.add_argument("--tick-size", type=int, default=12)
    parser.add_argument("--legend-size", type=int, default=13)
    parser.add_argument("--value-size", type=int, default=11)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_event_rows = []
    all_cell_rows = []

    for event in args.event:
        print(f"Comparing Event {event} against MRMS")
        event_df, cell_df = compare_event(event, args.out_dir)
        all_event_rows.append(event_df)
        all_cell_rows.append(cell_df)

    all_event_df = pd.concat(all_event_rows, ignore_index=True)
    all_cell_df = pd.concat(all_cell_rows, ignore_index=True)

    all_event_df.to_csv(args.out_dir / "All_events_metrics_against_MRMS.csv", index=False)
    all_cell_df.to_csv(args.out_dir / "All_events_cell_metrics_against_MRMS.csv", index=False)

    if args.make_figure:
        plot_all_events_metrics(
            all_event_df=all_event_df,
            out_dir=args.out_dir,
            dpi=args.dpi,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            label_size=args.label_size,
            tick_size=args.tick_size,
            legend_size=args.legend_size,
            value_size=args.value_size,
            no_title=args.no_title,
        )


if __name__ == "__main__":
    main()