#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK")
EVENT_TS_DIR = BASE_DIR / "01_Event_TimeSeries"
PLOT_DIR = BASE_DIR / "01_Correlation_Plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_pair_file(event: int, event_ts_dir: Path) -> pd.DataFrame:
    fp = event_ts_dir / f"Event_{event}_pair_correlations.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Pair correlation file not found: {fp}")

    df = pd.read_csv(fp)

    needed = ["station_1", "station_2", "distance_km", "correlation"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{fp} is missing required columns: {missing}")

    df["station_1"] = df["station_1"].astype(str)
    df["station_2"] = df["station_2"].astype(str)
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")
    df["correlation"] = pd.to_numeric(df["correlation"], errors="coerce")
    df = df.dropna(subset=["distance_km", "correlation"]).copy()

    return df

def parse_station_list(raw: str) -> set[str]:
    if raw is None:
        return set()

    s = str(raw).strip()
    if s == "":
        return set()

    out = set()
    for part in s.split(","):
        val = str(part).strip()
        if val != "":
            try:
                out.add(str(int(float(val))))
            except Exception:
                out.add(val)
    return out
def drop_ignored_stations(df: pd.DataFrame, ignored: set[str]) -> pd.DataFrame:
    if not ignored:
        return df.copy()

    out = df[
        (~df["station_1"].astype(str).isin(ignored))
        & (~df["station_2"].astype(str).isin(ignored))
    ].copy()

    return out
def select_suspicious_pairs(
    df: pd.DataFrame,
    mode: str,
    percentile: float,
    distance_min: float | None = None,
    distance_max: float | None = None,
) -> tuple[pd.DataFrame, float]:
    work = df.copy()

    if distance_min is not None:
        work = work[work["distance_km"] >= float(distance_min)].copy()
    if distance_max is not None:
        work = work[work["distance_km"] <= float(distance_max)].copy()

    if mode == "negative":
        base = work[work["correlation"] < 0].copy()
        if base.empty:
            return base, np.nan

        # keep the most negative tail
        cutoff = 0
        suspicious = base[base["correlation"] <= cutoff].copy()
        return suspicious, float(cutoff)

    elif mode == "lowtail":
        cutoff = np.percentile(work["correlation"].to_numpy(float), float(percentile))
        suspicious = work[work["correlation"] <= cutoff].copy()
        return suspicious, float(cutoff)

    else:
        raise ValueError("mode must be 'negative' or 'lowtail'")


def count_station_frequency(df_suspicious: pd.DataFrame) -> pd.DataFrame:
    if df_suspicious.empty:
        return pd.DataFrame(columns=["station_id", "count"])

    station_list = pd.concat(
        [
            df_suspicious["station_1"].rename("station_id"),
            df_suspicious["station_2"].rename("station_id"),
        ],
        axis=0,
        ignore_index=True,
    )

    freq = (
        station_list.value_counts()
        .rename_axis("station_id")
        .reset_index(name="count")
        .sort_values(["count", "station_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return freq


def save_outputs(
    event: int,
    all_pairs: pd.DataFrame,
    suspicious: pd.DataFrame,
    station_freq: pd.DataFrame,
    cutoff: float,
    mode: str,
    percentile: float,
    out_dir: Path,
    top_n: int,
    ignored_stations: set[str] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ignored_stations = ignored_stations or set()
    ignore_tag = ""
    ignore_text = "None"
    if ignored_stations:
        ignore_tag = "_ignored_" + "-".join(sorted(ignored_stations))
        ignore_text = ", ".join(sorted(ignored_stations))

    # ------------------------------------------------------------
    # Figure 1: scatter with suspicious pairs highlighted
    # ------------------------------------------------------------
    plt.figure(figsize=(11, 6))
    plt.scatter(
        all_pairs["distance_km"],
        all_pairs["correlation"],
        s=6,
        alpha=0.35,
        label="All accepted pairs",
    )

    if not suspicious.empty:
        plt.scatter(
            suspicious["distance_km"],
            suspicious["correlation"],
            s=10,
            alpha=0.9,
            label=f"Selected suspicious pairs ({len(suspicious)})",
        )

    if np.isfinite(cutoff):
        plt.axhline(cutoff, linestyle="--", linewidth=1.5, label=f"Cutoff = {cutoff:.3f}")

    plt.xlabel("Distance (km)")
    plt.ylabel("Correlation")
    plt.title(
        f"Event {event}: updated correlation plot\n"
        f"Mode={mode}, percentile={percentile}, ignored={ignore_text}"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig1 = out_dir / f"Event_{event}_updated_correlation_plot_{mode}_p{int(percentile)}{ignore_tag}.png"
    plt.savefig(fig1, dpi=300, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------
    # Figure 2: histogram of suspicious correlation values
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    if not suspicious.empty:
        plt.hist(suspicious["correlation"], bins=30)
        if np.isfinite(cutoff):
            plt.axvline(cutoff, linestyle="--", linewidth=1.5, label=f"Cutoff = {cutoff:.3f}")
            plt.legend()
    plt.xlabel("Correlation")
    plt.ylabel("Count")
    plt.title(
        f"Event {event}: histogram of suspicious correlations\n"
        f"Mode={mode}, percentile={percentile}, ignored={ignore_text}"
    )
    plt.grid(True, alpha=0.3)
    fig2 = out_dir / f"Event_{event}_diagnostic_corr_hist_{mode}_p{int(percentile)}{ignore_tag}.png"
    plt.savefig(fig2, dpi=300, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------
    # Figure 3: station frequency bar chart
    # ------------------------------------------------------------
    plt.figure(figsize=(11, 6))
    plot_df = station_freq.head(int(top_n)).copy()

    if not plot_df.empty:
        x = np.arange(len(plot_df))
        plt.bar(x, plot_df["count"].to_numpy())
        plt.xticks(x, plot_df["station_id"].astype(str).tolist(), rotation=90)

    plt.xlabel("Station ID")
    plt.ylabel("Frequency in suspicious pairs")
    plt.title(
        f"Event {event}: stations contributing to suspicious branch\n"
        f"Top {top_n}, mode={mode}, percentile={percentile}, ignored={ignore_text}"
    )
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig3 = out_dir / f"Event_{event}_diagnostic_station_bar_{mode}_p{int(percentile)}{ignore_tag}.png"
    plt.savefig(fig3, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved updated correlation plot: {fig1}")
    print(f"Saved correlation histogram: {fig2}")
    print(f"Saved station bar chart: {fig3}")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Diagnose the suspicious correlation branch for one event. "
            "By default it isolates the negative-correlation branch and counts "
            "which stations appear most often in those pairs."
        )
    )
    ap.add_argument("--event", type=int, required=True)
    ap.add_argument("--base-dir", type=Path, default=BASE_DIR)
    ap.add_argument(
        "--mode",
        choices=["negative", "lowtail"],
        default="negative",
        help=(
            "negative: analyze only correlation < 0 and keep the most negative tail; "
            "lowtail: analyze the lowest tail of all correlations"
        ),
    )
    ap.add_argument(
        "--percentile",
        type=float,
        default=30.0,
        help=(
            "Percentile used to define the suspicious tail. "
            "For mode=negative, this is the percentile within negative correlations only. "
            "For mode=lowtail, this is the percentile within all accepted correlations."
        ),
    )
    ap.add_argument(
        "--ignore-stations",
        type=str,
        default="",
        help="Comma-separated station IDs to ignore, e.g. 16005,16047,16050",
    )
    ap.add_argument("--distance-min", type=float, default=None)
    ap.add_argument("--distance-max", type=float, default=None)
    ap.add_argument("--top-n", type=int, default=30)
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    event_ts_dir = base_dir / "01_Event_TimeSeries"
    out_dir = base_dir / "01_Correlation_Plots"

    df_all = load_pair_file(args.event, event_ts_dir)

    ignored_stations = parse_station_list(args.ignore_stations)
    df = drop_ignored_stations(df_all, ignored_stations)

    suspicious, cutoff = select_suspicious_pairs(
        df=df,
        mode=str(args.mode),
        percentile=float(args.percentile),
        distance_min=args.distance_min,
        distance_max=args.distance_max,
    )

    station_freq = count_station_frequency(suspicious)

    print("=" * 80)
    print(f"Event: {args.event}")
    print(f"Mode: {args.mode}")
    print(f"Percentile: {args.percentile}")
    print(f"Ignored stations: {sorted(ignored_stations) if ignored_stations else 'None'}")
    print(f"Total accepted pairs before ignoring: {len(df_all)}")
    print(f"Total accepted pairs after ignoring: {len(df)}")
    print(f"Suspicious pairs selected: {len(suspicious)}")
    print(f"Cutoff correlation: {cutoff}")
    print("=" * 80)

    if suspicious.empty:
        print("No suspicious pairs found with the current settings.")
        return

    print("\nTop stations contributing to the suspicious branch:")
    print(station_freq.head(20).to_string(index=False))

    save_outputs(
        event=int(args.event),
        all_pairs=df,
        suspicious=suspicious,
        station_freq=station_freq,
        cutoff=cutoff,
        mode=str(args.mode),
        percentile=float(args.percentile),
        out_dir=out_dir,
        top_n=int(args.top_n),
    )


if __name__ == "__main__":
    main()