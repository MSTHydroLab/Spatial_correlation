#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


BASE_OK = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK")
BASE_IDW = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_IDW")

PRODUCT_PATHS = {
    "OK": BASE_OK / "03_Interpolated_Rain/Event_{event}_grid_rain_hourly_mm.csv",
    "RZ": BASE_OK / "Radar_Event_TimeSeries/RZ/Event_{event}/Event_{event}_grid_rain_hourly_mm_RZ.csv",
    "RKDP": BASE_OK / "Radar_Event_TimeSeries/RKDP/Event_{event}/Event_{event}_grid_rain_hourly_mm_RKDP.csv",
    "RA": BASE_OK / "Radar_Event_TimeSeries/RA/Event_{event}/Event_{event}_grid_rain_hourly_mm_RA.csv",
    "Composite_2": BASE_OK / "Radar_Event_TimeSeries/Composite_2/Event_{event}/Event_{event}_grid_rain_hourly_mm_Composite_2.csv",
}

REFERENCE_PATH = BASE_IDW / "03_Interpolated_Rain/Event_{event}_grid_rain_hourly_mm.csv"
OUT_DIR = BASE_OK / "11_Compare_Against_IDW"


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


def compute_metrics(ref: np.ndarray, sim: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(ref) & np.isfinite(sim)
    n = int(mask.sum())

    if n == 0:
        return {k: np.nan for k in [
            "ref_sum_mm", "sim_sum_mm", "sum_diff_mm", "sum_diff_pct",
            "bias_ratio", "mean_error_mm", "mae_mm", "mse_mm2",
            "rmse_mm", "correlation"
        ]} | {"n": 0}

    r = ref[mask]
    s = sim[mask]
    err = s - r

    ref_sum = float(np.sum(r))
    sim_sum = float(np.sum(s))

    return {
        "n": n,
        "ref_sum_mm": ref_sum,
        "sim_sum_mm": sim_sum,
        "sum_diff_mm": sim_sum - ref_sum,
        "sum_diff_pct": 100.0 * (sim_sum - ref_sum) / ref_sum if ref_sum != 0 else np.nan,
        "bias_ratio": sim_sum / ref_sum if ref_sum != 0 else np.nan,
        "mean_error_mm": float(np.mean(err)),
        "mae_mm": float(np.mean(np.abs(err))),
        "mse_mm2": float(np.mean(err ** 2)),
        "rmse_mm": float(np.sqrt(np.mean(err ** 2))),
        "correlation": float(np.corrcoef(r, s)[0, 1]) if n >= 2 else np.nan,
    }


def align_to_reference(ref: pd.DataFrame, sim: pd.DataFrame):
    common_time = ref.index.intersection(sim.index)
    common_cols = ref.columns.intersection(sim.columns)

    if len(common_time) == 0:
        raise ValueError("No common timestamps")
    if len(common_cols) == 0:
        raise ValueError("No common grid cells")

    return ref.loc[common_time, common_cols], sim.loc[common_time, common_cols]


def compare_event(event: int, out_dir: Path):
    ref_path = Path(str(REFERENCE_PATH).format(event=event))
    ref = load_grid_rain_csv(ref_path)

    event_rows = []
    cell_rows = []

    for product, template in PRODUCT_PATHS.items():
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

    event_df.to_csv(out_dir / f"Event_{event}_metrics_against_IDW.csv", index=False)
    cell_df.to_csv(out_dir / f"Event_{event}_cell_metrics_against_IDW.csv", index=False)

    return event_df, cell_df


def main():
    parser = argparse.ArgumentParser(
        description="Compare OK and radar rainfall grids against IDW reference."
    )
    parser.add_argument(
        "--event",
        type=int,
        nargs="+",
        required=True,
        help="One or more events, e.g. --event 1 or --event 4 7",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_event_rows = []
    all_cell_rows = []

    for event in args.event:
        print(f"Comparing Event {event} against IDW")
        event_df, cell_df = compare_event(event, args.out_dir)
        all_event_rows.append(event_df)
        all_cell_rows.append(cell_df)

    pd.concat(all_event_rows, ignore_index=True).to_csv(
        args.out_dir / "All_events_metrics_against_IDW.csv", index=False
    )

    pd.concat(all_cell_rows, ignore_index=True).to_csv(
        args.out_dir / "All_events_cell_metrics_against_IDW.csv", index=False
    )


if __name__ == "__main__":
    main()