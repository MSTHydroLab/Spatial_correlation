#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

# ---------------- PATHS ----------------
OK_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/03_Interpolated_Rain")
C2_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/Radar_Event_TimeSeries/Composite_2")
C3_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/Radar_Event_TimeSeries/Composite_3")
OUT_BASE = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/04_Comparison")

OK_GLOB = "Event_*_grid_rain_hourly_mm.csv"
C2_FMT = "Event_{event}_grid_rain_hourly_mm_composite2.csv"
C3_FMT = "Event_{event}_grid_rain_hourly_mm_composite3.csv"


# ---------------- HELPERS ----------------
def event_num_from_name(path: Path) -> int | None:
    m = re.search(r"Event_(\d+)_grid_rain_hourly_mm\.csv$", path.name)
    return int(m.group(1)) if m else None


def normalize_grid_col(col) -> str:
    s = str(col).strip()
    if s.lower().startswith("unnamed"):
        return s
    try:
        # convert 123.0 -> 123
        f = float(s)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return s
    except Exception:
        return s


def load_rain_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if df.shape[1] < 2:
        raise ValueError(f"{path} has fewer than 2 columns")

    # first column = timestamp
    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    bad_time = df[time_col].isna().sum()
    if bad_time > 0:
        raise ValueError(f"{path} has {bad_time} unparseable timestamps in column {time_col}")

    df = df.set_index(time_col)

    # normalize column names to strings
    df.columns = [normalize_grid_col(c) for c in df.columns]

    # remove accidental unnamed columns
    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # convert all data to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # sort
    df = df.sort_index()
    df = df.sort_index(axis=1)

    return df


def compute_metrics(obs: np.ndarray, est: np.ndarray):
    mask = np.isfinite(obs) & np.isfinite(est)
    n = int(mask.sum())

    if n == 0:
        return {
            "n": 0,
            "bias": np.nan,
            "cc": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "mean_diff": np.nan,
        }

    o = obs[mask]
    e = est[mask]

    so = np.sum(o)
    bias = np.sum(e) / so if so != 0 else np.nan

    if n >= 2:
        cc = np.corrcoef(o, e)[0, 1]
    else:
        cc = np.nan

    rmse = np.sqrt(np.mean((e - o) ** 2))
    mae = np.mean(np.abs(e - o))
    mean_diff = np.mean(e - o)

    return {
        "n": n,
        "bias": bias,
        "cc": cc,
        "rmse": rmse,
        "mae": mae,
        "mean_diff": mean_diff,
    }


def compare_event(event: int):
    ok_file = OK_DIR / f"Event_{event}_grid_rain_hourly_mm.csv"
    c2_file = C2_DIR / C2_FMT.format(event=event)
    c3_file = C3_DIR / C3_FMT.format(event=event)
    event=event
    out_dir = OUT_BASE / f"Event_{event}"
    out_dir.mkdir(parents=True, exist_ok=True)

    status = {
        "event": event,
        "ok_file": str(ok_file),
        "c2_file": str(c2_file),
        "c3_file": str(c3_file),
        "ok_exists": ok_file.exists(),
        "c2_exists": c2_file.exists(),
        "c3_exists": c3_file.exists(),
        "ok_rows": np.nan,
        "c2_rows": np.nan,
        "c3_rows": np.nan,
        "ok_cols": np.nan,
        "c2_cols": np.nan,
        "c3_cols": np.nan,
        "common_time": np.nan,
        "common_cols": np.nan,
        "status": "",
        "note": "",
    }

    if not ok_file.exists() or not c2_file.exists() or not c3_file.exists():
        status["status"] = "missing_file"
        pd.DataFrame([status]).to_csv(out_dir / f"comparison_status_event{event}.csv", index=False)
        return status

    try:
        ok = load_rain_df(ok_file)
        c2 = load_rain_df(c2_file)
        c3 = load_rain_df(c3_file)
    except Exception as e:
        status["status"] = "load_error"
        status["note"] = str(e)
        pd.DataFrame([status]).to_csv(out_dir / f"comparison_status_event{event}.csv", index=False)
        return status

    status["ok_rows"], status["ok_cols"] = ok.shape
    status["c2_rows"], status["c2_cols"] = c2.shape
    status["c3_rows"], status["c3_cols"] = c3.shape

    common_time = ok.index.intersection(c2.index).intersection(c3.index)
    common_cols = ok.columns.intersection(c2.columns).intersection(c3.columns)

    status["common_time"] = len(common_time)
    status["common_cols"] = len(common_cols)

    # save diagnostic previews
    pd.DataFrame({"ok_time": ok.index.astype(str)}).head(10).to_csv(out_dir / f"debug_ok_times_head_event{event}.csv", index=False)
    pd.DataFrame({"c2_time": c2.index.astype(str)}).head(10).to_csv(out_dir / f"debug_c2_times_head_event{event}.csv", index=False)
    pd.DataFrame({"c3_time": c3.index.astype(str)}).head(10).to_csv(out_dir / f"debug_c3_times_head_event{event}.csv", index=False)

    pd.DataFrame({"ok_cols": list(ok.columns[:20])}).to_csv(out_dir / f"debug_ok_cols_head_event{event}.csv", index=False)
    pd.DataFrame({"c2_cols": list(c2.columns[:20])}).to_csv(out_dir / f"debug_c2_cols_head_event{event}.csv", index=False)
    pd.DataFrame({"c3_cols": list(c3.columns[:20])}).to_csv(out_dir / f"debug_c3_cols_head_event{event}.csv", index=False)

    if len(common_time) == 0:
        status["status"] = "no_common_time"
        status["note"] = "No overlapping timestamps across OK, Composite_2, Composite_3"
        pd.DataFrame([status]).to_csv(out_dir / f"comparison_status_event{event}.csv", index=False)
        return status

    if len(common_cols) == 0:
        status["status"] = "no_common_cols"
        status["note"] = "No overlapping grid columns across OK, Composite_2, Composite_3"
        pd.DataFrame([status]).to_csv(out_dir / f"comparison_status_event{event}.csv", index=False)
        return status

    ok = ok.loc[common_time, common_cols].copy()
    c2 = c2.loc[common_time, common_cols].copy()
    c3 = c3.loc[common_time, common_cols].copy()

    # ---------------- 1) cell-by-cell metrics ----------------
    rows = []
    for gid in common_cols:
        m2 = compute_metrics(ok[gid].to_numpy(dtype=float), c2[gid].to_numpy(dtype=float))
        m3 = compute_metrics(ok[gid].to_numpy(dtype=float), c3[gid].to_numpy(dtype=float))

        rows.append({
            "grid_id": gid,
            "n_c2": m2["n"],
            "bias_c2": m2["bias"],
            "cc_c2": m2["cc"],
            "rmse_c2": m2["rmse"],
            "mae_c2": m2["mae"],
            "mean_diff_c2": m2["mean_diff"],
            "n_c3": m3["n"],
            "bias_c3": m3["bias"],
            "cc_c3": m3["cc"],
            "rmse_c3": m3["rmse"],
            "mae_c3": m3["mae"],
            "mean_diff_c3": m3["mean_diff"],
        })

    df_cell = pd.DataFrame(rows)
    df_cell.to_csv(out_dir / f"cell_metrics_event{event}.csv", index=False)

    # ---------------- 2) domain average time series ----------------
    ok_mean = ok.mean(axis=1, skipna=True)
    c2_mean = c2.mean(axis=1, skipna=True)
    c3_mean = c3.mean(axis=1, skipna=True)

    m2_dom = compute_metrics(ok_mean.to_numpy(dtype=float), c2_mean.to_numpy(dtype=float))
    m3_dom = compute_metrics(ok_mean.to_numpy(dtype=float), c3_mean.to_numpy(dtype=float))

    df_domain_metrics = pd.DataFrame([
        {"comparison": "OK_vs_Composite2", **m2_dom},
        {"comparison": "OK_vs_Composite3", **m3_dom},
    ])
    df_domain_metrics.to_csv(out_dir / f"domain_metrics_event{event}.csv", index=False)

    df_domain_ts = pd.DataFrame({
        "timestamp": ok_mean.index,
        "ok_domain_mean_mm": ok_mean.values,
        "composite2_domain_mean_mm": c2_mean.values,
        "composite3_domain_mean_mm": c3_mean.values,
    })
    df_domain_ts.to_csv(out_dir / f"domain_mean_timeseries_event{event}.csv", index=False)

    # ---------------- 3) event total per cell ----------------
    ok_total = ok.sum(axis=0, skipna=True)
    c2_total = c2.sum(axis=0, skipna=True)
    c3_total = c3.sum(axis=0, skipna=True)

    df_total = pd.DataFrame({
        "grid_id": common_cols,
        "ok_total_mm": ok_total.values,
        "composite2_total_mm": c2_total.values,
        "composite3_total_mm": c3_total.values,
        "diff_c2_minus_ok_mm": c2_total.values - ok_total.values,
        "diff_c3_minus_ok_mm": c3_total.values - ok_total.values,
    })
    df_total.to_csv(out_dir / f"event_total_per_cell_event{event}.csv", index=False)

    # ---------------- 4) flattened overall comparison ----------------
    ok_flat = ok.to_numpy(dtype=float).ravel()
    c2_flat = c2.to_numpy(dtype=float).ravel()
    c3_flat = c3.to_numpy(dtype=float).ravel()

    m2_all = compute_metrics(ok_flat, c2_flat)
    m3_all = compute_metrics(ok_flat, c3_flat)

    df_overall = pd.DataFrame([
        {"comparison": "OK_vs_Composite2_all_gridtime_pairs", **m2_all},
        {"comparison": "OK_vs_Composite3_all_gridtime_pairs", **m3_all},
    ])
    df_overall.to_csv(out_dir / f"overall_gridtime_metrics_event{event}.csv", index=False)

    status["status"] = "ok"
    status["note"] = "Comparison completed"
    pd.DataFrame([status]).to_csv(out_dir / f"comparison_status_event{event}.csv", index=False)
    return status


def main():
    ok_files = sorted(OK_DIR.glob(OK_GLOB))
    events = [event_num_from_name(p) for p in ok_files]
    events = [e for e in events if e is not None]

    if not events:
        print("No OK event files found.")
        return

    all_status = []
    for event in events:
        print(f"Processing Event {event} ...")
        s = compare_event(event)
        all_status.append(s)
        print(f"  status = {s['status']}, common_time = {s['common_time']}, common_cols = {s['common_cols']}")

    df_status = pd.DataFrame(all_status)
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    df_status.to_csv(OUT_BASE / f"comparison_summary_all_events_event{event}.csv", index=False)

    print("\nDone.")
    print(f"Summary written to: {OUT_BASE / f'comparison_summary_all_events_event{event}.csv'}")


if __name__ == "__main__":
    main()