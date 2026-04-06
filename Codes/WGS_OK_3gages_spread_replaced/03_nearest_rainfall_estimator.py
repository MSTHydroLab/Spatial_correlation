#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK_3gages_spread_replaced")
DEP_DIR = BASE_DIR / "01_Event_TimeSeries"
WEIGHTS_DIR = BASE_DIR / "02_OK_Weights"
OUT_DIR = BASE_DIR / "03_Interpolated_Rain"
RAIN_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/Compiled_rain/compiled_rawstyle_hourly_mm/per_station_hourly/")

FILE_SUFFIX = ".hourly_mm.csv"
TIME_LOCAL_COL = "time_local"
TIME_UTC_COL = "time_utc"
RAIN_COL = "rain_mm"
LOCAL_TZ = "America/Chicago"


def make_window(start_str: str, end_str: str):
    start = pd.to_datetime(start_str, errors="raise")
    end = pd.to_datetime(end_str, errors="raise")
    if start.tzinfo is not None:
        start = start.tz_convert(LOCAL_TZ).tz_localize(None)
    if end.tzinfo is not None:
        end = end.tz_convert(LOCAL_TZ).tz_localize(None)
    if end < start:
        raise ValueError("event_end must be >= event_start")
    return start, end, pd.date_range(start, end, freq="1h")


def load_event_window(event_number: int, event_meta_dir: Path):
    fp = event_meta_dir / f"Event_{event_number}_Stations_correlation.csv"
    meta = pd.read_csv(fp)
    start_str = str(meta["event_start"].dropna().iloc[0]).strip()
    end_str = str(meta["event_end"].dropna().iloc[0]).strip()
    return make_window(start_str, end_str)


def load_station_series_local(station_id: str, start: pd.Timestamp, end: pd.Timestamp, rain_dir: Path):
    fp = rain_dir / f"{station_id}{FILE_SUFFIX}"
    stats = {
        "station_id": str(station_id),
        "file_exists": fp.exists(),
        "rows_read": 0,
        "n_duplicates_collapsed": 0,
        "n_bad_time_rows_dropped": 0,
    }
    if not fp.exists():
        return pd.Series(dtype=float), stats

    df = pd.read_csv(fp, usecols=[TIME_LOCAL_COL, TIME_UTC_COL, RAIN_COL])
    stats["rows_read"] = len(df)
    t_utc = pd.to_datetime(df[TIME_UTC_COL], utc=True, errors="coerce")
    off = df[TIME_LOCAL_COL].astype(str).str.extract(r"([+-]\d{2})\d{2}$")[0]
    off_hours = pd.to_numeric(off, errors="coerce")
    t_local = (t_utc + pd.to_timedelta(off_hours, unit="h")).dt.tz_localize(None)

    rain = pd.to_numeric(df[RAIN_COL], errors="coerce").to_numpy()
    s = pd.Series(rain, index=t_local, name=str(station_id))
    n0 = len(s)
    s = s[~s.index.isna()]
    stats["n_bad_time_rows_dropped"] = int(n0 - len(s))
    n0 = len(s)
    s = s.groupby(level=0).mean()
    stats["n_duplicates_collapsed"] = int(n0 - len(s))
    s = s.sort_index()
    s = s[(s.index >= start) & (s.index <= end)]
    s = s.groupby(s.index.floor("h")).mean()
    return s, stats


def load_weights(event_number: int, n_gauges: int, weights_dir: Path):
    fp = weights_dir / f"Event_{event_number}_nearest{n_gauges}_weights.csv"
    W = pd.read_csv(fp)
    W["id"] = W["id"].astype(str)
    for k in range(1, 5):
        if f"g{k}" in W.columns:
            W[f"g{k}"] = W[f"g{k}"].fillna("").astype(str)
            W.loc[W[f"g{k}"].str.lower() == "nan", f"g{k}"] = ""
        if f"w{k}" in W.columns:
            W[f"w{k}"] = pd.to_numeric(W[f"w{k}"], errors="coerce").fillna(0.0)
    return W


def compute_grid_rain(weights_df: pd.DataFrame, rain_df: pd.DataFrame) -> pd.DataFrame:
    station_cols = {str(c): i for i, c in enumerate(rain_df.columns)}
    Rmat = rain_df.to_numpy(dtype=float)
    grid_ids = weights_df["id"].to_numpy()
    out = np.zeros((rain_df.shape[0], len(grid_ids)), dtype=float)

    for k in range(1, 5):
        gcol = f"g{k}"
        wcol = f"w{k}"
        if gcol not in weights_df.columns or wcol not in weights_df.columns:
            continue
        gids = weights_df[gcol].to_numpy()
        w = weights_df[wcol].to_numpy(dtype=float)
        idx = np.array([station_cols.get(g, -1) if g != "" else -1 for g in gids], dtype=int)
        term = np.zeros_like(out)
        valid = idx >= 0
        if np.any(valid):
            term[:, valid] = Rmat[:, idx[valid]]
        out += term * w[None, :]

    return pd.DataFrame(out, index=rain_df.index, columns=grid_ids)


def main():
    parser = argparse.ArgumentParser(description="Apply nearest-gauge OK weights to station rainfall time series.")
    parser.add_argument("--event", type=int, required=True)
    parser.add_argument("--n-gauges", type=int, choices=[3, 4], default=4)
    parser.add_argument("--event-meta-dir", default=str(DEP_DIR))
    parser.add_argument("--weights-dir", default=str(WEIGHTS_DIR))
    parser.add_argument("--rain-dir", default=str(RAIN_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    event_meta_dir = Path(args.event_meta_dir)
    weights_dir = Path(args.weights_dir)
    rain_dir = Path(args.rain_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start, end, event_idx = load_event_window(args.event, event_meta_dir)
    W = load_weights(args.event, args.n_gauges, weights_dir)

    gcols = [c for c in ["g1", "g2", "g3", "g4"] if c in W.columns]
    gauges = pd.unique(pd.concat([W[c] for c in gcols], axis=0))
    gauges = sorted(set(g for g in gauges if isinstance(g, str) and g != ""))

    series = []
    stats_rows = []
    for gid in gauges:
        s, st = load_station_series_local(gid, start, end, rain_dir)
        stats_rows.append(st)
        aligned = s.reindex(event_idx) if not s.empty else pd.Series(index=event_idx, data=np.nan, name=gid)
        aligned.name = gid
        series.append(aligned)

    R = pd.concat(series, axis=1) if series else pd.DataFrame(index=event_idx)
    missing_counts = R.isna().sum(axis=0).astype(int) if not R.empty else pd.Series(dtype=int)
    R_filled = R.fillna(0.0)

    grid_rain = compute_grid_rain(W, R_filled)

    out_rain = out_dir / f"Event_{args.event}_grid_rain_hourly_mm_nearest{args.n_gauges}.csv"
    out_df = grid_rain.copy()
    out_df.insert(0, "time_local", out_df.index.astype(str))
    out_df.to_csv(out_rain, index=False)

    out_station = out_dir / f"Event_{args.event}_station_rain_used_hourly_mm_nearest{args.n_gauges}.csv"
    out_station_df = R.copy()
    out_station_df.insert(0, "time_local", out_station_df.index.astype(str))
    out_station_df.to_csv(out_station, index=False)

    miss_rows = []
    for st in stats_rows:
        gid = str(st["station_id"])
        miss_rows.append({
            **st,
            "event": int(args.event),
            "n_missing_hours_in_event": int(missing_counts.get(gid, 0)),
        })
    pd.DataFrame(miss_rows).to_csv(
        out_dir / f"Event_{args.event}_missing_report_nearest{args.n_gauges}.csv", index=False
    )

    print(f"Saved: {out_rain}")
    print(f"Saved: {out_station}")


if __name__ == "__main__":
    main()
