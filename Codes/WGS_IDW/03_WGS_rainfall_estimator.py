#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_IDW")
DEP_DIR = BASE_DIR / "01_Event_TimeSeries"
WEIGHTS_DIR = BASE_DIR / "02_IDW_Weights"
OUT_DIR = BASE_DIR / "03_Interpolated_Rain"
RAIN_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/Compiled_rain/compiled_rawstyle_hourly_mm/per_station_hourly/")

FILE_SUFFIX = ".hourly_mm.csv"
TIME_LOCAL_COL = "time_local"
TIME_UTC_COL = "time_utc"
RAIN_COL = "rain_mm"
N_GAUGES = 4
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



def load_weights(event_number: int, weights_dir: Path):
    fp = weights_dir / f"Event_{event_number}_weights.csv"
    W = pd.read_csv(fp)
    req = ["id", "Latitude", "Longitude"]
    for k in range(1, N_GAUGES + 1):
        req += [f"g{k}", f"w{k}"]
    missing = [c for c in req if c not in W.columns]
    if missing:
        raise ValueError(f"{fp} missing required columns: {missing}")
    W = W[req].copy()
    W["id"] = W["id"].astype(str)

    def clean_gid(x):
        if pd.isna(x):
            return ""
        s = str(x).strip()
        try:
            f = float(s)
            if np.isfinite(f) and abs(f - int(f)) < 1e-9:
                return str(int(f))
        except Exception:
            pass
        return s

    for k in range(1, N_GAUGES + 1):
        W[f"g{k}"] = W[f"g{k}"].apply(clean_gid)
        W[f"w{k}"] = pd.to_numeric(W[f"w{k}"], errors="coerce").fillna(0.0)
    return W



def compute_grid_rain(weights_df: pd.DataFrame, rain_df: pd.DataFrame) -> pd.DataFrame:
    station_cols = {str(c): i for i, c in enumerate(rain_df.columns)}
    Rmat = rain_df.to_numpy(dtype=float)
    grid_ids = weights_df["id"].to_numpy()

    n_time = Rmat.shape[0]
    n_grid = len(grid_ids)

    out = np.zeros((n_time, n_grid), dtype=float)

    for j in range(n_grid):
        vals = np.zeros(n_time, dtype=float)
        weights = []
        data = []

        for k in range(1, N_GAUGES + 1):
            gid = weights_df.iloc[j][f"g{k}"]
            w = weights_df.iloc[j][f"w{k}"]

            if gid == "" or gid not in station_cols:
                continue

            col_idx = station_cols[gid]
            r = Rmat[:, col_idx]

            weights.append(w)
            data.append(r)

        if len(data) == 0:
            continue

        weights = np.array(weights, dtype=float)
        data = np.array(data)  # shape: (n_gauges, n_time)

        # --- key change ---
        for t in range(n_time):
            r_t = data[:, t]
            w_t = weights.copy()

            valid = ~np.isnan(r_t)

            if np.sum(valid) == 0:
                vals[t] = 0.0
                continue

            r_valid = r_t[valid]
            w_valid = w_t[valid]

            # renormalize weights
            w_valid = w_valid / np.sum(w_valid)

            vals[t] = np.sum(r_valid * w_valid)

        out[:, j] = vals

    return pd.DataFrame(out, index=rain_df.index, columns=grid_ids)



def main():
    parser = argparse.ArgumentParser(description="Apply WGS84 kriging weights to station rainfall time series.")
    parser.add_argument("--event", type=int, nargs="+", required=True)
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

    for ev in args.event:

        print(f"\nProcessing Event {ev}")

        start, end, event_idx = load_event_window(ev, event_meta_dir)
        W = load_weights(ev, weights_dir)

        gcols = [f"g{k}" for k in range(1, N_GAUGES + 1)]
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

        # ---------------- outputs ----------------
        out_rain = out_dir / f"Event_{ev}_grid_rain_hourly_mm.csv"
        out_df = grid_rain.copy()
        out_df.insert(0, "time_local", out_df.index.astype(str))
        out_df.to_csv(out_rain, index=False)

        grid_meta = W[["id", "Latitude", "Longitude"]].drop_duplicates().copy()
        grid_meta_out = out_dir / f"Event_{ev}_grid_metadata.csv"
        grid_meta.to_csv(grid_meta_out, index=False)

        stats_df = pd.DataFrame(stats_rows)
        if not stats_df.empty:
            stats_df["n_missing_after_reindex"] = stats_df["station_id"].map(lambda x: int(missing_counts.get(x, 0)))
            stats_df["n_filled_as_zero"] = stats_df["n_missing_after_reindex"]

        summary = pd.DataFrame([{
            "event": ev,
            "event_start_local": start.strftime("%Y-%m-%d %H:%M:%S"),
            "event_end_local": end.strftime("%Y-%m-%d %H:%M:%S"),
            "n_event_hours": len(event_idx),
            "n_grids": len(W),
            "n_gauges_used": len(gauges),
            "n_gauges_per_grid": N_GAUGES,
        }])

        out_missing = out_dir / f"Event_{ev}_missing_report.csv"
        with open(out_missing, "w", newline="") as f:
            summary.to_csv(f, index=False)
            f.write("\n")
            if not stats_df.empty:
                stats_df.sort_values(["file_exists", "station_id"], ascending=[False, True]).to_csv(f, index=False)

        print(f"Saved rainfall: {out_rain}")


if __name__ == "__main__":
    main()
