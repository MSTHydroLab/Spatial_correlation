from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import re
# ---------------- Paths / columns ----------------
WEIGHTS_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/02_OK_Weights")
EVENT_META_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/01_Correlation_and_Variogram")
RAIN_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/Compiled_rain/compiled_rawstyle_hourly_mm/per_station_hourly/")
OUT_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/03_Interpolated_Rain")

FILE_SUFFIX = ".hourly_mm.csv"
TIME_LOCAL_COL = "time_local"
TIME_UTC_COL   = "time_utc"
RAIN_COL       = "rain_mm"



# ---------------------------------------------------------
# 1) Window defined by explicit START and END (local clock)
#    Inputs: "yyyymmddHH"
#    event_end is inclusive
# ---------------------------------------------------------
def make_window(start_str: str, end_str: str):
    start = pd.to_datetime(start_str, format="%Y%m%d%H")
    end   = pd.to_datetime(end_str,   format="%Y%m%d%H")
    if end < start:
        raise ValueError("end_str must be >= start_str")
    idx = pd.date_range(start, end, freq="1h")
    return start, end, idx

def detect_gauge_pairs(columns):
    """
    Detect available (gk, wk) pairs from weights file columns.
    Returns a sorted list like:
    [("g1", "w1"), ("g2", "w2"), ("g3", "w3")]
    """
    cols = set(columns)
    pairs = []

    gauge_nums = []
    for c in cols:
        m = re.fullmatch(r"g(\d+)", str(c))
        if m:
            gauge_nums.append(int(m.group(1)))

    for k in sorted(gauge_nums):
        gcol = f"g{k}"
        wcol = f"w{k}"
        if wcol in cols:
            pairs.append((gcol, wcol))

    if not pairs:
        raise ValueError("No gauge/weight column pairs like g1/w1, g2/w2 found.")

    return pairs

def load_event_window(event_number: int):
    fp = EVENT_META_DIR / f"Event_{event_number}_Stations_correlation.csv"
    meta = pd.read_csv(fp)

    if "event_start" not in meta.columns or "event_end" not in meta.columns:
        raise ValueError(f"{fp} must contain event_start and event_end columns")

    start_str = str(meta["event_start"].dropna().iloc[0]).strip()
    end_str   = str(meta["event_end"].dropna().iloc[0]).strip()
    return make_window(start_str, end_str)


# ---------------------------------------------------------
# 2) Load station rainfall indexed by naive LOCAL time
#    using your logic: time_utc + offset parsed from time_local
# ---------------------------------------------------------
def load_station_series_local(station_id: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, dict]:
    fp = RAIN_DIR / f"{station_id}{FILE_SUFFIX}"

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

    # Parse UTC (always tz-aware UTC)
    t_utc = pd.to_datetime(df[TIME_UTC_COL], utc=True, errors="coerce")

    # Extract offset like -0600 -> -06 hours
    off = df[TIME_LOCAL_COL].astype(str).str.extract(r"([+-]\d{2})\d{2}$")[0]
    off_hours = pd.to_numeric(off, errors="coerce")

    # UTC -> local naive timestamp
    t_local = (t_utc + pd.to_timedelta(off_hours, unit="h")).dt.tz_localize(None)

    rain = pd.to_numeric(df[RAIN_COL], errors="coerce").to_numpy()
    s = pd.Series(rain, index=t_local)
    s.name = str(station_id)

    # Drop NaT timestamps
    n0 = len(s)
    s = s[~s.index.isna()]
    stats["n_bad_time_rows_dropped"] = int(n0 - len(s))

    # Collapse duplicates (DST fall-back)
    n0 = len(s)
    s = s.groupby(level=0).mean()
    stats["n_duplicates_collapsed"] = int(n0 - len(s))

    # Robust subset
    s = s.sort_index()
    s = s[(s.index >= start) & (s.index <= end)]

    # Force hourly clock
    s = s.groupby(s.index.floor("h")).mean()

    return s, stats


# ---------------------------------------------------------
# 3) Load weights: id, g1..g4, w1..w4
# ---------------------------------------------------------
def load_weights(event_number: int) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    fp = WEIGHTS_DIR / f"Event_{event_number}_weights.csv"
    W = pd.read_csv(fp)

    if "id" not in W.columns:
        raise ValueError(f"{fp} must contain column 'id'")

    gauge_pairs = detect_gauge_pairs(W.columns)

    req = ["id"]
    for gcol, wcol in gauge_pairs:
        req += [gcol, wcol]

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

    for gcol, wcol in gauge_pairs:
        W[gcol] = W[gcol].apply(clean_gid)
        W[wcol] = pd.to_numeric(W[wcol], errors="coerce").fillna(0.0)

    return W, gauge_pairs


# ---------------------------------------------------------
# 4) Compute rainfall timeseries for all grid ids
# ---------------------------------------------------------
def compute_grid_rain(W: pd.DataFrame, R: pd.DataFrame, gauge_pairs: list[tuple[str, str]]) -> pd.DataFrame:
    station_cols = {str(c): i for i, c in enumerate(R.columns)}
    Rmat = R.to_numpy(dtype=float)

    grid_ids = W["id"].to_numpy()
    T = R.shape[0]
    ngrids = len(grid_ids)

    out = np.zeros((T, ngrids), dtype=float)

    for gcol, wcol in gauge_pairs:
        gids = W[gcol].to_numpy()
        w = W[wcol].to_numpy(dtype=float)

        idx = np.array([station_cols.get(g, -1) if g != "" else -1 for g in gids], dtype=int)

        term = np.zeros((T, ngrids), dtype=float)
        valid = idx >= 0
        if np.any(valid):
            term[:, valid] = Rmat[:, idx[valid]]

        out += term * w[None, :]

    return pd.DataFrame(out, index=R.index, columns=grid_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", type=int, required=True)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    start, end, event_idx = load_event_window(args.event)
    W, gauge_pairs = load_weights(args.event)

    # All unique gauges referenced in weights
    gcols = [gcol for gcol, _ in gauge_pairs]
    gauges = pd.unique(pd.concat([W[gcol] for gcol in gcols], axis=0))
    gauges = [g for g in gauges if isinstance(g, str) and g != ""]
    gauges = sorted(set(gauges))

    # Load station series, align to event index
    series = []
    stats_rows = []

    for gid in gauges:
        s, st = load_station_series_local(gid, start, end)
        stats_rows.append(st)

        if s.empty:
            aligned = pd.Series(index=event_idx, data=np.nan, name=gid)
        else:
            aligned = s.reindex(event_idx)
            aligned.name = gid

        series.append(aligned)

    R = pd.concat(series, axis=1)

    missing_counts = R.isna().sum(axis=0).astype(int)

    # Treat missing as 0
    R_filled = R.fillna(0.0)
    # Export rainfall used from all unique stations in this event
    out_station_rain = OUT_DIR / f"Event_{args.event}_station_rain_used_hourly_mm.csv"
    station_df = R_filled.copy()
    station_df.insert(0, "time_local", station_df.index.astype(str))
    station_df.to_csv(out_station_rain, index=False)

    # Interpolate
    grid_rain = compute_grid_rain(W, R_filled, gauge_pairs)

    # Export rainfall file
    out_rain = OUT_DIR / f"Event_{args.event}_grid_rain_hourly_mm.csv"
    out_df = grid_rain.copy()
    out_df.insert(0, "time_local", out_df.index.astype(str))
    out_df.to_csv(out_rain, index=False)

    # Export missing report
    stats_df = pd.DataFrame(stats_rows)
    stats_df["n_missing_after_reindex"] = stats_df["station_id"].map(lambda x: int(missing_counts.get(x, 0)))
    stats_df["n_filled_as_zero"] = stats_df["n_missing_after_reindex"]

    summary = pd.DataFrame([{
        "event": args.event,
        "event_start_local": start.strftime("%Y-%m-%d %H:%M:%S"),
        "event_end_local": end.strftime("%Y-%m-%d %H:%M:%S"),
        "n_event_hours": len(event_idx),
        "n_grids": len(W),
        "n_gauges_used": len(gauges),
        "n_gauges_per_grid": len(gauge_pairs),
    }])

    out_missing = OUT_DIR / f"Event_{args.event}_missing_report.csv"
    with open(out_missing, "w", newline="") as f:
        summary.to_csv(f, index=False)
        f.write("\n")
        stats_df.sort_values(["file_exists", "station_id"], ascending=[False, True]).to_csv(f, index=False)

    print(f"Wrote: {out_rain}")
    print(f"Wrote: {out_missing}")
    print(f"Wrote: {out_station_rain}")


if __name__ == "__main__":
    main()