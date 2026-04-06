#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK")
DEP_DIR = BASE_DIR / "01_Event_TimeSeries"
WEIGHTS_DIR = BASE_DIR / "02_OK_Weights"
OUT_DIR = BASE_DIR / "03_Interpolated_Rain"
RAIN_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/Compiled_rain/compiled_rawstyle_hourly_mm/per_station_hourly/")

FILE_SUFFIX = ".hourly_mm.csv"
TIME_LOCAL_COL = "time_local"
TIME_UTC_COL = "time_utc"
RAIN_COL = "rain_mm"
N_GAUGES = 4
LOCAL_TZ = "America/Chicago"


# -----------------------------
# Event and rainfall loading
# -----------------------------

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


# -----------------------------
# Primary rainfall and weights
# -----------------------------

def load_primary_rainfall(event_number: int, out_dir: Path, event_idx: pd.DatetimeIndex) -> pd.DataFrame:
    fp = out_dir / f"Event_{event_number}_grid_rain_hourly_mm.csv"
    if not fp.exists():
        raise FileNotFoundError(
            f"Primary rainfall file not found: {fp}. Run 03_WGS_rainfall_estimator.py first."
        )

    df = pd.read_csv(fp)
    if "time_local" not in df.columns:
        raise ValueError(f"{fp} missing required column: time_local")

    df["time_local"] = pd.to_datetime(df["time_local"], errors="coerce")
    bad = int(df["time_local"].isna().sum())
    if bad > 0:
        raise ValueError(f"{fp} has {bad} bad time_local rows")

    df = df.set_index("time_local")
    df.columns = [str(c) for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.reindex(event_idx)
    df.index = event_idx
    return df


def load_primary_weights(event_number: int, weights_dir: Path):
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


def load_weight_options(event_number: int, weights_dir: Path, max_options: int | None = None):
    fp = weights_dir / f"Event_{event_number}_weight_options.csv"
    W = pd.read_csv(fp)
    req = ["id", "Latitude", "Longitude", "option_rank", "ranking_rule", "weight_method", "remarks"]
    for k in range(1, N_GAUGES + 1):
        req += [f"g{k}", f"w{k}"]
    missing = [c for c in req if c not in W.columns]
    if missing:
        raise ValueError(f"{fp} missing required columns: {missing}")
    W = W[req].copy()
    W["id"] = W["id"].astype(str)
    W["option_rank"] = pd.to_numeric(W["option_rank"], errors="coerce").astype(int)

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

    W = W.sort_values(["id", "option_rank"]).reset_index(drop=True)
    if max_options is not None:
        W = W[W["option_rank"] <= int(max_options)].copy()
    return W


def collect_all_gauges(options_df: pd.DataFrame) -> list[str]:
    gauges = set()
    for k in range(1, N_GAUGES + 1):
        gauges.update(str(x) for x in options_df[f"g{k}"].tolist() if str(x).strip() != "")
    return sorted(gauges)


def compute_candidate_value(row: pd.Series, rain_row: pd.Series) -> float:
    vals = []
    weights = []

    for k in range(1, N_GAUGES + 1):
        gid = str(row[f"g{k}"]).strip()
        if gid == "":
            continue

        w = float(row[f"w{k}"])
        r = rain_row.get(gid, np.nan)

        if pd.isna(r):
            continue

        vals.append(float(r))
        weights.append(w)

    if len(vals) == 0:
        return 0.0  # or np.nan if you prefer strict handling

    vals = np.array(vals, dtype=float)
    weights = np.array(weights, dtype=float)

    # renormalize weights
    weights = weights / np.sum(weights)

    return float(np.sum(vals * weights))


# -----------------------------
# Grid neighbors
# -----------------------------

def build_grid_neighbors(grid_df: pd.DataFrame, decimals: int = 6) -> dict[str, list[str]]:
    grid = grid_df[["id", "Latitude", "Longitude"]].copy()
    grid["id"] = grid["id"].astype(str)
    grid["lat_r"] = grid["Latitude"].round(decimals)
    grid["lon_r"] = grid["Longitude"].round(decimals)

    unique_lats = np.sort(grid["lat_r"].unique())
    unique_lons = np.sort(grid["lon_r"].unique())
    lat_to_r = {v: i for i, v in enumerate(unique_lats)}
    lon_to_c = {v: i for i, v in enumerate(unique_lons)}

    grid["r"] = grid["lat_r"].map(lat_to_r)
    grid["c"] = grid["lon_r"].map(lon_to_c)

    rc_to_id = {(int(r), int(c)): str(i) for i, r, c in zip(grid["id"], grid["r"], grid["c"])}
    id_to_rc = {str(i): (int(r), int(c)) for i, r, c in zip(grid["id"], grid["r"], grid["c"])}

    out = {}
    for cid in grid["id"]:
        r, c = id_to_rc[str(cid)]
        ids = []
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                nid = rc_to_id.get((rr, cc))
                if nid is not None:
                    ids.append(nid)
        out[str(cid)] = ids
    return out


# -----------------------------
# Screening
# -----------------------------

def compute_stat(values: np.ndarray, mode: str) -> float:
    if values.size == 0:
        return np.nan
    if mode == "mean":
        return float(np.mean(values))
    return float(np.median(values))


def compute_scale(values: np.ndarray, stat_value: float) -> float:
    if values.size == 0:
        return np.nan
    return float(np.median(np.abs(values - stat_value)))


def run_screen(primary_grid: pd.DataFrame,
               options_df: pd.DataFrame,
               rain_df: pd.DataFrame,
               neighbors: dict[str, list[str]],
               min_valid_in_window: int,
               abs_threshold_mm: float,
               require_improvement: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    adjusted = primary_grid.copy()
    cid_list = [str(c) for c in adjusted.columns]
    switch_rows = []

    option_lookup = {}
    for _, row in options_df.iterrows():
        option_lookup.setdefault(str(row["id"]), []).append(row)
    for cid in option_lookup:
        option_lookup[cid] = sorted(option_lookup[cid], key=lambda r: int(r["option_rank"]))

    for t_idx, ts in enumerate(adjusted.index):
        base_snapshot = adjusted.iloc[t_idx].copy()
        rain_row = rain_df.iloc[t_idx] if not rain_df.empty else pd.Series(dtype=float)

        for cid in cid_list:
            window_ids = neighbors.get(cid, [])
            if len(window_ids) < 2:
                continue

            window_vals_all = base_snapshot.reindex(window_ids).to_numpy(dtype=float)
            valid_mask_all = np.isfinite(window_vals_all)
            if int(np.sum(valid_mask_all)) < int(min_valid_in_window):
                continue

            neighbor_ids = [x for x in window_ids if x != cid]
            neighbor_vals = base_snapshot.reindex(neighbor_ids).to_numpy(dtype=float)
            neighbor_vals = neighbor_vals[np.isfinite(neighbor_vals)]
            if neighbor_vals.size == 0:
                continue

            local_min = float(np.min(neighbor_vals))
            local_max = float(np.max(neighbor_vals))
            lower_bound = local_min - float(abs_threshold_mm)
            upper_bound = local_max + float(abs_threshold_mm)

            current_val = float(base_snapshot.loc[cid])

            # only screen if outside the allowed neighbor range
            if lower_bound <= current_val <= upper_bound:
                continue

            # how far outside the allowed range is the current value?
            if current_val < lower_bound:
                current_excess = lower_bound - current_val
            else:
                current_excess = current_val - upper_bound

            candidates = option_lookup.get(cid, [])
            if len(candidates) <= 1:
                continue

            best_val = current_val
            best_row = None
            best_excess = current_excess
            best_passed = False

            for opt_row in candidates[1:]:
                cand_val = compute_candidate_value(opt_row, rain_row)

                if lower_bound <= cand_val <= upper_bound:
                    cand_excess = 0.0
                    cand_passed = True
                elif cand_val < lower_bound:
                    cand_excess = lower_bound - cand_val
                    cand_passed = False
                else:
                    cand_excess = cand_val - upper_bound
                    cand_passed = False

                if cand_passed:
                    if (not require_improvement) or (cand_excess < best_excess):
                        best_val = cand_val
                        best_row = opt_row
                        best_excess = cand_excess
                        best_passed = True
                        break
                elif (not best_passed) and (cand_excess < best_excess):
                    best_val = cand_val
                    best_row = opt_row
                    best_excess = cand_excess

            if best_row is not None and ((not require_improvement) or (best_excess < current_excess)):
                adjusted.iloc[t_idx, adjusted.columns.get_loc(cid)] = best_val
                switch_rows.append({
                    "time_local": str(ts),
                    "id": cid,
                    "primary_value_mm": current_val,
                    "replacement_value_mm": best_val,
                    "neighbor_min_mm": local_min,
                    "neighbor_max_mm": local_max,
                    "allowed_lower_mm": lower_bound,
                    "allowed_upper_mm": upper_bound,
                    "threshold_mm": float(abs_threshold_mm),
                    "primary_excess_mm": current_excess,
                    "replacement_excess_mm": best_excess,
                    "replacement_passed_rule": bool(best_passed),
                    "chosen_option_rank": int(best_row["option_rank"]),
                    "chosen_ranking_rule": str(best_row["ranking_rule"]),
                    "chosen_weight_method": str(best_row["weight_method"]),
                    "chosen_remarks": str(best_row["remarks"]),
                })
                adjusted.iloc[t_idx, adjusted.columns.get_loc(cid)] = best_val

                print(
                    f"[SWITCH] time={ts} | cell={cid} | "
                    f"{current_val:.2f} → {best_val:.2f} mm | "
                    f"range=[{local_min:.2f}, {local_max:.2f}] | "
                    f"allowed=[{lower_bound:.2f}, {upper_bound:.2f}] | "
                    f"primary_excess={current_excess:.2f} | new_excess={best_excess:.2f} | "
                    f"option={int(best_row['option_rank'])} ({best_row['weight_method']})"
                )

    switch_df = pd.DataFrame(switch_rows)
    if switch_df.empty:
        summary_df = pd.DataFrame(columns=["id", "n_switches", "first_switch_time", "last_switch_time"])
    else:
        summary_df = (
            switch_df.groupby("id", as_index=False)
            .agg(
                n_switches=("id", "size"),
                first_switch_time=("time_local", "min"),
                last_switch_time=("time_local", "max"),
            )
        )

    return adjusted, switch_df, summary_df

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Apply fallback weight options only where the already-built primary interpolated rainfall looks locally inconsistent.")
    parser.add_argument("--event", type=int, required=True)
    parser.add_argument("--event-meta-dir", default=str(DEP_DIR))
    parser.add_argument("--weights-dir", default=str(WEIGHTS_DIR))
    parser.add_argument("--rain-dir", default=str(RAIN_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--max-options", type=int, default=5)
    parser.add_argument("--min-valid-in-window", type=int, default=8)
    parser.add_argument("--abs-threshold-mm", type=float, default=5.0,
                    help="Allowed amount outside the local neighbor min/max range before trying fallback weights.")
    parser.add_argument("--no-require-improvement", action="store_true")
    args = parser.parse_args()

    event_meta_dir = Path(args.event_meta_dir)
    weights_dir = Path(args.weights_dir)
    rain_dir = Path(args.rain_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start, end, event_idx = load_event_window(args.event, event_meta_dir)
    primary_weights = load_primary_weights(args.event, weights_dir)
    option_weights = load_weight_options(args.event, weights_dir, max_options=args.max_options)
    primary_grid = load_primary_rainfall(args.event, out_dir, event_idx)

    # Keep only centroids that actually exist in the primary rainfall file and in the weights table.
    valid_ids = [str(x) for x in primary_weights["id"].tolist() if str(x) in primary_grid.columns]
    primary_weights = primary_weights[primary_weights["id"].isin(valid_ids)].copy()
    option_weights = option_weights[option_weights["id"].isin(valid_ids)].copy()
    primary_grid = primary_grid[valid_ids].copy()

    gauges = collect_all_gauges(option_weights)
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

    neighbors = build_grid_neighbors(primary_weights[["id", "Latitude", "Longitude"]])

    adjusted_grid, switch_df, summary_df = run_screen(
        primary_grid=primary_grid,
        options_df=option_weights,
        rain_df=R_filled,
        neighbors=neighbors,
        min_valid_in_window=args.min_valid_in_window,
        abs_threshold_mm=args.abs_threshold_mm,
        require_improvement=not args.no_require_improvement,
    )

    out_adjusted = adjusted_grid.copy()
    out_adjusted.insert(0, "time_local", out_adjusted.index.astype(str))
    out_adjusted.to_csv(out_dir / f"Event_{args.event}_grid_rain_hourly_mm.csv", index=False)

    primary_weights[["id", "Latitude", "Longitude"]].drop_duplicates().to_csv(
        out_dir / f"Event_{args.event}_grid_metadata_screened.csv", index=False
    )

    switch_df.to_csv(out_dir / f"Event_{args.event}_screened_switch_log.csv", index=False)
    summary_df.to_csv(out_dir / f"Event_{args.event}_screened_switch_summary.csv", index=False)

    stats_df = pd.DataFrame(stats_rows)
    if not stats_df.empty:
        stats_df["n_missing_after_reindex"] = stats_df["station_id"].map(lambda x: int(missing_counts.get(x, 0)))
        stats_df["n_filled_as_zero"] = stats_df["n_missing_after_reindex"]

    summary = pd.DataFrame([{
        "event": args.event,
        "event_start_local": start.strftime("%Y-%m-%d %H:%M:%S"),
        "event_end_local": end.strftime("%Y-%m-%d %H:%M:%S"),
        "n_event_hours": len(event_idx),
        "n_grids": primary_grid.shape[1],
        "n_gauges_loaded_for_fallbacks": len(gauges),
        "n_option_rows": len(option_weights),
        "screen_rule": "outside_neighbor_minmax_by_threshold",
        "min_valid_in_window": int(args.min_valid_in_window),
        "abs_threshold_mm": float(args.abs_threshold_mm),
        "n_switches_total": int(len(switch_df)),
    }])

    with open(out_dir / f"Event_{args.event}_screened_missing_report.csv", "w", newline="") as f:
        summary.to_csv(f, index=False)
        f.write("\n")
        if not stats_df.empty:
            stats_df.sort_values(["file_exists", "station_id"], ascending=[False, True]).to_csv(f, index=False)

    print(f"Saved screened rainfall: {out_dir / f'Event_{args.event}_grid_rain_hourly_mm_screened.csv'}")
    print(f"Used existing primary rainfall: {out_dir / f'Event_{args.event}_grid_rain_hourly_mm.csv'}")
    print(f"Saved switch log: {out_dir / f'Event_{args.event}_screened_switch_log.csv'}")
    print(f"Saved switch summary: {out_dir / f'Event_{args.event}_screened_switch_summary.csv'}")


if __name__ == "__main__":
    main()
