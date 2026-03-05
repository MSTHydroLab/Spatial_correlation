#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Default paths
# -----------------------
EVENT_META_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/01_Correlation_and_Variogram")
WEIGHTS_DIR    = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/02_OK_Weights")
RAIN_DIR       = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/Compiled_rain/compiled_rawstyle_hourly_mm/per_station_hourly/")
STATIONS_FILE  = EVENT_META_DIR / "Stations_df.csv"

FILE_SUFFIX    = ".hourly_mm.csv"
TIME_LOCAL_COL = "time_local"
TIME_UTC_COL   = "time_utc"
RAIN_COL       = "rain_mm"

OUT_DIR_DEFAULT = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/diagnostics")


# -----------------------
# Helpers
# -----------------------
def make_window(start_str: str, end_str: str) -> tuple[pd.Timestamp, pd.Timestamp, pd.DatetimeIndex]:
    start = pd.to_datetime(start_str, format="%Y%m%d%H")
    end   = pd.to_datetime(end_str,   format="%Y%m%d%H")
    if end < start:
        raise ValueError("end_str must be >= start_str")
    idx = pd.date_range(start, end, freq="1h")
    return start, end, idx


def load_event_window(event_number: int) -> tuple[pd.Timestamp, pd.Timestamp, pd.DatetimeIndex, dict]:
    fp = EVENT_META_DIR / f"Event_{event_number}_Stations_correlation.csv"
    meta = pd.read_csv(fp)

    start_str = str(meta["event_start"].dropna().iloc[0]).strip()
    end_str   = str(meta["event_end"].dropna().iloc[0]).strip()

    a_km = float(meta["corr_a_km"].dropna().iloc[0]) if "corr_a_km" in meta.columns else np.nan
    b    = float(meta["corr_b"].dropna().iloc[0]) if "corr_b" in meta.columns else np.nan

    start, end, idx = make_window(start_str, end_str)
    info = {"event_start": start_str, "event_end": end_str, "corr_a_km": a_km, "corr_b": b}
    return start, end, idx, info


def _clean_gid(x) -> str:
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


def load_weights_row(event: int, grid_id: int, n_gauges: int) -> pd.Series:
    """
    Tries common filenames:
      Event_{event}_weights_{n}gauges.csv
      Event_{event}_weights.csv
    """
    cand = [
        WEIGHTS_DIR / f"Event_{event}_weights_{n_gauges}gauges.csv",
        WEIGHTS_DIR / f"Event_{event}_weights.csv",
    ]
    fp = None
    for c in cand:
        if c.exists():
            fp = c
            break
    if fp is None:
        raise FileNotFoundError(f"No weights file found. Tried: {[str(x) for x in cand]}")

    W = pd.read_csv(fp)
    if "id" not in W.columns:
        raise ValueError(f"{fp} missing 'id' column")

    # grid_id compare as int if possible
    # (your files sometimes store id as int-like)
    W["id_int"] = pd.to_numeric(W["id"], errors="coerce").astype("Int64")
    row = W[W["id_int"] == int(grid_id)]
    if row.empty:
        raise KeyError(f"grid_id={grid_id} not found in {fp}")
    row = row.iloc[0]

    # normalize gauge ids and weights
    for k in range(1, n_gauges + 1):
        gcol = f"g{k}"
        wcol = f"w{k}"
        if gcol not in row.index or wcol not in row.index:
            raise ValueError(f"{fp} row missing {gcol}/{wcol}")
    return row


def load_station_series_local(station_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    fp = RAIN_DIR / f"{station_id}{FILE_SUFFIX}"
    if not fp.exists():
        return pd.Series(dtype=float, name=str(station_id))

    df = pd.read_csv(fp, usecols=[TIME_LOCAL_COL, TIME_UTC_COL, RAIN_COL])

    # Parse UTC
    t_utc = pd.to_datetime(df[TIME_UTC_COL], utc=True, errors="coerce")

    # Extract offset like -0600 -> -06 hours
    off = df[TIME_LOCAL_COL].astype(str).str.extract(r"([+-]\d{2})\d{2}$")[0]
    off_hours = pd.to_numeric(off, errors="coerce")

    # UTC -> local naive timestamp
    t_local = (t_utc + pd.to_timedelta(off_hours, unit="h")).dt.tz_localize(None)

    rain = pd.to_numeric(df[RAIN_COL], errors="coerce").to_numpy()
    s = pd.Series(rain, index=t_local, name=str(station_id))

    # Drop NaT timestamps
    s = s[~s.index.isna()]

    # Collapse duplicates (DST fall-back) using mean
    s = s.groupby(level=0).mean()

    # subset and hourly-floor
    s = s.sort_index()
    s = s[(s.index >= start) & (s.index <= end)]
    s = s.groupby(s.index.floor("h")).mean()

    return s


def write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


# -----------------------
# Main diagnostic
# -----------------------
def run_diagnostic(event: int, grid_id: int, n_gauges: int, out_dir: Path) -> None:
    start, end, idx, meta = load_event_window(event)
    row = load_weights_row(event, grid_id, n_gauges)

    gauge_ids = []
    weights = []
    for k in range(1, n_gauges + 1):
        gauge_ids.append(_clean_gid(row[f"g{k}"]))
        weights.append(float(pd.to_numeric(row[f"w{k}"], errors="coerce")))

    weights = np.array(weights, dtype=float)

    # Load rainfall for gauges (aligned)
    series = []
    per_gauge_stats = []
    for g in gauge_ids:
        if g == "":
            s = pd.Series(index=idx, data=np.nan, name=g)
            series.append(s)
            per_gauge_stats.append({"gauge": g, "exists": False, "n_nan": len(idx), "max_mm": np.nan})
            continue

        s = load_station_series_local(g, start, end).reindex(idx)
        series.append(s)
        per_gauge_stats.append({
            "gauge": g,
            "exists": True,
            "n_nan": int(s.isna().sum()),
            "max_mm": float(np.nanmax(s.values)) if not s.dropna().empty else np.nan,
        })

    R = pd.concat(series, axis=1)
    R.columns = gauge_ids

    # Fill missing as 0 (same as your estimator)
    R_filled = R.fillna(0.0)

    # Contribution series
    contrib = {}
    for i, g in enumerate(gauge_ids):
        contrib[f"contrib_{g}"] = R_filled[g].values * weights[i]

    contrib_df = pd.DataFrame(contrib, index=idx)

    grid_rain = contrib_df.sum(axis=1)
    grid_df = pd.DataFrame({"grid_rain_mm": grid_rain}, index=idx)

    # QC metrics
    wsum = float(np.nansum(weights))
    n_neg = int(np.sum(weights < 0))
    n_gt1 = int(np.sum(weights > 1))
    w_min = float(np.nanmin(weights))
    w_max = float(np.nanmax(weights))

    grid_max = float(np.nanmax(grid_rain.values)) if len(grid_rain) else np.nan
    grid_p99 = float(np.nanpercentile(grid_rain.values, 99)) if len(grid_rain) else np.nan

    # Identify the hour of max grid rainfall and contribution split
    tmax = grid_rain.idxmax()
    split_at_max = contrib_df.loc[tmax].sort_values(ascending=False)

    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"Event{event}_Grid{grid_id}_ng{n_gauges}"

    out_csv = out_dir / f"{base}_timeseries.csv"
    out_report = out_dir / f"{base}_report.txt"
    out_png = out_dir / f"{base}_plot.png"

    out_ts = pd.concat([grid_df, contrib_df, R_filled.add_prefix("rain_")], axis=1)
    out_ts.insert(0, "time_local", out_ts.index.astype(str))
    out_ts.to_csv(out_csv, index=False)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(grid_rain.index, grid_rain.values, linewidth=1.5)
    plt.title(f"Event {event} | Grid {grid_id} | OK rainfall (mm/hr)")
    plt.xlabel("time_local")
    plt.ylabel("grid_rain_mm")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    # Report text
    lines = []
    lines.append(f"Event: {event}")
    lines.append(f"Grid: {grid_id}")
    lines.append(f"Window (local): {meta['event_start']} -> {meta['event_end']}  (n_hours={len(idx)})")
    lines.append(f"Correlation params from meta: a_km={meta.get('corr_a_km', np.nan)}  b={meta.get('corr_b', np.nan)}")
    lines.append("")
    lines.append("Weights QC:")
    for i, g in enumerate(gauge_ids):
        lines.append(f"  g{i+1}={g:>8s}  w{i+1}={weights[i]: .6f}")
    lines.append(f"  sum(weights) = {wsum:.6f}")
    lines.append(f"  min/max weight = {w_min:.6f} / {w_max:.6f}")
    lines.append(f"  count(weight<0) = {n_neg}")
    lines.append(f"  count(weight>1) = {n_gt1}")
    lines.append("")
    lines.append("Gauge rainfall QC (within window, after alignment; before fill):")
    for st in per_gauge_stats:
        lines.append(f"  gauge={st['gauge']:>8s}  exists={st['exists']}  n_nan={st['n_nan']}  max_mm={st['max_mm']}")
    lines.append("")
    lines.append("Grid rainfall QC (after filling missing as 0):")
    lines.append(f"  max(grid_rain_mm) = {grid_max}")
    lines.append(f"  p99(grid_rain_mm) = {grid_p99}")
    lines.append(f"  time of max = {tmax}")
    lines.append("")
    lines.append("Contribution split at max hour (largest to smallest):")
    for k, v in split_at_max.items():
        lines.append(f"  {k}: {float(v):.6f}")
    lines.append("")
    lines.append(f"Wrote timeseries: {out_csv}")
    lines.append(f"Wrote plot:       {out_png}")

    write_report(out_report, lines)

    print(f"[OK] {base}")
    print(f"  {out_report}")
    print(f"  {out_csv}")
    print(f"  {out_png}")


def main():
    ap = argparse.ArgumentParser(description="Diagnose OK interpolation for one event + one grid.")
    ap.add_argument("--event", type=int, required=True)
    ap.add_argument("--grid-id", type=int, required=True)
    ap.add_argument("--n-gauges", type=int, default=3, choices=[3, 4])
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR_DEFAULT))
    args = ap.parse_args()

    run_diagnostic(
        event=args.event,
        grid_id=args.grid_id,
        n_gauges=args.n_gauges,
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()