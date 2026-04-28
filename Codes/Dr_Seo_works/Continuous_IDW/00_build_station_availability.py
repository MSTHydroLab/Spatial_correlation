#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd

LOCAL_TZ = "America/Chicago"
BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Continuous_IDW")
DEFAULT_RAIN_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/Compiled_rain/compiled_rawstyle_hourly_mm/per_station_hourly/")
DEFAULT_OUT_DIR = BASE_DIR / "00_station_availability"
FILE_SUFFIX = ".hourly_mm.csv"
TIME_LOCAL_COL = "time_local"
TIME_UTC_COL = "time_utc"
RAIN_COL = "rain_mm"
N_RECORDS_COL = "n_records"


# -----------------------------
# Helpers
# -----------------------------

def norm_station_id(x) -> str:
    s = str(x).strip().strip("'").strip('"')
    if s == "" or s.lower() == "nan":
        return ""
    try:
        return str(int(float(s)))
    except Exception:
        return s


def parse_station_id_from_path(path: Path) -> str:
    stem = path.stem.strip()
    m = re.search(r"(\d+)", stem)
    if m:
        return norm_station_id(m.group(1))
    return norm_station_id(stem)


def make_analysis_window(start_str: str | None, end_str: str | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start = pd.to_datetime(start_str, errors="raise") if start_str else None
    end = pd.to_datetime(end_str, errors="raise") if end_str else None

    if start is not None and getattr(start, "tzinfo", None) is not None:
        start = start.tz_convert(LOCAL_TZ).tz_localize(None)
    if end is not None and getattr(end, "tzinfo", None) is not None:
        end = end.tz_convert(LOCAL_TZ).tz_localize(None)

    if start is not None and end is not None and end < start:
        raise ValueError("end must be >= start")
    return start, end


def load_station_series_local(path: Path) -> tuple[pd.Series, dict]:
    stats = {
        "source_file": str(path),
        "rows_read": 0,
        "n_bad_time_rows_dropped": 0,
        "n_duplicates_collapsed": 0,
        "first_time": pd.NaT,
        "last_time": pd.NaT,
    }

    usecols = [c for c in [TIME_LOCAL_COL, TIME_UTC_COL, RAIN_COL, N_RECORDS_COL] if c is not None]
    df = pd.read_csv(path, usecols=lambda c: c in usecols)
    stats["rows_read"] = len(df)

    if TIME_UTC_COL in df.columns:
        t_utc = pd.to_datetime(df[TIME_UTC_COL], utc=True, errors="coerce")
        if TIME_LOCAL_COL in df.columns:
            off = df[TIME_LOCAL_COL].astype(str).str.extract(r"([+-]\d{2})\d{2}$")[0]
            off_hours = pd.to_numeric(off, errors="coerce")
            t_local = (t_utc + pd.to_timedelta(off_hours, unit="h")).dt.tz_localize(None)
        else:
            t_local = t_utc.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    elif TIME_LOCAL_COL in df.columns:
        t_local = pd.to_datetime(df[TIME_LOCAL_COL], errors="coerce")
        if getattr(t_local.dt, "tz", None) is not None:
            t_local = t_local.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    else:
        raise ValueError(f"{path} must contain {TIME_LOCAL_COL} or {TIME_UTC_COL}")

    rain = pd.to_numeric(df[RAIN_COL], errors="coerce") if RAIN_COL in df.columns else pd.Series(np.nan, index=df.index)

    s = pd.Series(rain.to_numpy(dtype=float), index=t_local, name=parse_station_id_from_path(path))
    n0 = len(s)
    s = s[~s.index.isna()]
    stats["n_bad_time_rows_dropped"] = int(n0 - len(s))

    n0 = len(s)
    s = s.groupby(level=0).mean()
    stats["n_duplicates_collapsed"] = int(n0 - len(s))

    s = s.sort_index()
    if len(s) > 0:
        stats["first_time"] = s.index.min()
        stats["last_time"] = s.index.max()
    return s, stats


def series_to_structural_periods(
    s: pd.Series,
    gap_threshold_hours: int,
    analysis_start: pd.Timestamp | None,
    analysis_end: pd.Timestamp | None,
) -> tuple[pd.DataFrame, dict]:
    summary = {
        "n_hours_with_rows": 0,
        "analysis_start": analysis_start,
        "analysis_end": analysis_end,
        "first_record_time": pd.NaT,
        "last_record_time": pd.NaT,
        "n_available_periods": 0,
        "n_missing_periods": 0,
        "total_available_hours": 0,
        "total_missing_hours": 0,
        "longest_missing_gap_hours": 0,
        "gap_threshold_hours": int(gap_threshold_hours),
    }

    if s.empty:
        periods = []
        if analysis_start is not None and analysis_end is not None:
            hours = int((analysis_end - analysis_start) / pd.Timedelta(hours=1)) + 1
            periods.append({
                "start_time": analysis_start,
                "end_time": analysis_end,
                "status": "missing",
                "reason": "no_station_rows_in_window",
                "duration_hours": hours,
            })
            summary.update({
                "n_missing_periods": 1,
                "total_missing_hours": hours,
                "longest_missing_gap_hours": hours,
            })
        return pd.DataFrame(periods), summary

    summary["first_record_time"] = s.index.min()
    summary["last_record_time"] = s.index.max()

    if analysis_start is None:
        analysis_start = s.index.min().floor("h")
    if analysis_end is None:
        analysis_end = s.index.max().floor("h")

    s = s[(s.index >= analysis_start) & (s.index <= analysis_end)]
    if s.empty:
        hours = int((analysis_end - analysis_start) / pd.Timedelta(hours=1)) + 1
        periods = pd.DataFrame([{
            "start_time": analysis_start,
            "end_time": analysis_end,
            "status": "missing",
            "reason": "no_station_rows_in_window",
            "duration_hours": hours,
        }])
        summary.update({
            "analysis_start": analysis_start,
            "analysis_end": analysis_end,
            "n_missing_periods": 1,
            "total_missing_hours": hours,
            "longest_missing_gap_hours": hours,
        })
        return periods, summary

    observed_hours = pd.DatetimeIndex(s.index.floor("h").unique()).sort_values()
    summary["n_hours_with_rows"] = len(observed_hours)

    periods: list[dict] = []

    # Leading missing period
    if analysis_start < observed_hours[0]:
        lead_hours = int((observed_hours[0] - analysis_start) / pd.Timedelta(hours=1))
        if lead_hours > 0:
            periods.append({
                "start_time": analysis_start,
                "end_time": observed_hours[0] - pd.Timedelta(hours=1),
                "status": "missing",
                "reason": "leading_gap",
                "duration_hours": lead_hours,
            })

    # Available blocks separated by structural gaps only
    block_start = observed_hours[0]
    prev = observed_hours[0]

    for t in observed_hours[1:]:
        diff_hours = int((t - prev) / pd.Timedelta(hours=1))
        if diff_hours > gap_threshold_hours:
            periods.append({
                "start_time": block_start,
                "end_time": prev,
                "status": "available",
                "reason": "continuous_record",
                "duration_hours": int((prev - block_start) / pd.Timedelta(hours=1)) + 1,
            })
            periods.append({
                "start_time": prev + pd.Timedelta(hours=1),
                "end_time": t - pd.Timedelta(hours=1),
                "status": "missing",
                "reason": "long_gap",
                "duration_hours": diff_hours - 1,
            })
            block_start = t
        prev = t

    periods.append({
        "start_time": block_start,
        "end_time": prev,
        "status": "available",
        "reason": "continuous_record",
        "duration_hours": int((prev - block_start) / pd.Timedelta(hours=1)) + 1,
    })

    # Trailing missing period
    if prev < analysis_end:
        trail_hours = int((analysis_end - prev) / pd.Timedelta(hours=1))
        if trail_hours > 0:
            periods.append({
                "start_time": prev + pd.Timedelta(hours=1),
                "end_time": analysis_end,
                "status": "missing",
                "reason": "trailing_gap",
                "duration_hours": trail_hours,
            })

    periods_df = pd.DataFrame(periods)
    if not periods_df.empty:
        periods_df = periods_df.sort_values(["start_time", "end_time"]).reset_index(drop=True)

    avail_df = periods_df.loc[periods_df["status"] == "available"]
    miss_df = periods_df.loc[periods_df["status"] == "missing"]

    summary.update({
        "analysis_start": analysis_start,
        "analysis_end": analysis_end,
        "n_available_periods": int(len(avail_df)),
        "n_missing_periods": int(len(miss_df)),
        "total_available_hours": int(avail_df["duration_hours"].sum()) if not avail_df.empty else 0,
        "total_missing_hours": int(miss_df["duration_hours"].sum()) if not miss_df.empty else 0,
        "longest_missing_gap_hours": int(miss_df["duration_hours"].max()) if not miss_df.empty else 0,
    })
    return periods_df, summary


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export one structural-availability CSV per station. "
            "Zero rainfall is treated as valid data. Only long gaps in record coverage "
            "are treated as structural missing periods."
        )
    )
    parser.add_argument("--rain-dir", type=Path, default=DEFAULT_RAIN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start", type=str, default=None, help="Optional analysis start in local time")
    parser.add_argument("--end", type=str, default=None, help="Optional analysis end in local time")
    parser.add_argument(
        "--gap-threshold-days",
        type=float,
        default=7.0,
        help="Gap longer than this is treated as structural missing. Default: 7 days.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default=f"*{FILE_SUFFIX}",
        help="File pattern for station rainfall CSVs",
    )
    args = parser.parse_args()

    rain_dir = Path(args.rain_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start, end = make_analysis_window(args.start, args.end)
    gap_threshold_hours = int(round(float(args.gap_threshold_days) * 24.0))

    station_files = sorted(rain_dir.glob(args.glob))
    if not station_files:
        raise FileNotFoundError(f"No station files found under {rain_dir} matching {args.glob}")

    print("=" * 80)
    print("Building station structural availability files")
    print(f"Rain directory        : {rain_dir}")
    print(f"Output directory      : {out_dir}")
    print(f"Requested start       : {start}")
    print(f"Requested end         : {end}")
    print(f"Structural gap rule   : gaps longer than {gap_threshold_hours} hours")
    print(f"Station files found   : {len(station_files)}")
    print("Zero rainfall counts as valid data. Only missing record coverage creates outages.")
    print("=" * 80)

    summary_rows = []

    for i, fp in enumerate(station_files, start=1):
        sid = parse_station_id_from_path(fp)
        if sid == "":
            print(f"[{i}/{len(station_files)}] Skipping file with unreadable station id: {fp.name}")
            continue

        print(f"\n[{i}/{len(station_files)}] Station {sid}")
        print(f"  Reading file        : {fp.name}")
        s, load_stats = load_station_series_local(fp)
        print(f"  Rows read           : {load_stats['rows_read']}")
        print(f"  Bad time rows drop  : {load_stats['n_bad_time_rows_dropped']}")
        print(f"  Duplicate times fix : {load_stats['n_duplicates_collapsed']}")

        periods_df, summary = series_to_structural_periods(
            s=s,
            gap_threshold_hours=gap_threshold_hours,
            analysis_start=start,
            analysis_end=end,
        )

        station_csv = out_dir / f"{sid}_availability.csv"
        if not periods_df.empty:
            out_df = periods_df.copy()
            out_df["start_time"] = pd.to_datetime(out_df["start_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            out_df["end_time"] = pd.to_datetime(out_df["end_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            out_df.to_csv(station_csv, index=False)
        else:
            pd.DataFrame(columns=["start_time", "end_time", "status", "reason", "duration_hours"]).to_csv(station_csv, index=False)

        print(f"  Availability periods: {summary['n_available_periods']} available, {summary['n_missing_periods']} missing")
        print(f"  Missing hours total : {summary['total_missing_hours']}")
        print(f"  Longest missing gap : {summary['longest_missing_gap_hours']} hours")
        print(f"  Saved               : {station_csv.name}")

        summary_rows.append({
            "station_id": sid,
            "source_file": fp.name,
            "rows_read": load_stats["rows_read"],
            "n_bad_time_rows_dropped": load_stats["n_bad_time_rows_dropped"],
            "n_duplicates_collapsed": load_stats["n_duplicates_collapsed"],
            "first_record_time": summary["first_record_time"],
            "last_record_time": summary["last_record_time"],
            "analysis_start": summary["analysis_start"],
            "analysis_end": summary["analysis_end"],
            "n_hours_with_rows": summary["n_hours_with_rows"],
            "n_available_periods": summary["n_available_periods"],
            "n_missing_periods": summary["n_missing_periods"],
            "total_available_hours": summary["total_available_hours"],
            "total_missing_hours": summary["total_missing_hours"],
            "longest_missing_gap_hours": summary["longest_missing_gap_hours"],
            "gap_threshold_hours": summary["gap_threshold_hours"],
            "availability_csv": station_csv.name,
        })

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        for c in ["first_record_time", "last_record_time", "analysis_start", "analysis_end"]:
            summary_df[c] = pd.to_datetime(summary_df[c], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    summary_csv = out_dir / "station_availability_summary.csv"
    summary_df.sort_values("station_id").to_csv(summary_csv, index=False)

    print("\n" + "=" * 80)
    print("Finished building station availability files")
    print(f"Per-station CSV folder: {out_dir}")
    print(f"Summary CSV           : {summary_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
