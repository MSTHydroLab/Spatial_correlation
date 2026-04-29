#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


LOCAL_TZ = "America/Chicago"
FILE_SUFFIX = ".hourly_mm.csv"
TIME_LOCAL_COL = "time_local"
TIME_UTC_COL = "time_utc"
RAIN_COL = "rain_mm"

'''python 02_generate_continuous_idw_per_bin.py \
  --bins-meta-csv /mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Continuous_IDW/01_grid_information/availability_bins_2013010100_2024123123.csv \
  --manifest-csv /mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Continuous_IDW/01_grid_information/bin_grid_manifest_2013010100_2024123123.csv \
  --top8-dir /mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Continuous_IDW/02_top8_available \
  --rain-dir /mnt/12TB/Sujan/Spatial_correlation/Codes/Compiled_rain/compiled_rawstyle_hourly_mm/per_station_hourly \
  --start "2013-01-01 00:00:00" \
  --end "2024-12-31 00:00:00" \
  --out-rain-csv /mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Continuous_IDW/03_Interpolated_Rain/continuous_idw_rainfall.csv \
  --bin-rain-dir /mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Continuous_IDW/03_Interpolated_Rain/bin_rainfall \
  --max-dist-km 5 \
  --max-idw-gauges 4 \
  --start-bin 31'''
def norm_station_id(x) -> str:
    s = str(x).strip().strip("'").strip('"')
    if s == "" or s.lower() == "nan":
        return ""
    try:
        return str(int(float(s)))
    except Exception:
        return s


def make_window(start_str: str, end_str: str) -> tuple[pd.Timestamp, pd.Timestamp, pd.DatetimeIndex]:
    start = pd.to_datetime(start_str, errors="raise")
    end = pd.to_datetime(end_str, errors="raise")

    if getattr(start, "tzinfo", None) is not None:
        start = start.tz_convert(LOCAL_TZ).tz_localize(None)
    if getattr(end, "tzinfo", None) is not None:
        end = end.tz_convert(LOCAL_TZ).tz_localize(None)

    if end < start:
        raise ValueError("end must be >= start")

    return start, end, pd.date_range(start, end, freq="1h")


def compute_idw_weights(dists_m: np.ndarray, power: float = 2.0) -> np.ndarray:
    d = np.asarray(dists_m, dtype=float)

    if d.size == 0:
        return np.array([], dtype=float)

    if np.any(d <= 0):
        w = np.zeros_like(d, dtype=float)
        w[np.argmin(d)] = 1.0
        return w

    inv = 1.0 / np.power(d, power)
    s = np.sum(inv)

    if s <= 0:
        return np.zeros_like(d, dtype=float)

    return inv / s


def load_station_series_local(
    station_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    rain_dir: Path,
) -> pd.Series:
    fp = rain_dir / f"{station_id}{FILE_SUFFIX}"
    idx = pd.date_range(start, end, freq="1h")

    if not fp.exists():
        return pd.Series(index=idx, dtype=float, name=station_id)

    df = pd.read_csv(fp, usecols=[TIME_LOCAL_COL, TIME_UTC_COL, RAIN_COL])

    t_utc = pd.to_datetime(df[TIME_UTC_COL], utc=True, errors="coerce")
    off = df[TIME_LOCAL_COL].astype(str).str.extract(r"([+-]\d{2})\d{2}$")[0]
    off_hours = pd.to_numeric(off, errors="coerce")

    t_local = (t_utc + pd.to_timedelta(off_hours, unit="h")).dt.tz_localize(None)

    rain = pd.to_numeric(df[RAIN_COL], errors="coerce").to_numpy()
    s = pd.Series(rain, index=t_local, name=str(station_id))

    s = s[~s.index.isna()]
    s = s.groupby(level=0).mean().sort_index()
    s = s[(s.index >= start) & (s.index <= end)]
    s = s.groupby(s.index.floor("h")).mean()

    return s.reindex(idx)


def load_bins_meta(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    req = ["bin_id", "start_time", "end_time"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    df["start_time"] = pd.to_datetime(df["start_time"], errors="raise")
    df["end_time"] = pd.to_datetime(df["end_time"], errors="raise")

    return df.sort_values("bin_id").reset_index(drop=True)


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    req = ["bin_id", "grid_csv"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    return df[req].copy().sort_values("bin_id").reset_index(drop=True)


def confidence_flag(n_used: int) -> str:
    if n_used >= 4:
        return "high"
    if n_used == 3:
        return "medium"
    if n_used == 2:
        return "low"
    if n_used == 1:
        return "very_low"
    return "missing"


def read_bin_top8(fp: Path, max_dist_m: float) -> pd.DataFrame:
    df = pd.read_csv(fp)

    if "id" not in df.columns:
        raise ValueError(f"{fp} missing id column")

    df["id"] = df["id"].astype(str)

    k = 0
    while f"g{k + 1}" in df.columns:
        k += 1

    if k == 0:
        raise ValueError(f"{fp} does not contain g1..gN columns")

    for kk in range(1, k + 1):
        gcol = f"g{kk}"
        dcol = f"d{kk}_m"

        df[gcol] = df[gcol].astype("string")

        if dcol in df.columns:
            df[dcol] = pd.to_numeric(df[dcol], errors="coerce")

            too_far = df[dcol].isna() | (df[dcol] > max_dist_m)
            df.loc[too_far, gcol] = ""
            df.loc[too_far, dcol] = np.nan

    return df


def unique_stations_in_bin(df: pd.DataFrame) -> list[str]:
    out: set[str] = set()

    k = 0
    while f"g{k + 1}" in df.columns:
        k += 1

    for kk in range(1, k + 1):
        out.update(
            norm_station_id(x)
            for x in df[f"g{kk}"].tolist()
            if norm_station_id(x) != ""
        )

    return sorted(out)


def build_station_cache(
    station_ids: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    rain_dir: Path,
) -> Dict[str, pd.Series]:
    cache: Dict[str, pd.Series] = {}

    for i, sid in enumerate(station_ids, start=1):
        cache[sid] = load_station_series_local(sid, start, end, rain_dir)

        if i <= 5 or i == len(station_ids) or i % 25 == 0:
            print(f"        station {i}/{len(station_ids)} loaded: {sid}")

    return cache


def compute_bin_rainfall(
    top8_df: pd.DataFrame,
    bin_times: pd.DatetimeIndex,
    station_cache: Dict[str, pd.Series],
    power: float,
    max_idw_gauges: int,
    make_diagnostics: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    grid_ids = top8_df["id"].astype(str).tolist()

    out = np.full((len(bin_times), len(grid_ids)), np.nan, dtype=float)
    diag_rows: list[dict] = []

    for j, (_, row) in enumerate(top8_df.iterrows()):
        gid = str(row["id"])

        cand_ids: list[str] = []
        cand_dists: list[float] = []

        kk = 0
        while f"g{kk + 1}" in row.index:
            kk += 1

        for k in range(1, kk + 1):
            sid = norm_station_id(row.get(f"g{k}", ""))
            dist_m = pd.to_numeric(
                pd.Series([row.get(f"d{k}_m", np.nan)]),
                errors="coerce",
            ).iloc[0]

            if sid == "" or pd.isna(dist_m):
                continue

            cand_ids.append(sid)
            cand_dists.append(float(dist_m))

        if not cand_ids:
            if make_diagnostics:
                for ts in bin_times:
                    diag_rows.append({
                        "time_local": str(ts),
                        "grid_id": gid,
                        "n_gauges_used": 0,
                        "nearest_distance_used_m": np.nan,
                        "max_distance_used_m": np.nan,
                        "confidence_flag": "missing",
                        "used_station_ids": "",
                    })
            continue

        for i, ts in enumerate(bin_times):
            used_ids: list[str] = []
            used_vals: list[float] = []
            used_dists: list[float] = []

            for sid, dist_m in zip(cand_ids, cand_dists):
                sval = station_cache[sid].iloc[i]

                if pd.isna(sval):
                    continue

                used_ids.append(sid)
                used_vals.append(float(sval))
                used_dists.append(float(dist_m))

                if len(used_ids) >= max_idw_gauges:
                    break

            if not used_ids:
                if make_diagnostics:
                    diag_rows.append({
                        "time_local": str(ts),
                        "grid_id": gid,
                        "n_gauges_used": 0,
                        "nearest_distance_used_m": np.nan,
                        "max_distance_used_m": np.nan,
                        "confidence_flag": "missing",
                        "used_station_ids": "",
                    })
                continue

            w = compute_idw_weights(np.asarray(used_dists, dtype=float), power=power)
            out[i, j] = float(np.sum(np.asarray(used_vals, dtype=float) * w))

            if make_diagnostics:
                diag_rows.append({
                    "time_local": str(ts),
                    "grid_id": gid,
                    "n_gauges_used": len(used_ids),
                    "nearest_distance_used_m": float(np.min(used_dists)),
                    "max_distance_used_m": float(np.max(used_dists)),
                    "confidence_flag": confidence_flag(len(used_ids)),
                    "used_station_ids": ",".join(used_ids),
                })

        if j < 3 or j == len(grid_ids) - 1 or (j + 1) % 500 == 0:
            print(f"        grid {j + 1}/{len(grid_ids)} done: {gid}")

    rain_df = pd.DataFrame(out, index=bin_times, columns=grid_ids)
    rain_df.insert(0, "time_local", rain_df.index.astype(str))

    diag_df = pd.DataFrame(diag_rows) if make_diagnostics else None

    return rain_df, diag_df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate continuous IDW rainfall per bin using prebuilt top8 bin files."
    )

    ap.add_argument("--bins-meta-csv", type=Path, required=True)
    ap.add_argument("--manifest-csv", type=Path, required=True)
    ap.add_argument("--top8-dir", type=Path, required=True)
    ap.add_argument("--rain-dir", type=Path, required=True)

    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)

    ap.add_argument("--out-rain-csv", type=Path, required=True)
    ap.add_argument("--out-diag-csv", type=Path, default=None)

    ap.add_argument("--bin-rain-dir", type=Path, default=None)
    ap.add_argument("--bin-diag-dir", type=Path, default=None)

    ap.add_argument("--power", type=float, default=2.0)
    ap.add_argument("--max-idw-gauges", type=int, default=4)
    ap.add_argument("--max-dist-km", type=float, default=5.0)

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--only-bin", type=int, default=None)
    ap.add_argument("--start-bin", type=int, default=None)

    ap.add_argument(
        "--write-diagnostics",
        action="store_true",
        help="Write per-bin diagnostics. This can be very large.",
    )

    args = ap.parse_args()

    start, end, _ = make_window(args.start, args.end)

    bins_meta = load_bins_meta(args.bins_meta_csv)
    manifest = load_manifest(args.manifest_csv)

    bins = bins_meta.merge(manifest, on="bin_id", how="inner")
    if bins.empty:
        raise ValueError("No bins found after joining metadata and manifest")

    args.out_rain_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.out_diag_csv is not None:
        args.out_diag_csv.parent.mkdir(parents=True, exist_ok=True)

    bin_rain_dir = args.bin_rain_dir or (args.out_rain_csv.parent / "bin_rainfall")
    bin_diag_dir = args.bin_diag_dir or (args.out_rain_csv.parent / "bin_diagnostics")

    Path(bin_rain_dir).mkdir(parents=True, exist_ok=True)
    Path(bin_diag_dir).mkdir(parents=True, exist_ok=True)

    max_dist_m = float(args.max_dist_km) * 1000.0

    written_rain_files: list[str] = []
    written_diag_files: list[str] = []

    print("[1/4] Processing bins one by one")

    for _, row in bins.iterrows():
        bin_id = int(row["bin_id"])

        if args.only_bin is not None and bin_id != args.only_bin:
            continue

        if args.start_bin is not None and bin_id < args.start_bin:
            continue

        bstart = max(pd.to_datetime(row["start_time"]), start)
        bend = min(pd.to_datetime(row["end_time"]), end)

        if bend < bstart:
            continue

        bin_times = pd.date_range(bstart, bend, freq="1h")
        if len(bin_times) == 0:
            continue

        top8_fp = args.top8_dir / str(row["grid_csv"])

        print(f"    Bin {bin_id:04d} | {bstart} to {bend} | hours={len(bin_times)}")
        print(f"      reading top8 file: {top8_fp.name}")

        bin_rain_fp = Path(bin_rain_dir) / f"bin_{bin_id:04d}_rainfall.csv"
        bin_diag_fp = Path(bin_diag_dir) / f"bin_{bin_id:04d}_diagnostics.csv"

        if bin_rain_fp.exists() and not args.overwrite:
            print(f"      skipping existing bin rainfall: {bin_rain_fp}")
            written_rain_files.append(str(bin_rain_fp))

            if args.write_diagnostics and bin_diag_fp.exists():
                written_diag_files.append(str(bin_diag_fp))

            continue

        top8_df = read_bin_top8(top8_fp, max_dist_m=max_dist_m)

        station_ids = unique_stations_in_bin(top8_df)
        print(f"      active stations in top8 table: {len(station_ids)}")

        station_cache = build_station_cache(
            station_ids=station_ids,
            start=bstart,
            end=bend,
            rain_dir=args.rain_dir,
        )

        rain_df, diag_df = compute_bin_rainfall(
            top8_df=top8_df,
            bin_times=bin_times,
            station_cache=station_cache,
            power=float(args.power),
            max_idw_gauges=int(args.max_idw_gauges),
            make_diagnostics=bool(args.write_diagnostics),
        )

        rain_df.to_csv(bin_rain_fp, index=False)
        print(f"      saved bin rainfall: {bin_rain_fp}")
        written_rain_files.append(str(bin_rain_fp))

        del rain_df

        if args.write_diagnostics and diag_df is not None:
            diag_df.insert(1, "bin_id", bin_id)
            diag_df.to_csv(bin_diag_fp, index=False)
            print(f"      saved bin diagnostics: {bin_diag_fp}")
            written_diag_files.append(str(bin_diag_fp))
            del diag_df

        del station_cache
        del top8_df

    if not written_rain_files:
        raise ValueError("No bin rainfall outputs were produced")

    print("[2/4] Skipping full rainfall combine to avoid memory kill")

    rain_manifest = pd.DataFrame({
        "rain_file": written_rain_files,
    })

    rain_manifest_fp = args.out_rain_csv.with_name("bin_rainfall_manifest.csv")
    rain_manifest.to_csv(rain_manifest_fp, index=False)
    print(f"      saved rainfall manifest: {rain_manifest_fp}")

    if args.write_diagnostics:
        print("[3/4] Skipping full diagnostics combine to avoid memory kill")

        diag_manifest = pd.DataFrame({
            "diag_file": written_diag_files,
        })

        diag_manifest_fp = (
            args.out_diag_csv.with_name("bin_diagnostics_manifest.csv")
            if args.out_diag_csv is not None
            else args.out_rain_csv.with_name("bin_diagnostics_manifest.csv")
        )

        diag_manifest.to_csv(diag_manifest_fp, index=False)
        print(f"      saved diagnostics manifest: {diag_manifest_fp}")
    else:
        print("[3/4] Diagnostics skipped")

    print("[4/4] Done")
    print("      Outputs are saved as per-bin rainfall CSVs.")
    print("      A manifest CSV lists all written bin rainfall files.")
    print("      Full combine was intentionally skipped to avoid memory kill.")


if __name__ == "__main__":
    main()