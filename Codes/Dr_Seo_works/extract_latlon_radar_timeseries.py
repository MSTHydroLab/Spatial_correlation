#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

LOCAL_TZ = "America/Chicago"

RADAR_DIR = Path("/mnt/12TB/Sujan/Radar_products/LATLON/LATLON/")
OUT_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/")
WGS_GRID_CSV = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/dependent_files/grid_centers_wgs84.csv")

CATCHMENT_SHP_PATHS = [
    Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6893390/6893390.shp"),
    Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/06893080/6893080.shp"),
    Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6892513/6892513.shp"),
]

FNAME_RE = re.compile(
    r".*?_(?P<accum_min>\d+)_(?P<date>\d{2}[A-Z]{3}\d{4})_(?P<time>\d{6})\.out$",
    re.IGNORECASE,
)


@dataclass
class RadarHeader:
    ncols: int
    nrows: int
    xllcorner: float
    yllcorner: float
    cellsize: float
    nodata: float
    accumulation_seconds: int | None
    source_file: Path

    @property
    def x_center0(self) -> float:
        return self.xllcorner + 0.5 * self.cellsize

    @property
    def y_center_top(self) -> float:
        return self.yllcorner + (self.nrows - 0.5) * self.cellsize


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Extract LATLON radar time series over the three standard catchments using existing WGS grid IDs."
    )
    ap.add_argument("--event-start", required=True, help="Event start in local time, e.g. '2017-07-23 00:00:00'")
    ap.add_argument("--event-end", required=True, help="Event end in local time, e.g. '2017-07-23 12:00:00'")
    ap.add_argument("--radar-dir", type=Path, default=RADAR_DIR)
    ap.add_argument("--grid-csv", type=Path, default=WGS_GRID_CSV)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--output-prefix", default="latlon_radar")
    ap.add_argument("--catchments", nargs="*", type=Path, default=CATCHMENT_SHP_PATHS)
    ap.add_argument("--write-grid-centers-csv", action="store_true")
    return ap.parse_args()


def make_window(start_str: str, end_str: str) -> tuple[pd.Timestamp, pd.Timestamp, pd.DatetimeIndex]:
    start = pd.to_datetime(start_str, errors="raise")
    end = pd.to_datetime(end_str, errors="raise")

    if getattr(start, "tzinfo", None) is not None:
        start = start.tz_convert(LOCAL_TZ).tz_localize(None)
    if getattr(end, "tzinfo", None) is not None:
        end = end.tz_convert(LOCAL_TZ).tz_localize(None)

    if end < start:
        raise ValueError("event_end must be >= event_start")

    return start, end, pd.date_range(start, end, freq="1h")


def parse_radar_filename_time(path: Path) -> tuple[pd.Timestamp, int | None]:
    m = FNAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Unrecognized radar filename format: {path.name}")

    ts = pd.to_datetime(
        f"{m.group('date').upper()} {m.group('time')}",
        format="%d%b%Y %H%M%S",
        errors="raise",
    )

    accum_min = int(m.group("accum_min"))
    return ts, accum_min * 60


def _parse_header_number_after_colon(line: str) -> float:
    return float(line.split(":", 1)[1].strip().split()[0])


def read_radar_header(path: Path) -> tuple[RadarHeader, int]:
    header_lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                header_lines.append(line.rstrip("\n"))
            else:
                break

    vals: dict[str, float | int | None] = {
        "ncols": None,
        "nrows": None,
        "xllcorner": None,
        "yllcorner": None,
        "cellsize": None,
        "nodata": None,
        "accumulation_seconds": None,
    }

    for raw in header_lines:
        s = raw.strip()
        sl = s.lower()
        if sl.startswith("# accumulation time:"):
            vals["accumulation_seconds"] = int(round(_parse_header_number_after_colon(s)))
        elif sl.startswith("# number of columns:"):
            vals["ncols"] = int(round(_parse_header_number_after_colon(s)))
        elif sl.startswith("# number of rows:"):
            vals["nrows"] = int(round(_parse_header_number_after_colon(s)))
        elif sl.startswith("# xllcorner [lon]:"):
            vals["xllcorner"] = float(_parse_header_number_after_colon(s))
        elif sl.startswith("# yllcorner [lat]:"):
            vals["yllcorner"] = float(_parse_header_number_after_colon(s))
        elif sl.startswith("# cellsize [dec deg]:"):
            vals["cellsize"] = float(_parse_header_number_after_colon(s))
        elif sl.startswith("# no data value:"):
            vals["nodata"] = float(_parse_header_number_after_colon(s))

    missing = [k for k, v in vals.items() if k != "accumulation_seconds" and v is None]
    if missing:
        raise ValueError(f"Header parse failed for {path}. Missing: {missing}")

    header = RadarHeader(
        ncols=int(vals["ncols"]),
        nrows=int(vals["nrows"]),
        xllcorner=float(vals["xllcorner"]),
        yllcorner=float(vals["yllcorner"]),
        cellsize=float(vals["cellsize"]),
        nodata=float(vals["nodata"]),
        accumulation_seconds=int(vals["accumulation_seconds"]) if vals["accumulation_seconds"] is not None else None,
        source_file=path,
    )
    return header, len(header_lines)


def validate_headers(paths: Iterable[Path]) -> RadarHeader:
    baseline: RadarHeader | None = None
    checked = 0
    for p in paths:
        h, _ = read_radar_header(p)
        checked += 1
        if baseline is None:
            baseline = h
            continue

        attrs = ["ncols", "nrows", "xllcorner", "yllcorner", "cellsize", "nodata"]
        mismatches = []
        for a in attrs:
            v0 = getattr(baseline, a)
            v1 = getattr(h, a)
            same = math.isclose(v0, v1, rel_tol=0.0, abs_tol=1e-12) if isinstance(v0, float) else (v0 == v1)
            if not same:
                mismatches.append((a, v0, v1))
        if mismatches:
            parts = ", ".join(f"{a}: {v0} != {v1}" for a, v0, v1 in mismatches)
            raise ValueError(f"Radar header mismatch between {baseline.source_file.name} and {p.name}: {parts}")

    if baseline is None:
        raise ValueError("No radar files found to validate")

    print(f"[header-check] validated {checked} files in {baseline.source_file.parent}")
    return baseline


def discover_files_in_window(radar_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> dict[pd.Timestamp, Path]:
    found: dict[pd.Timestamp, Path] = {}
    for p in sorted(radar_dir.glob("*.out")):
        try:
            t_local, _ = parse_radar_filename_time(p)
        except Exception:
            continue
        if start <= t_local <= end:
            found[t_local] = p
    return found


def load_catchment_union(shp_paths: list[Path]):
    geoms = []
    for shp in shp_paths:
        if not shp.exists():
            raise FileNotFoundError(f"Catchment shapefile not found: {shp}")
        gdf = gpd.read_file(shp)
        if gdf.empty:
            continue
        if gdf.crs is None:
            raise ValueError(f"Catchment shapefile has no CRS: {shp}")
        gdf = gdf.to_crs("EPSG:4326")
        geoms.extend(list(gdf.geometry.dropna()))

    if not geoms:
        raise ValueError("No valid catchment geometries could be loaded")

    return gpd.GeoSeries(geoms, crs="EPSG:4326").union_all()


def load_existing_wgs_grid(grid_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(grid_csv)
    required = ["id", "Latitude", "Longitude"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Grid CSV missing required columns: {missing}")

    df = df[required].copy()
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["id", "Latitude", "Longitude"]).copy()
    df["id"] = df["id"].astype(int)
    return df.sort_values(["Latitude", "Longitude", "id"]).reset_index(drop=True)


def infer_grid_spacing(values: np.ndarray, name: str) -> float:
    vals = np.sort(np.unique(np.round(values.astype(float), 10)))
    diffs = np.diff(vals)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        raise ValueError(f"Could not infer {name} spacing from grid coordinates")
    rounded = np.round(diffs, 10)
    counts = pd.Series(rounded).value_counts().sort_values(ascending=False)
    return float(counts.index[0])


def subset_existing_grid_to_catchments(grid_df: pd.DataFrame, catchment_union) -> pd.DataFrame:
    lat_step = infer_grid_spacing(grid_df["Latitude"].to_numpy(), "latitude")
    lon_step = infer_grid_spacing(grid_df["Longitude"].to_numpy(), "longitude")
    half_lat = lat_step / 2.0
    half_lon = lon_step / 2.0

    geoms = [
        box(lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)
        for lat, lon in zip(grid_df["Latitude"], grid_df["Longitude"])
    ]

    gdf = gpd.GeoDataFrame(grid_df.copy(), geometry=geoms, crs="EPSG:4326")
    mask = gdf.intersects(catchment_union)
    sub = gdf.loc[mask, ["id", "Latitude", "Longitude"]].copy()

    if sub.empty:
        raise ValueError("No WGS grid cells intersect the catchment union")

    sub["id"] = sub["id"].astype(str)
    return sub.sort_values(["Latitude", "Longitude", "id"]).reset_index(drop=True)


def add_radar_indices(grid_df: pd.DataFrame, header: RadarHeader, tol_factor: float = 0.51) -> pd.DataFrame:
    x0 = header.x_center0
    y_top = header.y_center_top
    cs = header.cellsize

    cols = np.rint((grid_df["Longitude"].to_numpy(float) - x0) / cs).astype(int)
    rows = np.rint((y_top - grid_df["Latitude"].to_numpy(float)) / cs).astype(int)

    lon_center = x0 + cols * cs
    lat_center = y_top - rows * cs

    dlon = np.abs(grid_df["Longitude"].to_numpy(float) - lon_center)
    dlat = np.abs(grid_df["Latitude"].to_numpy(float) - lat_center)
    tol = tol_factor * cs

    out_of_bounds = (cols < 0) | (cols >= header.ncols) | (rows < 0) | (rows >= header.nrows)
    off_center = (dlon > tol) | (dlat > tol)
    bad = out_of_bounds | off_center
    if np.any(bad):
        bad_df = grid_df.loc[bad, ["id", "Latitude", "Longitude"]].copy()
        raise ValueError(
            "Some WGS grid centers do not map cleanly to radar cells. "
            f"Count={len(bad_df)}. First few:\n{bad_df.head(10).to_string(index=False)}"
        )

    out = grid_df.copy()
    out["radar_row"] = rows
    out["radar_col"] = cols
    out["radar_lon_center"] = lon_center
    out["radar_lat_center"] = lat_center
    out["lon_offset_deg"] = grid_df["Longitude"].to_numpy(float) - lon_center
    out["lat_offset_deg"] = grid_df["Latitude"].to_numpy(float) - lat_center
    return out


def read_selected_cells(path: Path, header: RadarHeader, skiprows: int, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    arr = np.loadtxt(path, dtype=float, comments=None, skiprows=skiprows)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape != (header.nrows, header.ncols):
        raise ValueError(f"Unexpected array shape in {path}: got {arr.shape}, expected {(header.nrows, header.ncols)}")

    vals = arr[rows, cols].astype(float)
    vals[np.isclose(vals, header.nodata, rtol=0.0, atol=1e-12)] = np.nan
    return vals


def build_event_cube(
    radar_dir: Path,
    out_dir: Path,
    output_prefix: str,
    grid_map: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    event_idx: pd.DatetimeIndex,
    write_grid_centers_csv: bool,
) -> None:
    files = discover_files_in_window(radar_dir, start, end)
    if not files:
        raise FileNotFoundError(f"No radar files found in window under {radar_dir}")

    first_header = validate_headers(files.values())
    grid_map = add_radar_indices(grid_map, first_header)

    rows = grid_map["radar_row"].to_numpy(int)
    cols = grid_map["radar_col"].to_numpy(int)
    grid_ids = grid_map["id"].tolist()

    out = np.full((len(event_idx), len(grid_map)), np.nan, dtype=float)
    present_files = 0
    parsed_times: list[pd.Timestamp] = []

    for i, ts in enumerate(event_idx):
        fp = files.get(ts)
        if fp is None:
            continue

        h, skiprows = read_radar_header(fp)
        attrs = ["ncols", "nrows", "xllcorner", "yllcorner", "cellsize", "nodata"]
        for a in attrs:
            v0 = getattr(first_header, a)
            v1 = getattr(h, a)
            same = math.isclose(v0, v1, rel_tol=0.0, abs_tol=1e-12) if isinstance(v0, float) else (v0 == v1)
            if not same:
                raise ValueError(f"Header changed within {radar_dir} for {fp.name}: field {a} differs")

        out[i, :] = read_selected_cells(fp, h, skiprows, rows, cols)
        parsed_times.append(ts)
        present_files += 1

    out_dir.mkdir(parents=True, exist_ok=True)

    rain_df = pd.DataFrame(out, index=event_idx, columns=grid_ids)
    rain_df.insert(0, "time_local", event_idx.astype(str))

    meta_df = grid_map[[
        "id", "Latitude", "Longitude", "radar_row", "radar_col",
        "radar_lat_center", "radar_lon_center", "lat_offset_deg", "lon_offset_deg",
    ]].copy()
    meta_df.insert(1, "source", "WGS_grid_centers_wgs84")
    meta_df.insert(2, "cellsize_deg", float(first_header.cellsize))
    meta_df.insert(3, "xllcorner_lon", float(first_header.xllcorner))
    meta_df.insert(4, "yllcorner_lat", float(first_header.yllcorner))

    missing_hours = sorted(set(event_idx) - set(parsed_times))
    missing_df = pd.DataFrame({
        "time_local": [t.strftime("%Y-%m-%d %H:%M:%S") for t in missing_hours],
        "reason": "file_not_found_in_radar_folder",
    })

    summary_df = pd.DataFrame([{
        "event_start_local": start.strftime("%Y-%m-%d %H:%M:%S"),
        "event_end_local": end.strftime("%Y-%m-%d %H:%M:%S"),
        "n_event_hours_expected": int(len(event_idx)),
        "n_hours_with_file": int(present_files),
        "n_hours_missing_file": int(len(missing_hours)),
        "n_grid_cells": int(len(grid_map)),
        "ncols": int(first_header.ncols),
        "nrows": int(first_header.nrows),
        "cellsize_deg": float(first_header.cellsize),
        "xllcorner_lon": float(first_header.xllcorner),
        "yllcorner_lat": float(first_header.yllcorner),
        "nodata_value": float(first_header.nodata),
        "accumulation_seconds_header": int(first_header.accumulation_seconds) if first_header.accumulation_seconds is not None else np.nan,
        "radar_dir": str(radar_dir),
        "grid_csv": str(WGS_GRID_CSV),
    }])

    rain_path = out_dir / f"{output_prefix}_grid_rain_timeseries_catchments_only.csv"
    meta_path = out_dir / f"{output_prefix}_grid_metadata_catchments_only.csv"
    missing_path = out_dir / f"{output_prefix}_missing_hours_catchments_only.csv"
    summary_path = out_dir / f"{output_prefix}_summary_catchments_only.csv"

    rain_df.to_csv(rain_path, index=False)
    meta_df.to_csv(meta_path, index=False)
    missing_df.to_csv(missing_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"[saved] {rain_path}")
    print(f"[saved] {meta_path}")
    print(f"[saved] {missing_path}")
    print(f"[saved] {summary_path}")

    if write_grid_centers_csv:
        grid_centers_path = out_dir / f"{output_prefix}_selected_wgs_grid_centers.csv"
        grid_map[["id", "Latitude", "Longitude"]].to_csv(grid_centers_path, index=False)
        print(f"[saved] {grid_centers_path}")


def main() -> None:
    args = parse_args()

    if not args.radar_dir.exists():
        raise FileNotFoundError(f"Radar folder not found: {args.radar_dir}")
    if not args.grid_csv.exists():
        raise FileNotFoundError(f"WGS grid CSV not found: {args.grid_csv}")

    start, end, event_idx = make_window(args.event_start, args.event_end)
    print(f"[window] {start} to {end} ({len(event_idx)} hourly steps)")

    files = discover_files_in_window(args.radar_dir, start, end)
    if not files:
        raise FileNotFoundError(f"No radar files found between {start} and {end} in {args.radar_dir}")

    reference_header = validate_headers(files.values())
    print(
        f"[radar-grid] ncols={reference_header.ncols}, nrows={reference_header.nrows}, "
        f"xllcorner={reference_header.xllcorner}, yllcorner={reference_header.yllcorner}, "
        f"cellsize={reference_header.cellsize}"
    )

    catchment_union = load_catchment_union(list(args.catchments))
    wgs_grid = load_existing_wgs_grid(args.grid_csv)
    grid_sub = subset_existing_grid_to_catchments(wgs_grid, catchment_union)
    print(f"[grid] selected {len(grid_sub)} WGS grid centers intersecting the 3 catchments")

    build_event_cube(
        radar_dir=args.radar_dir,
        out_dir=args.out_dir,
        output_prefix=args.output_prefix,
        grid_map=grid_sub,
        start=start,
        end=end,
        event_idx=event_idx,
        write_grid_centers_csv=args.write_grid_centers_csv,
    )


if __name__ == "__main__":
    main()
