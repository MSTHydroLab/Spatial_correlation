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

DEFAULT_RADAR_DIR = Path("/mnt/12TB/Sujan/Radar_products/LATLON/LATLON/")
DEFAULT_OUT_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/")

DEFAULT_CATCHMENT_SHP_PATHS = [
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
        description="Build total rainfall ASCII raster over event period, restricted to radar cells intersecting catchments."
    )
    ap.add_argument("--event-start", required=True, help="Event start in local time, e.g. '2017-07-23 00:00:00'")
    ap.add_argument("--event-end", required=True, help="Event end in local time, e.g. '2017-07-23 12:00:00'")
    ap.add_argument("--radar-dir", type=Path, default=DEFAULT_RADAR_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--output-prefix", default="latlon_radar_total")
    ap.add_argument("--catchments", nargs="*", type=Path, default=DEFAULT_CATCHMENT_SHP_PATHS)
    ap.add_argument(
        "--full-grid",
        action="store_true",
        help="Write full raster extent. Default is catchment cells only, outside set to NODATA."
    )
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


def build_radar_cell_map(header: RadarHeader) -> gpd.GeoDataFrame:
    cs = header.cellsize
    half = cs / 2.0

    rows = np.arange(header.nrows, dtype=int)
    cols = np.arange(header.ncols, dtype=int)

    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")

    lon_center = header.x_center0 + col_grid * cs
    lat_center = header.y_center_top - row_grid * cs

    lon_center = lon_center.ravel()
    lat_center = lat_center.ravel()
    row_flat = row_grid.ravel()
    col_flat = col_grid.ravel()

    geoms = [
        box(lon - half, lat - half, lon + half, lat + half)
        for lon, lat in zip(lon_center, lat_center)
    ]

    gdf = gpd.GeoDataFrame(
        {
            "radar_row": row_flat,
            "radar_col": col_flat,
            "Longitude": lon_center,
            "Latitude": lat_center,
        },
        geometry=geoms,
        crs="EPSG:4326",
    )
    return gdf


def subset_radar_cells_to_catchments(cell_gdf: gpd.GeoDataFrame, catchment_union) -> gpd.GeoDataFrame:
    mask = cell_gdf.intersects(catchment_union)
    sub = cell_gdf.loc[mask].copy()
    if sub.empty:
        raise ValueError("No radar cells intersect the catchment union")
    sub = sub.sort_values(["radar_row", "radar_col"]).reset_index(drop=True)
    return sub


def read_full_raster(path: Path, header: RadarHeader, skiprows: int) -> np.ndarray:
    arr = np.loadtxt(path, dtype=float, comments=None, skiprows=skiprows)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape != (header.nrows, header.ncols):
        raise ValueError(
            f"Unexpected array shape in {path}: got {arr.shape}, expected {(header.nrows, header.ncols)}"
        )
    arr[np.isclose(arr, header.nodata, rtol=0.0, atol=1e-12)] = np.nan
    return arr


def write_ascii_grid(out_path: Path, arr: np.ndarray, header: RadarHeader) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"ncols {header.ncols}\n")
        f.write(f"nrows {header.nrows}\n")
        f.write(f"xllcorner {header.xllcorner}\n")
        f.write(f"yllcorner {header.yllcorner}\n")
        f.write(f"cellsize {header.cellsize}\n")
        f.write(f"NODATA_value {header.nodata}\n")

        for row in arr:
            vals = [
                f"{header.nodata:.2f}" if not np.isfinite(v) else f"{v:.2f}"
                for v in row
            ]
            f.write(" ".join(vals) + "\n")


def build_total_raster(
    radar_dir: Path,
    out_dir: Path,
    output_prefix: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    catchments: list[Path],
    full_grid: bool,
) -> None:
    files = discover_files_in_window(radar_dir, start, end)
    if not files:
        raise FileNotFoundError(f"No radar files found between {start} and {end} in {radar_dir}")

    first_header = validate_headers(files.values())

    if first_header.accumulation_seconds is not None and first_header.accumulation_seconds != 3600:
        print(f"[warning] accumulation time from header is {first_header.accumulation_seconds} sec, not 3600 sec")

    catchment_union = load_catchment_union(catchments)
    radar_cells = build_radar_cell_map(first_header)
    radar_sub = subset_radar_cells_to_catchments(radar_cells, catchment_union)

    print(f"[radar-grid] ncols={first_header.ncols}, nrows={first_header.nrows}, "
          f"xllcorner={first_header.xllcorner}, yllcorner={first_header.yllcorner}, "
          f"cellsize={first_header.cellsize}")
    print(f"[catchments] selected {len(radar_sub)} radar cells intersecting catchments")

    total = np.full((first_header.nrows, first_header.ncols), np.nan, dtype=float)
    valid_count = np.zeros((first_header.nrows, first_header.ncols), dtype=np.int32)

    selected_rows = radar_sub["radar_row"].to_numpy(int)
    selected_cols = radar_sub["radar_col"].to_numpy(int)

    parsed_times = []
    for i, ts in enumerate(sorted(files.keys())):
        fp = files[ts]
        h, skiprows = read_radar_header(fp)

        attrs = ["ncols", "nrows", "xllcorner", "yllcorner", "cellsize", "nodata"]
        for a in attrs:
            v0 = getattr(first_header, a)
            v1 = getattr(h, a)
            same = math.isclose(v0, v1, rel_tol=0.0, abs_tol=1e-12) if isinstance(v0, float) else (v0 == v1)
            if not same:
                raise ValueError(f"Header changed within {radar_dir} for {fp.name}: field {a} differs")

        arr = read_full_raster(fp, h, skiprows)

        vals = arr[selected_rows, selected_cols]
        mask = np.isfinite(vals)

        rr = selected_rows[mask]
        cc = selected_cols[mask]
        vv = vals[mask]

        need_init = ~np.isfinite(total[rr, cc])
        if np.any(need_init):
            total[rr[need_init], cc[need_init]] = 0.0

        total[rr, cc] += vv
        valid_count[rr, cc] += 1

        parsed_times.append(ts)
        print(f"[{i+1}/{len(files)}] processed {fp.name}")

    if full_grid:
        out_arr = total.copy()
    else:
        out_arr = np.full((first_header.nrows, first_header.ncols), np.nan, dtype=float)
        out_arr[selected_rows, selected_cols] = total[selected_rows, selected_cols]

    out_dir.mkdir(parents=True, exist_ok=True)

    asc_path = out_dir / f"{output_prefix}_total_event_rain_catchments_only.asc"
    meta_path = out_dir / f"{output_prefix}_total_event_rain_catchments_only_metadata.csv"
    summary_path = out_dir / f"{output_prefix}_total_event_rain_catchments_only_summary.csv"

    write_ascii_grid(asc_path, out_arr, first_header)

    meta_df = radar_sub[["radar_row", "radar_col", "Latitude", "Longitude"]].copy()
    meta_df["n_valid_hours"] = valid_count[selected_rows, selected_cols]
    meta_df["event_total_mm"] = out_arr[selected_rows, selected_cols]
    meta_df["cellsize_deg"] = float(first_header.cellsize)
    meta_df["xllcorner_lon"] = float(first_header.xllcorner)
    meta_df["yllcorner_lat"] = float(first_header.yllcorner)
    meta_df.to_csv(meta_path, index=False)

    used_vals = out_arr[np.isfinite(out_arr)]
    summary_df = pd.DataFrame([{
        "event_start_local": start.strftime("%Y-%m-%d %H:%M:%S"),
        "event_end_local": end.strftime("%Y-%m-%d %H:%M:%S"),
        "n_hours_with_file": int(len(parsed_times)),
        "ncols": int(first_header.ncols),
        "nrows": int(first_header.nrows),
        "cellsize_deg": float(first_header.cellsize),
        "xllcorner_lon": float(first_header.xllcorner),
        "yllcorner_lat": float(first_header.yllcorner),
        "nodata_value": float(first_header.nodata),
        "accumulation_seconds_header": (
            int(first_header.accumulation_seconds)
            if first_header.accumulation_seconds is not None else np.nan
        ),
        "n_selected_cells": int(len(radar_sub)),
        "min_total_mm": float(np.nanmin(used_vals)) if used_vals.size else np.nan,
        "max_total_mm": float(np.nanmax(used_vals)) if used_vals.size else np.nan,
        "mean_total_mm": float(np.nanmean(used_vals)) if used_vals.size else np.nan,
        "sum_total_mm_over_selected_cells": float(np.nansum(used_vals)) if used_vals.size else np.nan,
        "radar_dir": str(radar_dir),
    }])
    summary_df.to_csv(summary_path, index=False)

    print(f"[saved] {asc_path}")
    print(f"[saved] {meta_path}")
    print(f"[saved] {summary_path}")


def main() -> None:
    args = parse_args()

    if not args.radar_dir.exists():
        raise FileNotFoundError(f"Radar folder not found: {args.radar_dir}")

    start, end, _ = make_window(args.event_start, args.event_end)
    print(f"[window] {start} to {end}")

    build_total_raster(
        radar_dir=args.radar_dir,
        out_dir=args.out_dir,
        output_prefix=args.output_prefix,
        start=start,
        end=end,
        catchments=list(args.catchments),
        full_grid=args.full_grid,
    )


if __name__ == "__main__":
    main()