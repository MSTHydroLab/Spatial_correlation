#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LOCAL_TZ = "America/Chicago"

BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK")
DEFAULT_GRID_CSV = BASE_DIR / "dependent_files" / "grid_centers_wgs84.csv"
DEFAULT_EVENT_META_DIR = BASE_DIR / "01_Event_TimeSeries"
DEFAULT_OUT_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_IDW/Radar_Event_TimeSeries")
DEFAULT_PRODUCT_DIRS = [
    Path("/mnt/12TB/Sujan/Radar_products/Composite_2"),
    Path("/mnt/12TB/Sujan/Radar_products/Composite_3"),
    Path("/mnt/12TB/Sujan/Radar_products/RA"),
    Path("/mnt/12TB/Sujan/Radar_products/RKDP"),
    Path("/mnt/12TB/Sujan/Radar_products/RZ"),
]

GENERIC_FNAME_RE = re.compile(
    r".*?_(?P<accum_sec>\d+)_(?P<date>\d{2}[A-Z]{3}\d{4})_(?P<time>\d{6})\.out$",
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
    accumulation_sec: int | None
    source_file: Path

    @property
    def x_center0(self) -> float:
        return self.xllcorner + 0.5 * self.cellsize

    @property
    def y_center0_from_bottom(self) -> float:
        return self.yllcorner + 0.5 * self.cellsize

    @property
    def y_center_top(self) -> float:
        return self.yllcorner + (self.nrows - 0.5) * self.cellsize


def make_window(start_str: str, end_str: str) -> tuple[pd.Timestamp, pd.Timestamp, pd.DatetimeIndex]:
    start = pd.to_datetime(start_str, errors="raise")
    end = pd.to_datetime(end_str, errors="raise")

    if getattr(start, "tzinfo", None) is not None:
        start = start.tz_convert(LOCAL_TZ).tz_localize(None)
    if getattr(end, "tzinfo", None) is not None:
        end = end.tz_convert(LOCAL_TZ).tz_localize(None)

    if end < start:
        raise ValueError("event_end must be >= event_start")

    idx = pd.date_range(start, end, freq="1h")
    return start, end, idx


def load_event_window(event_number: int, event_meta_dir: Path) -> tuple[pd.Timestamp, pd.Timestamp, pd.DatetimeIndex]:
    fp = event_meta_dir / f"Event_{event_number}_Stations_correlation.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing event metadata file: {fp}")

    meta = pd.read_csv(fp)
    if "event_start" not in meta.columns or "event_end" not in meta.columns:
        raise ValueError(f"{fp} must contain event_start and event_end columns")

    start_str = str(meta["event_start"].dropna().iloc[0]).strip()
    end_str = str(meta["event_end"].dropna().iloc[0]).strip()
    return make_window(start_str, end_str)


def parse_radar_filename_time(path: Path) -> tuple[pd.Timestamp, int | None]:
    m = GENERIC_FNAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Unrecognized radar filename format: {path.name}")

    ts = pd.to_datetime(
        f"{m.group('date').upper()} {m.group('time')}",
        format="%d%b%Y %H%M%S",
        errors="raise",
    )
    return ts, int(m.group("accum_sec"))


def sanitize_product_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(name).strip())
    s = s.strip("_")
    return s or "radar"


def _parse_header_line_num(line: str) -> float:
    return float(line.split(":", 1)[1].strip())


def read_radar_header(path: Path) -> tuple[RadarHeader, int]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                lines.append(line.rstrip("\n"))
            else:
                break

    vals: dict[str, float | int | None] = {
        "ncols": None,
        "nrows": None,
        "xllcorner": None,
        "yllcorner": None,
        "cellsize": None,
        "nodata": None,
        "accumulation_sec": None,
    }

    for raw in lines:
        s = raw.strip()
        sl = s.lower()
        if sl.startswith("# accumulation time [sec]:"):
            vals["accumulation_sec"] = int(round(_parse_header_line_num(s)))
        elif sl.startswith("# number of columns:"):
            vals["ncols"] = int(round(_parse_header_line_num(s)))
        elif sl.startswith("# number of rows:"):
            vals["nrows"] = int(round(_parse_header_line_num(s)))
        elif sl.startswith("# xllcorner [lon]:"):
            vals["xllcorner"] = float(_parse_header_line_num(s))
        elif sl.startswith("# yllcorner [lat]:"):
            vals["yllcorner"] = float(_parse_header_line_num(s))
        elif sl.startswith("# cellsize [dec deg]:"):
            vals["cellsize"] = float(_parse_header_line_num(s))
        elif sl.startswith("# no data value:"):
            vals["nodata"] = float(_parse_header_line_num(s))

    missing = [k for k, v in vals.items() if k != "accumulation_sec" and v is None]
    if missing:
        raise ValueError(f"Header parse failed for {path}. Missing: {missing}")

    header = RadarHeader(
        ncols=int(vals["ncols"]),
        nrows=int(vals["nrows"]),
        xllcorner=float(vals["xllcorner"]),
        yllcorner=float(vals["yllcorner"]),
        cellsize=float(vals["cellsize"]),
        nodata=float(vals["nodata"]),
        accumulation_sec=int(vals["accumulation_sec"]) if vals["accumulation_sec"] is not None else None,
        source_file=path,
    )
    return header, len(lines)


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
            if isinstance(v0, float):
                if not math.isclose(v0, v1, rel_tol=0.0, abs_tol=1e-12):
                    mismatches.append((a, v0, v1))
            else:
                if v0 != v1:
                    mismatches.append((a, v0, v1))

        if mismatches:
            parts = ", ".join(f"{a}: {v0} != {v1}" for a, v0, v1 in mismatches)
            raise ValueError(f"Radar header mismatch between {baseline.source_file.name} and {p.name}: {parts}")

    if baseline is None:
        raise ValueError("No radar files found to validate")
    print(f"[header-check] validated {checked} files in {baseline.source_file.parent}")
    return baseline


def load_and_subset_grid(
    grid_csv: Path,
    start_lat: float,
    end_lat: float,
    start_lon: float,
    end_lon: float,
) -> pd.DataFrame:
    grid = pd.read_csv(grid_csv)
    req = ["id", "Latitude", "Longitude"]
    missing = [c for c in req if c not in grid.columns]
    if missing:
        raise ValueError(f"Grid CSV missing required columns: {missing}")

    lat_min, lat_max = sorted([start_lat, end_lat])
    lon_min, lon_max = sorted([start_lon, end_lon])
    sub = grid.loc[
        grid["Latitude"].between(lat_min, lat_max)
        & grid["Longitude"].between(lon_min, lon_max),
        req,
    ].copy()
    if sub.empty:
        raise ValueError("No grid centers found inside the requested lat/lon box")

    sub["id"] = pd.to_numeric(sub["id"], errors="coerce").astype("Int64")
    sub = sub.dropna(subset=["id"]).copy()
    sub["id"] = sub["id"].astype(int).astype(str)
    sub = sub.sort_values(["Latitude", "Longitude", "id"]).reset_index(drop=True)
    return sub


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
            "Some selected grid centers do not map cleanly to radar cells. "
            f"Count={len(bad_df)}. First few:\n{bad_df.head(10).to_string(index=False)}"
        )

    out = grid_df.copy()
    out["radar_col"] = cols
    out["radar_row"] = rows
    out["radar_lon_center"] = lon_center
    out["radar_lat_center"] = lat_center
    out["lon_offset_deg"] = grid_df["Longitude"].to_numpy(float) - lon_center
    out["lat_offset_deg"] = grid_df["Latitude"].to_numpy(float) - lat_center
    return out


def discover_files_in_window(prod_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> dict[pd.Timestamp, Path]:
    found: dict[pd.Timestamp, Path] = {}
    for p in sorted(prod_dir.glob("*.out")):
        try:
            t_local, _ = parse_radar_filename_time(p)
        except Exception:
            continue
        if start <= t_local <= end:
            found[t_local] = p
    return found


def read_selected_cells(path: Path, header: RadarHeader, skiprows: int, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    arr = np.loadtxt(path, dtype=float, comments=None, skiprows=skiprows)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape != (header.nrows, header.ncols):
        raise ValueError(
            f"Unexpected array shape in {path}: got {arr.shape}, expected {(header.nrows, header.ncols)}"
        )
    vals = arr[rows, cols].astype(float)
    vals[np.isclose(vals, header.nodata, rtol=0.0, atol=1e-12)] = np.nan
    return vals


def build_event_cube(
    event_number: int,
    prod_dir: Path,
    out_dir: Path,
    grid_map: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    event_idx: pd.DatetimeIndex,
) -> None:
    files = discover_files_in_window(prod_dir, start, end)
    if not files:
        raise FileNotFoundError(f"No radar files found in window for event {event_number} under {prod_dir}")

    first_header = validate_headers(files.values())
    grid_map = add_radar_indices(grid_map, first_header)

    rows = grid_map["radar_row"].to_numpy(int)
    cols = grid_map["radar_col"].to_numpy(int)
    grid_ids = grid_map["id"].tolist()
    product_name = sanitize_product_name(prod_dir.name)

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
                raise ValueError(f"Header changed within {prod_dir} for {fp.name}: field {a} differs")

        out[i, :] = read_selected_cells(fp, h, skiprows, rows, cols)
        present_files += 1
        parsed_times.append(ts)

    out_df = pd.DataFrame(out, index=event_idx, columns=grid_ids)
    out_df.insert(0, "time_local", event_idx.astype(str))

    meta_df = grid_map.copy()
    meta_df.insert(1, "product_name", product_name)
    meta_df.insert(2, "event", int(event_number))

    missing_hours = sorted(set(event_idx) - set(parsed_times))
    missing_df = pd.DataFrame({
        "time_local": [t.strftime("%Y-%m-%d %H:%M:%S") for t in missing_hours],
        "reason": "file_not_found_in_product_folder",
    })

    summary_df = pd.DataFrame([{
        "event": int(event_number),
        "product_name": product_name,
        "event_start_local": start.strftime("%Y-%m-%d %H:%M:%S"),
        "event_end_local": end.strftime("%Y-%m-%d %H:%M:%S"),
        "n_event_hours_expected": int(len(event_idx)),
        "n_hours_with_file": int(present_files),
        "n_hours_missing_file": int(len(missing_hours)),
        "n_grid_cells": int(len(grid_map)),
        "cellsize_deg": float(first_header.cellsize),
        "ncols": int(first_header.ncols),
        "nrows": int(first_header.nrows),
        "xllcorner_lon": float(first_header.xllcorner),
        "yllcorner_lat": float(first_header.yllcorner),
        "nodata_value": float(first_header.nodata),
        "accumulation_sec_header": int(first_header.accumulation_sec) if first_header.accumulation_sec is not None else np.nan,
        "source_dir": str(prod_dir),
    }])

    event_dir = out_dir / f"Event_{event_number}"
    event_dir.mkdir(parents=True, exist_ok=True)

    out_rain = event_dir / f"Event_{event_number}_grid_rain_hourly_mm_{product_name}.csv"
    out_meta = event_dir / f"Event_{event_number}_grid_metadata_{product_name}.csv"
    out_missing = event_dir / f"Event_{event_number}_missing_hours_{product_name}.csv"
    out_summary = event_dir / f"Event_{event_number}_summary_{product_name}.csv"

    out_df.to_csv(out_rain, index=False)
    meta_df.to_csv(out_meta, index=False)
    missing_df.to_csv(out_missing, index=False)
    summary_df.to_csv(out_summary, index=False)

    print(f"[saved] {out_rain}")
    print(f"[saved] {out_meta}")
    print(f"[saved] {out_missing}")
    print(f"[saved] {out_summary}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Extract radar time series at existing WGS grid centers for a given event. Supports rainfall composites and other radar products."
    )
    ap.add_argument("--event", type=int, required=True, help="Event number, e.g. 2")
    ap.add_argument("--grid-csv", default=str(DEFAULT_GRID_CSV))
    ap.add_argument("--event-meta-dir", default=str(DEFAULT_EVENT_META_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument(
        "--product-dirs",
        nargs="+",
        default=[str(p) for p in DEFAULT_PRODUCT_DIRS],
        help="One or more radar product folders. Separate outputs are written for each folder.",
    )
    ap.add_argument("--start-lat", type=float, default=38.7293249)
    ap.add_argument("--end-lat", type=float, default=39.0418499)
    ap.add_argument("--start-lon", type=float, default=-94.899835)
    ap.add_argument("--end-lon", type=float, default=-94.591477)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    grid_csv = Path(args.grid_csv)
    event_meta_dir = Path(args.event_meta_dir)
    out_root = Path(args.out_dir)
    product_dirs = [Path(p) for p in args.product_dirs]

    start, end, event_idx = load_event_window(args.event, event_meta_dir)
    print(f"[event] {args.event}")
    print(f"[window] {start} to {end} ({len(event_idx)} hourly steps)")

    grid_sub = load_and_subset_grid(
        grid_csv=grid_csv,
        start_lat=args.start_lat,
        end_lat=args.end_lat,
        start_lon=args.start_lon,
        end_lon=args.end_lon,
    )
    print(f"[grid] selected {len(grid_sub)} grid centers from {grid_csv}")

    for prod_dir in product_dirs:
        if not prod_dir.exists():
            raise FileNotFoundError(f"Radar product folder not found: {prod_dir}")
        print(f"\n[product] processing {prod_dir}")
        build_event_cube(
            event_number=args.event,
            prod_dir=prod_dir,
            out_dir=out_root / sanitize_product_name(prod_dir.name),
            grid_map=grid_sub,
            start=start,
            end=end,
            event_idx=event_idx,
        )


if __name__ == "__main__":
    main()
