#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

LOCAL_TZ = "America/Chicago"

'''
python /mnt/12TB/Sujan/Spatial_correlation/Codes/extract_mrms_grib_grid_timeseries.py \
  --event 7 \
  --var-name unknown \
  --negative-mode nan \
  --reader auto \
  --progress \
  --debug
'''
# Default paths for the /mnt server layout.
# You can still override any of these from the command line.
BASE_CODES_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes")
BASE_OK_DIR = BASE_CODES_DIR / "WGS_OK"

DEFAULT_MRMS_DIR = Path("/mnt/12TB/Hari/MRMS_gribs")
DEFAULT_DEP_DIR = BASE_OK_DIR / "dependent_files"
DEFAULT_GRID_CSV = DEFAULT_DEP_DIR / "grid_centers_wgs84.csv"
DEFAULT_EVENT_META_DIR = BASE_OK_DIR / "01_Event_TimeSeries"
DEFAULT_CATCHMENTS = [
    BASE_CODES_DIR / "dependent_files" / "6892513" / "6892513.shp",
    BASE_CODES_DIR / "dependent_files" / "06893080" / "6893080.shp",
    BASE_CODES_DIR / "dependent_files" / "6893390" / "6893390.shp",
]
DEFAULT_OUT_DIR = BASE_OK_DIR / "Radar_Event_TimeSeries" / "MRMS_Event_TimeSeries"

# Anchored. This intentionally ignores cropped_20170726-22.grib2 unless --allow-cropped is used.
FILENAME_RE = re.compile(r"^(?P<ymd>\d{8})[-_](?P<hour>\d{2})\.grib2?$", re.IGNORECASE)


@dataclass(frozen=True)
class EventWindow:
    event: int
    start_local: pd.Timestamp
    end_local: pd.Timestamp
    hourly_index_local: pd.DatetimeIndex


@dataclass(frozen=True)
class MrmsGridInfo:
    lat: np.ndarray
    lon: np.ndarray
    y_dim: str
    x_dim: str
    lat_name: str
    lon_name: str
    var_name: str
    raw_lon_min: float
    raw_lon_max: float
    norm_lon_min: float
    norm_lon_max: float
    lat_min: float
    lat_max: float


@dataclass(frozen=True)
class DomainBbox:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


def normalize_grid_id(value) -> str:
    s = str(value).strip()
    try:
        f = float(s)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
    except Exception:
        pass
    return s


def normalize_lon_to_180(lon: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon, dtype=float)
    if np.nanmax(lon) > 180.0:
        return ((lon + 180.0) % 360.0) - 180.0
    return lon


def lon_to_360(lon: float) -> float:
    return float(lon % 360.0)


def parse_file_time(path: Path, file_timezone: str, output_timezone: str) -> pd.Timestamp:
    m = FILENAME_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse MRMS filename time: {path.name}")
    naive = pd.to_datetime(f"{m.group('ymd')} {m.group('hour')}:00:00", format="%Y%m%d %H:%M:%S")
    src = naive.tz_localize(file_timezone)
    return src.tz_convert(output_timezone).tz_localize(None)


def make_window(start_str: str, end_str: str, event: int, local_tz: str) -> EventWindow:
    start = pd.to_datetime(start_str, errors="raise")
    end = pd.to_datetime(end_str, errors="raise")
    if getattr(start, "tzinfo", None) is not None:
        start = start.tz_convert(local_tz).tz_localize(None)
    if getattr(end, "tzinfo", None) is not None:
        end = end.tz_convert(local_tz).tz_localize(None)
    if end < start:
        raise ValueError("--end must be >= --start")
    return EventWindow(event=event, start_local=start, end_local=end, hourly_index_local=pd.date_range(start, end, freq="1h"))


def load_event_window_from_meta(event: int, event_meta_dir: Path, local_tz: str) -> EventWindow:
    fp = event_meta_dir / f"Event_{event}_Stations_correlation.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Event metadata not found: {fp}. Use --start and --end instead.")
    meta = pd.read_csv(fp)
    required = ["event_start", "event_end"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(f"{fp} missing required columns: {missing}")
    return make_window(str(meta["event_start"].dropna().iloc[0]).strip(), str(meta["event_end"].dropna().iloc[0]).strip(), event, local_tz)


def load_catchments(paths: Iterable[Path]) -> gpd.GeoDataFrame:
    rows = []
    for shp in paths:
        shp = Path(shp)
        if not shp.exists():
            raise FileNotFoundError(f"Missing catchment shapefile: {shp}")
        gdf = gpd.read_file(shp)
        if gdf.empty:
            continue
        if gdf.crs is None:
            raise ValueError(f"Catchment shapefile has no CRS: {shp}")
        gdf = gdf.to_crs("EPSG:4326")
        geom = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union
        rows.append({"catchment": shp.stem, "path": str(shp), "geometry": geom})
    if not rows:
        raise ValueError("No valid catchment polygons were loaded.")
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def infer_grid_spacing(values: pd.Series) -> float:
    vals = np.sort(pd.to_numeric(values, errors="coerce").dropna().unique())
    diffs = np.diff(vals)
    diffs = diffs[diffs > 1e-12]
    if diffs.size == 0:
        raise ValueError("Could not infer grid spacing from grid centers.")
    return float(np.median(diffs))


def load_grid_cells(grid_csv: Path) -> gpd.GeoDataFrame:
    grid = pd.read_csv(grid_csv)
    required = ["id", "Latitude", "Longitude"]
    missing = [c for c in required if c not in grid.columns]
    if missing:
        raise ValueError(f"{grid_csv} missing required columns: {missing}")
    grid = grid[required].copy()
    grid["id"] = grid["id"].apply(normalize_grid_id)
    grid["Latitude"] = pd.to_numeric(grid["Latitude"], errors="coerce")
    grid["Longitude"] = pd.to_numeric(grid["Longitude"], errors="coerce")
    grid = grid.dropna(subset=["id", "Latitude", "Longitude"]).copy()
    dlat = infer_grid_spacing(grid["Latitude"])
    dlon = infer_grid_spacing(grid["Longitude"])
    grid["geometry"] = [
        box(lon - 0.5 * dlon, lat - 0.5 * dlat, lon + 0.5 * dlon, lat + 0.5 * dlat)
        for lat, lon in zip(grid["Latitude"], grid["Longitude"])
    ]
    grid["cell_height_deg"] = dlat
    grid["cell_width_deg"] = dlon
    return gpd.GeoDataFrame(grid, geometry="geometry", crs="EPSG:4326")


def select_catchment_cells(grid_cells: gpd.GeoDataFrame, catchments: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict[str, list[str]]]:
    catch_union = catchments.geometry.union_all() if hasattr(catchments.geometry, "union_all") else catchments.geometry.unary_union
    selected = grid_cells.loc[grid_cells.geometry.intersects(catch_union)].copy()
    if selected.empty:
        raise ValueError("No grid cells intersect the catchment polygons.")
    masks: dict[str, list[str]] = {}
    for _, row in catchments.iterrows():
        ids = selected.loc[selected.geometry.intersects(row.geometry), "id"].astype(str).tolist()
        masks[str(row["catchment"])] = ids
    return selected.reset_index(drop=True), masks


def compute_selected_bbox(selected_cells: gpd.GeoDataFrame, pad_deg: float) -> DomainBbox:
    minx, miny, maxx, maxy = selected_cells.total_bounds
    return DomainBbox(
        lon_min=float(minx - pad_deg),
        lon_max=float(maxx + pad_deg),
        lat_min=float(miny - pad_deg),
        lat_max=float(maxy + pad_deg),
    )


def discover_files(mrms_dir: Path, window: EventWindow, file_timezone: str, output_timezone: str, allow_cropped: bool, debug: bool) -> dict[pd.Timestamp, Path]:
    out: dict[pd.Timestamp, Path] = {}
    skipped_cropped = 0
    skipped_unparsed = 0
    for p in sorted(mrms_dir.glob("*.grib2")):
        if (not allow_cropped) and p.name.lower().startswith("cropped_"):
            skipped_cropped += 1
            continue
        try:
            t_local = parse_file_time(p, file_timezone=file_timezone, output_timezone=output_timezone)
        except Exception:
            skipped_unparsed += 1
            continue
        if window.start_local <= t_local <= window.end_local:
            out.setdefault(t_local.floor("h"), p)
    if debug:
        print(f"[debug] Skipped cropped files: {skipped_cropped}")
        print(f"[debug] Skipped unparsed GRIB files: {skipped_unparsed}")
    return dict(sorted(out.items()))


def open_grib(path: Path) -> xr.Dataset:
    # A unique cfgrib index path avoids stale index issues on shared servers.
    tmp = tempfile.NamedTemporaryFile(prefix="cfgrib_", suffix=".idx", delete=True)
    tmp.close()
    try:
        return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": tmp.name})
    except Exception:
        return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})


def choose_var(ds: xr.Dataset, var_name: str | None) -> str:
    if var_name:
        if var_name not in ds.data_vars:
            raise ValueError(f"Variable {var_name!r} not found. Available: {list(ds.data_vars)}")
        return var_name
    candidates = [name for name, da in ds.data_vars.items() if da.ndim >= 2 and np.issubdtype(da.dtype, np.number)]
    if not candidates:
        raise ValueError(f"No numeric 2D data variable found. Available: {list(ds.data_vars)}")
    return candidates[0]


def find_lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    lat_name = next((c for c in ["latitude", "lat", "Latitude", "LAT"] if c in ds.variables), None)
    lon_name = next((c for c in ["longitude", "lon", "Longitude", "LON"] if c in ds.variables), None)
    if lat_name is None or lon_name is None:
        raise ValueError(f"Could not find latitude/longitude variables. Variables: {list(ds.variables)}")
    return lat_name, lon_name


def read_mrms_grid_info(path: Path, var_name: str | None, verbose: bool) -> MrmsGridInfo:
    with open_grib(path) as ds:
        chosen_var = choose_var(ds, var_name)
        lat_name, lon_name = find_lat_lon_names(ds)
        da = ds[chosen_var]
        lat_raw = np.asarray(ds[lat_name].values, dtype=float)
        lon_raw = np.asarray(ds[lon_name].values, dtype=float)
        raw_lon_min = float(np.nanmin(lon_raw))
        raw_lon_max = float(np.nanmax(lon_raw))
        lon_norm = normalize_lon_to_180(lon_raw)
        if lat_raw.ndim == 1 and lon_norm.ndim == 1:
            y_dim = ds[lat_name].dims[0]
            x_dim = ds[lon_name].dims[0]
            lon2, lat2 = np.meshgrid(lon_norm, lat_raw)
            lat = lat2
            lon = lon2
        elif lat_raw.ndim == 2 and lon_norm.ndim == 2:
            y_dim, x_dim = ds[lat_name].dims
            lat = lat_raw
            lon = lon_norm
        else:
            raise ValueError("Latitude/longitude must both be 1D or both be 2D.")
        if y_dim not in da.dims or x_dim not in da.dims:
            y_dim, x_dim = da.dims[-2], da.dims[-1]
        if verbose:
            print("\n[debug] Available GRIB variables:")
            for name, data_array in ds.data_vars.items():
                print(f"  {name}: dims={data_array.dims}, shape={data_array.shape}, attrs={dict(data_array.attrs)}")
            print(f"[debug] Using MRMS variable: {chosen_var}")
            print(f"[debug] Raw longitude range: {raw_lon_min} to {raw_lon_max}")
            print(f"[debug] Normalized longitude range: {float(np.nanmin(lon))} to {float(np.nanmax(lon))}")
            print(f"[debug] Latitude range: {float(np.nanmin(lat))} to {float(np.nanmax(lat))}")
    return MrmsGridInfo(
        lat=lat,
        lon=lon,
        y_dim=y_dim,
        x_dim=x_dim,
        lat_name=lat_name,
        lon_name=lon_name,
        var_name=chosen_var,
        raw_lon_min=raw_lon_min,
        raw_lon_max=raw_lon_max,
        norm_lon_min=float(np.nanmin(lon)),
        norm_lon_max=float(np.nanmax(lon)),
        lat_min=float(np.nanmin(lat)),
        lat_max=float(np.nanmax(lat)),
    )


def build_nearest_lookup(selected_cells: pd.DataFrame, mrms_info: MrmsGridInfo) -> pd.DataFrame:
    target = selected_cells[["Longitude", "Latitude"]].to_numpy(dtype=float)
    source = np.column_stack([mrms_info.lon.ravel(), mrms_info.lat.ravel()])
    finite = np.isfinite(source).all(axis=1)
    source_valid = source[finite]
    flat_valid = np.flatnonzero(finite)
    if cKDTree is not None:
        tree = cKDTree(source_valid)
        dist_deg, idx_valid = tree.query(target, k=1)
    else:
        idx_valid = []
        dist_deg = []
        for pt in target:
            d2 = np.sum((source_valid - pt) ** 2, axis=1)
            i = int(np.argmin(d2))
            idx_valid.append(i)
            dist_deg.append(float(np.sqrt(d2[i])))
        idx_valid = np.asarray(idx_valid, dtype=int)
        dist_deg = np.asarray(dist_deg, dtype=float)
    flat_idx = flat_valid[idx_valid]
    rows, cols = np.unravel_index(flat_idx, mrms_info.lat.shape)
    out = selected_cells[["id", "Latitude", "Longitude"]].copy()
    out["mrms_row"] = rows.astype(int)
    out["mrms_col"] = cols.astype(int)
    out["mrms_lat"] = mrms_info.lat[rows, cols]
    out["mrms_lon"] = mrms_info.lon[rows, cols]
    out["nearest_distance_deg"] = dist_deg
    return out


def apply_negative_mode(vals: np.ndarray, mode: str) -> np.ndarray:
    out = np.asarray(vals, dtype=float).copy()
    out[~np.isfinite(out)] = np.nan
    if mode == "nan":
        out[out < 0] = np.nan
    elif mode == "zero":
        out[out < 0] = 0.0
    elif mode == "keep":
        pass
    else:
        raise ValueError(f"Unknown negative mode: {mode}")
    return out


def read_grid_array(path: Path, mrms_info: MrmsGridInfo) -> np.ndarray:
    with open_grib(path) as ds:
        arr = np.asarray(ds[mrms_info.var_name].squeeze().values)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[-2], arr.shape[-1])
    return arr.astype(float, copy=False)


def read_values_nearest(path: Path, mrms_info: MrmsGridInfo, lookup: pd.DataFrame, negative_mode: str) -> tuple[np.ndarray, dict[str, float]]:
    arr = read_grid_array(path, mrms_info)
    vals = arr[lookup["mrms_row"].to_numpy(dtype=int), lookup["mrms_col"].to_numpy(dtype=int)]
    vals = apply_negative_mode(vals, negative_mode)
    stats = {
        "sampled_min": float(np.nanmin(vals)) if np.isfinite(vals).any() else np.nan,
        "sampled_max": float(np.nanmax(vals)) if np.isfinite(vals).any() else np.nan,
        "sampled_sum": float(np.nansum(vals)),
        "sampled_nonzero_count": int(np.nansum(vals > 0)),
    }
    return vals, stats


def write_mask_tables(out_dir: Path, selected_cells: gpd.GeoDataFrame, catchment_masks: dict[str, list[str]], lookup: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    domain_ids = set(selected_cells["id"].astype(str))
    lookup_small = lookup.set_index("id")
    rows = []
    for _, row in selected_cells.iterrows():
        gid = str(row["id"])
        rec = {
            "grid_id": gid,
            "Latitude": float(row["Latitude"]),
            "Longitude": float(row["Longitude"]),
            "in_any_watershed": gid in domain_ids,
        }
        for name, ids in catchment_masks.items():
            rec[f"in_{name}"] = gid in set(ids)
        if gid in lookup_small.index:
            hit = lookup_small.loc[gid]
            rec.update({
                "mrms_row": int(hit["mrms_row"]),
                "mrms_col": int(hit["mrms_col"]),
                "mrms_lat": float(hit["mrms_lat"]),
                "mrms_lon": float(hit["mrms_lon"]),
                "nearest_distance_deg": float(hit["nearest_distance_deg"]),
            })
        rows.append(rec)
    pd.DataFrame(rows).to_csv(out_dir / "watershed_cell_mask.csv", index=False)
    summary = [{"mask_name": "all_watersheds_union", "n_cells": len(domain_ids)}]
    for name, ids in catchment_masks.items():
        summary.append({"mask_name": name, "n_cells": len(ids)})
    pd.DataFrame(summary).to_csv(out_dir / "watershed_mask_summary.csv", index=False)


def build_catchment_summary(rain_df: pd.DataFrame, catchment_masks: dict[str, list[str]]) -> pd.DataFrame:
    rows = pd.DataFrame({"time_local": rain_df.index})
    available = set(rain_df.columns.astype(str))
    for name, ids in catchment_masks.items():
        cols = [str(gid) for gid in ids if str(gid) in available]
        sub = rain_df[cols] if cols else pd.DataFrame(index=rain_df.index)
        rows[f"{name}_n_cells"] = len(cols)
        rows[f"{name}_sum_mm_over_cells"] = sub.sum(axis=1, skipna=True).values if cols else np.nan
        rows[f"{name}_mean_mm"] = sub.mean(axis=1, skipna=True).values if cols else np.nan
    return rows


def get_wgrib2_path(user_path: str | None) -> str | None:
    if user_path:
        p = shutil.which(user_path) or user_path
        return p if Path(p).exists() or shutil.which(p) else None
    return shutil.which("wgrib2")


def crop_with_wgrib2(src: Path, dst: Path, bbox: DomainBbox, source_uses_360_lon: bool, wgrib2_path: str, debug: bool) -> bool:
    if source_uses_360_lon:
        lon1 = lon_to_360(bbox.lon_min)
        lon2 = lon_to_360(bbox.lon_max)
    else:
        lon1 = bbox.lon_min
        lon2 = bbox.lon_max
    lon_min, lon_max = sorted([lon1, lon2])
    lat_min, lat_max = sorted([bbox.lat_min, bbox.lat_max])
    cmd = [
        wgrib2_path,
        str(src),
        "-small_grib",
        f"{lon_min:.6f}:{lon_max:.6f}",
        f"{lat_min:.6f}:{lat_max:.6f}",
        str(dst),
    ]
    if debug:
        print("[debug] crop:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        if debug:
            print(f"[warning] wgrib2 crop failed for {src.name}: {proc.stderr.strip() or proc.stdout.strip()}")
        return False
    return dst.exists() and dst.stat().st_size > 0


def prepare_reader_first_file(first_file: Path, selected_cells: gpd.GeoDataFrame, bbox: DomainBbox, args: argparse.Namespace, event_out: Path) -> tuple[Path, tempfile.TemporaryDirectory | None, Path | None, bool]:
    """Return file to use for grid metadata plus temp state.

    The fastest path is to make temporary basin-sized GRIB files with wgrib2. This does not use
    your existing cropped_* files. It creates temporary crops from the full-domain files and deletes
    them unless --keep-temp-crops is used.
    """
    if args.reader == "cfgrib_full":
        return first_file, None, None, False

    wgrib2_path = get_wgrib2_path(args.wgrib2)
    if wgrib2_path is None:
        if args.reader == "wgrib2_crop":
            raise RuntimeError("--reader wgrib2_crop requested, but wgrib2 was not found in PATH. Install wgrib2 or use --reader cfgrib_full.")
        print("[warning] wgrib2 not found. Falling back to full-domain cfgrib reads, which can be very slow.")
        return first_file, None, None, False

    tmp_manager = None
    if args.keep_temp_crops:
        crop_dir = event_out / "_tmp_mrms_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp_manager = tempfile.TemporaryDirectory(prefix="mrms_crop_")
        crop_dir = Path(tmp_manager.name)

    # Need one quick metadata read on the full file to know whether the source uses 0..360 longitude.
    full_info = read_mrms_grid_info(first_file, var_name=args.var_name, verbose=args.debug)
    source_uses_360_lon = full_info.raw_lon_max > 180.0
    first_crop = crop_dir / first_file.name
    ok = crop_with_wgrib2(first_file, first_crop, bbox, source_uses_360_lon, wgrib2_path, args.debug)
    if not ok:
        if args.reader == "wgrib2_crop":
            raise RuntimeError(f"wgrib2 crop failed for first file: {first_file}")
        print("[warning] wgrib2 crop failed. Falling back to full-domain cfgrib reads.")
        if tmp_manager is not None:
            tmp_manager.cleanup()
        return first_file, None, None, False
    return first_crop, tmp_manager, crop_dir, source_uses_360_lon


def run_event(args: argparse.Namespace, window: EventWindow) -> None:
    event_out = Path(args.out_dir) / f"Event_{window.event}"
    event_out.mkdir(parents=True, exist_ok=True)

    files = discover_files(Path(args.mrms_dir), window, args.file_timezone, args.local_tz, args.allow_cropped, args.debug)
    if not files:
        raise FileNotFoundError(f"No full-domain MRMS GRIB2 files found in {args.mrms_dir} for {window.start_local} to {window.end_local}.")

    print(f"[debug] Files found in event window: {len(files)}")
    print(f"[debug] First file in window: {next(iter(files.values()))}")

    catchments = load_catchments(args.catchments)
    all_cells = load_grid_cells(Path(args.grid_csv))
    selected_cells, catchment_masks = select_catchment_cells(all_cells, catchments)
    bbox = compute_selected_bbox(selected_cells, pad_deg=args.crop_pad_deg)
    print(f"[debug] Selected WGS cells: {len(selected_cells)}")
    print(f"[debug] Basin crop bbox: lon {bbox.lon_min:.6f} to {bbox.lon_max:.6f}, lat {bbox.lat_min:.6f} to {bbox.lat_max:.6f}")

    first_full_file = next(iter(files.values()))
    first_read_file, tmp_manager, crop_dir, source_uses_360_lon = prepare_reader_first_file(first_full_file, selected_cells, bbox, args, event_out)

    try:
        mrms_info = read_mrms_grid_info(first_read_file, var_name=args.var_name, verbose=args.debug)
        lookup = build_nearest_lookup(selected_cells, mrms_info)
        write_mask_tables(event_out, selected_cells, catchment_masks, lookup)
        print(f"[debug] Maximum WGS-grid to MRMS nearest distance in degrees: {float(lookup['nearest_distance_deg'].max())}")
        print(f"[debug] Mean WGS-grid to MRMS nearest distance in degrees: {float(lookup['nearest_distance_deg'].mean())}")
        if float(lookup["nearest_distance_deg"].max()) > 0.05:
            print("[warning] Large nearest distance. Check MRMS domain, crop bbox, or grid CSV.")

        grid_ids = lookup["id"].astype(str).tolist()
        out_values = []
        status_rows = []
        wgrib2_path = get_wgrib2_path(args.wgrib2)
        use_crop = crop_dir is not None

        for i, ts in enumerate(window.hourly_index_local, start=1):
            f = files.get(ts.floor("h"))
            if f is None:
                out_values.append(np.full(len(grid_ids), np.nan, dtype=float))
                status_rows.append({"time_local": ts, "file": "", "read_file": "", "status": "missing_file"})
                continue

            read_file = f
            if use_crop:
                assert crop_dir is not None and wgrib2_path is not None
                read_file = crop_dir / f.name
                # First crop may already exist. Other hours are generated here.
                if not read_file.exists():
                    ok = crop_with_wgrib2(f, read_file, bbox, source_uses_360_lon, wgrib2_path, args.debug)
                    if not ok:
                        out_values.append(np.full(len(grid_ids), np.nan, dtype=float))
                        status_rows.append({"time_local": ts, "file": str(f), "read_file": str(read_file), "status": "crop_error"})
                        continue

            try:
                vals, stats = read_values_nearest(read_file, mrms_info=mrms_info, lookup=lookup, negative_mode=args.negative_mode)
                out_values.append(vals)
                status_rows.append({"time_local": ts, "file": str(f), "read_file": str(read_file), "status": "ok", **stats})
                if args.progress:
                    print(f"[{i:03d}/{len(window.hourly_index_local):03d}] {ts} ok sum={stats['sampled_sum']:.3f} max={stats['sampled_max']:.3f}", flush=True)
            except Exception as e:
                out_values.append(np.full(len(grid_ids), np.nan, dtype=float))
                status_rows.append({"time_local": ts, "file": str(f), "read_file": str(read_file), "status": f"read_error: {e}"})
                print(f"[warning] read error for {ts}: {e}", flush=True)

        rain = pd.DataFrame(out_values, index=window.hourly_index_local, columns=grid_ids)
        rain.index.name = "time_local"
        if args.fill_missing_with_zero:
            rain = rain.fillna(0.0)

        out_csv = event_out / f"Event_{window.event}_grid_rain_hourly_mm_MRMS.csv"
        rain.reset_index().to_csv(out_csv, index=False)

        summary_csv = event_out / f"Event_{window.event}_MRMS_hourly_catchment_summary.csv"
        build_catchment_summary(rain, catchment_masks).to_csv(summary_csv, index=False)

        status_csv = event_out / f"Event_{window.event}_MRMS_file_status.csv"
        pd.DataFrame(status_rows).to_csv(status_csv, index=False)

        arr = rain.to_numpy(dtype=float)
        meta_csv = event_out / f"Event_{window.event}_MRMS_run_metadata.csv"
        pd.DataFrame([{
            "event": window.event,
            "event_start_local": window.start_local,
            "event_end_local": window.end_local,
            "n_requested_hours": len(window.hourly_index_local),
            "n_files_found_in_window": len(files),
            "n_selected_grid_cells": len(grid_ids),
            "mrms_dir": str(args.mrms_dir),
            "grid_csv": str(args.grid_csv),
            "mrms_variable": mrms_info.var_name,
            "file_timezone_assumed": args.file_timezone,
            "local_timezone": args.local_tz,
            "reader": args.reader,
            "used_wgrib2_temp_crop": bool(use_crop),
            "crop_pad_deg": args.crop_pad_deg,
            "negative_mode": args.negative_mode,
            "overall_output_sum": float(np.nansum(arr)),
            "overall_output_max": float(np.nanmax(arr)) if np.isfinite(arr).any() else np.nan,
            "source_lon_min_raw": mrms_info.raw_lon_min,
            "source_lon_max_raw": mrms_info.raw_lon_max,
            "source_lon_min_norm": mrms_info.norm_lon_min,
            "source_lon_max_norm": mrms_info.norm_lon_max,
            "source_lat_min": mrms_info.lat_min,
            "source_lat_max": mrms_info.lat_max,
            "note": "Full-domain cropped_* input files are ignored by default. If wgrib2 is available, temporary basin-sized GRIBs are created from full-domain files only and deleted unless --keep-temp-crops is used.",
        }]).to_csv(meta_csv, index=False)

        print("Saved:")
        print(f"  {out_csv}")
        print(f"  {summary_csv}")
        print(f"  {event_out / 'watershed_cell_mask.csv'}")
        print(f"  {status_csv}")
        print(f"  {meta_csv}")
        print("\n[debug] Output check:")
        print(f"  overall output sum: {float(np.nansum(arr))}")
        print(f"  overall output max: {float(np.nanmax(arr)) if np.isfinite(arr).any() else np.nan}")
        print(f"  nonzero count: {int(np.nansum(arr > 0))}")
    finally:
        if tmp_manager is not None:
            tmp_manager.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fast MRMS GRIB2 to WGS-grid rainfall time series extractor for "
            "catchment-intersecting cells. Event start/end are read automatically "
            "from Event_X_Stations_correlation.csv unless --start/--end are provided."
        )
    )
    parser.add_argument(
        "--event",
        type=int,
        nargs="+",
        required=True,
        help="One or more event numbers, e.g. --event 4 or --event 4 7. Start/end are read from event metadata by default.",
    )
    parser.add_argument(
        "--start",
        default="",
        help="Optional local event start. Only allowed when one event is supplied. If omitted, read from event metadata.",
    )
    parser.add_argument(
        "--end",
        default="",
        help="Optional local event end. Only allowed when one event is supplied. If omitted, read from event metadata.",
    )
    parser.add_argument(
        "--event-meta-dir",
        type=Path,
        default=DEFAULT_EVENT_META_DIR,
        help="Folder containing Event_X_Stations_correlation.csv files with event_start and event_end columns.",
    )
    parser.add_argument("--mrms-dir", type=Path, default=DEFAULT_MRMS_DIR)
    parser.add_argument("--grid-csv", type=Path, default=DEFAULT_GRID_CSV)
    parser.add_argument("--catchments", nargs="+", type=Path, default=DEFAULT_CATCHMENTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--var-name", default="", help="GRIB variable name. Your files show this as 'unknown'.")
    parser.add_argument("--file-timezone", default="UTC")
    parser.add_argument("--local-tz", default=LOCAL_TZ)
    parser.add_argument("--negative-mode", choices=["nan", "zero", "keep"], default="nan")
    parser.add_argument("--fill-missing-with-zero", action="store_true")
    parser.add_argument("--allow-cropped", action="store_true", help="Allow existing cropped_*.grib2 inputs. Default is false.")
    parser.add_argument(
        "--reader",
        choices=["auto", "wgrib2_crop", "cfgrib_full"],
        default="auto",
        help="auto uses temporary wgrib2 crops when wgrib2 exists, otherwise falls back to slow full-domain cfgrib.",
    )
    parser.add_argument("--wgrib2", default="", help="Optional path to wgrib2 executable.")
    parser.add_argument("--crop-pad-deg", type=float, default=0.10, help="Padding around selected WGS cells for temporary wgrib2 crop.")
    parser.add_argument("--keep-temp-crops", action="store_true", help="Keep internally generated temporary crops under Event_X/_tmp_mrms_crops for debugging.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--progress", action="store_true", help="Print one progress line per hour.")
    args = parser.parse_args()
    args.var_name = args.var_name.strip() or None
    args.wgrib2 = args.wgrib2.strip() or None
    return args

def build_window_for_event(args: argparse.Namespace, event: int) -> EventWindow:
    if args.start or args.end:
        if not (args.start and args.end):
            raise SystemExit("Provide both --start and --end, or omit both to read from event metadata.")
        if len(args.event) != 1:
            raise SystemExit("--start/--end can only be used when one event is provided.")
        return make_window(args.start, args.end, event=event, local_tz=args.local_tz)

    return load_event_window_from_meta(
        event=event,
        event_meta_dir=Path(args.event_meta_dir),
        local_tz=args.local_tz,
    )


def main() -> None:
    args = parse_args()

    for event in args.event:
        window = build_window_for_event(args, event)

        print("=" * 80)
        print(f"Event {window.event}: {window.start_local} to {window.end_local}")
        print(f"Event metadata folder: {args.event_meta_dir}")
        print(f"MRMS folder: {args.mrms_dir}")
        print(f"Output folder: {args.out_dir}")
        print(f"Reader: {args.reader}")
        print(f"Allow cropped files: {args.allow_cropped}")
        print("=" * 80)

        run_event(args, window)


if __name__ == "__main__":
    main()
