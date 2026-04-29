#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from shapely.geometry import box

LOCAL_TZ = "America/Chicago"

DEFAULT_NC = "/mnt/12TB/Sujan/ngiab_preprocess_output/06892513/forcings/forcings.nc"
DEFAULT_GPKGS = [
    "/mnt/12TB/Sujan/ngiab_preprocess_output/06892513/config/06892513_subset.gpkg",
    "/mnt/12TB/Sujan/ngiab_preprocess_output/06893080/config/06893080_subset.gpkg",
    "/mnt/12TB/Sujan/ngiab_preprocess_output/06893390/config/06893390_subset.gpkg",
]
DEFAULT_GRID_CSV = "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/dependent_files/grid_centers_wgs84.csv"
DEFAULT_BIN_DIR = "/mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Continuous_IDW/03_Interpolated_Rain/bin_rainfall"
DEFAULT_OUT_DIR = "/mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Continuous_IDW/04_IDW_to_Subcatchments_nc_file"

''' python 04_IDW_to_Subcatchments_nc_file.py\
    --start-date 20130101 --end-date 20241231\
        --nc /mnt/12TB/Sujan/ngiab_preprocess_output/06893080/forcings/forcings.nc\
            --gpkg "/mnt/12TB/Sujan/ngiab_preprocess_output/06893080/config/06893080_subset.gpkg"
                
'''
# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def norm_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def clean_grid_id(x) -> str:
    s = norm_str(x)
    if s == "":
        return ""
    try:
        f = float(s)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
    except Exception:
        pass
    return s


def infer_grid_delta(grid_df: pd.DataFrame) -> tuple[float, float]:
    lats = np.sort(pd.to_numeric(grid_df["Latitude"], errors="coerce").dropna().unique())
    lons = np.sort(pd.to_numeric(grid_df["Longitude"], errors="coerce").dropna().unique())

    def min_pos_diff(vals: np.ndarray) -> float:
        if len(vals) < 2:
            raise ValueError("Not enough unique coordinates to infer grid spacing")
        diffs = np.diff(vals)
        diffs = diffs[diffs > 1e-12]
        if len(diffs) == 0:
            raise ValueError("Could not infer positive grid spacing")
        return float(np.min(diffs))

    return min_pos_diff(lats), min_pos_diff(lons)


def parse_time_local_to_unix(series: pd.Series, tz_name: str) -> np.ndarray:
    s = series.astype(str).str.strip()

    # Remove any trailing timezone offset like -0500, -0600, -05:00, -06:00
    s = s.str.replace(r'([+-]\d{2}:?\d{2})$', '', regex=True).str.strip()

    t = pd.to_datetime(s, errors="coerce", utc=True)

    if t.isna().any():
        bad = int(t.isna().sum())
        raise ValueError(f"Found {bad} bad timestamps in time_local column")

    return (t.astype("int64") // 10**9).to_numpy(np.int64)


def build_time_lookup_from_nc(nc: Dataset) -> tuple[np.ndarray, dict[int, int]]:
    if "Time" not in nc.variables:
        raise RuntimeError("NetCDF missing Time variable")
    time_var = nc.variables["Time"][:]
    if time_var.ndim != 2:
        raise RuntimeError(f"Expected Time to be 2D, got shape {time_var.shape}")
    tvec = np.asarray(time_var[0, :], dtype=np.int64)
    return tvec, {int(v): int(i) for i, v in enumerate(tvec)}


def detect_vector_layer(gpkg: Path) -> str:
    import fiona

    layers = list(fiona.listlayers(gpkg))
    if not layers:
        raise ValueError(f"No layers found in {gpkg}")
    # Prefer likely divide/catchment layers.
    ranked = sorted(
        layers,
        key=lambda x: (
            0 if any(k in x.lower() for k in ["divide", "cat", "catch", "nexus", "flowpath"]) else 1,
            x.lower(),
        ),
    )
    return ranked[0]


def detect_id_field(gdf: gpd.GeoDataFrame, nc_ids: set[str]) -> str:
    preferred = [
        "divide_id", "divide-id", "id", "ids", "cat", "cat_id", "catchment_id",
        "feature_id", "realized_catchment", "divide", "toid",
    ]
    cols = list(gdf.columns)

    # First try explicit likely names with direct match to NC ids.
    for col in preferred:
        for actual in cols:
            if actual.lower() == col.lower():
                vals = set(gdf[actual].astype(str).str.strip())
                if vals & nc_ids:
                    return actual

    # Then any object/string column with overlapping values.
    best_col = None
    best_overlap = -1
    for col in cols:
        if col == gdf.geometry.name:
            continue
        try:
            vals = set(gdf[col].astype(str).str.strip())
        except Exception:
            continue
        overlap = len(vals & nc_ids)
        if overlap > best_overlap:
            best_overlap = overlap
            best_col = col

    if best_col is None or best_overlap <= 0:
        raise ValueError(
            "Could not find a catchment ID field in the GeoPackage that matches NetCDF ids"
        )
    return best_col


def load_catchments(gpkg_paths, nc_ids, layer_name):
    nc_id_set = {str(x).strip() for x in nc_ids}
    parts = []

    for gpkg in gpkg_paths:
        print(f"Reading {gpkg} (layer={layer_name})")

        gdf = gpd.read_file(gpkg, layer=layer_name)

        if not isinstance(gdf, gpd.GeoDataFrame):
            raise ValueError(f"{gpkg} did not return a GeoDataFrame")

        if gdf.geometry is None:
            raise ValueError(f"{gpkg} has no geometry column")

        if gdf.empty:
            continue

        # detect ID column
        id_field = detect_id_field(gdf, nc_id_set)

        gdf = gdf[[id_field, gdf.geometry.name]].copy()
        gdf = gdf.rename(columns={id_field: "nc_id"})

        gdf["nc_id"] = gdf["nc_id"].astype(str).str.strip()

        # keep only matching IDs
        gdf = gdf[gdf["nc_id"].isin(nc_id_set)].copy()

        # remove bad geometries
        gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()

        if not gdf.empty:
            parts.append(gdf)

    if not parts:
        raise ValueError("No matching catchments found in provided GeoPackages")

    out = pd.concat(parts, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=parts[0].crs)

    # convert to WGS84 (matches your grid)
    out = out.to_crs(epsg=4326)

    # dissolve duplicates across multiple gpkg files
    out = out.dissolve(by="nc_id", as_index=False)

    return out


def build_grid_cells(grid_csv: Path) -> gpd.GeoDataFrame:
    grid = pd.read_csv(grid_csv)
    req = ["id", "Latitude", "Longitude"]
    missing = [c for c in req if c not in grid.columns]
    if missing:
        raise ValueError(f"Grid CSV missing required columns: {missing}")

    grid = grid[req].copy()
    grid["id"] = grid["id"].apply(clean_grid_id)
    grid["Latitude"] = pd.to_numeric(grid["Latitude"], errors="coerce")
    grid["Longitude"] = pd.to_numeric(grid["Longitude"], errors="coerce")
    grid = grid.dropna(subset=["id", "Latitude", "Longitude"]).copy()

    dlat, dlon = infer_grid_delta(grid)
    half_lat = 0.5 * dlat
    half_lon = 0.5 * dlon

    grid["geometry"] = [
        box(lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)
        for lat, lon in zip(grid["Latitude"], grid["Longitude"])
    ]
    gdf = gpd.GeoDataFrame(grid[["id", "Latitude", "Longitude", "geometry"]], geometry="geometry", crs="EPSG:4326")
    return gdf


def build_area_weight_map(grid_cells: gpd.GeoDataFrame, catchments: gpd.GeoDataFrame) -> pd.DataFrame:
    # Area-weight in a projected CRS.
    metric_crs = "EPSG:26915"
    grid_m = grid_cells.to_crs(metric_crs)
    cat_m = catchments.to_crs(metric_crs)

    inter = gpd.overlay(
        grid_m[["id", "geometry"]],
        cat_m[["nc_id", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    if inter.empty:
        raise ValueError("No overlap found between grid cells and catchments")

    inter["area_m2"] = inter.geometry.area
    inter = inter[inter["area_m2"] > 0].copy()
    if inter.empty:
        raise ValueError("Intersections were found but all intersection areas are zero")

    weight_df = inter.groupby(["nc_id", "id"], as_index=False)["area_m2"].sum()
    weight_df["weight"] = weight_df.groupby("nc_id")["area_m2"].transform(lambda s: s / s.sum())
    return weight_df[["nc_id", "id", "weight"]].copy()


def find_bin_files(bin_dir: Path) -> list[Path]:
    files = sorted(bin_dir.glob("bin_*_rainfall.csv"))
    if not files:
        raise FileNotFoundError(f"No bin rainfall files found in {bin_dir}")
    return files


def load_idw_timeseries(bin_files: list[Path], tz_name: str) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    frames = []
    summaries = []

    for fp in bin_files:
        df = pd.read_csv(fp)
        if "time_local" not in df.columns:
            raise ValueError(f"{fp} missing time_local column")

        orig_cols = list(df.columns)
        rename_map = {c: clean_grid_id(c) for c in df.columns if c != "time_local"}
        df = df.rename(columns=rename_map)
        grid_cols = [c for c in df.columns if c != "time_local"]
        for c in grid_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        unix = parse_time_local_to_unix(df["time_local"], tz_name)
        df.insert(1, "time_unix", unix)
        df = df.drop(columns=[c for c in df.columns if c == "time_local"])
        df = df.groupby("time_unix", as_index=False).mean(numeric_only=True)
        frames.append(df)

        summaries.append({
            "file": str(fp),
            "n_rows_in": len(unix),
            "n_rows_out": len(df),
            "n_grid_cols": len(grid_cols),
            "first_time_unix": int(df["time_unix"].min()) if len(df) else np.nan,
            "last_time_unix": int(df["time_unix"].max()) if len(df) else np.nan,
            "first_grid_col": grid_cols[0] if grid_cols else "",
            "last_grid_col": grid_cols[-1] if grid_cols else "",
            "orig_first_cols": ", ".join(orig_cols[:5]),
        })

    full = pd.concat(frames, ignore_index=True, sort=False)
    full = full.groupby("time_unix", as_index=False).mean(numeric_only=True)
    full = full.sort_values("time_unix").reset_index(drop=True)

    rain = full.set_index("time_unix")
    rain.columns = [clean_grid_id(c) for c in rain.columns]
    rain = rain.loc[:, [c for c in rain.columns if c != ""]]

    return full["time_unix"].to_numpy(np.int64), rain, pd.DataFrame(summaries)


def aggregate_to_catchments(rain_df: pd.DataFrame, weight_df: pd.DataFrame, nc_ids: list[str]) -> pd.DataFrame:
    common_grid_ids = sorted(set(rain_df.columns.astype(str)) & set(weight_df["id"].astype(str)))
    if not common_grid_ids:
        raise ValueError("No overlapping grid IDs between rainfall CSVs and grid-to-catchment map")

    W = weight_df[weight_df["id"].isin(common_grid_ids)].copy()
    W["id"] = W["id"].astype(str)

    # Re-normalize in case some grid ids are missing from rainfall file.
    W["weight"] = W.groupby("nc_id")["weight"].transform(lambda s: s / s.sum())

    out = pd.DataFrame(index=rain_df.index)
    for nc_id in nc_ids:
        sub = W.loc[W["nc_id"] == nc_id, ["id", "weight"]].copy()
        if sub.empty:
            out[nc_id] = np.nan
            continue
        vals = rain_df[sub["id"].tolist()].to_numpy(dtype=float)
        w = sub["weight"].to_numpy(dtype=float)

        # Weighted mean, ignoring NaNs and renormalizing by available weights.
        valid = np.isfinite(vals)
        weighted_vals = np.where(valid, vals * w[None, :], 0.0)
        weight_sum = np.where(valid, w[None, :], 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            series = weighted_vals.sum(axis=1) / weight_sum
        series[weight_sum <= 0] = np.nan
        out[nc_id] = series

    out = out[nc_ids].copy()
    return out


def write_to_netcdf(
    nc_in: Path,
    nc_out: Path,
    catch_rain: pd.DataFrame,
    weight_df: pd.DataFrame,
    gpkg_paths: list[Path],
    grid_csv: Path,
    bin_files: list[Path],
    tz_name: str,
) -> pd.DataFrame:
    nc_out.parent.mkdir(parents=True, exist_ok=True)
    if nc_out.resolve() != nc_in.resolve():
        import shutil
        shutil.copy2(nc_in, nc_out)

    with Dataset(nc_out, mode="r+") as nc:
        apcp = nc.variables["APCP_surface"]
        prate = nc.variables["precip_rate"] if "precip_rate" in nc.variables else None
        ids = [str(x) for x in nc.variables["ids"][:]]
        time_vec, time_lookup = build_time_lookup_from_nc(nc)

        if apcp.shape[0] != len(ids):
            raise RuntimeError("APCP_surface first dimension does not match ids length")

        dt = np.diff(time_vec).astype(np.float64)
        matched_rows = 0
        matched_cols = 0
        touched_times = []
        missing_times = []

        arr = apcp[:, :]
        if isinstance(arr, np.ma.MaskedArray):
            arr = arr.filled(np.nan)
        arr = np.asarray(arr, dtype=np.float32)

        for unix_time, row in catch_rain.iterrows():
            idx = time_lookup.get(int(unix_time))
            if idx is None:
                missing_times.append(int(unix_time))
                continue
            vals = row.to_numpy(dtype=np.float32)
            arr[:, idx] = vals
            matched_rows += 1
            matched_cols += int(np.isfinite(vals).sum())
            touched_times.append(int(unix_time))

        apcp[:, :] = arr

        if prate is not None:
            pr = prate[:, :]
            if isinstance(pr, np.ma.MaskedArray):
                pr = pr.filled(np.nan)
            pr = np.asarray(pr, dtype=np.float32)

            # Match existing file note: APCP_surface converted to mm/s by dividing by dt.
            # For water, 1 kg/m^2 == 1 mm numerically.
            pr[:, :] = np.nan
            with np.errstate(invalid="ignore", divide="ignore"):
                pr[:, 1:] = arr[:, 1:] / dt[None, :]
                pr[:, 0] = arr[:, 0] / dt[0]
            prate[:, :] = pr
            prate.source_note = (
                "APCP_surface overwritten from Continuous IDW grid rainfall aggregated to subcatchments; "
                "precip_rate recomputed as APCP_surface(t) / dt in mm s^-1 "
                "(numerically equal to kg m^-2 s^-1 for water)."
            )

        apcp.source_note = (
            "APCP_surface overwritten from Continuous IDW grid rainfall CSV bins aggregated to subcatchments by "
            "area-weighted grid-cell overlap. Source grid rainfall was in local time and converted to Unix seconds using "
            f"timezone {tz_name}."
        )
        apcp.idw_bin_files = "; ".join(str(p) for p in bin_files)
        apcp.subcatchment_gpkgs = "; ".join(str(p) for p in gpkg_paths)
        apcp.grid_centers_csv = str(grid_csv)
        apcp.grid_overlap_method = "area_weighted_grid_cell_intersection"

    summary = pd.DataFrame([
        {
            "nc_in": str(nc_in),
            "nc_out": str(nc_out),
            "n_subcatchments": len(catch_rain.columns),
            "n_times_in_rain_csv": len(catch_rain.index),
            "n_times_written_to_nc": matched_rows,
            "n_values_written": matched_cols,
            "first_written_unix": min(touched_times) if touched_times else np.nan,
            "last_written_unix": max(touched_times) if touched_times else np.nan,
            "n_rain_times_missing_in_nc": len(missing_times),
            "first_missing_unix": missing_times[0] if missing_times else np.nan,
        }
    ])
    return summary

def parse_date_to_unix(date_str: str) -> int:
    date_str = str(date_str).strip()

    # support YYYYMMDD
    if re.match(r"^\d{8}$", date_str):
        dt = pd.to_datetime(date_str, format="%Y%m%d", utc=True)
    else:
        dt = pd.to_datetime(date_str, utc=True)

    return int(dt.value // 10**9)
# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate Continuous IDW grid rainfall to NextGen subcatchments and overwrite APCP_surface / precip_rate in a forcing NetCDF."
    )
    p.add_argument("--start-date", type=str, default=None, help="Start date (YYYYMMDD or YYYY-MM-DD)")
    p.add_argument("--end-date", type=str, default=None, help="End date (YYYYMMDD or YYYY-MM-DD)")
    p.add_argument("--nc", default=DEFAULT_NC, help="Input NextGen forcing NetCDF")
    p.add_argument("--gpkg", nargs="+", default=DEFAULT_GPKGS, help="One or more GeoPackages containing subcatchments")
    p.add_argument("--grid-csv", default=DEFAULT_GRID_CSV, help="Grid center CSV with id, Latitude, Longitude")
    p.add_argument("--bin-dir", default=DEFAULT_BIN_DIR, help="Folder containing bin_XXXX_rainfall.csv files")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output folder")
    p.add_argument("--out-nc-name", default="forcings_idw_subcatchment.nc", help="Output NetCDF filename")
    p.add_argument("--timezone", default=LOCAL_TZ, help="Timezone for time_local in rainfall CSVs")
    p.add_argument("--keep-weight-map", action="store_true", help="Save full grid-to-subcatchment weight map CSV")
    p.add_argument("--layer",default="divides",help="GeoPackage layer name containing subcatchment polygons (default: divides)",)
    return p.parse_args()


def main() -> None:
    a = parse_args()

    nc_path = Path(a.nc)
    gpkg_paths = [Path(x) for x in a.gpkg]
    grid_csv = Path(a.grid_csv)
    bin_dir = Path(a.bin_dir)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nc = out_dir / a.out_nc_name

    for p in [nc_path, grid_csv, bin_dir, *gpkg_paths]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required path: {p}")

    print("Reading NetCDF ids...")
    with Dataset(nc_path) as nc:
        nc_ids = [str(x) for x in nc.variables["ids"][:]]

    print("Loading catchments from GeoPackages...")
    catchments = load_catchments(gpkg_paths, nc_ids, a.layer)
    catchments.to_file(out_dir / "subcatchments_used.gpkg", driver="GPKG")
    pd.DataFrame({"nc_id": nc_ids}).to_csv(out_dir / "nc_ids.csv", index=False)

    print("Building grid cell polygons from centroid CSV...")
    grid_cells = build_grid_cells(grid_csv)

    print("Building area-weight map from grid cells to subcatchments...")
    weight_df = build_area_weight_map(grid_cells, catchments)
    if a.keep_weight_map:
        weight_df.to_csv(out_dir / "grid_to_subcatchment_weights.csv", index=False)

    print("Reading Continuous IDW rainfall bins...")
    bin_files = find_bin_files(bin_dir)
    time_unix, rain_df, bin_summary = load_idw_timeseries(bin_files, a.timezone)
    if a.start_date or a.end_date:
        print("Applying time filter...")

        start_unix = parse_date_to_unix(a.start_date) if a.start_date else None
        end_unix   = parse_date_to_unix(a.end_date) if a.end_date else None

        mask = np.ones(len(rain_df), dtype=bool)

        if start_unix is not None:
            mask &= (rain_df.index >= start_unix)

        if end_unix is not None:
            mask &= (rain_df.index <= end_unix)

        rain_df = rain_df.loc[mask].copy()

        print(f"Filtered rainfall rows: {len(rain_df)}")
    bin_summary.to_csv(out_dir / "bin_file_summary.csv", index=False)

    print("Aggregating grid rainfall to subcatchments...")
    catch_rain = aggregate_to_catchments(rain_df, weight_df, nc_ids)
    catch_rain_out = catch_rain.copy()
    catch_rain_out.insert(0, "time_unix", catch_rain_out.index.astype(np.int64))
    catch_rain_out.to_csv(out_dir / "idw_rain_subcatchments.csv", index=False)

    print("Writing updated forcing NetCDF...")
    write_summary = write_to_netcdf(
        nc_in=nc_path,
        nc_out=out_nc,
        catch_rain=catch_rain,
        weight_df=weight_df,
        gpkg_paths=gpkg_paths,
        grid_csv=grid_csv,
        bin_files=bin_files,
        tz_name=a.timezone,
    )
    write_summary.to_csv(out_dir / "write_summary.csv", index=False)

    print("Done.")
    print(f"Output NetCDF: {out_nc}")
    print(f"Output folder : {out_dir}")


if __name__ == "__main__":
    main()
