#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK")
OK_DIR = BASE_DIR / "03_Interpolated_Rain"
OUT_BASE = BASE_DIR / "04_Comparison"
GRID_CSV = BASE_DIR / "dependent_files/grid_centers_wgs84.csv"

CATCHMENT_SHP_PATHS = [
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6893390/6893390.shp",
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/06893080/6893080.shp",
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6892513/6892513.shp",
]

OK_GLOB = "Event_*_grid_rain_hourly_mm.csv"
PRODUCTS = ["Composite_2", "Composite_3", "RA", "RKDP", "RZ"]
RADAR_BASE = BASE_DIR / "Radar_Event_TimeSeries"




# ---------------- HELPERS ----------------
def event_num_from_name(path: Path) -> int | None:
    m = re.search(r"Event_(\d+)_grid_rain_hourly_mm\.csv$", path.name)
    return int(m.group(1)) if m else None



def normalize_grid_col(col) -> str:
    s = str(col).strip()
    if s.lower().startswith("unnamed"):
        return s
    try:
        f = float(s)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return s
    except Exception:
        return s



def load_rain_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if df.shape[1] < 2:
        raise ValueError(f"{path} has fewer than 2 columns")

    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    bad_time = int(df[time_col].isna().sum())
    if bad_time > 0:
        raise ValueError(f"{path} has {bad_time} unparseable timestamps in column {time_col}")

    df = df.set_index(time_col)
    df.columns = [normalize_grid_col(c) for c in df.columns]

    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_index()
    df = df.sort_index(axis=1)
    return df



def compute_metrics(obs: np.ndarray, est: np.ndarray):
    mask = np.isfinite(obs) & np.isfinite(est)
    n = int(mask.sum())

    if n == 0:
        return {
            "n": 0,
            "bias": np.nan,
            "cc": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "mean_diff": np.nan,
        }

    o = obs[mask]
    e = est[mask]

    so = np.sum(o)
    bias = np.sum(e) / so if so != 0 else np.nan
    cc = np.corrcoef(o, e)[0, 1] if n >= 2 else np.nan
    rmse = np.sqrt(np.mean((e - o) ** 2))
    mae = np.mean(np.abs(e - o))
    mean_diff = np.mean(e - o)

    return {
        "n": n,
        "bias": bias,
        "cc": cc,
        "rmse": rmse,
        "mae": mae,
        "mean_diff": mean_diff,
    }



def infer_grid_spacing(values: pd.Series) -> float:
    arr = np.sort(pd.to_numeric(values, errors="coerce").dropna().unique())
    diffs = np.diff(arr)
    diffs = diffs[diffs > 1e-12]
    if len(diffs) == 0:
        raise ValueError("Could not infer grid spacing from grid centers")
    return float(np.median(diffs))



def load_grid_cells(grid_csv: Path) -> gpd.GeoDataFrame:
    grid = pd.read_csv(grid_csv)
    req = ["id", "Latitude", "Longitude"]
    missing = [c for c in req if c not in grid.columns]
    if missing:
        raise ValueError(f"{grid_csv} missing required columns: {missing}")

    grid["id"] = grid["id"].apply(normalize_grid_col)
    grid["Latitude"] = pd.to_numeric(grid["Latitude"], errors="coerce")
    grid["Longitude"] = pd.to_numeric(grid["Longitude"], errors="coerce")
    grid = grid.dropna(subset=["Latitude", "Longitude"]).copy()

    dlat = infer_grid_spacing(grid["Latitude"])
    dlon = infer_grid_spacing(grid["Longitude"])
    half_lat = dlat / 2.0
    half_lon = dlon / 2.0

    grid["geometry"] = [
        box(lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)
        for lat, lon in zip(grid["Latitude"], grid["Longitude"])
    ]
    gdf = gpd.GeoDataFrame(grid[["id", "Latitude", "Longitude", "geometry"]], geometry="geometry", crs="EPSG:4326")
    return gdf



def load_catchments(shp_paths: list[str]) -> gpd.GeoDataFrame:
    rows = []
    for p in shp_paths:
        shp = Path(p)
        if not shp.exists():
            print(f"[catchment] missing: {shp}")
            continue
        g = gpd.read_file(shp)
        if g.empty:
            continue
        if g.crs is None:
            raise ValueError(f"Catchment shapefile has no CRS: {shp}")
        g = g.to_crs(epsg=4326)
        geom = g.geometry.union_all()
        rows.append({"catchment": shp.parent.name, "path": str(shp), "geometry": geom})

    if not rows:
        raise FileNotFoundError("No valid catchment shapefiles found")

    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")



def build_masks(common_cols: pd.Index, grid_cells: gpd.GeoDataFrame, catchments: gpd.GeoDataFrame):
    grid_sub = grid_cells[grid_cells["id"].isin(common_cols)].copy()
    if grid_sub.empty:
        raise ValueError("None of the common rainfall grid IDs were found in grid_centers_wgs84.csv")

    union_geom = catchments.geometry.union_all()
    domain_mask = grid_sub.geometry.intersects(union_geom)
    selected = grid_sub.loc[domain_mask].copy()
    if selected.empty:
        raise ValueError("No grid cells intersect/touch the watershed polygons")

    domain_ids = selected["id"].astype(str).tolist()

    catchment_masks = {}
    for _, row in catchments.iterrows():
        m = grid_sub.geometry.intersects(row.geometry)
        ids = grid_sub.loc[m, "id"].astype(str).tolist()
        catchment_masks[str(row["catchment"])] = ids

    return domain_ids, catchment_masks, grid_sub



def save_mask_tables(out_dir: Path, grid_sub: gpd.GeoDataFrame, domain_ids: list[str], catchment_masks: dict[str, list[str]]):
    domain_set = set(domain_ids)
    per_cell_rows = []
    for _, row in grid_sub.iterrows():
        gid = str(row["id"])
        rec = {
            "grid_id": gid,
            "Latitude": row["Latitude"],
            "Longitude": row["Longitude"],
            "in_any_watershed": gid in domain_set,
        }
        for name, ids in catchment_masks.items():
            rec[f"in_{name}"] = gid in set(ids)
        per_cell_rows.append(rec)
    pd.DataFrame(per_cell_rows).to_csv(out_dir / "watershed_cell_mask.csv", index=False)

    summary = [{"mask_name": "all_watersheds_union", "n_cells": len(domain_ids)}]
    for name, ids in catchment_masks.items():
        summary.append({"mask_name": name, "n_cells": len(ids)})
    pd.DataFrame(summary).to_csv(out_dir / "watershed_mask_summary.csv", index=False)



def build_hourly_sum_ts(df: pd.DataFrame, cols: list[str], prefix: str) -> pd.DataFrame:
    sub = df[cols].copy() if cols else pd.DataFrame(index=df.index)
    out = pd.DataFrame({
        "timestamp": df.index,
        f"{prefix}_sum_mm_over_cells": sub.sum(axis=1, skipna=True).values if len(cols) else np.nan,
        f"{prefix}_mean_mm": sub.mean(axis=1, skipna=True).values if len(cols) else np.nan,
    })
    return out



def compare_pair_by_cells(obs_df: pd.DataFrame, est_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for gid in cols:
        m = compute_metrics(obs_df[gid].to_numpy(dtype=float), est_df[gid].to_numpy(dtype=float))
        rows.append({"grid_id": gid, **m})
    return pd.DataFrame(rows)



def compare_pair_overall(obs_df: pd.DataFrame, est_df: pd.DataFrame, cols: list[str], label: str) -> pd.DataFrame:
    obs = obs_df[cols]
    est = est_df[cols]

    mean_metrics = compute_metrics(
        obs.mean(axis=1, skipna=True).to_numpy(dtype=float),
        est.mean(axis=1, skipna=True).to_numpy(dtype=float),
    )
    sum_metrics = compute_metrics(
        obs.sum(axis=1, skipna=True).to_numpy(dtype=float),
        est.sum(axis=1, skipna=True).to_numpy(dtype=float),
    )
    flat_metrics = compute_metrics(
        obs.to_numpy(dtype=float).ravel(),
        est.to_numpy(dtype=float).ravel(),
    )

    return pd.DataFrame([
        {"comparison": label, "aggregation": "hourly_mean_over_cells", **mean_metrics},
        {"comparison": label, "aggregation": "hourly_sum_over_cells", **sum_metrics},
        {"comparison": label, "aggregation": "all_gridtime_pairs", **flat_metrics},
    ])



def compare_event(event: int, grid_cells: gpd.GeoDataFrame, catchments: gpd.GeoDataFrame):
    ok_file = OK_DIR / f"Event_{event}_grid_rain_hourly_mm.csv"

    product_files = {
        p: RADAR_BASE / p / f"Event_{event}" / f"Event_{event}_grid_rain_hourly_mm_{p}.csv"
        for p in PRODUCTS
    }

    out_dir = OUT_BASE / f"Event_{event}"
    out_dir.mkdir(parents=True, exist_ok=True)

    status = {
        "event": event,
        "ok_file": str(ok_file),
        "ok_exists": ok_file.exists(),
        "ok_rows": np.nan,
        "ok_cols": np.nan,
        "common_time": np.nan,
        "common_cols": np.nan,
        "watershed_cols": np.nan,
        "status": "",
        "note": "",
    }

    for pname, pfile in product_files.items():
        status[f"{pname}_file"] = str(pfile)
        status[f"{pname}_exists"] = pfile.exists()
        status[f"{pname}_rows"] = np.nan
        status[f"{pname}_cols"] = np.nan

    available_products = [p for p, f in product_files.items() if f.exists()]

    if not ok_file.exists():
        status["status"] = "missing_ok_file"
        pd.DataFrame([status]).to_csv(out_dir / f"comparison_status_event{event}.csv", index=False)
        return status, None

    if not available_products:
        status["status"] = "missing_radar_files"
        status["note"] = "No radar product files found for this event"
        pd.DataFrame([status]).to_csv(out_dir / f"comparison_status_event{event}.csv", index=False)
        return status, None

    try:
        ok = load_rain_df(ok_file)
    except Exception as e:
        status["status"] = "load_error"
        status["note"] = f"OK load failed: {e}"
        pd.DataFrame([status]).to_csv(out_dir / f"comparison_status_event{event}.csv", index=False)
        return status, None

    product_dfs = {}
    for pname in available_products:
        try:
            product_dfs[pname] = load_rain_df(product_files[pname])
        except Exception as e:
            print(f"[skip] failed to load {pname}: {e}")

    if not product_dfs:
        status["status"] = "load_error"
        status["note"] = "No radar product file could be loaded"
        pd.DataFrame([status]).to_csv(out_dir / f"comparison_status_event{event}.csv", index=False)
        return status, None

    status["ok_rows"], status["ok_cols"] = ok.shape
    for pname, df in product_dfs.items():
        status[f"{pname}_rows"], status[f"{pname}_cols"] = df.shape

    common_time = ok.index
    common_cols = ok.columns

    for df in product_dfs.values():
        common_time = common_time.intersection(df.index)
        common_cols = common_cols.intersection(df.columns)

    status["common_time"] = len(common_time)
    status["common_cols"] = len(common_cols)

    if len(common_time) == 0:
        status["status"] = "no_common_time"
        status["note"] = "No overlapping timestamps across OK and available radar products"
        pd.DataFrame([status]).to_csv(out_dir / f"comparison_status_event{event}.csv", index=False)
        return status, None

    if len(common_cols) == 0:
        status["status"] = "no_common_cols"
        status["note"] = "No overlapping grid columns across OK and available radar products"
        pd.DataFrame([status]).to_csv(out_dir / f"comparison_status_event{event}.csv", index=False)
        return status, None

    ok = ok.loc[common_time, common_cols].copy()
    for pname in list(product_dfs.keys()):
        product_dfs[pname] = product_dfs[pname].loc[common_time, common_cols].copy()

    try:
        watershed_cols, catchment_masks, grid_sub = build_masks(common_cols, grid_cells, catchments)
    except Exception as e:
        status["status"] = "watershed_mask_error"
        status["note"] = str(e)
        pd.DataFrame([status]).to_csv(out_dir / f"comparison_status_event{event}.csv", index=False)
        return status, None

    status["watershed_cols"] = len(watershed_cols)
    ok = ok[watershed_cols].copy()
    for pname in list(product_dfs.keys()):
        product_dfs[pname] = product_dfs[pname][watershed_cols].copy()
    save_mask_tables(out_dir, grid_sub, watershed_cols, catchment_masks)

    # 1) cell-by-cell metrics only for watershed-intersecting cells
    df_cell = pd.DataFrame({"grid_id": watershed_cols})

    for pname, dfp in product_dfs.items():
        dfp_cell = compare_pair_by_cells(ok, dfp, watershed_cols)
        suffix = pname.lower()
        dfp_cell = dfp_cell.rename(columns={
            "n": f"n_{suffix}",
            "bias": f"bias_{suffix}",
            "cc": f"cc_{suffix}",
            "rmse": f"rmse_{suffix}",
            "mae": f"mae_{suffix}",
            "mean_diff": f"mean_diff_{suffix}",
        })
        df_cell = df_cell.merge(dfp_cell, on="grid_id", how="left")

    df_cell.to_csv(out_dir / f"cell_metrics_watershed_only_event{event}.csv", index=False)
    
    # 2) whole-watershed hourly sums and means
    df_hourly = pd.DataFrame({"timestamp": ok.index})
    df_hourly["ok_sum_mm_over_cells"] = ok.sum(axis=1, skipna=True).values
    df_hourly["ok_mean_mm"] = ok.mean(axis=1, skipna=True).values

    for pname, dfp in product_dfs.items():
        suffix = pname.lower()
        df_hourly[f"{suffix}_sum_over_cells"] = dfp.sum(axis=1, skipna=True).values
        df_hourly[f"{suffix}_mean"] = dfp.mean(axis=1, skipna=True).values

    df_hourly.to_csv(out_dir / f"watershed_hourly_totals_event{event}.csv", index=False)

    # 3) whole-watershed summary metrics
    metrics_list = []
    for pname, dfp in product_dfs.items():
        metrics_list.append(compare_pair_overall(ok, dfp, watershed_cols, f"OK_vs_{pname}"))

    df_domain_metrics = pd.concat(metrics_list, ignore_index=True)
    df_domain_metrics.to_csv(out_dir / f"watershed_metrics_event{event}.csv", index=False)

    # 4) event totals per cell within watershed only
    df_total = pd.DataFrame({
        "grid_id": watershed_cols,
        "ok_total_mm": ok.sum(axis=0, skipna=True).values,
    })

    for pname, dfp in product_dfs.items():
        suffix = pname.lower()
        df_total[f"{suffix}_total"] = dfp.sum(axis=0, skipna=True).values
        df_total[f"diff_{suffix}_minus_ok"] = df_total[f"{suffix}_total"] - df_total["ok_total_mm"]

    df_total.to_csv(out_dir / f"event_total_per_cell_watershed_only_event{event}.csv", index=False)
        # 4b) whole-event total rainfall summary across all watershed cells and all event hours
    event_total_row = {
        "event": event,
        "n_time_steps": len(ok.index),
        "n_watershed_cells": len(watershed_cols),
        "ok_event_total_mm_over_cells": float(ok.to_numpy(dtype=float).sum()),
    }

    for pname, dfp in product_dfs.items():
        suffix = pname.lower()
        event_total_row[f"{suffix}_event_total_mm_over_cells"] = float(dfp.to_numpy(dtype=float).sum())
        event_total_row[f"diff_{suffix}_minus_ok_event_total"] = (
            event_total_row[f"{suffix}_event_total_mm_over_cells"]
            - event_total_row["ok_event_total_mm_over_cells"]
        )

    df_event_total_summary = pd.DataFrame([event_total_row])
    df_event_total_summary.to_csv(out_dir / f"event_total_rainfall_summary_event{event}.csv", index=False)
    
    # 5) each catchment separately
    catchment_summary_rows = []
    for catchment_name, ids in catchment_masks.items():
        ids = [gid for gid in ids if gid in watershed_cols]
        if not ids:
            continue

        catch_ts = pd.DataFrame({"timestamp": ok.index})
        catch_ts["ok_sum_mm_over_cells"] = ok[ids].sum(axis=1, skipna=True).values
        catch_ts["ok_mean_mm"] = ok[ids].mean(axis=1, skipna=True).values

        for pname, dfp in product_dfs.items():
            suffix = pname.lower()
            catch_ts[f"{suffix}_sum_over_cells"] = dfp[ids].sum(axis=1, skipna=True).values
            catch_ts[f"{suffix}_mean"] = dfp[ids].mean(axis=1, skipna=True).values

        catch_ts.to_csv(out_dir / f"hourly_totals_{catchment_name}_event{event}.csv", index=False)

        pair_metrics_list = []
        for pname, dfp in product_dfs.items():
            pair_metrics_list.append(compare_pair_overall(ok, dfp, ids, f"{catchment_name}: OK_vs_{pname}"))

        pair_metrics = pd.concat(pair_metrics_list, ignore_index=True)
        pair_metrics.to_csv(out_dir / f"metrics_{catchment_name}_event{event}.csv", index=False)

        for _, r in pair_metrics.iterrows():
            catchment_summary_rows.append({
                "catchment": catchment_name,
                "n_cells": len(ids),
                **r.to_dict(),
            })

    pd.DataFrame(catchment_summary_rows).to_csv(out_dir / f"catchment_metrics_summary_event{event}.csv", index=False)

    status["status"] = "ok"
    status["note"] = "Watershed-filtered comparison completed"
    pd.DataFrame([status]).to_csv(out_dir / f"comparison_status_event{event}.csv", index=False)
    return status, event_total_row



def main():
    grid_cells = load_grid_cells(GRID_CSV)
    catchments = load_catchments(CATCHMENT_SHP_PATHS)

    ok_files = sorted(OK_DIR.glob(OK_GLOB))
    events = [event_num_from_name(p) for p in ok_files]
    events = [e for e in events if e is not None]

    if not events:
        print("No OK event files found.")
        return

    all_status = []
    all_event_totals = []

    for event in events:
        print(f"Processing Event {event} ...")
        s, event_total_row = compare_event(event, grid_cells=grid_cells, catchments=catchments)
        all_status.append(s)

        if event_total_row is not None:
            all_event_totals.append(event_total_row)

        print(
            f"  status = {s['status']}, common_time = {s['common_time']}, "
            f"common_cols = {s['common_cols']}, watershed_cols = {s['watershed_cols']}"
        )

    if all_event_totals:
        df_event_totals = pd.DataFrame(all_event_totals)
        df_event_totals = df_event_totals.sort_values("event").reset_index(drop=True)
        df_event_totals.to_csv(OUT_BASE / "event_total_rainfall_summary_all_events.csv", index=False)

        print("\nEvent total rainfall summary:")
        print(df_event_totals.to_string(index=False))
    
    df_status = pd.DataFrame(all_status)
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    df_status.to_csv(OUT_BASE / "comparison_summary_all_events.csv", index=False)
    print("\nDone.")
    print(f"Summary written to: {OUT_BASE / 'comparison_summary_all_events.csv'}")


if __name__ == "__main__":
    main()
