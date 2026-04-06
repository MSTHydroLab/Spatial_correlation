#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


DEFAULT_CATCHMENT_SHP_PATHS = [
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6893390/6893390.shp",
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/06893080/6893080.shp",
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6892513/6892513.shp",
]


def norm_id(x) -> str:
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return ""
    try:
        f = float(s)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
    except Exception:
        pass
    return s


def load_catchments(shp_paths: list[str]) -> gpd.GeoDataFrame:
    gdfs = []
    for p in shp_paths:
        fp = Path(p)
        if not fp.exists():
            print(f"[warn] catchment file not found: {fp}")
            continue
        try:
            g = gpd.read_file(fp)
            if g.empty:
                print(f"[warn] catchment file empty: {fp}")
                continue
            if g.crs is None:
                raise ValueError(f"Missing CRS in shapefile: {fp}")
            g = g.to_crs("EPSG:4326")
            gdfs.append(g[["geometry"]].copy())
        except Exception as e:
            print(f"[warn] failed reading {fp}: {e}")

    if not gdfs:
        raise FileNotFoundError("No valid catchment shapefiles could be loaded.")

    out = pd.concat(gdfs, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")
    out = out[out.geometry.notnull()].copy()
    return out


def ensure_latlon(weights_df: pd.DataFrame, grid_csv: Path) -> pd.DataFrame:
    out = weights_df.copy()
    out["id"] = out["id"].apply(norm_id)

    have_latlon = ("Latitude" in out.columns) and ("Longitude" in out.columns)
    if have_latlon:
        out["Latitude"] = pd.to_numeric(out["Latitude"], errors="coerce")
        out["Longitude"] = pd.to_numeric(out["Longitude"], errors="coerce")
        if out["Latitude"].notna().all() and out["Longitude"].notna().all():
            return out

    if not grid_csv.exists():
        raise FileNotFoundError(
            f"Weights file is missing Latitude/Longitude, and grid CSV was not found: {grid_csv}"
        )

    grid = pd.read_csv(grid_csv)
    req = ["id", "Latitude", "Longitude"]
    missing = [c for c in req if c not in grid.columns]
    if missing:
        raise ValueError(f"{grid_csv} is missing required columns: {missing}")

    grid = grid[req].copy()
    grid["id"] = grid["id"].apply(norm_id)
    grid["Latitude"] = pd.to_numeric(grid["Latitude"], errors="coerce")
    grid["Longitude"] = pd.to_numeric(grid["Longitude"], errors="coerce")

    keep_cols = [c for c in out.columns if c not in ["Latitude", "Longitude"]]
    out = out[keep_cols].merge(grid, on="id", how="left")

    if out["Latitude"].isna().any() or out["Longitude"].isna().any():
        bad = out.loc[out["Latitude"].isna() | out["Longitude"].isna(), "id"].tolist()[:10]
        raise ValueError(
            "Could not recover Latitude/Longitude for some centroid ids from grid CSV. "
            f"Example ids: {bad}"
        )

    return out


def count_negative_weights(df: pd.DataFrame, n_gauges: int) -> pd.DataFrame:
    out = df.copy()
    wcols = [f"w{k}" for k in range(1, n_gauges + 1)]
    missing = [c for c in wcols if c not in out.columns]
    if missing:
        raise ValueError(f"Weights file missing required columns: {missing}")

    for c in wcols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["n_negative_weights"] = (out[wcols] < 0).sum(axis=1)
    out["has_negative_weight"] = out["n_negative_weights"] > 0
    out["min_weight"] = out[wcols].min(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Compute percent of catchment centroids with negative kriging weights."
    )
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--event", type=int, required=True)
    ap.add_argument("--n-gauges", type=int, choices=[3, 4], required=True)
    ap.add_argument("--weights-file", default="")
    ap.add_argument("--grid-csv", default="")
    ap.add_argument("--catchments", nargs="*", default=DEFAULT_CATCHMENT_SHP_PATHS)
    args = ap.parse_args()

    base_dir = Path(args.base_dir)

    weights_file = (
        Path(args.weights_file)
        if args.weights_file
        else base_dir / "02_OK_Weights" / f"Event_{args.event}_nearest{args.n_gauges}_weights.csv"
    )
    grid_csv = (
        Path(args.grid_csv)
        if args.grid_csv
        else base_dir / "dependent_files" / "grid_centers_wgs84.csv"
    )

    if not weights_file.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_file}")

    D = pd.read_csv(weights_file)
    if "id" not in D.columns:
        raise ValueError(f"{weights_file} must contain an 'id' column")

    D["id"] = D["id"].apply(norm_id)
    D = ensure_latlon(D, grid_csv)
    D = count_negative_weights(D, args.n_gauges)

    D["Latitude"] = pd.to_numeric(D["Latitude"], errors="coerce")
    D["Longitude"] = pd.to_numeric(D["Longitude"], errors="coerce")
    D = D.dropna(subset=["Latitude", "Longitude"]).copy()

    catchments = load_catchments(args.catchments)

    pts = gpd.GeoDataFrame(
        D.copy(),
        geometry=[Point(xy) for xy in zip(D["Longitude"], D["Latitude"])],
        crs="EPSG:4326",
    )

    catch_union = catchments.union_all()
    pts["inside_catchment"] = pts.geometry.within(catch_union) | pts.geometry.touches(catch_union)

    inside = pts.loc[pts["inside_catchment"]].copy()

    n_inside = int(len(inside))
    n_negative = int(inside["has_negative_weight"].sum())
    pct_negative = 100.0 * n_negative / n_inside if n_inside > 0 else np.nan

    summary = pd.DataFrame([{
        "event": args.event,
        "n_gauges": args.n_gauges,
        "weights_file": str(weights_file),
        "grid_csv": str(grid_csv),
        "n_total_centroids": int(len(pts)),
        "n_centroids_inside_catchment": n_inside,
        "n_centroids_with_negative_weight_inside_catchment": n_negative,
        "pct_centroids_with_negative_weight_inside_catchment": pct_negative,
    }])

    out_dir = base_dir / "02_OK_Weights"
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_out = out_dir / f"Event_{args.event}_nearest{args.n_gauges}_negative_weight_catchment_detail.csv"
    summary_out = out_dir / f"Event_{args.event}_nearest{args.n_gauges}_negative_weight_catchment_stats.csv"

    detail_cols = [
        "id", "Latitude", "Longitude",
        "inside_catchment",
        "has_negative_weight",
        "n_negative_weights",
        "min_weight",
    ] + [f"g{k}" for k in range(1, args.n_gauges + 1)] + [f"w{k}" for k in range(1, args.n_gauges + 1)]

    keep_detail = [c for c in detail_cols if c in pts.columns]
    pts[keep_detail].to_csv(detail_out, index=False)
    summary.to_csv(summary_out, index=False)

    print("\nDone.")
    print(f"Detail : {detail_out}")
    print(f"Summary: {summary_out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()