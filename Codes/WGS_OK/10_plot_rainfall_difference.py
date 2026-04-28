#!/usr/bin/env python3
import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import box
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon

'''python 10_plot_rainfall_difference.py   --csv1 "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/03_Interpolated_Rain/Event_7_grid_rain_hourly_mm.csv"   --csv2 "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_IDW/03_Interpolated_Rain/Event_7_grid_rain_hourly_mm.csv"   --grid-csv "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/dependent_files/grid_centers_wgs84.csv"   --catchments     "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6893390/6893390.shp"     "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/06893080/6893080.shp"     "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6892513/6892513.shp"   --out diff_plot_event7.png
'''
# -------------------------
# Helpers
# -------------------------

def normalize_grid_col(col):
    s = str(col).strip()
    try:
        f = float(s)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return s
    except:
        return s


def load_grid(grid_csv):
    df = pd.read_csv(grid_csv)
    df = df[['id', 'Latitude', 'Longitude']].copy()
    df['id'] = df['id'].astype(int).astype(str)
    return df


def build_cells(grid_df):
    lat_vals = np.sort(grid_df['Latitude'].unique())
    lon_vals = np.sort(grid_df['Longitude'].unique())

    dlat = np.min(np.diff(lat_vals))
    dlon = np.min(np.diff(lon_vals))

    half_lat = dlat / 2
    half_lon = dlon / 2

    geoms = [
        box(lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)
        for lat, lon in zip(grid_df['Latitude'], grid_df['Longitude'])
    ]

    return gpd.GeoDataFrame(grid_df.copy(), geometry=geoms, crs="EPSG:4326")


def load_catchments(paths):
    gdfs = []
    for p in paths:
        g = gpd.read_file(p).to_crs("EPSG:4326")
        gdfs.append(g[['geometry']])
    out = pd.concat(gdfs, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")


def load_sum(csv_path):
    df = pd.read_csv(csv_path)

    time_col = df.columns[0]
    data = df.drop(columns=[time_col])

    data.columns = [normalize_grid_col(c) for c in data.columns]

    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    sums = data.sum(axis=0, skipna=True)
    sums.index = sums.index.astype(str)

    return sums


def shapely_to_patches(geom):
    patches = []
    if geom.is_empty:
        return patches
    if geom.geom_type == "Polygon":
        patches.append(MplPolygon(np.asarray(geom.exterior.coords)))
    elif geom.geom_type == "MultiPolygon":
        for g in geom.geoms:
            patches.extend(shapely_to_patches(g))
    return patches


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv1", required=True, help="First rainfall CSV (e.g., IDW)")
    ap.add_argument("--csv2", required=True, help="Second rainfall CSV (e.g., AORC)")
    ap.add_argument("--grid-csv", required=True)
    ap.add_argument("--catchments", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    grid = load_grid(args.grid_csv)
    cells = build_cells(grid)
    catch = load_catchments(args.catchments)
    catch_union = catch.geometry.union_all()

    sum1 = load_sum(args.csv1)
    sum2 = load_sum(args.csv2)

    # difference
    diff = sum1.subtract(sum2, fill_value=np.nan)

    plot = cells.copy()
    plot["diff"] = plot["id"].map(diff)

    plot = plot[plot.geometry.intersects(catch_union)]
    plot = plot[plot["diff"].notna()]

    vals = plot["diff"].values
    vmax = np.nanmax(np.abs(vals))
    vmin = -vmax  # symmetric

    fig, ax = plt.subplots(figsize=(9, 8))

    patches = []
    patch_vals = []

    for _, row in plot.iterrows():
        p = shapely_to_patches(row.geometry)
        patches.extend(p)
        patch_vals.extend([row["diff"]] * len(p))

    pc = PatchCollection(
        patches,
        cmap="RdBu_r",
        edgecolor="none",
        linewidth=0,
    )
    pc.set_array(np.array(patch_vals))
    pc.set_clim(vmin, vmax)

    ax.add_collection(pc)

    catch.boundary.plot(ax=ax, color="black", linewidth=1.2)

    ax.set_aspect("equal")
    ax.set_xlabel("Longitude", fontsize=16, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=16, fontweight="bold")

    cbar = plt.colorbar(pc, ax=ax)
    cbar.set_label("Rainfall Difference (mm)", fontsize=14)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    plt.close()

    print(f"[saved] {args.out}")
    print(f"range: {np.nanmin(vals):.2f} to {np.nanmax(vals):.2f} mm")


if __name__ == "__main__":
    main()