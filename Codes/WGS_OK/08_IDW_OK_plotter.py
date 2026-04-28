#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point, box


BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK")
GRID_CSV = BASE_DIR / "dependent_files" / "grid_centers_wgs84.csv"
STATIONS_CSV = BASE_DIR / "dependent_files" / "Stations_df.csv"
DEFAULT_OUT_DIR = BASE_DIR / "07_IDW_OK_Avg_method_results"
PROJECT_EPSG = 26915

CATCHMENT_SHP_PATHS = [
    Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6893390/6893390.shp"),
    Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/06893080/6893080.shp"),
    Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6892513/6892513.shp"),
]


def load_grid(grid_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(grid_csv)
    required = ["id", "Latitude", "Longitude"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Grid CSV is missing required columns: {missing}")

    df = df[required].copy()
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["id", "Latitude", "Longitude"]).copy()
    df["id"] = df["id"].astype(int)
    return df.sort_values(["Latitude", "Longitude", "id"]).reset_index(drop=True)


def load_stations(stations_csv: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(stations_csv)
    required = ["ID", "Latitude", "Longitude"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Stations CSV is missing required columns: {missing}")

    df = df[required].copy()
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["ID", "Latitude", "Longitude"]).copy()
    df["ID"] = df["ID"].astype(int)

    geoms = [Point(lon, lat) for lat, lon in zip(df["Latitude"], df["Longitude"])]
    return gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")


def infer_grid_spacing(values: np.ndarray, name: str) -> float:
    vals = np.sort(np.unique(np.round(values.astype(float), 10)))
    diffs = np.diff(vals)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        raise ValueError(f"Could not infer {name} spacing from grid coordinates.")

    rounded = np.round(diffs, 10)
    counts = pd.Series(rounded).value_counts().sort_values(ascending=False)
    return float(counts.index[0])


def build_cell_polygons(grid_df: pd.DataFrame, lat_step: float, lon_step: float) -> gpd.GeoDataFrame:
    half_lat = lat_step / 2.0
    half_lon = lon_step / 2.0

    geoms = [
        box(lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)
        for lat, lon in zip(grid_df["Latitude"], grid_df["Longitude"])
    ]

    gdf = gpd.GeoDataFrame(grid_df.copy(), geometry=geoms, crs="EPSG:4326")
    gdf["cell_width_deg"] = lon_step
    gdf["cell_height_deg"] = lat_step
    return gdf


def load_catchments(shp_paths: list[Path]) -> gpd.GeoDataFrame:
    frames = []
    for shp in shp_paths:
        if not shp.exists():
            raise FileNotFoundError(f"Catchment shapefile not found: {shp}")
        gdf = gpd.read_file(shp)
        if gdf.empty:
            continue
        if gdf.crs is None:
            raise ValueError(f"Catchment shapefile has no CRS: {shp}")
        gdf = gdf.to_crs("EPSG:4326")
        gdf = gdf[["geometry"]].copy()
        gdf["catchment_name"] = shp.stem
        frames.append(gdf)

    if not frames:
        raise ValueError("No catchment polygons could be loaded.")

    out = pd.concat(frames, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")
    out = out[out.geometry.notna()].copy()
    out["catchment_id"] = np.arange(1, len(out) + 1)
    return out[["catchment_id", "catchment_name", "geometry"]]


def select_boundary_touching_cells_by_catchment(
    cells_gdf: gpd.GeoDataFrame,
    catchments_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    rows: list[gpd.GeoDataFrame] = []

    for _, crow in catchments_gdf.iterrows():
        poly = crow.geometry
        boundary = poly.boundary

        subset = cells_gdf.loc[
            cells_gdf.geometry.intersects(boundary)
        ].copy()

        if subset.empty:
            continue

        subset["catchment_id"] = int(crow["catchment_id"])
        subset["catchment_name"] = str(crow["catchment_name"])
        subset["cell_centroid"] = subset.geometry.centroid
        subset["centroid_inside_catchment"] = subset["cell_centroid"].within(poly)
        rows.append(subset.drop(columns=["cell_centroid"]))

    if not rows:
        return gpd.GeoDataFrame(
            columns=list(cells_gdf.columns) + ["catchment_id", "catchment_name", "centroid_inside_catchment"],
            geometry="geometry",
            crs=cells_gdf.crs,
        )

    out = pd.concat(rows, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=cells_gdf.crs)
    out["boundary_touching"] = True
    out = out.drop_duplicates(subset=["id", "catchment_id"]).reset_index(drop=True)
    return out


def select_cells_intersecting_catchments_by_catchment(
    cells_gdf: gpd.GeoDataFrame,
    catchments_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    rows: list[gpd.GeoDataFrame] = []

    for _, crow in catchments_gdf.iterrows():
        poly = crow.geometry
        subset = cells_gdf.loc[cells_gdf.geometry.intersects(poly)].copy()
        if subset.empty:
            continue
        subset["catchment_id"] = int(crow["catchment_id"])
        subset["catchment_name"] = str(crow["catchment_name"])
        rows.append(subset)

    if not rows:
        return gpd.GeoDataFrame(
            columns=list(cells_gdf.columns) + ["catchment_id", "catchment_name"],
            geometry="geometry",
            crs=cells_gdf.crs,
        )

    out = pd.concat(rows, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=cells_gdf.crs)
    out = out.drop_duplicates(subset=["id", "catchment_id"]).reset_index(drop=True)
    return out


def select_stations_inside_catchments_by_catchment(
    stations_gdf: gpd.GeoDataFrame,
    catchments_gdf: gpd.GeoDataFrame,
    include_boundary_points: bool,
) -> gpd.GeoDataFrame:
    predicate = "intersects" if include_boundary_points else "within"
    joined = gpd.sjoin(
        stations_gdf,
        catchments_gdf[["catchment_id", "catchment_name", "geometry"]],
        how="inner",
        predicate=predicate,
    ).drop(columns=["index_right"], errors="ignore")

    if joined.empty:
        return gpd.GeoDataFrame(
            columns=list(stations_gdf.columns) + ["catchment_id", "catchment_name"],
            geometry="geometry",
            crs=stations_gdf.crs,
        )

    joined["inside_catchment"] = True
    joined = joined.drop_duplicates(subset=["ID", "catchment_id"]).reset_index(drop=True)
    return joined


def compute_nearest_station_for_boundary_cells(
    boundary_cells: gpd.GeoDataFrame,
    all_stations: gpd.GeoDataFrame,
    project_epsg: int,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    if boundary_cells.empty:
        empty_cells = boundary_cells.copy()
        empty_summary = pd.DataFrame(columns=[
            "catchment_id",
            "catchment_name",
            "n_boundary_cells",
            "n_unique_nearest_stations",
            "mean_nearest_distance_m",
            "mean_nearest_distance_km",
        ])
        empty_nearest = gpd.GeoDataFrame(
            columns=list(all_stations.columns) + ["used_by_boundary_cells"],
            geometry="geometry",
            crs=all_stations.crs,
        )
        return empty_cells, empty_summary, empty_nearest

    if all_stations.empty:
        out = boundary_cells.copy()
        out["nearest_station_id"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        out["nearest_distance_m"] = np.nan
        out["nearest_distance_km"] = np.nan
        out["nearest_station_lon"] = np.nan
        out["nearest_station_lat"] = np.nan
        summary_df = pd.DataFrame(columns=[
            "catchment_id",
            "catchment_name",
            "n_boundary_cells",
            "n_unique_nearest_stations",
            "mean_nearest_distance_m",
            "mean_nearest_distance_km",
        ])
        empty_nearest = gpd.GeoDataFrame(
            columns=list(all_stations.columns) + ["used_by_boundary_cells"],
            geometry="geometry",
            crs=all_stations.crs,
        )
        return out, summary_df, empty_nearest

    bc_proj = boundary_cells.to_crs(epsg=project_epsg).copy()
    st_proj = all_stations.to_crs(epsg=project_epsg).copy()

    st_ids = st_proj["ID"].to_numpy()
    st_geoms = st_proj.geometry.to_numpy()

    nearest_ids = []
    nearest_dists_m = []
    nearest_lons = []
    nearest_lats = []

    for _, brow in bc_proj.iterrows():
        centroid = brow.geometry.centroid
        dists = np.array([centroid.distance(g) for g in st_geoms], dtype=float)
        idx = int(np.argmin(dists))
        nearest_ids.append(int(st_ids[idx]))
        nearest_dists_m.append(float(dists[idx]))

        hit = all_stations.loc[all_stations["ID"] == int(st_ids[idx])].iloc[0]
        nearest_lons.append(float(hit["Longitude"]))
        nearest_lats.append(float(hit["Latitude"]))

    bc_proj["nearest_station_id"] = pd.Series(nearest_ids, dtype="Int64")
    bc_proj["nearest_distance_m"] = nearest_dists_m
    bc_proj["nearest_distance_km"] = bc_proj["nearest_distance_m"] / 1000.0
    bc_proj["nearest_station_lon"] = nearest_lons
    bc_proj["nearest_station_lat"] = nearest_lats

    summary_df = (
        bc_proj.groupby(["catchment_id", "catchment_name"], as_index=False)
        .agg(
            n_boundary_cells=("id", "size"),
            n_unique_nearest_stations=("nearest_station_id", lambda s: int(pd.Series(s).dropna().nunique())),
            mean_nearest_distance_m=("nearest_distance_m", "mean"),
            mean_nearest_distance_km=("nearest_distance_km", "mean"),
        )
        .sort_values("catchment_name")
        .reset_index(drop=True)
    )

    used_ids = sorted(pd.Series(nearest_ids).dropna().astype(int).unique().tolist())
    nearest_stations_gdf = all_stations.loc[all_stations["ID"].isin(used_ids)].copy()
    nearest_stations_gdf["used_by_boundary_cells"] = True

    return bc_proj.to_crs("EPSG:4326"), summary_df, nearest_stations_gdf


def compute_mean_distance_to_nearest_n_gauges(
    cells_for_stats: gpd.GeoDataFrame,
    all_stations: gpd.GeoDataFrame,
    n_nearest: int,
    project_epsg: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cells_for_stats.empty:
        cell_stats = pd.DataFrame(columns=[
            "id",
            "catchment_id",
            "catchment_name",
            "mean_distance_to_nearest_n_m",
            "mean_distance_to_nearest_n_km",
        ])
        summary = pd.DataFrame(columns=[
            "scope",
            "n_cells",
            "n_gauges_considered",
            "n_nearest",
            "overall_mean_distance_m",
            "overall_mean_distance_km",
        ])
        return cell_stats, summary

    if all_stations.empty:
        cell_stats = cells_for_stats[["id", "catchment_id", "catchment_name"]].copy()
        cell_stats["mean_distance_to_nearest_n_m"] = np.nan
        cell_stats["mean_distance_to_nearest_n_km"] = np.nan
        summary = pd.DataFrame([{
            "scope": "catchment_intersecting_cells",
            "n_cells": int(len(cells_for_stats)),
            "n_gauges_considered": 0,
            "n_nearest": int(n_nearest),
            "overall_mean_distance_m": np.nan,
            "overall_mean_distance_km": np.nan,
        }])
        return cell_stats, summary

    n_use = min(int(n_nearest), int(len(all_stations)))
    cells_proj = cells_for_stats.to_crs(epsg=project_epsg).copy()
    st_proj = all_stations.to_crs(epsg=project_epsg).copy()

    st_geoms = st_proj.geometry.to_numpy()
    out_rows = []

    for _, row in cells_proj.iterrows():
        centroid = row.geometry.centroid
        dists = np.array([centroid.distance(g) for g in st_geoms], dtype=float)
        nearest = np.sort(dists)[:n_use]
        mean_m = float(np.mean(nearest)) if nearest.size > 0 else np.nan

        out_rows.append({
            "id": int(row["id"]),
            "catchment_id": int(row["catchment_id"]),
            "catchment_name": str(row["catchment_name"]),
            "mean_distance_to_nearest_n_m": mean_m,
            "mean_distance_to_nearest_n_km": mean_m / 1000.0 if pd.notna(mean_m) else np.nan,
        })

    cell_stats_df = pd.DataFrame(out_rows)
    overall_mean_m = float(cell_stats_df["mean_distance_to_nearest_n_m"].mean()) if not cell_stats_df.empty else np.nan

    summary_df = pd.DataFrame([{
        "scope": "catchment_intersecting_cells",
        "n_cells": int(len(cell_stats_df)),
        "n_gauges_considered": int(len(all_stations)),
        "n_nearest": int(n_use),
        "overall_mean_distance_m": overall_mean_m,
        "overall_mean_distance_km": overall_mean_m / 1000.0 if pd.notna(overall_mean_m) else np.nan,
    }])

    return cell_stats_df, summary_df


def make_annotation_text(avg4_summary_df: pd.DataFrame) -> str:
    if avg4_summary_df.empty:
        return "Average distance to nearest 4 rain gages: not available"

    row = avg4_summary_df.iloc[0]
    if pd.isna(row["overall_mean_distance_km"]):
        return "Average distance to nearest 4 rain gages: not available"

    return (
        f"Average distance "
        f": {row['overall_mean_distance_km']:.2f} km"
    )


def save_outputs(
    cells_gdf: gpd.GeoDataFrame,
    boundary_cells_gdf: gpd.GeoDataFrame,
    catchments_gdf: gpd.GeoDataFrame,
    stations_in_gdf: gpd.GeoDataFrame,
    nearest_boundary_stations_gdf: gpd.GeoDataFrame,
    boundary_summary_df: pd.DataFrame,
    avg4_cell_stats_df: pd.DataFrame,
    avg4_summary_df: pd.DataFrame,
    out_dir: Path,
    figure_width: float,
    figure_height: float,
    dpi: int,
    axis_label_fontsize: float,
    tick_fontsize: float,
    annotation_fontsize: float,
    station_marker_size: float,
    nearest_station_marker_size: float,
    boundary_cell_linewidth: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    cells_csv = out_dir / "grid_cells_touching_catchment_boundaries.csv"
    stations_csv = out_dir / "gage_stations_inside_catchments.csv"
    nearest_stations_csv = out_dir / "nearest_stations_for_boundary_cells.csv"
    boundary_summary_csv = out_dir / "boundary_cell_nearest_station_summary.csv"
    avg4_cells_csv = out_dir / "cell_average_distance_to_nearest_4_gages.csv"
    avg4_summary_csv = out_dir / "overall_average_distance_to_nearest_4_gages.csv"
    gpkg_path = out_dir / "grid_cells_touching_catchment_boundaries.gpkg"
    png_path = out_dir / "grid_cells_touching_catchment_boundaries.png"

    boundary_cells_gdf.drop(columns="geometry").to_csv(cells_csv, index=False)
    stations_in_gdf.drop(columns="geometry").to_csv(stations_csv, index=False)
    nearest_boundary_stations_gdf.drop(columns="geometry").to_csv(nearest_stations_csv, index=False)
    boundary_summary_df.to_csv(boundary_summary_csv, index=False)
    avg4_cell_stats_df.to_csv(avg4_cells_csv, index=False)
    # ---- histogram of average distance ----
    hist_png = out_dir / "hist_avg_distance_to_nearest_4_gages.png"

    vals = avg4_cell_stats_df["mean_distance_to_nearest_n_km"].dropna().to_numpy()

    if vals.size > 0:
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.hist(vals, bins=20, alpha=0.7)

        ax.set_xlabel("Average distance to nearest 4 gages (km)", fontsize=14)
        ax.set_ylabel("Number of cells", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(hist_png, dpi=dpi)
        plt.close(fig)

        print(f"  Histogram                   : {hist_png}")
    avg4_summary_df.to_csv(avg4_summary_csv, index=False)

    cells_gdf.to_file(gpkg_path, layer="all_grid_cells", driver="GPKG")
    boundary_cells_gdf.to_file(gpkg_path, layer="boundary_touching_cells", driver="GPKG")
    catchments_gdf.to_file(gpkg_path, layer="catchments", driver="GPKG")
    stations_in_gdf.to_file(gpkg_path, layer="stations_inside_catchments", driver="GPKG")
    if not nearest_boundary_stations_gdf.empty:
        nearest_boundary_stations_gdf.to_file(gpkg_path, layer="nearest_stations_for_boundary_cells", driver="GPKG")

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    catchment_colors = {
        "6893390": "#1f77b4",
        "6893080": "#2ca02c",
        "6892513": "#ff7f0e",
    }

    cells_gdf.boundary.plot(ax=ax, linewidth=0.2, color="lightgray", zorder=1)

    for _, row in catchments_gdf.iterrows():
        cname = row["catchment_name"]
        c = catchment_colors.get(cname, "lightgray")
        gpd.GeoSeries([row.geometry], crs=catchments_gdf.crs).plot(
            ax=ax,
            facecolor=c,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.22,
            zorder=1.5,
        )

    if not boundary_cells_gdf.empty:
        inside = boundary_cells_gdf.loc[boundary_cells_gdf["centroid_inside_catchment"] == True].copy()
        outside = boundary_cells_gdf.loc[boundary_cells_gdf["centroid_inside_catchment"] == False].copy()

        if not inside.empty:
            inside.plot(
                ax=ax,
                facecolor="red",
                edgecolor="red",
                alpha=0.18,
                linewidth=boundary_cell_linewidth,
                zorder=3,
            )

        if not outside.empty:
            outside.plot(
                ax=ax,
                facecolor="red",
                edgecolor="red",
                alpha=0.12,
                linewidth=boundary_cell_linewidth,
                zorder=3,
            )

    if not stations_in_gdf.empty:
        stations_in_gdf.plot(
            ax=ax,
            marker="o",
            color="grey",
            edgecolor="black",
            markersize=station_marker_size,
            zorder=5,
        )

    if not nearest_boundary_stations_gdf.empty:
        nearest_boundary_stations_gdf.plot(
            ax=ax,
            marker="o",
            color="grey",
            edgecolor="black",
            markersize=nearest_station_marker_size,
            zorder=6,
        )

    ax.set_xlabel("Longitude", fontsize=axis_label_fontsize)
    ax.set_ylabel("Latitude", fontsize=axis_label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    annotation_text = make_annotation_text(avg4_summary_df)
    ax.text(
        0.98,
        0.98,
        annotation_text,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=annotation_fontsize,
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85, boxstyle="round,pad=0.3"),
    )

    plt.tight_layout()
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(f"  CSV (boundary cells)        : {cells_csv}")
    print(f"  CSV (inside gauges)         : {stations_csv}")
    print(f"  CSV (nearest boundary stn)  : {nearest_stations_csv}")
    print(f"  CSV (boundary summary)      : {boundary_summary_csv}")
    print(f"  CSV (avg nearest 4 by cell) : {avg4_cells_csv}")
    print(f"  CSV (avg nearest 4 overall) : {avg4_summary_csv}")
    print(f"  GPKG                        : {gpkg_path}")
    print(f"  PNG                         : {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select WGS grid cells touching catchment boundaries, show all gauges inside catchments, "
            "highlight stations that are nearest to any border cell, and compute average distance from "
            "each catchment-intersecting pixel to the nearest 4 rain gages."
        )
    )
    parser.add_argument("--grid-csv", type=Path, default=GRID_CSV)
    parser.add_argument("--stations-csv", type=Path, default=STATIONS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--project-epsg", type=int, default=PROJECT_EPSG)
    parser.add_argument("--include-boundary-stations", action="store_true", help="Use intersects instead of within for gauge selection inside catchments.")
    parser.add_argument("--n-nearest", type=int, default=4, help="Number of nearest gauges to use for pixel-average distance calculation.")
    parser.add_argument("--figure-width", type=float, default=12.0)
    parser.add_argument("--figure-height", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--axis-label-fontsize", type=float, default=24.0)
    parser.add_argument("--tick-fontsize", type=float, default=20.0)
    parser.add_argument("--annotation-fontsize", type=float, default=20.0)
    parser.add_argument("--station-marker-size", type=float, default=100.0)
    parser.add_argument("--nearest-station-marker-size", type=float, default=120.0)
    parser.add_argument("--boundary-cell-linewidth", type=float, default=0.9)
    parser.add_argument(
        "--catchments",
        nargs="*",
        type=Path,
        default=CATCHMENT_SHP_PATHS,
        help="One or more catchment shapefiles. Defaults to the usual three.",
    )
    args = parser.parse_args()

    grid_df = load_grid(args.grid_csv)
    stations_gdf = load_stations(args.stations_csv)
    lat_step = infer_grid_spacing(grid_df["Latitude"].to_numpy(), "latitude")
    lon_step = infer_grid_spacing(grid_df["Longitude"].to_numpy(), "longitude")

    print(f"Inferred cell height (deg): {lat_step}")
    print(f"Inferred cell width  (deg): {lon_step}")

    cells_gdf = build_cell_polygons(grid_df, lat_step=lat_step, lon_step=lon_step)
    catchments_gdf = load_catchments(list(args.catchments))

    boundary_cells_gdf = select_boundary_touching_cells_by_catchment(cells_gdf, catchments_gdf)
    catchment_pixels_gdf = select_cells_intersecting_catchments_by_catchment(cells_gdf, catchments_gdf)

    stations_in_gdf = select_stations_inside_catchments_by_catchment(
        stations_gdf,
        catchments_gdf,
        include_boundary_points=args.include_boundary_stations,
    )

    boundary_cells_gdf, boundary_summary_df, nearest_boundary_stations_gdf = compute_nearest_station_for_boundary_cells(
        boundary_cells=boundary_cells_gdf,
        all_stations=stations_gdf,
        project_epsg=args.project_epsg,
    )

    avg4_cell_stats_df, avg4_summary_df = compute_mean_distance_to_nearest_n_gauges(
        cells_for_stats=catchment_pixels_gdf,
        all_stations=stations_gdf,
        n_nearest=args.n_nearest,
        project_epsg=args.project_epsg,
    )

    print(f"Total grid cells                          : {len(cells_gdf)}")
    print(f"Boundary-touching cell records            : {len(boundary_cells_gdf)}")
    print(f"Catchment-intersecting pixels             : {len(catchment_pixels_gdf)}")
    print(f"Stations inside catchments                : {len(stations_in_gdf)}")
    print(f"Unique nearest stations for boundary cells: {len(nearest_boundary_stations_gdf)}")

    if not boundary_summary_df.empty:
        print("\nBoundary-cell nearest-station summary:")
        print(boundary_summary_df.to_string(index=False))

    if not avg4_summary_df.empty:
        print("\nAverage distance:")
        print(avg4_summary_df.to_string(index=False))

    save_outputs(
        cells_gdf=cells_gdf,
        boundary_cells_gdf=boundary_cells_gdf,
        catchments_gdf=catchments_gdf,
        stations_in_gdf=stations_in_gdf,
        nearest_boundary_stations_gdf=nearest_boundary_stations_gdf,
        boundary_summary_df=boundary_summary_df,
        avg4_cell_stats_df=avg4_cell_stats_df,
        avg4_summary_df=avg4_summary_df,
        out_dir=args.out_dir,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        dpi=args.dpi,
        axis_label_fontsize=args.axis_label_fontsize,
        tick_fontsize=args.tick_fontsize,
        annotation_fontsize=args.annotation_fontsize,
        station_marker_size=args.station_marker_size,
        nearest_station_marker_size=args.nearest_station_marker_size,
        boundary_cell_linewidth=args.boundary_cell_linewidth,
    )


if __name__ == "__main__":
    main()