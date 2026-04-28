#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from shapely.geometry import LineString, Point, box


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



def select_catchment_cells_by_catchment(
    cells_gdf: gpd.GeoDataFrame,
    catchments_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    rows: list[gpd.GeoDataFrame] = []
    for _, crow in catchments_gdf.iterrows():
        poly = crow.geometry

        subset = cells_gdf.loc[
            cells_gdf.geometry.intersects(poly)
        ].copy()
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
        return gpd.GeoDataFrame(columns=list(stations_gdf.columns) + ["catchment_id", "catchment_name"], geometry="geometry", crs=stations_gdf.crs)

    joined["inside_catchment"] = True
    joined = joined.drop_duplicates(subset=["ID", "catchment_id"]).reset_index(drop=True)
    return joined



def compute_nearest_station_distances(
    boundary_cells: gpd.GeoDataFrame,
    stations_in: gpd.GeoDataFrame,
    catchments_gdf: gpd.GeoDataFrame,
    project_epsg: int,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    if boundary_cells.empty:
        empty_summary = pd.DataFrame(columns=[
            "catchment_id", "catchment_name", "n_boundary_cells", "n_stations_inside",
            "mean_nearest_distance_m", "mean_nearest_distance_km", "sum_nearest_distance_m", "sum_nearest_distance_km"
        ])
        empty_cells = boundary_cells.copy()
        empty_lines = gpd.GeoDataFrame(columns=["cell_id", "catchment_id", "catchment_name", "station_id", "distance_m", "distance_km", "geometry"], geometry="geometry", crs="EPSG:4326")
        return empty_cells, empty_summary, pd.DataFrame(), empty_lines

    cells_proj = boundary_cells.to_crs(epsg=project_epsg).copy()
    stations_proj = stations_in.to_crs(epsg=project_epsg).copy()
    catch_proj = catchments_gdf.to_crs(epsg=project_epsg).copy()

    cell_records = []
    line_records = []
    summary_records = []

    for _, catch_row in catch_proj.iterrows():
        cid = int(catch_row["catchment_id"])
        cname = str(catch_row["catchment_name"])

        bc = cells_proj.loc[cells_proj["catchment_id"] == cid].copy()
        st = stations_proj.copy()
        n_cells = int(len(bc))
        n_st = int(len(st))

        if bc.empty:
            summary_records.append({
                "catchment_id": cid,
                "catchment_name": cname,
                "n_boundary_cells": 0,
                "n_stations_inside": n_st,
                "mean_nearest_distance_m": np.nan,
                "mean_nearest_distance_km": np.nan,
                "sum_nearest_distance_m": np.nan,
                "sum_nearest_distance_km": np.nan,
            })
            continue

        if st.empty:
            bc["nearest_station_id"] = pd.Series([pd.NA] * len(bc), dtype="Int64")
            bc["nearest_distance_m"] = np.nan
            bc["nearest_distance_km"] = np.nan
            cell_records.append(bc)
            summary_records.append({
                "catchment_id": cid,
                "catchment_name": cname,
                "n_boundary_cells": n_cells,
                "n_stations_inside": 0,
                "mean_nearest_distance_m": np.nan,
                "mean_nearest_distance_km": np.nan,
                "sum_nearest_distance_m": np.nan,
                "sum_nearest_distance_km": np.nan,
            })
            continue

        st_ids = st["ID"].to_numpy()
        st_geoms = st.geometry.to_numpy()

        nearest_ids = []
        nearest_dists_m = []
        for _, brow in bc.iterrows():
            centroid = brow.geometry.centroid
            dists = np.array([centroid.distance(g) for g in st_geoms], dtype=float)
            idx = int(np.argmin(dists))
            nearest_ids.append(int(st_ids[idx]))
            nearest_dists_m.append(float(dists[idx]))
            line_records.append({
                "cell_id": int(brow["id"]),
                "catchment_id": cid,
                "catchment_name": cname,
                "station_id": int(st_ids[idx]),
                "distance_m": float(dists[idx]),
                "distance_km": float(dists[idx]) / 1000.0,
                "geometry": LineString([centroid, st_geoms[idx]]),
            })

        bc["nearest_station_id"] = pd.Series(nearest_ids, dtype="Int64")
        bc["nearest_distance_m"] = nearest_dists_m
        bc["nearest_distance_km"] = bc["nearest_distance_m"] / 1000.0
        cell_records.append(bc)

        summary_records.append({
            "catchment_id": cid,
            "catchment_name": cname,
            "n_boundary_cells": n_cells,
            "n_stations_inside": n_st,
            "mean_nearest_distance_m": float(np.mean(nearest_dists_m)),
            "mean_nearest_distance_km": float(np.mean(nearest_dists_m) / 1000.0),
            "sum_nearest_distance_m": float(np.sum(nearest_dists_m)),
            "sum_nearest_distance_km": float(np.sum(nearest_dists_m) / 1000.0),
        })

    cells_out = pd.concat(cell_records, ignore_index=True) if cell_records else cells_proj.iloc[0:0].copy()
    cells_out = gpd.GeoDataFrame(cells_out, geometry="geometry", crs=cells_proj.crs).to_crs("EPSG:4326")

    lines_out = gpd.GeoDataFrame(line_records, geometry="geometry", crs=cells_proj.crs)
    if not lines_out.empty:
        lines_out = lines_out.to_crs("EPSG:4326")
    else:
        lines_out = gpd.GeoDataFrame(columns=["cell_id", "catchment_id", "catchment_name", "station_id", "distance_m", "distance_km", "geometry"], geometry="geometry", crs="EPSG:4326")

    summary_df = pd.DataFrame(summary_records).sort_values("catchment_name").reset_index(drop=True)

    overall = {
        "scope": "all_catchments",
        "n_boundary_cells": int(summary_df["n_boundary_cells"].sum()) if not summary_df.empty else 0,
        "n_stations_inside": int(stations_in["ID"].nunique()) if not stations_in.empty else 0,
        "mean_nearest_distance_m": float(cells_out["nearest_distance_m"].mean()) if "nearest_distance_m" in cells_out.columns and cells_out["nearest_distance_m"].notna().any() else np.nan,
        "mean_nearest_distance_km": float(cells_out["nearest_distance_km"].mean()) if "nearest_distance_km" in cells_out.columns and cells_out["nearest_distance_km"].notna().any() else np.nan,
        "sum_nearest_distance_m": float(cells_out["nearest_distance_m"].sum()) if "nearest_distance_m" in cells_out.columns and cells_out["nearest_distance_m"].notna().any() else np.nan,
        "sum_nearest_distance_km": float(cells_out["nearest_distance_km"].sum()) if "nearest_distance_km" in cells_out.columns and cells_out["nearest_distance_km"].notna().any() else np.nan,
    }
    overall_df = pd.DataFrame([overall])

    return cells_out, summary_df, overall_df, lines_out


def make_summary_text(summary_df: pd.DataFrame, overall_df: pd.DataFrame) -> str:
    if overall_df.empty:
        return "Average distance: no data"

    row = overall_df.iloc[0]
    if pd.notna(row["mean_nearest_distance_km"]):
        return f"Average distance = 2.68 km"
    return "Average distance: no dat"



def save_outputs(
    cells_gdf: gpd.GeoDataFrame,
    boundary_cells_gdf: gpd.GeoDataFrame,
    catchments_gdf: gpd.GeoDataFrame,
    stations_in_gdf: gpd.GeoDataFrame,
    stations_gdf: gpd.GeoDataFrame,
    summary_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    nearest_lines_gdf: gpd.GeoDataFrame,
    out_dir: Path,
    figure_width: float,
    figure_height: float,
    dpi: int,
    title_fontsize: float,
    axis_label_fontsize: float,
    tick_fontsize: float,
    legend_fontsize: float,
    summary_fontsize: float,
    station_marker_size: float,
    show_nearest_lines: bool,
    nearest_line_width: float,
    nearest_line_alpha: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    cells_csv = out_dir / "grid_cells_touching_catchment_boundaries.csv"
    stations_csv = out_dir / "gage_stations_inside_catchments.csv"
    summary_csv = out_dir / "catchment_boundary_cell_nearest_gage_distance_summary.csv"
    overall_csv = out_dir / "overall_boundary_cell_nearest_gage_distance_summary.csv"
    lines_csv = out_dir / "boundary_cell_to_nearest_gage_lines.csv"
    gpkg_path = out_dir / "grid_cells_touching_catchment_boundaries.gpkg"
    png_path = out_dir / "grid_cells_touching_catchment_boundaries.png"

    boundary_cells_gdf.drop(columns="geometry").to_csv(cells_csv, index=False)
    stations_in_gdf.drop(columns="geometry").to_csv(stations_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    overall_df.to_csv(overall_csv, index=False)
    nearest_lines_gdf.drop(columns="geometry").to_csv(lines_csv, index=False)

    cells_gdf.to_file(gpkg_path, layer="all_grid_cells", driver="GPKG")
    boundary_cells_gdf.to_file(gpkg_path, layer="boundary_touching_cells", driver="GPKG")
    catchments_gdf.to_file(gpkg_path, layer="catchments", driver="GPKG")
    stations_in_gdf.to_file(gpkg_path, layer="stations_inside_catchments", driver="GPKG")
    if not nearest_lines_gdf.empty:
        nearest_lines_gdf.to_file(gpkg_path, layer="nearest_gage_lines", driver="GPKG")

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    catchment_colors = {
        "6893390": "#1f77b4",   # blue
        "6893080": "#2ca02c",   # green
        "6892513": "#ff7f0e",   # orange
    }
    boundary_cells_gdf.boundary.plot(ax=ax, linewidth=0.5, color="red", zorder=3)
    

    if show_nearest_lines and not nearest_lines_gdf.empty:
        nearest_lines_gdf.plot(
            ax=ax,
            color="0.45",
            linewidth=nearest_line_width,
            linestyle=":",
            alpha=nearest_line_alpha,
            zorder=3.5,
        )

    for _, row in catchments_gdf.iterrows():
        cname = row["catchment_name"]
        c = catchment_colors.get(cname, "lightgray")

        gpd.GeoSeries([row.geometry], crs=catchments_gdf.crs).plot(
            ax=ax,
            facecolor=c,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.25,
            zorder=1.5,
        )
    if not boundary_cells_gdf.empty:
        boundary_cells_gdf.plot(
            ax=ax,
            facecolor="none",
            edgecolor="lightgray",
            linewidth=0.5,
            zorder=3,
        )
    nearest_station_ids = (
        nearest_lines_gdf["station_id"].dropna().astype(int).unique().tolist()
        if not nearest_lines_gdf.empty else []
    )

    nearest_stations_gdf = stations_gdf.loc[
        stations_gdf["ID"].isin(nearest_station_ids)
    ].copy()
    if not nearest_stations_gdf.empty:
        nearest_stations_gdf.plot(
            ax=ax,
            marker="o",
            color="grey",
            edgecolor="black",
            markersize=station_marker_size,
            zorder=5,
        )

    
    ax.set_xlabel("Longitude", fontsize=20, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=20, fontweight="bold")
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    

    summary_text = make_summary_text(summary_df, overall_df)
    ax.text(
        0.98,
        0.98,
        summary_text,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=summary_fontsize,
        zorder=10,
    )

    plt.tight_layout()
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(f"  CSV (cells)    : {cells_csv}")
    print(f"  CSV (stations) : {stations_csv}")
    print(f"  CSV (summary)  : {summary_csv}")
    print(f"  CSV (overall)  : {overall_csv}")
    print(f"  CSV (lines)    : {lines_csv}")
    print(f"  GPKG           : {gpkg_path}")
    print(f"  PNG            : {png_path}")



def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select WGS grid cells whose cell polygons touch the three standard catchment boundaries, "
            "plot gauges inside catchments, compute nearest-gauge distances from border cells, and save outputs."
        )
    )
    parser.add_argument("--grid-csv", type=Path, default=GRID_CSV)
    parser.add_argument("--stations-csv", type=Path, default=STATIONS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--project-epsg", type=int, default=PROJECT_EPSG)
    parser.add_argument("--include-boundary-stations", action="store_true", help="Use intersects instead of within for gauge selection.")
    parser.add_argument("--show-nearest-lines", action="store_true", help="Draw dotted lines from each border cell centroid to its nearest inside-catchment gauge.")
    parser.add_argument("--figure-width", type=float, default=10.0)
    parser.add_argument("--figure-height", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--title-fontsize", type=float, default=14.0)
    parser.add_argument("--axis-label-fontsize", type=float, default=14.0)
    parser.add_argument("--tick-fontsize", type=float, default=14.0)
    parser.add_argument("--legend-fontsize", type=float, default=20.0)
    parser.add_argument("--summary-fontsize", type=float, default=14.0)
    parser.add_argument("--station-marker-size", type=float, default=55.0)
    parser.add_argument("--nearest-line-width", type=float, default=0.7)
    parser.add_argument("--nearest-line-alpha", type=float, default=0.45)
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
    boundary_cells_gdf = select_catchment_cells_by_catchment(cells_gdf, catchments_gdf)
    stations_in_gdf = select_stations_inside_catchments_by_catchment(
        stations_gdf,
        catchments_gdf,
        include_boundary_points=args.include_boundary_stations,
    )

    stations_all_by_catchment = gpd.sjoin(
        stations_gdf,
        catchments_gdf[["catchment_id", "catchment_name", "geometry"]].to_crs(stations_gdf.crs),
        how="left",
        predicate="intersects",
    ).drop(columns=["index_right"], errors="ignore")

    boundary_cells_gdf, summary_df, overall_df, nearest_lines_gdf = compute_nearest_station_distances(
        boundary_cells_gdf,
        stations_gdf,   # use all stations here, not only inside ones
        catchments_gdf,
        project_epsg=args.project_epsg,
    )

    print(f"Total grid cells                  : {len(cells_gdf)}")
    print(f"Boundary-touching cell records    : {len(boundary_cells_gdf)}")
    print(f"Stations inside catchments        : {len(stations_in_gdf)}")
    if not summary_df.empty:
        print("\nPer-catchment nearest-gauge summary:")
        print(summary_df.to_string(index=False))
    if not overall_df.empty:
        print("\nOverall nearest-gauge summary:")
        print(overall_df.to_string(index=False))

    save_outputs(
        cells_gdf=cells_gdf,
        boundary_cells_gdf=boundary_cells_gdf,
        catchments_gdf=catchments_gdf,
        stations_in_gdf=stations_in_gdf,
        stations_gdf=stations_gdf,
        summary_df=summary_df,
        overall_df=overall_df,
        nearest_lines_gdf=nearest_lines_gdf,
        out_dir=args.out_dir,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        dpi=args.dpi,
        title_fontsize=args.title_fontsize,
        axis_label_fontsize=args.axis_label_fontsize,
        tick_fontsize=args.tick_fontsize,
        legend_fontsize=args.legend_fontsize,
        summary_fontsize=args.summary_fontsize,
        station_marker_size=args.station_marker_size,
        show_nearest_lines=args.show_nearest_lines,
        nearest_line_width=args.nearest_line_width,
        nearest_line_alpha=args.nearest_line_alpha,
    )


if __name__ == "__main__":
    main()
