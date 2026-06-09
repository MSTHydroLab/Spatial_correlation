#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import box
'''
  
  python 10_plot_rainfall_difference.py \
  --csv1 "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/Radar_Event_TimeSeries/MRMS_Event_TimeSeries/Event_7/Event_7_grid_rain_hourly_mm_MRMS.csv" \
  --csv2 \
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/03_Interpolated_Rain/Event_7_grid_rain_hourly_mm.csv"\
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_IDW/03_Interpolated_Rain/Event_7_grid_rain_hourly_mm.csv"\
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/Radar_Event_TimeSeries/RA/Event_7/Event_7_grid_rain_hourly_mm_RA.csv" \
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/Radar_Event_TimeSeries/RZ/Event_7/Event_7_grid_rain_hourly_mm_RZ.csv"\
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/Radar_Event_TimeSeries/RKDP/Event_7/Event_7_grid_rain_hourly_mm_RKDP.csv"\
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/Radar_Event_TimeSeries/Composite_2/Event_7/Event_7_grid_rain_hourly_mm_Composite_2.csv" \
  --grid-csv "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/dependent_files/grid_centers_wgs84.csv"\
  --catchments \
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/06893080/6893080.shp" \
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6892513/6892513.shp" \
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6893390/6893390.shp" \
  --out "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/10_difference_plots/Event_7/all_methods_minus_mrms.png" \
  --one-figure \
  --subplot-ncols 2 \
  --fig-width 11 \
  --fig-height 12
  
  python 10_plot_rainfall_difference.py \
  --csv1 "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/Radar_Event_TimeSeries/MRMS_Event_TimeSeries/Event_4/Event_4_grid_rain_hourly_mm_MRMS.csv" \
  --csv2 \
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/03_Interpolated_Rain/Event_4_grid_rain_hourly_mm.csv"\
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_IDW/03_Interpolated_Rain/Event_4_grid_rain_hourly_mm.csv"\
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/Radar_Event_TimeSeries/RA/Event_4/Event_4_grid_rain_hourly_mm_RA.csv" \
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/Radar_Event_TimeSeries/RZ/Event_4/Event_4_grid_rain_hourly_mm_RZ.csv"\
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/Radar_Event_TimeSeries/RKDP/Event_4/Event_4_grid_rain_hourly_mm_RKDP.csv"\
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/Radar_Event_TimeSeries/Composite_2/Event_4/Event_4_grid_rain_hourly_mm_Composite_2.csv" \
  --grid-csv "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/dependent_files/grid_centers_wgs84.csv"\
  --catchments \
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/06893080/6893080.shp" \
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6892513/6892513.shp" \
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6893390/6893390.shp" \
  --out "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/10_difference_plots/Event_4/all_methods_minus_mrms.png" \
  --one-figure \
  --subplot-ncols 2 \
  --fig-width 11 \
  --fig-height 12
  '''

# -------------------------
# Helpers
# -------------------------

def normalize_grid_col(col) -> str:
    s = str(col).strip()
    try:
        f = float(s)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return s
    except Exception:
        return s


def safe_stem(path: Path) -> str:
    """Create a clean filename stem from a CSV path."""
    stem = path.stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    return stem.strip("_") or "csv2"


def load_grid(grid_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(grid_csv)

    required = ["id", "Latitude", "Longitude"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{grid_csv} missing required columns: {missing}")

    df = df[required].copy()
    df["id"] = df["id"].apply(normalize_grid_col)
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["id", "Latitude", "Longitude"]).copy()

    return df


def infer_spacing(values: pd.Series, name: str) -> float:
    vals = np.sort(pd.to_numeric(values, errors="coerce").dropna().unique())
    diffs = np.diff(vals)
    diffs = diffs[diffs > 1e-12]

    if diffs.size == 0:
        raise ValueError(f"Could not infer {name} spacing from grid centers.")

    return float(np.median(diffs))


def build_cells(grid_df: pd.DataFrame) -> gpd.GeoDataFrame:
    dlat = infer_spacing(grid_df["Latitude"], "latitude")
    dlon = infer_spacing(grid_df["Longitude"], "longitude")

    half_lat = dlat / 2.0
    half_lon = dlon / 2.0

    geoms = [
        box(lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)
        for lat, lon in zip(grid_df["Latitude"], grid_df["Longitude"])
    ]

    return gpd.GeoDataFrame(grid_df.copy(), geometry=geoms, crs="EPSG:4326")


def load_catchments(paths: list[Path]) -> gpd.GeoDataFrame:
    gdfs = []

    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Catchment shapefile not found: {p}")

        g = gpd.read_file(p)
        if g.empty:
            continue
        if g.crs is None:
            raise ValueError(f"Catchment shapefile has no CRS: {p}")

        g = g.to_crs("EPSG:4326")
        gdfs.append(g[["geometry"]].copy())

    if not gdfs:
        raise ValueError("No valid catchment geometries loaded.")

    out = pd.concat(gdfs, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")


def load_event_sum(csv_path: Path) -> pd.Series:
    """
    Load event rainfall CSV and sum rainfall through time for each grid cell.

    Expected format:
        first column = time column
        remaining columns = grid IDs
    """
    df = pd.read_csv(csv_path)

    if df.shape[1] < 2:
        raise ValueError(f"{csv_path} must have at least one time column and one grid column.")

    time_col = df.columns[0]
    data = df.drop(columns=[time_col]).copy()

    data.columns = [normalize_grid_col(c) for c in data.columns]

    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    sums = data.sum(axis=0, skipna=True, min_count=1)
    sums.index = sums.index.astype(str)

    return sums


def shapely_to_patches(geom):
    patches = []

    if geom is None or geom.is_empty:
        return patches

    if geom.geom_type == "Polygon":
        patches.append(MplPolygon(np.asarray(geom.exterior.coords)))
    elif geom.geom_type == "MultiPolygon":
        for g in geom.geoms:
            patches.extend(shapely_to_patches(g))

    return patches


def resolve_output_path(out_arg: Path, csv1: Path, csv2: Path, n_csv2: int) -> Path:
    """
    If --out is a folder, create one file per csv2.
    If --out is a file and only one csv2 is given, use it directly.
    If --out is a file and multiple csv2 files are given, use its parent folder.
    """
    out_arg = Path(out_arg)

    if out_arg.suffix.lower() in [".png", ".jpg", ".jpeg", ".pdf", ".tif", ".tiff"]:
        if n_csv2 == 1:
            return out_arg

        out_dir = out_arg.parent
    else:
        out_dir = out_arg

    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir / f"diff_{safe_stem(csv2)}_minus_{safe_stem(csv1)}.png"

def compute_global_diff_limit(
    *,
    csv1: Path,
    csv2_list: list[Path],
    cells: gpd.GeoDataFrame,
    catch_union,
) -> float:
    sum1 = load_event_sum(csv1)

    global_vmax = 0.0

    for csv2 in csv2_list:
        sum2 = load_event_sum(csv2)

        common_ids = sum1.index.intersection(sum2.index)
        if len(common_ids) == 0:
            continue

        diff = sum2.loc[common_ids].subtract(sum1.loc[common_ids], fill_value=np.nan)

        plot = cells.copy()
        plot["diff"] = plot["id"].astype(str).map(diff)
        plot = plot[plot.geometry.intersects(catch_union)].copy()
        plot = plot[plot["diff"].notna()].copy()

        if plot.empty:
            continue

        vals = plot["diff"].to_numpy(dtype=float)
        this_vmax = np.nanmax(np.abs(vals))

        if np.isfinite(this_vmax):
            global_vmax = max(global_vmax, float(this_vmax))

    if global_vmax <= 0 or not np.isfinite(global_vmax):
        global_vmax = 1.0

    return global_vmax

def plot_difference(
    *,
    csv1: Path,
    csv2: Path,
    cells: gpd.GeoDataFrame,
    catchments: gpd.GeoDataFrame,
    catch_union,
    out_png: Path,
    label_csv1: str | None = None,
    label_csv2: str | None = None,
    global_vmax: float | None = None,
    fig_width: float = 9.0,
    fig_height: float = 8.0,
    dpi: int = 300,
    axis_label_size: float = 16.0,
    tick_label_size: float = 13.0,
    colorbar_label_size: float = 14.0,
    colorbar_tick_size: float = 12.0,
    boundary_linewidth: float = 1.2,
    ) -> None:
    sum1 = load_event_sum(csv1)
    sum2 = load_event_sum(csv2)

    common_ids = sum1.index.intersection(sum2.index)
    if len(common_ids) == 0:
        raise ValueError(f"No common grid IDs between:\n  csv1={csv1}\n  csv2={csv2}")

    # Difference requested: csv2 - csv1
    diff = sum2.loc[common_ids].subtract(sum1.loc[common_ids], fill_value=np.nan)

    plot = cells.copy()
    plot["diff"] = plot["id"].astype(str).map(diff)

    plot = plot[plot.geometry.intersects(catch_union)].copy()
    plot = plot[plot["diff"].notna()].copy()

    if plot.empty:
        raise ValueError(f"No valid plotted cells for csv2={csv2}")

    vals = plot["diff"].to_numpy(dtype=float)

    if global_vmax is None:
        vmax = float(np.nanmax(np.abs(vals)))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
    else:
        vmax = float(global_vmax)

    vmin = -vmax

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    patches = []
    patch_vals = []

    for _, row in plot.iterrows():
        geom_patches = shapely_to_patches(row.geometry)
        patches.extend(geom_patches)
        patch_vals.extend([row["diff"]] * len(geom_patches))

    pc = PatchCollection(
        patches,
        cmap="RdBu_r",
        edgecolor="none",
        linewidth=0,
    )
    pc.set_array(np.asarray(patch_vals, dtype=float))
    pc.set_clim(vmin, vmax)

    ax.add_collection(pc)
    catchments.boundary.plot(ax=ax, color="black", linewidth=boundary_linewidth)

    ax.set_aspect("equal")
    ax.set_xlabel("Longitude", fontsize=axis_label_size, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=axis_label_size, fontweight="bold")
    ax.tick_params(axis="both", labelsize=tick_label_size)

    cbar = plt.colorbar(
        pc,
        ax=ax,
        shrink=0.70,      # shorter colorbar
        fraction=0.045,   # thinner colorbar
        pad=0.03,         # distance from plot
    )
    cbar.set_label("Rainfall Difference (mm)\nCSV2 - CSV1",fontsize=colorbar_label_size,fontweight="bold",)
    cbar.ax.tick_params(labelsize=colorbar_tick_size)

    # Keep title optional by not adding one. The filename carries the comparison.
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    name1 = label_csv1 or csv1.name
    name2 = label_csv2 or csv2.name

    print(f"[saved] {out_png}")
    print(f"  csv1/main : {name1}")
    print(f"  csv2/other: {name2}")
    print(f"  plotted   : csv2 - csv1")
    print(f"  cells     : {len(plot)}")
    print(f"  range     : {np.nanmin(vals):.2f} to {np.nanmax(vals):.2f} mm")
    print(f"  mean diff : {np.nanmean(vals):.2f} mm")
    print()


# -------------------------
# Main
# -------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Plot event-total rainfall difference maps. "
            "csv1 is the main/reference file. Each csv2 file is plotted as csv2 - csv1."
        )
    )

    ap.add_argument("--csv1", required=True, type=Path, help="Main/reference rainfall CSV.")
    ap.add_argument(
        "--csv2",
        required=True,
        nargs="+",
        type=Path,
        help="One or more rainfall CSVs to compare against csv1. Difference is csv2 - csv1.",
    )
    ap.add_argument("--grid-csv", required=True, type=Path)
    ap.add_argument("--catchments", nargs="+", required=True, type=Path)
    ap.add_argument(
        "--out",
        required=True,
        type=Path,
        help=(
            "Output PNG path if one csv2 is used, or output folder if multiple csv2 files are used."
        ),
    )
    ap.add_argument(
        "--one-figure",
        action="store_true",
        help="Save all csv2 comparisons in one combined figure with a shared colorbar.",
    )

    ap.add_argument(
        "--subplot-ncols",
        type=int,
        default=2,
        help="Number of subplot columns when using --one-figure.",
    )
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--fig-width", type=float, default=9.0)
    ap.add_argument("--fig-height", type=float, default=8.0)
    ap.add_argument("--axis-label-size", type=float, default=16.0)
    ap.add_argument("--tick-label-size", type=float, default=13.0)
    ap.add_argument("--colorbar-label-size", type=float, default=14.0)
    ap.add_argument("--colorbar-tick-size", type=float, default=12.0)
    ap.add_argument("--boundary-linewidth", type=float, default=1.2)

    return ap.parse_args()

def plot_all_differences_one_figure(
    *,
    csv1: Path,
    csv2_list: list[Path],
    cells: gpd.GeoDataFrame,
    catchments: gpd.GeoDataFrame,
    catch_union,
    out_png: Path,
    fig_width: float = 12.0,
    fig_height: float = 12.0,
    dpi: int = 300,
    axis_label_size: float = 16.0,
    tick_label_size: float = 13.0,
    colorbar_label_size: float = 14.0,
    colorbar_tick_size: float = 12.0,
    boundary_linewidth: float = 1.2,
    ncols: int = 2,
) -> None:
    n = len(csv2_list)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    global_vmax = compute_global_diff_limit(
        csv1=csv1,
        csv2_list=csv2_list,
        cells=cells,
        catch_union=catch_union,
    )
    vmin, vmax = -global_vmax, global_vmax

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=False,
    )

    # Leave clean space on the right for the shared colorbar.
    fig.subplots_adjust(
        left=0.08,
        right=0.86,
        bottom=0.08,
        top=0.98,
        wspace=0.08,
        hspace=0.10,
    )

    # Dedicated colorbar axis: [left, bottom, width, height]
    cax = fig.add_axes([0.89, 0.25, 0.025, 0.50])

    last_pc = None
    sum1 = load_event_sum(csv1)

    for i, csv2 in enumerate(csv2_list):
        row_i = i // ncols
        col_i = i % ncols
        ax = axes[row_i][col_i]

        sum2 = load_event_sum(csv2)
        common_ids = sum1.index.intersection(sum2.index)

        if len(common_ids) == 0:
            ax.text(0.5, 0.5, "No common grid IDs", ha="center", va="center")
            ax.axis("off")
            continue

        # Difference: csv2 - csv1
        diff = sum2.loc[common_ids].subtract(sum1.loc[common_ids], fill_value=np.nan)

        plot = cells.copy()
        plot["diff"] = plot["id"].astype(str).map(diff)
        plot = plot[plot.geometry.intersects(catch_union)].copy()
        plot = plot[plot["diff"].notna()].copy()

        patches = []
        patch_vals = []

        for _, prow in plot.iterrows():
            geom_patches = shapely_to_patches(prow.geometry)
            patches.extend(geom_patches)
            patch_vals.extend([prow["diff"]] * len(geom_patches))

        pc = PatchCollection(
            patches,
            cmap="RdBu_r",
            edgecolor="none",
            linewidth=0,
        )
        pc.set_array(np.asarray(patch_vals, dtype=float))
        pc.set_clim(vmin, vmax)

        ax.add_collection(pc)
        catchments.boundary.plot(ax=ax, color="black", linewidth=boundary_linewidth)

        ax.set_aspect("equal")

        # No title and no filename label. Add (a), (b), etc. yourself later if needed.
        # If you want automatic labels later, this is the clean place to add them.

        # Only left-column panels get y-axis label.
        if col_i == 0:
            ax.set_ylabel("Latitude", fontsize=axis_label_size, fontweight="bold")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

        # Only bottom-row panels get x-axis label.
        if row_i == nrows - 1:
            ax.set_xlabel("Longitude", fontsize=axis_label_size, fontweight="bold")
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)

        ax.tick_params(axis="both", labelsize=tick_label_size, length=4)

        # Remove the subplot box.
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Tighten each panel extent to the plotted basin cells.
        xmin, ymin, xmax, ymax = plot.total_bounds
        xpad = 0.01 * (xmax - xmin)
        ypad = 0.01 * (ymax - ymin)
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)

        last_pc = pc

    # Turn off unused axes.
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    if last_pc is not None:
        cbar = fig.colorbar(last_pc, cax=cax)
        cbar.set_label(
            "Rainfall Difference (mm)\nCSV2 - CSV1",
            fontsize=colorbar_label_size,
            fontweight="bold",
        )
        cbar.ax.tick_params(labelsize=colorbar_tick_size)

        # Remove colorbar box too.
        cbar.outline.set_visible(False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved combined figure] {out_png}")
    print(f"[info] Common colorbar range: {vmin:.2f} to {vmax:.2f} mm")
    
def main() -> None:
    args = parse_args()

    grid = load_grid(args.grid_csv)
    cells = build_cells(grid)

    catchments = load_catchments(args.catchments)
    catch_union = (
        catchments.geometry.union_all()
        if hasattr(catchments.geometry, "union_all")
        else catchments.geometry.unary_union
    )
    global_vmax = compute_global_diff_limit(
        csv1=args.csv1,
        csv2_list=args.csv2,
        cells=cells,
        catch_union=catch_union,
    )

    print(f"[info] Common colorbar range: {-global_vmax:.2f} to {global_vmax:.2f} mm")
    if args.one_figure:
        out_png = args.out

        if out_png.suffix.lower() not in [".png", ".jpg", ".jpeg", ".pdf", ".tif", ".tiff"]:
            out_png = out_png / f"all_differences_minus_{safe_stem(args.csv1)}.png"

        plot_all_differences_one_figure(
            csv1=args.csv1,
            csv2_list=args.csv2,
            cells=cells,
            catchments=catchments,
            catch_union=catch_union,
            out_png=out_png,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            dpi=args.dpi,
            axis_label_size=args.axis_label_size,
            tick_label_size=args.tick_label_size,
            colorbar_label_size=args.colorbar_label_size,
            colorbar_tick_size=args.colorbar_tick_size,
            boundary_linewidth=args.boundary_linewidth,
            ncols=args.subplot_ncols,
        )
        return
    for csv2_path in args.csv2:
        out_png = resolve_output_path(
            out_arg=args.out,
            csv1=args.csv1,
            csv2=csv2_path,
            n_csv2=len(args.csv2),
        )

        plot_difference(
            csv1=args.csv1,
            csv2=csv2_path,
            cells=cells,
            catchments=catchments,
            catch_union=catch_union,
            out_png=out_png,
            global_vmax=global_vmax,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            dpi=args.dpi,
            axis_label_size=args.axis_label_size,
            tick_label_size=args.tick_label_size,
            colorbar_label_size=args.colorbar_label_size,
            colorbar_tick_size=args.colorbar_tick_size,
            boundary_linewidth=args.boundary_linewidth,
        )


if __name__ == "__main__":
    main()