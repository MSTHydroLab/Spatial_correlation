'''python 09_plot_event_sum_catchments.py --event-csv "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/03_Interpolated_Rain/Event_7_grid_rain_hourly_mm.csv"'''

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from shapely.geometry import box

BASE_DIR = Path('/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK')
DEFAULT_GRID_CSV = BASE_DIR / 'dependent_files' / 'grid_centers_wgs84.csv'
STATIONS_CSV = BASE_DIR / 'dependent_files' / 'Stations_df.csv'
DEFAULT_CATCHMENTS = [
    Path('/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6893390/6893390.shp'),
    Path('/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/06893080/6893080.shp'),
    Path('/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6892513/6892513.shp'),
]
default_out=Path('/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/09_Sum_rainfall/')
def load_stations(stations_csv: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(stations_csv)
    df = df[['ID', 'Latitude', 'Longitude']].copy()

    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df = df.dropna(subset=['Latitude', 'Longitude'])

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']),
        crs='EPSG:4326'
    )
    return gdf
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description='Plot total rainfall summed over all times in an event CSV, restricted to the three catchments.'
    )
    ap.add_argument('--event-csv', type=Path, required=True, help='Event rainfall CSV. First column should be time_local, other columns grid IDs.')
    ap.add_argument('--grid-csv', type=Path, default=DEFAULT_GRID_CSV, help='Grid centers CSV with id, Latitude, Longitude.')
    ap.add_argument('--catchments', nargs='*', type=Path, default=DEFAULT_CATCHMENTS, help='Catchment shapefiles.')
    ap.add_argument('--out-png', type=Path, default=default_out, help='Output PNG path. Default: same folder as event CSV with _sum_catchments.png suffix.')
    ap.add_argument('--title', type=str, default=None, help='Optional custom title.')
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--fig-width', type=float, default=9.0)
    ap.add_argument('--fig-height', type=float, default=8.0)
    ap.add_argument('--title-fontsize', type=float, default=14.0)
    ap.add_argument('--axis-label-fontsize', type=float, default=20.0)
    ap.add_argument('--tick-fontsize', type=float, default=16.0)
    ap.add_argument('--colorbar-fontsize', type=float, default=17.0)
    ap.add_argument('--boundary-linewidth', type=float, default=1.2)
    ap.add_argument('--cell-edge-width', type=float, default=0.0)
    ap.add_argument('--cmap', type=str, default='jet')
    return ap.parse_args()


def load_grid(grid_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(grid_csv)
    req = ['id', 'Latitude', 'Longitude']
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f'Grid CSV missing required columns: {missing}')
    df = df[req].copy()
    df['id'] = pd.to_numeric(df['id'], errors='coerce').astype('Int64')
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df = df.dropna(subset=['id', 'Latitude', 'Longitude']).copy()
    df['id'] = df['id'].astype(int).astype(str)
    return df.sort_values(['Latitude', 'Longitude', 'id']).reset_index(drop=True)


def infer_spacing(values: np.ndarray, name: str) -> float:
    vals = np.sort(np.unique(np.round(values.astype(float), 10)))
    diffs = np.diff(vals)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        raise ValueError(f'Could not infer {name} spacing from grid coordinates.')
    rounded = np.round(diffs, 10)
    counts = pd.Series(rounded).value_counts().sort_values(ascending=False)
    return float(counts.index[0])


def build_cells(grid_df: pd.DataFrame) -> gpd.GeoDataFrame:
    lat_step = infer_spacing(grid_df['Latitude'].to_numpy(), 'latitude')
    lon_step = infer_spacing(grid_df['Longitude'].to_numpy(), 'longitude')
    half_lat = lat_step / 2.0
    half_lon = lon_step / 2.0
    geoms = [
        box(lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)
        for lat, lon in zip(grid_df['Latitude'], grid_df['Longitude'])
    ]
    out = gpd.GeoDataFrame(grid_df.copy(), geometry=geoms, crs='EPSG:4326')
    out['cell_height_deg'] = lat_step
    out['cell_width_deg'] = lon_step
    return out


def load_catchments(paths: list[Path]) -> gpd.GeoDataFrame:
    frames = []
    for shp in paths:
        if not shp.exists():
            raise FileNotFoundError(f'Catchment shapefile not found: {shp}')
        gdf = gpd.read_file(shp)
        if gdf.empty:
            continue
        if gdf.crs is None:
            raise ValueError(f'Catchment shapefile has no CRS: {shp}')
        gdf = gdf.to_crs('EPSG:4326')
        gdf = gdf[['geometry']].copy()
        gdf['catchment_name'] = shp.stem
        frames.append(gdf)
    if not frames:
        raise ValueError('No catchments could be loaded.')
    out = pd.concat(frames, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry='geometry', crs='EPSG:4326')
    out = out[out.geometry.notna()].copy().reset_index(drop=True)
    return out


def load_event_sums(event_csv: Path) -> pd.Series:
    df = pd.read_csv(event_csv)
    if df.shape[1] < 2:
        raise ValueError('Event CSV must have at least 2 columns: time_local and one grid column.')

    time_col = df.columns[0]
    if str(time_col).strip().lower() != 'time_local':
        print(f'[warning] First column is {time_col!r}, not time_local. Using it as the time column anyway.')

    data = df.drop(columns=[time_col]).copy()
    data.columns = [normalize_grid_col(c) for c in data.columns]
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors='coerce')
    sums = data.sum(axis=0, skipna=True, min_count=1)
    sums.index = [str(c) for c in sums.index]
    return sums


def normalize_grid_col(col) -> str:
    s = str(col).strip()
    try:
        f = float(s)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return s
    except Exception:
        return s


def shapely_to_patches(geom):
    patches = []
    if geom is None or geom.is_empty:
        return patches
    if geom.geom_type == 'Polygon':
        patches.append(MplPolygon(np.asarray(geom.exterior.coords), closed=True))
    elif geom.geom_type == 'MultiPolygon':
        for poly in geom.geoms:
            patches.extend(shapely_to_patches(poly))
    return patches

def get_nearest_boundary_stations(cells_gdf: gpd.GeoDataFrame,
                                  catch_union,
                                  stations_gdf: gpd.GeoDataFrame,
                                  project_epsg: int = 26915) -> gpd.GeoDataFrame:
    boundary_cells = cells_gdf[cells_gdf.geometry.intersects(catch_union.boundary)].copy()

    if boundary_cells.empty or stations_gdf.empty:
        return stations_gdf.iloc[0:0].copy()

    bc_proj = boundary_cells.to_crs(epsg=project_epsg)
    st_proj = stations_gdf.to_crs(epsg=project_epsg)

    st_ids = st_proj["ID"].to_numpy()
    st_geoms = st_proj.geometry.to_numpy()

    nearest_ids = []
    for _, row in bc_proj.iterrows():
        cen = row.geometry.centroid
        dists = np.array([cen.distance(g) for g in st_geoms], dtype=float)
        idx = int(np.argmin(dists))
        nearest_ids.append(int(st_ids[idx]))

    nearest_ids = sorted(set(nearest_ids))
    return stations_gdf[stations_gdf["ID"].isin(nearest_ids)].copy()

def main() -> None:
    args = parse_args()

    if args.out_png.is_dir() or args.out_png.suffix == "":
        args.out_png = args.out_png / f"{args.event_csv.stem}_sum_catchments.png"

    grid_df = load_grid(args.grid_csv)
    cells_gdf = build_cells(grid_df)
    catch_gdf = load_catchments(list(args.catchments))
    stations_gdf = load_stations(STATIONS_CSV)
    catch_union = catch_gdf.geometry.union_all()
    stations_gdf = load_stations(STATIONS_CSV)
    stations_in = stations_gdf[stations_gdf.geometry.within(catch_union)].copy()
    nearest_stations = get_nearest_boundary_stations(cells_gdf, catch_union, stations_gdf)
    stations_in = stations_gdf[stations_gdf.geometry.within(catch_union)].copy()
    sums = load_event_sums(args.event_csv)

    plot_gdf = cells_gdf.copy()
    plot_gdf['rain_sum_mm'] = plot_gdf['id'].map(sums)
    plot_gdf = plot_gdf[plot_gdf.geometry.intersects(catch_union)].copy()
    plot_gdf = plot_gdf[plot_gdf['rain_sum_mm'].notna()].copy()
    plot_gdf = plot_gdf[plot_gdf['rain_sum_mm'] > 0].copy()

    if plot_gdf.empty:
        raise ValueError('No grid cells intersect the catchments.')

    vals = plot_gdf['rain_sum_mm'].to_numpy(dtype=float)
    finite = np.isfinite(vals)
    if not np.any(finite):
        raise ValueError('No finite rainfall totals found for the selected catchment cells.')

    valid = plot_gdf['rain_sum_mm'][plot_gdf['rain_sum_mm'] > 0]

    vmin = float(valid.min())
    vmax = float(valid.max())

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))

    patches = []
    patch_vals = []
    for _, row in plot_gdf.iterrows():
        row_patches = shapely_to_patches(row.geometry)
        patches.extend(row_patches)
        patch_vals.extend([row['rain_sum_mm']] * len(row_patches))

    valid = plot_gdf['rain_sum_mm'][plot_gdf['rain_sum_mm'] > 0]

    pc = PatchCollection(
        patches,
        cmap='jet',
        edgecolor='none',
        linewidth=0.0,
    )
    pc.set_array(np.asarray(patch_vals, dtype=float))
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    if not stations_in.empty:
        stations_in.plot(
            ax=ax,
            marker='o',
            color='grey',
            edgecolor='black',
            markersize=80,
            zorder=4
        )
    if not nearest_stations.empty:
        nearest_stations.plot(
            ax=ax,
            marker='o',
            color='grey',
            edgecolor='black',
            markersize=70,
            zorder=5
        )

    catch_gdf.boundary.plot(ax=ax, color='black', linewidth=args.boundary_linewidth, zorder=3)

    minx, miny, maxx, maxy = catch_gdf.total_bounds
    padx = 0.02 * (maxx - minx) if maxx > minx else 0.01
    pady = 0.02 * (maxy - miny) if maxy > miny else 0.01
    ax.set_xlim(minx - padx, maxx + padx)
    ax.set_ylim(miny - pady, maxy + pady)
    ax.set_aspect('equal')

    title = args.title
    if title is None:
        title = f'Total rainfall sum, {args.event_csv.name}'
    #ax.set_title(title, fontsize=args.title_fontsize)
    ax.set_xlabel('Longitude', fontsize=20,fontweight="bold")
    ax.set_ylabel('Latitude', fontsize=20,fontweight="bold")
    ax.tick_params(axis='both', labelsize=args.tick_fontsize)
    ax.grid(True, alpha=0.25)

    cbar = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Rain (mm)', fontsize=20,fontweight="bold")
    cbar.ax.tick_params(labelsize=args.colorbar_fontsize)

    plt.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    print(f'[saved] {args.out_png}')
    print(f'[info] plotted {len(plot_gdf)} catchment-intersecting cells')
    print(f'[info] rainfall sum range: {vmin:.3f} to {vmax:.3f} mm')


if __name__ == '__main__':
    main()
