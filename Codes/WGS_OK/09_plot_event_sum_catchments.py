'''python 09_plot_event_sum_catchments.py --event-csv "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/03_Interpolated_Rain/Event_7_grid_rain_hourly_mm.csv"'''

#!/usr/bin/env python3
from __future__ import annotations
from matplotlib.colors import LinearSegmentedColormap, Normalize
import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from shapely.geometry import box
import re

# ============================================================

# ============================================================
ACC_COLORS = [
    "#cdedff",
    "#6cd1ff",
    "#009bff",
    "#0eb800",
    "#008800",
    "#ffdb00",
    "#ffad00",
    "#ff8000",
    "#ff3f26",
    "#ff27ff",
]

EVENT_WINDOWS = {
    1: ("2013-05-30 12:00:00", "2013-05-31 12:00:00"),
    2: ("2014-08-06 12:00:00", "2014-08-07 12:00:00"),
    3: ("2015-09-10 11:00:00", "2015-09-11 12:00:00"),
    4: ("2016-04-26 07:00:00", "2016-04-27 12:00:00"),
    5: ("2016-08-26 06:00:00", "2016-08-27 12:00:00"),
    6: ("2017-07-23 00:00:00", "2017-07-23 12:00:00"),
    7: ("2017-07-26 17:00:00", "2017-07-27 12:00:00"),
    8: ("2017-08-21 12:00:00", "2017-08-22 12:00:00"),
    9: ("2018-07-17 21:00:00", "2018-07-18 16:00:00"),
    10: ("2019-06-23 01:00:00", "2019-06-23 20:00:00"),
    11: ("2019-08-25 11:00:00", "2019-08-26 17:00:00"),
    12: ("2020-05-28 01:00:00", "2020-05-29 00:00:00"),
    13: ("2020-07-03 19:00:00", "2020-07-04 03:00:00"),
    14: ("2021-08-13 00:00:00", "2021-08-13 13:00:00"),
    15: ("2022-03-30 01:00:00", "2022-03-30 11:00:00"),
}

def make_accumulated_rain_cmap_norm(vmin: float, vmax: float):
    """
    Continuous rainfall color scale using the preferred color scheme.
    Color range is stretched between event min and event max.
    """
    cmap = LinearSegmentedColormap.from_list(
        "custom_accumulated_rain",
        ACC_COLORS,
        N=256,
    )

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0

    if vmax <= vmin:
        vmax = vmin + 1.0

    norm = Normalize(vmin=vmin, vmax=vmax)

    return cmap, norm

BASE_DIR = Path('/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK')
DEFAULT_GRID_CSV = BASE_DIR / 'dependent_files' / 'grid_centers_wgs84.csv'
STATIONS_CSV = BASE_DIR / 'dependent_files' / 'Stations_df.csv'
DEFAULT_RAIN_DIR = Path(
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/"
    "Compiled_rain/compiled_rawstyle_hourly_mm/per_station_hourly"
)
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
    ap.add_argument(
        "--event",
        type=int,
        default=None,
        help="Event number. If omitted, inferred from event CSV name like Event_7_grid_rain_hourly_mm.csv.",
    )

    ap.add_argument(
        "--rain-dir",
        type=Path,
        default=DEFAULT_RAIN_DIR,
        help="Folder containing per-station hourly rainfall CSV files.",
    )

    ap.add_argument(
        "--buffer-km",
        type=float,
        default=10.0,
        help="Plot gauges within this distance from the catchment union.",
    )

    ap.add_argument(
        "--min-valid-hours",
        type=int,
        default=1,
        help="Minimum valid hourly gauge values required during the event.",
    )

    ap.add_argument(
        "--only-positive-gauges",
        action="store_true",
        help="Plot only gauges with event-total rainfall > 0.",
    )

    ap.add_argument(
        "--gauge-size",
        type=float,
        default=85.0,
        help="Gauge marker size.",
    )

    ap.add_argument(
        "--gauge-edge-width",
        type=float,
        default=0.8,
        help="Gauge marker outline width.",
    )

    ap.add_argument(
        "--label-gauges",
        action="store_true",
        help="Add station IDs beside gauge points.",
    )

    ap.add_argument("--fixed-vmin", type=float, default=None)
    ap.add_argument("--fixed-vmax", type=float, default=None)
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

def norm_station_id(x) -> str:
    s = str(x).strip().strip("'").strip('"')
    if s == "" or s.lower() == "nan":
        return ""
    try:
        return str(int(float(s)))
    except Exception:
        return s


def infer_event_from_csv(event_csv: Path) -> int:
    m = re.search(r"Event[_\-]?(\d+)", event_csv.stem, flags=re.IGNORECASE)
    if not m:
        raise ValueError(
            f"Could not infer event number from {event_csv.name}. "
            "Pass --event explicitly."
        )
    return int(m.group(1))


def to_utc_timestamp(value) -> pd.Timestamp:
    """
    Convert timestamp to UTC-aware timestamp.

    Assumption:
    - Naive event window timestamps in EVENT_WINDOWS are already UTC.
    - Timezone-aware timestamps are converted to UTC.
    """
    ts = pd.to_datetime(value, errors="raise")

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return pd.Timestamp(ts)


def get_event_window_utc(event: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    if event not in EVENT_WINDOWS:
        raise ValueError(
            f"Event {event} not found in EVENT_WINDOWS. "
            f"Available events: {sorted(EVENT_WINDOWS)}"
        )

    start_raw, end_raw = EVENT_WINDOWS[event]
    start_utc = to_utc_timestamp(start_raw)
    end_utc = to_utc_timestamp(end_raw)

    if end_utc < start_utc:
        raise ValueError(f"Invalid event window: {start_utc} to {end_utc}")

    return start_utc, end_utc


def parse_station_time_utc(df: pd.DataFrame) -> pd.Series:
    """
    Parse station timestamps using time_utc.
    Returns UTC-aware timestamps.
    """
    if "time_utc" not in df.columns:
        raise ValueError("Station file must contain time_utc.")

    return pd.to_datetime(df["time_utc"], errors="coerce", utc=True)


def build_rain_file_map(rain_dir: Path) -> dict[str, Path]:
    file_map = {}

    if not rain_dir.exists():
        raise FileNotFoundError(f"Gauge rainfall folder not found: {rain_dir}")

    for fp in rain_dir.glob("*.csv"):
        stem = fp.stem.strip()
        candidates = {stem}

        # Handles names like 16040.hourly_mm.csv
        if ".hourly_mm" in stem:
            candidates.add(stem.replace(".hourly_mm", ""))

        # Handles names like station_16040.csv
        m = re.search(r"(\d+)", stem)
        if m:
            candidates.add(str(int(m.group(1))))

        for key in candidates:
            sid = norm_station_id(key)
            if sid and sid not in file_map:
                file_map[sid] = fp

    return file_map


def load_station_event_sum(
    station_id: str,
    rain_file: Path,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    rain_col: str = "rain_mm",
) -> dict:
    try:
        df = pd.read_csv(rain_file)
    except Exception as exc:
        return {
            "ID": station_id,
            "status": f"read_error: {exc}",
            "rain_sum_mm": np.nan,
            "n_valid": 0,
            "n_positive": 0,
            "rain_max_mm": np.nan,
        }

    if rain_col not in df.columns:
        return {
            "ID": station_id,
            "status": f"missing_column_{rain_col}",
            "rain_sum_mm": np.nan,
            "n_valid": 0,
            "n_positive": 0,
            "rain_max_mm": np.nan,
        }

    try:
        t_utc = parse_station_time_utc(df)
    except Exception as exc:
        return {
            "ID": station_id,
            "status": f"time_parse_error: {exc}",
            "rain_sum_mm": np.nan,
            "n_valid": 0,
            "n_positive": 0,
            "rain_max_mm": np.nan,
        }

    rain = pd.to_numeric(df[rain_col], errors="coerce")

    s = pd.Series(rain.to_numpy(float), index=t_utc, name=station_id)
    s = s[~s.index.isna()]
    s = s.groupby(level=0).mean().sort_index()

    # Collapse to hourly UTC.
    s = s.groupby(s.index.floor("h")).mean()

    # Filter by UTC event window.
    s = s[(s.index >= start_utc) & (s.index <= end_utc)]

    if s.empty:
        return {
            "ID": station_id,
            "status": "no_data_in_event_window",
            "rain_sum_mm": np.nan,
            "n_valid": 0,
            "n_positive": 0,
            "rain_max_mm": np.nan,
        }

    # Negative gauge rainfall is invalid, not rainfall.
    s = s.where(s >= 0)

    return {
        "ID": station_id,
        "status": "ok",
        "rain_sum_mm": float(s.sum(skipna=True)),
        "n_valid": int(s.notna().sum()),
        "n_positive": int((s > 0).sum()),
        "rain_max_mm": float(s.max(skipna=True)) if s.notna().any() else np.nan,
    }


def load_gauge_event_sums(
    *,
    stations_csv: Path,
    rain_dir: Path,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    catch_gdf: gpd.GeoDataFrame,
    buffer_km: float,
    min_valid_hours: int,
    only_positive: bool,
) -> gpd.GeoDataFrame:
    stations = pd.read_csv(stations_csv)

    required = ["ID", "Latitude", "Longitude"]
    missing = [c for c in required if c not in stations.columns]
    if missing:
        raise ValueError(f"{stations_csv} missing columns: {missing}")

    stations = stations[required].copy()
    stations["ID"] = stations["ID"].apply(norm_station_id)
    stations["Latitude"] = pd.to_numeric(stations["Latitude"], errors="coerce")
    stations["Longitude"] = pd.to_numeric(stations["Longitude"], errors="coerce")
    stations = stations.dropna(subset=["ID", "Latitude", "Longitude"]).copy()

    rain_files = build_rain_file_map(rain_dir)

    rows = []
    for _, row in stations.iterrows():
        sid = row["ID"]
        fp = rain_files.get(sid)

        if fp is None:
            rec = {
                "ID": sid,
                "status": "missing_rain_file",
                "rain_sum_mm": np.nan,
                "n_valid": 0,
                "n_positive": 0,
                "rain_max_mm": np.nan,
            }
        else:
            rec = load_station_event_sum(
                station_id=sid,
                rain_file=fp,
                start_utc=start_utc,
                end_utc=end_utc,
                rain_col="rain_mm",
            )

        rec["Latitude"] = row["Latitude"]
        rec["Longitude"] = row["Longitude"]
        rec["rain_file"] = str(fp) if fp is not None else ""
        rows.append(rec)

    gdf = gpd.GeoDataFrame(
        pd.DataFrame(rows),
        geometry=gpd.points_from_xy(
            [r["Longitude"] for r in rows],
            [r["Latitude"] for r in rows],
        ),
        crs="EPSG:4326",
    )

    # Spatial filter using UTM 15N.
    catch_m = catch_gdf.to_crs(epsg=26915)
    gauge_m = gdf.to_crs(epsg=26915)

    catch_union_m = (
        catch_m.geometry.union_all()
        if hasattr(catch_m.geometry, "union_all")
        else catch_m.geometry.unary_union
    )

    buffer_geom = catch_union_m.buffer(float(buffer_km) * 1000.0)
    in_buffer = gauge_m.geometry.within(buffer_geom) | gauge_m.geometry.touches(buffer_geom)

    gdf["in_buffer"] = in_buffer.to_numpy()
    gdf["rain_sum_mm"] = pd.to_numeric(gdf["rain_sum_mm"], errors="coerce")
    gdf["n_valid"] = pd.to_numeric(gdf["n_valid"], errors="coerce").fillna(0).astype(int)
    gdf["n_positive"] = pd.to_numeric(gdf["n_positive"], errors="coerce").fillna(0).astype(int)

    # Final filtering for plotted gauges.
    out = gdf[gdf["in_buffer"]].copy()
    out = out[out["status"] == "ok"].copy()
    out = out[out["n_valid"] >= int(min_valid_hours)].copy()

    if only_positive:
        out = out[out["rain_sum_mm"] > 0].copy()

    print("\n[gauge summary]")
    print(f"  total stations      : {len(gdf)}")
    print(f"  stations in buffer  : {int(gdf['in_buffer'].sum())}")
    print(f"  gauges plotted      : {len(out)}")
    print("  status counts:")
    print(gdf["status"].value_counts(dropna=False).to_string())

    return out.reset_index(drop=True)

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
    event = args.event if args.event is not None else infer_event_from_csv(args.event_csv)
    start_utc, end_utc = get_event_window_utc(event)

    print(f"[info] Event {event} UTC window: {start_utc} to {end_utc}")

    gauge_gdf = load_gauge_event_sums(
        stations_csv=STATIONS_CSV,
        rain_dir=args.rain_dir,
        start_utc=start_utc,
        end_utc=end_utc,
        catch_gdf=catch_gdf,
        buffer_km=args.buffer_km,
        min_valid_hours=args.min_valid_hours,
        only_positive=args.only_positive_gauges,
    )

    gauge_summary_csv = args.out_png.parent / f"Event_{event}_gauge_event_rainfall_sums.csv"
    gauge_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    gauge_gdf.drop(columns="geometry").to_csv(gauge_summary_csv, index=False)
    print(f"[saved gauge summary] {gauge_summary_csv}")
    plot_gdf = cells_gdf.copy()
    plot_gdf['rain_sum_mm'] = plot_gdf['id'].map(sums)
    plot_gdf = plot_gdf[plot_gdf.geometry.intersects(catch_union)].copy()
    plot_gdf = plot_gdf[plot_gdf['rain_sum_mm'].notna()].copy()
    plot_gdf["rain_sum_mm_plot"] = plot_gdf["rain_sum_mm"]

    if plot_gdf.empty:
        raise ValueError('No grid cells intersect the catchments.')

    vals = plot_gdf['rain_sum_mm'].to_numpy(dtype=float)
    finite = np.isfinite(vals)
    if not np.any(finite):
        raise ValueError('No finite rainfall totals found for the selected catchment cells.')

    valid = plot_gdf["rain_sum_mm"][np.isfinite(plot_gdf["rain_sum_mm"])]

    grid_vals = plot_gdf["rain_sum_mm"].to_numpy(dtype=float)

    if not gauge_gdf.empty:
        gauge_vals = gauge_gdf["rain_sum_mm"].to_numpy(dtype=float)
    else:
        gauge_vals = np.array([], dtype=float)

    all_vals = np.concatenate([
        grid_vals[np.isfinite(grid_vals)],
        gauge_vals[np.isfinite(gauge_vals)],
    ])

    if all_vals.size == 0:
        raise ValueError("No finite rainfall values found for grid or gauges.")

    vmin = float(np.nanmin(all_vals)) if args.fixed_vmin is None else float(args.fixed_vmin)
    vmax = float(np.nanmax(all_vals)) if args.fixed_vmax is None else float(args.fixed_vmax)

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))

    patches = []
    patch_vals = []
    for _, row in plot_gdf.iterrows():
        row_patches = shapely_to_patches(row.geometry)
        patches.extend(row_patches)
        patch_vals.extend([row["rain_sum_mm_plot"]] * len(row_patches))

    valid = plot_gdf['rain_sum_mm'][plot_gdf['rain_sum_mm'] > 0]

    cmap, norm = make_accumulated_rain_cmap_norm(vmin, vmax)
    
    cmap.set_bad(color="white", alpha=1.0)

    pc = PatchCollection(
        patches,
        cmap=cmap,
        norm=norm,
        edgecolor="lightgray",
        linewidth=0.15,
    )

    pc.set_array(np.asarray(patch_vals, dtype=float))
    ax.add_collection(pc)
    if not gauge_gdf.empty:
        ax.scatter(
            gauge_gdf.geometry.x,
            gauge_gdf.geometry.y,
            c=gauge_gdf["rain_sum_mm"],
            cmap=cmap,
            norm=norm,
            s=args.gauge_size,
            marker="o",
            edgecolors="black",
            linewidths=args.gauge_edge_width,
            zorder=5,
        )

        if args.label_gauges:
            for _, row in gauge_gdf.iterrows():
                ax.text(
                    row.geometry.x,
                    row.geometry.y,
                    str(row["ID"]),
                    fontsize=max(args.tick_fontsize - 5, 8),
                    ha="left",
                    va="bottom",
                    color="black",
                    zorder=6,
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

    cbar = fig.colorbar(
        pc,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cbar.set_label(
        "Accumulated rainfall (mm)",
        fontsize=20,
        fontweight="bold",
    )

    cbar.ax.tick_params(labelsize=args.colorbar_fontsize)

    plt.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    print(f'[saved] {args.out_png}')
    print(f'[info] plotted {len(plot_gdf)} catchment-intersecting cells')
    print(f'[info] rainfall sum range: {vmin:.3f} to {vmax:.3f} mm')
    print(f"[info] gauges plotted: {len(gauge_gdf)}")
    if not gauge_gdf.empty:
        print(
            f"[info] gauge rainfall range: "
            f"{gauge_gdf['rain_sum_mm'].min():.3f} to {gauge_gdf['rain_sum_mm'].max():.3f} mm"
        )


if __name__ == '__main__':
    main()
