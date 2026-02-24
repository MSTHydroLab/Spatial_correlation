#!/usr/bin/env python3
"""
spatial_corr_tools.py

Reusable utilities for:
- selecting rain gauges within a buffer around a pilot point
- building an hourly rainfall matrix Z over a time window
- computing pairwise correlation and semivariance vs distance
- fitting simple exponential correlation / variogram models and plotting

Designed to be imported as a module OR run as a script.

Assumptions (match your notebook):
- Gauge coordinates are in NAD83 / UTM Zone 15N (EPSG:26915) in meters.
- pilot location is provided in lon/lat (EPSG:4326) by default and is transformed to EPSG:26915.
- Per-station hourly rainfall CSV files contain:
    time_local, time_utc, rain_mm
  where time_utc is parseable as UTC timestamps, and time_local ends with an offset like -0600.

Notes:
- If your rain_mm is *incremental* and you have duplicates within an hour, you may want sum instead of mean
  in `load_station_series_local(..., duplicate_agg="mean"|"sum")`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional mapping dependencies
try:
    import geopandas as gpd
    from shapely.geometry import Point
    import contextily as ctx
except Exception:  # pragma: no cover
    gpd = None
    Point = None
    ctx = None

from pyproj import Transformer
from scipy.optimize import curve_fit


@dataclass
class GridWindow:
    start: pd.Timestamp
    end: pd.Timestamp
    index: pd.DatetimeIndex


# -----------------------------
# CRS / geometry helpers
# -----------------------------

def get_transformer_lonlat_to_utm15n() -> Transformer:
    """Lon/Lat (EPSG:4326) -> NAD83 / UTM 15N (EPSG:26915), always_xy=True."""
    return Transformer.from_crs("EPSG:4326", "EPSG:26915", always_xy=True)


def make_geodataframe_utm15n(
    coordinate_df: pd.DataFrame,
    x_col: str = "NAD83_15N_Long",
    y_col: str = "NAD83_15N_Lat",
    crs: str = "EPSG:26915",
):
    """Build a GeoDataFrame from UTM15N coordinates."""
    if gpd is None or Point is None:
        raise ImportError("geopandas/shapely are required for mapping/geometry operations.")

    geometry = [Point(xy) for xy in zip(coordinate_df[x_col], coordinate_df[y_col])]
    return gpd.GeoDataFrame(coordinate_df.copy(), geometry=geometry, crs=crs)


def make_pilot_geodataframe_utm15n(
    pilot_lon: float,
    pilot_lat: float,
    name: str = "pilot",
    transformer: Optional[Transformer] = None,
    crs: str = "EPSG:26915",
):
    """Create a pilot point GeoDataFrame in EPSG:26915 from lon/lat degrees (EPSG:4326)."""
    if gpd is None or Point is None:
        raise ImportError("geopandas/shapely are required for mapping/geometry operations.")

    if transformer is None:
        transformer = get_transformer_lonlat_to_utm15n()

    x_utm, y_utm = transformer.transform(pilot_lon, pilot_lat)
    return gpd.GeoDataFrame({"name": [name]}, geometry=[Point(x_utm, y_utm)], crs=crs)


def select_stations_within_buffer(
    gdf_utm: "gpd.GeoDataFrame",
    pilot_utm: "gpd.GeoDataFrame",
    buffer_m: float,
):
    """Return subset of stations within buffer_m of pilot point (all in EPSG:26915)."""
    buf_geom = pilot_utm.geometry.iloc[0].buffer(buffer_m)
    return gdf_utm[gdf_utm.within(buf_geom)].copy()


def plot_buffer_map(
    gdf_utm: "gpd.GeoDataFrame",
    pilot_utm: "gpd.GeoDataFrame",
    buffer_m: float,
    *,
    basemap_provider=None,
    station_markersize: float = 30,
    pilot_markersize: float = 120,
    figsize: Tuple[int, int] = (9, 9),
):
    """Plot stations, pilot, and buffer on a web basemap."""
    if gpd is None or ctx is None:
        raise ImportError("geopandas + contextily are required for plotting basemap maps.")

    if basemap_provider is None:
        basemap_provider = ctx.providers.CartoDB.Voyager

    buffer_poly = pilot_utm.copy()
    buffer_poly["geometry"] = buffer_poly.geometry.buffer(buffer_m)

    gages_web = gdf_utm.to_crs(epsg=3857)
    pilot_web = pilot_utm.to_crs(epsg=3857)
    buffer_web = buffer_poly.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=figsize)
    buffer_web.plot(ax=ax, alpha=0.25, edgecolor="black", linewidth=2)
    gages_web.plot(ax=ax, markersize=station_markersize)
    pilot_web.plot(ax=ax, markersize=pilot_markersize, marker="^")

    ctx.add_basemap(ax, source=basemap_provider)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Distance matrix
# -----------------------------

def station_distance_matrix(
    coordinate_df: pd.DataFrame,
    x_col: str = "NAD83_15N_Long",
    y_col: str = "NAD83_15N_Lat",
    id_col: str = "ID",
) -> pd.DataFrame:
    """Return an NxN distance matrix (meters), labeled by id_col."""
    if id_col not in coordinate_df.columns:
        raise ValueError(f"{id_col} not found in dataframe.")

    df = coordinate_df.copy()
    labels = df[id_col].astype(str).values
    xy = df[[x_col, y_col]].to_numpy(dtype=float)

    dx = xy[:, 0][:, None] - xy[:, 0][None, :]
    dy = xy[:, 1][:, None] - xy[:, 1][None, :]
    dist = np.sqrt(dx**2 + dy**2)

    return pd.DataFrame(dist, index=labels, columns=labels)


# -----------------------------
# Time window + rainfall loading
# -----------------------------

def make_window(start_str: str, end_str: str, freq: str = "1h") -> GridWindow:
    """Window defined by explicit START and END using format yyyymmddHH."""
    start = pd.to_datetime(start_str, format="%Y%m%d%H")
    end = pd.to_datetime(end_str, format="%Y%m%d%H")
    if end < start:
        raise ValueError("end_str must be >= start_str")
    idx = pd.date_range(start, end, freq=freq)
    return GridWindow(start=start, end=end, index=idx)


def load_station_series_local(
    station_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    rain_dir: Path,
    file_suffix: str = ".hourly_mm.csv",
    time_local_col: str = "time_local",
    time_utc_col: str = "time_utc",
    rain_col: str = "rain_mm",
    duplicate_agg: str = "mean",
) -> pd.Series:
    """Load one station file and return a Series indexed by naive LOCAL time."""
    fp = Path(rain_dir) / f"{station_id}{file_suffix}"
    if not fp.exists():
        raise FileNotFoundError(fp)

    df = pd.read_csv(fp, usecols=[time_local_col, time_utc_col, rain_col])

    t_utc = pd.to_datetime(df[time_utc_col], utc=True, errors="coerce")
    off = df[time_local_col].astype(str).str.extract(r"([+-]\d{2})\d{2}$")[0]
    off_hours = pd.to_numeric(off, errors="coerce")

    t_local = (t_utc + pd.to_timedelta(off_hours, unit="h")).dt.tz_localize(None)
    s = pd.Series(pd.to_numeric(df[rain_col], errors="coerce").to_numpy(), index=t_local)

    if duplicate_agg not in {"mean", "sum"}:
        raise ValueError("duplicate_agg must be 'mean' or 'sum'")
    s = s.groupby(level=0).mean() if duplicate_agg == "mean" else s.groupby(level=0).sum()

    s = s.sort_index().loc[start:end]
    s = s.groupby(s.index.floor("h")).mean() if duplicate_agg == "mean" else s.groupby(s.index.floor("h")).sum()
    return s


def build_rain_matrix_for_period(
    start_str: str,
    end_str: str,
    station_ids: Iterable[str],
    *,
    rain_dir: Path,
    out_csv: Optional[Path] = None,
    file_suffix: str = ".hourly_mm.csv",
    min_stations: int = 3,
    fill_value: float = 0.0,
    duplicate_agg: str = "mean",
) -> pd.DataFrame:
    """Build Z (hours x stations) for station_ids and [start,end]."""
    win = make_window(start_str, end_str)
    cols = {}

    for sid in pd.Series(list(station_ids)).astype(str).unique().tolist():
        try:
            s = load_station_series_local(
                sid, win.start, win.end,
                rain_dir=rain_dir,
                file_suffix=file_suffix,
                duplicate_agg=duplicate_agg,
            )
        except FileNotFoundError:
            print(f"[WARN] missing file for station {sid}")
            continue

        cols[sid] = s.reindex(win.index)

    Z = pd.DataFrame(cols, index=win.index)

    Z = Z.dropna(axis=1, how="all")
    Z = Z.dropna(axis=0, how="all")
    Z = Z.fillna(fill_value)

    if Z.shape[1] < min_stations:
        raise RuntimeError(f"Too few stations with usable data ({Z.shape[1]} < {min_stations}).")

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        Z.to_csv(out_csv)

    print("Hours:", Z.shape[0], "Stations:", Z.shape[1])
    return Z


# -----------------------------
# Pair computations
# -----------------------------

def make_pairs_table(Dsub: pd.DataFrame) -> pd.DataFrame:
    """Pair table from a square distance matrix (no binning)."""
    ids = Dsub.index.astype(str).tolist()
    rows = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            d = float(Dsub.loc[i, j])
            rows.append((i, j, d))
    return pd.DataFrame(rows, columns=["id_i", "id_j", "dist_m"])


def compute_pair_stats(
    Z: pd.DataFrame,
    Dsub: pd.DataFrame,
    *,
    min_samples: int = 4,
    min_rain_hours: int = 8,
) -> pd.DataFrame:
    """Pairwise correlation + semivariance."""
    ids = Z.columns.astype(str).tolist()
    out = []

    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            if i not in Dsub.index or j not in Dsub.columns:
                continue

            d = float(Dsub.loc[i, j])
            xi = Z[i]
            xj = Z[j]

            m = xi.notna() & xj.notna()
            xi = xi[m]
            xj = xj[m]

            n = len(xi)
            if n < min_samples:
                continue

            if (xi > 0).sum() < min_rain_hours or (xj > 0).sum() < min_rain_hours:
                continue

            corr_ij = xi.corr(xj)
            gamma_ij = 0.5 * ((xi - xj) ** 2).mean()
            out.append((i, j, d, n, corr_ij, gamma_ij))

    return pd.DataFrame(out, columns=["id_i", "id_j", "dist_m", "n_overlap", "corr", "gamma"])


# -----------------------------
# Models + plotting
# -----------------------------

def exponential_corr_model(h_km: np.ndarray, a_km: float, b: float) -> np.ndarray:
    """rho(h) = exp(-(h/a)^b), h in km."""
    h_km = np.asarray(h_km, dtype=float)
    return np.exp(- (h_km / a_km) ** b)


def exponential_variogram_model(h_km: np.ndarray, sill: float, a_km: float) -> np.ndarray:
    """gamma(h) = sill * (1 - exp(-h/a)), h in km."""
    h_km = np.asarray(h_km, dtype=float)
    return sill * (1.0 - np.exp(-h_km / a_km))


def fit_and_plot_models(
    pair_stats: pd.DataFrame,
    *,
    title_prefix: str = "",
    corr_p0: Tuple[float, float] = (20.0, 1.0),
    vario_p0: Optional[Tuple[float, float]] = None,
    scatter_size: float = 2.0,
):
    """Fit and plot exponential correlation and exponential variogram."""
    p_corr = pair_stats.dropna(subset=["corr"])
    corr_params = None

    if len(p_corr) >= 3:
        x = p_corr["dist_m"].values / 1000.0
        y = p_corr["corr"].values

        popt, _ = curve_fit(exponential_corr_model, x, y, p0=list(corr_p0), maxfev=5000)
        a_est, b_est = float(popt[0]), float(popt[1])
        corr_params = (a_est, b_est)

        xs = np.linspace(0, float(np.nanmax(x)), 200)
        ys = exponential_corr_model(xs, a_est, b_est)

        plt.figure(figsize=(8, 4))
        plt.scatter(x, y, s=scatter_size)
        plt.plot(xs, ys, linewidth=2, color="red")
        plt.xlabel("Distance (km)")
        plt.ylabel("Correlation")
        plt.title(f"{title_prefix}Exponential correlation fit")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(f"[corr] Estimated range a = {a_est:.3f} km")
        print(f"[corr] Estimated shape b = {b_est:.3f}")
    else:
        print("Not enough correlation points to fit model (need >= 3).")

    x = pair_stats["dist_m"].values / 1000.0
    y = pair_stats["gamma"].values
    if vario_p0 is None:
        vario_p0 = (float(np.nanmax(y)), 20.0)

    popt, _ = curve_fit(exponential_variogram_model, x, y, p0=list(vario_p0), maxfev=5000)
    sill_est, a_est = float(popt[0]), float(popt[1])

    xs = np.linspace(0, float(np.nanmax(x)), 200)
    ys = exponential_variogram_model(xs, sill_est, a_est)

    plt.figure(figsize=(8, 4))
    plt.scatter(x, y, s=scatter_size)
    plt.plot(xs, ys, linewidth=2, color="red")
    plt.xlabel("Distance (km)")
    plt.ylabel("Semivariance γ(h)")
    plt.title(f"{title_prefix}Exponential variogram fit")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"[vario] Estimated sill = {sill_est:.3f}")
    print(f"[vario] Estimated range = {a_est:.3f} km")

    return corr_params, (sill_est, a_est)


# -----------------------------
# High-level runner
# -----------------------------

def run_period(
    start_str: str,
    end_str: str,
    station_ids: Iterable[str],
    distance_matrix: pd.DataFrame,
    *,
    rain_dir: Path,
    out_z_csv: Optional[Path] = None,
    min_samples: int = 4,
    min_rain_hours: int = 8,
    duplicate_agg: str = "mean",
    plot: bool = True,
    title_prefix: str = "",
):
    """End-to-end for a period."""
    Z = build_rain_matrix_for_period(
        start_str,
        end_str,
        station_ids,
        rain_dir=rain_dir,
        out_csv=out_z_csv,
        duplicate_agg=duplicate_agg,
    )

    ids = Z.columns.astype(str)
    Dsub = distance_matrix.loc[ids, ids]
    pair_stats = compute_pair_stats(Z, Dsub, min_samples=min_samples, min_rain_hours=min_rain_hours)

    corr_params = None
    vario_params = None
    if plot:
        corr_params, vario_params = fit_and_plot_models(pair_stats, title_prefix=title_prefix)

    return Z, pair_stats, corr_params, vario_params


# -----------------------------
# CLI
# -----------------------------

def _parse_events(events: List[str]) -> List[Tuple[str, str]]:
    out = []
    for e in events:
        parts = [p.strip() for p in e.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Event must be 'start,end' (yyyymmddHH,yyyymmddHH). Got: {e}")
        out.append((parts[0], parts[1]))
    return out


def main():
    p = argparse.ArgumentParser(description="Spatial correlation/variogram utilities.")
    p.add_argument("--latlon-csv", type=str, required=True, help="Path to raingage_lat_lon.csv")
    p.add_argument("--rain-dir", type=str, required=True, help="Folder with per-station hourly CSVs")
    p.add_argument("--buffer-m", type=float, default=50000, help="Buffer in meters (EPSG:26915)")
    p.add_argument("--pilot-lon", type=float, required=True, help="Pilot longitude (deg, EPSG:4326)")
    p.add_argument("--pilot-lat", type=float, required=True, help="Pilot latitude (deg, EPSG:4326)")
    p.add_argument("--id-col", type=str, default="ID", help="Station ID column name in latlon CSV")
    p.add_argument("--x-col", type=str, default="NAD83_15N_Long", help="Easting column (m) in latlon CSV")
    p.add_argument("--y-col", type=str, default="NAD83_15N_Lat", help="Northing column (m) in latlon CSV")
    p.add_argument(
        "--events",
        type=str,
        nargs="+",
        required=True,
        help="One or more events as 'start,end' (yyyymmddHH,yyyymmddHH)",
    )
    p.add_argument("--out-dir", type=str, default=None, help="Optional output folder for Z_*.csv and pair_stats_*.csv")
    p.add_argument("--no-plot", action="store_true", help="Disable plots")
    p.add_argument("--min-samples", type=int, default=4, help="Min overlapping samples per pair")
    p.add_argument("--min-rain-hours", type=int, default=8, help="Min rain>0 hours per station in overlap")
    p.add_argument(
        "--duplicate-agg",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help="How to aggregate duplicate timestamps",
    )
    p.add_argument("--make-map", action="store_true", help="Plot buffer/stations on a basemap (requires contextily)")
    args = p.parse_args()

    coord = pd.read_csv(args.latlon_csv)
    D = station_distance_matrix(coord, x_col=args.x_col, y_col=args.y_col, id_col=args.id_col)

    if gpd is None:
        raise ImportError("geopandas/shapely are required for buffer selection (install geopandas).")

    gdf = make_geodataframe_utm15n(coord, x_col=args.x_col, y_col=args.y_col)
    pilot = make_pilot_geodataframe_utm15n(args.pilot_lon, args.pilot_lat)

    stations_in_buf = select_stations_within_buffer(gdf, pilot, args.buffer_m)
    station_ids = stations_in_buf[args.id_col].astype(str).tolist()
    print(f"Stations within {args.buffer_m/1000:.1f} km: {len(station_ids)}")

    if args.make_map:
        plot_buffer_map(gdf, pilot, args.buffer_m)

    out_dir = Path(args.out_dir) if args.out_dir else None

    for start_str, end_str in _parse_events(args.events):
        out_z = (out_dir / f"Z_{start_str}.csv") if out_dir else None

        title_prefix = f"[{start_str} to {end_str}] "
        _, pair_stats, _, _ = run_period(
            start_str,
            end_str,
            station_ids,
            D,
            rain_dir=Path(args.rain_dir),
            out_z_csv=out_z,
            min_samples=args.min_samples,
            min_rain_hours=args.min_rain_hours,
            duplicate_agg=args.duplicate_agg,
            plot=not args.no_plot,
            title_prefix=title_prefix,
        )

        if out_dir:
            ps_fp = out_dir / f"pair_stats_{start_str}.csv"
            pair_stats.to_csv(ps_fp, index=False)
            print(f"Wrote: {ps_fp}")


if __name__ == "__main__":
    main()
