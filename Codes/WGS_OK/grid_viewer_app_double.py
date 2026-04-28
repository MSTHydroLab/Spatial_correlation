#!/usr/bin/env python3
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html
from dash.exceptions import PreventUpdate
from shapely.geometry import MultiPolygon, Polygon


BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK")

GRID_CSV = BASE_DIR / "dependent_files" / "grid_centers_wgs84.csv"
STATIONS_CSV = BASE_DIR / "dependent_files" / "Stations_df.csv"
WEIGHTS_DIR = BASE_DIR / "02_OK_Weights"
EVENT_TS_DIR = BASE_DIR / "01_Event_TimeSeries"

DEFAULT_LEFT_SOURCE = "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/03_Interpolated_Rain/"
DEFAULT_RIGHT_SOURCE = "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_IDW/03_Interpolated_Rain/"

CATCHMENT_SHP_PATHS = [
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6893390/6893390.shp",
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/06893080/6893080.shp",
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6892513/6892513.shp",
]

GRID_ID_COL = "id"
GRID_LAT_COL = "Latitude"
GRID_LON_COL = "Longitude"

STN_ID_COL = "ID"
STN_LAT_COL = "Latitude"
STN_LON_COL = "Longitude"

RAIN_COLORSCALE = [
    [0.00, "rgb(255,255,255)"],
    [0.20, "rgb(0,0,255)"],
    [0.40, "rgb(0,180,0)"],
    [0.70, "rgb(255,255,0)"],
    [1.00, "rgb(255,0,0)"],
]

COLORBAR_X_LEFT = 0.46
COLORBAR_X_RIGHT = 1.08


# -----------------------------------------------------------------------------
# Base loaders
# -----------------------------------------------------------------------------
def load_catchments(shp_paths, target_epsg=4326, simplify_tol_deg=0.0001):
    gdfs = []
    for p in shp_paths:
        try:
            g = gpd.read_file(p)
            if g.empty or g.crs is None:
                continue
            g = g.to_crs(epsg=target_epsg)
            if simplify_tol_deg is not None:
                g["geometry"] = g["geometry"].simplify(simplify_tol_deg, preserve_topology=True)
            gdfs.append(g[["geometry"]].copy())
        except Exception as e:
            print(f"[catchments] Failed to read {p}: {e}")

    if not gdfs:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=f"EPSG:{target_epsg}")

    out = pd.concat(gdfs, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=f"EPSG:{target_epsg}")
    out = out[out.geometry.notnull()].copy()
    return out


def polygon_to_traces(geom, line_width=1.5):
    traces = []

    def add_poly(poly: Polygon):
        x, y = poly.exterior.coords.xy
        traces.append(
            go.Scatter(
                x=list(x),
                y=list(y),
                mode="lines",
                line=dict(width=line_width, color="rgba(0,0,0,0.9)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if geom is None:
        return traces

    if isinstance(geom, Polygon):
        add_poly(geom)
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            add_poly(poly)

    return traces

def make_heatmap_arrays(grid_lon, grid_lat, grid_ids, z_1d):
    lon_vals = np.sort(np.unique(np.round(grid_lon, 10)))
    lat_vals = np.sort(np.unique(np.round(grid_lat, 10)))

    lon_to_ix = {v: i for i, v in enumerate(lon_vals)}
    lat_to_ix = {v: i for i, v in enumerate(lat_vals)}

    Z = np.full((len(lat_vals), len(lon_vals)), np.nan, dtype=float)
    ID = np.full((len(lat_vals), len(lon_vals)), "", dtype=object)

    for lon, lat, gid, val in zip(grid_lon, grid_lat, grid_ids, z_1d):
        i = lat_to_ix[round(float(lat), 10)]
        j = lon_to_ix[round(float(lon), 10)]
        Z[i, j] = val
        ID[i, j] = str(gid)

    return lon_vals, lat_vals, Z, ID

def circle_lonlat(clon, clat, r_km, n=360):
    t = np.linspace(0, 2 * np.pi, n)
    dlat = r_km / 111.32
    coslat = max(np.cos(np.deg2rad(clat)), 1e-6)
    dlon = r_km / (111.32 * coslat)
    x = clon + dlon * np.cos(t)
    y = clat + dlat * np.sin(t)
    return x, y


def load_data():
    grid_df = pd.read_csv(GRID_CSV)
    stn_df = pd.read_csv(STATIONS_CSV)

    grid_df[GRID_ID_COL] = pd.to_numeric(grid_df[GRID_ID_COL], errors="coerce").astype("Int64")
    stn_df[STN_ID_COL] = pd.to_numeric(stn_df[STN_ID_COL], errors="coerce").astype("Int64")

    grid_df[GRID_LON_COL] = pd.to_numeric(grid_df[GRID_LON_COL], errors="coerce")
    grid_df[GRID_LAT_COL] = pd.to_numeric(grid_df[GRID_LAT_COL], errors="coerce")
    stn_df[STN_LON_COL] = pd.to_numeric(stn_df[STN_LON_COL], errors="coerce")
    stn_df[STN_LAT_COL] = pd.to_numeric(stn_df[STN_LAT_COL], errors="coerce")

    grid_df = grid_df.dropna(subset=[GRID_ID_COL, GRID_LON_COL, GRID_LAT_COL]).copy()
    stn_df = stn_df.dropna(subset=[STN_ID_COL, STN_LON_COL, STN_LAT_COL]).copy()

    grid_df[GRID_ID_COL] = grid_df[GRID_ID_COL].astype(int)
    stn_df[STN_ID_COL] = stn_df[STN_ID_COL].astype(int)
    return grid_df, stn_df


GRID_DF, STN_DF = load_data()
CATCH_GDF = load_catchments(CATCHMENT_SHP_PATHS, target_epsg=4326, simplify_tol_deg=0.0001)

GRID_IDS = sorted(GRID_DF[GRID_ID_COL].unique().tolist())

GRID_LON = GRID_DF[GRID_LON_COL].to_numpy(float)
GRID_LAT = GRID_DF[GRID_LAT_COL].to_numpy(float)
GRID_ID_ARR = GRID_DF[GRID_ID_COL].to_numpy(int)

STN_LON = STN_DF[STN_LON_COL].to_numpy(float)
STN_LAT = STN_DF[STN_LAT_COL].to_numpy(float)
STN_ID_ARR = STN_DF[STN_ID_COL].to_numpy(int)
STN_ID_SET = set(STN_ID_ARR.tolist())

xmin = min(GRID_LON.min(), STN_LON.min())
xmax = max(GRID_LON.max(), STN_LON.max())
ymin = min(GRID_LAT.min(), STN_LAT.min())
ymax = max(GRID_LAT.max(), STN_LAT.max())
padx = 0.03 * (xmax - xmin) if xmax > xmin else 0.02
pady = 0.03 * (ymax - ymin) if ymax > ymin else 0.02
FULL_XRANGE = [float(xmin - padx), float(xmax + padx)]
FULL_YRANGE = [float(ymin - pady), float(ymax + pady)]


# -----------------------------------------------------------------------------
# Source and event helpers
# -----------------------------------------------------------------------------
def normalize_path_str(path_like: str | Path | None) -> str:
    if path_like is None:
        return ""
    return str(Path(str(path_like).strip()).expanduser())


@lru_cache(maxsize=256)
def list_event_files_in_source(source_path: str) -> tuple[tuple[int, str], ...]:
    source_path = normalize_path_str(source_path)
    if source_path == "":
        return tuple()

    p = Path(source_path)
    out: dict[int, str] = {}

    if p.is_file() and p.suffix.lower() == ".csv":
        m = re.search(r"Event_(\d+)", p.name, re.IGNORECASE)
        if m:
            out[int(m.group(1))] = str(p)
        return tuple(sorted(out.items()))

    if not p.exists() or not p.is_dir():
        return tuple()

    for fp in sorted(p.rglob("*.csv")):
        m = re.search(r"Event_(\d+)", fp.name, re.IGNORECASE)
        if not m:
            continue
        event_no = int(m.group(1))
        low = fp.name.lower()
        if "grid_rain_hourly_mm" not in low:
            continue
        if event_no not in out:
            out[event_no] = str(fp)

    return tuple(sorted(out.items()))


@lru_cache(maxsize=256)
def get_available_events_for_source(source_path: str) -> tuple[int, ...]:
    pairs = list_event_files_in_source(source_path)
    return tuple(sorted({int(ev) for ev, _ in pairs}))


def get_union_available_events(left_source: str, right_source: str) -> list[int]:
    left = set(get_available_events_for_source(left_source))
    right = set(get_available_events_for_source(right_source))
    return sorted(left | right)


@lru_cache(maxsize=512)
def resolve_event_file(source_path: str, event_no: int) -> str:
    source_path = normalize_path_str(source_path)
    p = Path(source_path)

    if p.is_file():
        return str(p)

    mapping = dict(list_event_files_in_source(source_path))
    if int(event_no) in mapping:
        return mapping[int(event_no)]

    candidates = [
        p / f"Event_{event_no}_grid_rain_hourly_mm.csv",
        p / f"Event_{event_no}_grid_rain_hourly_mm_composite2.csv",
        p / f"Event_{event_no}_grid_rain_hourly_mm_composite3.csv",
        p / f"Event_{event_no}_grid_rain_hourly_mm_RA.csv",
        p / f"Event_{event_no}_grid_rain_hourly_mm_RKDP.csv",
        p / f"Event_{event_no}_grid_rain_hourly_mm_RZ.csv",
    ]
    for fp in candidates:
        if fp.exists():
            return str(fp)

    raise FileNotFoundError(f"No event file found for event {event_no} in {source_path}")


@lru_cache(maxsize=512)
def source_has_weights(source_path: str) -> bool:
    source_path = normalize_path_str(source_path).lower()
    return ("wgs_ok" in source_path) or ("wgs_idw" in source_path)


@lru_cache(maxsize=512)
def get_source_label(source_path: str) -> str:
    source_path = normalize_path_str(source_path)
    if source_path == "":
        return "(empty source)"
    p = Path(source_path)
    if p.is_file():
        return p.stem
    return p.name if p.name else str(p)


# -----------------------------------------------------------------------------
# Event and weight loaders
# -----------------------------------------------------------------------------
@lru_cache(maxsize=256)
def load_event_matrix_from_source(source_path: str, event_no: int):
    f = Path(resolve_event_file(source_path, int(event_no)))
    df = pd.read_csv(f)
    if "time_local" not in df.columns:
        raise ValueError(f"{f} must contain a 'time_local' column.")

    t = pd.to_datetime(df["time_local"], errors="coerce")
    if t.isna().all():
        raise ValueError(f"Could not parse time_local in {f}.")

    times = pd.DatetimeIndex(t)

    def clean_id(x):
        try:
            return int(float(str(x).strip()))
        except Exception:
            return None

    col_lookup = {}
    for c in df.columns:
        if c == "time_local":
            continue
        gid = clean_id(c)
        if gid is not None:
            col_lookup[gid] = c

    ntime = len(df)
    ngrid = len(GRID_ID_ARR)
    rain_mat = np.full((ntime, ngrid), np.nan, dtype=float)

    for j, gid in enumerate(GRID_ID_ARR):
        c = col_lookup.get(int(gid), None)
        if c is not None:
            rain_mat[:, j] = pd.to_numeric(df[c], errors="coerce").to_numpy(float)

    order = np.argsort(times.values)
    times = times[order]
    rain_mat = rain_mat[order, :]

    finite = rain_mat[np.isfinite(rain_mat)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = 0.0, float(np.nanmax(finite))
        if vmax <= 0:
            vmax = 1.0

    return times, rain_mat, vmin, vmax, str(f)


@lru_cache(maxsize=64)
def load_event_station_matrix(event_no: int):
    f = EVENT_TS_DIR / f"Event_{int(event_no)}_all_used_station_timeseries.csv"
    if not f.exists():
        raise FileNotFoundError(f"Station event file not found: {f}")

    df = pd.read_csv(f)
    if "time_local" not in df.columns:
        raise ValueError(f"{f} must contain a 'time_local' column.")

    t = pd.to_datetime(df["time_local"], errors="coerce")
    if t.isna().all():
        raise ValueError(f"Could not parse time_local in {f}.")

    times = pd.DatetimeIndex(t)

    col_ids = []
    for c in df.columns:
        if c == "time_local":
            continue
        try:
            col_ids.append(int(c))
        except Exception:
            pass

    col_lookup = {int(c): str(c) for c in col_ids}
    ntime = len(df)
    nstn = len(STN_ID_ARR)
    rain_mat = np.full((ntime, nstn), np.nan, dtype=float)

    for j, sid in enumerate(STN_ID_ARR):
        c = col_lookup.get(int(sid), None)
        if c is not None:
            rain_mat[:, j] = pd.to_numeric(df[c], errors="coerce").to_numpy(float)

    order = np.argsort(times.values)
    times = times[order]
    rain_mat = rain_mat[order, :]

    finite = rain_mat[np.isfinite(rain_mat)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = 0.0, float(np.nanmax(finite))
        if vmax <= 0:
            vmax = 1.0

    return times, rain_mat, vmin, vmax


@lru_cache(maxsize=64)
def load_event_station_ids(event_no: int):
    f = EVENT_TS_DIR / f"Event_{int(event_no)}_all_used_station_timeseries.csv"
    if not f.exists():
        raise FileNotFoundError(f"Station event file not found: {f}")

    df = pd.read_csv(f)
    station_ids = set()
    for c in df.columns:
        if c == "time_local":
            continue
        try:
            station_ids.add(int(float(str(c).strip())))
        except Exception:
            pass
    return station_ids


@lru_cache(maxsize=64)
def load_event_weights_map_for_source(source_path: str, event_no: int):
    if not source_has_weights(source_path):
        return {}

    source_low = normalize_path_str(source_path).lower()
    weights_file: Path | None = None

    if "wgs_idw" in source_low:
        source_dir = Path(source_path)
        if source_dir.is_file():
            source_dir = source_dir.parent
        candidate = source_dir.parent / "02_IDW_Weights" / f"Event_{int(event_no)}_weights.csv"
        if candidate.exists():
            weights_file = candidate

    if weights_file is None:
        candidate = WEIGHTS_DIR / f"Event_{int(event_no)}_weights.csv"
        if candidate.exists():
            weights_file = candidate

    if weights_file is None or not weights_file.exists():
        raise FileNotFoundError(f"Weights file not found for source={source_path}, event={event_no}")

    df = pd.read_csv(weights_file)
    if "id" not in df.columns:
        raise ValueError(f"{weights_file} must contain an 'id' column.")

    g_cols = [c for c in df.columns if re.fullmatch(r"g\d+", str(c))]
    g_cols = sorted(g_cols, key=lambda x: int(x[1:]))

    weights_map = {}
    for _, row in df.iterrows():
        try:
            gid = int(float(row["id"]))
        except Exception:
            continue

        chosen = []
        for gcol in g_cols:
            idx = gcol[1:]
            wcol = f"w{idx}"
            if gcol not in row.index or wcol not in row.index or pd.isna(row[gcol]):
                continue
            try:
                gauge_id = int(float(row[gcol]))
            except Exception:
                continue
            try:
                weight = float(row[wcol]) if pd.notna(row[wcol]) else np.nan
            except Exception:
                weight = np.nan
            chosen.append({"gauge_id": gauge_id, "weight": weight})

        weights_map[gid] = chosen

    return weights_map


# -----------------------------------------------------------------------------
# Figure helpers
# -----------------------------------------------------------------------------
def extract_ranges(relayout):
    if not relayout:
        return None

    if "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
        y0 = relayout.get("yaxis.range[0]", FULL_YRANGE[0])
        y1 = relayout.get("yaxis.range[1]", FULL_YRANGE[1])
        return {
            "xrange": [relayout["xaxis.range[0]"], relayout["xaxis.range[1]"],],
            "yrange": [y0, y1],
        }

    if relayout.get("xaxis.autorange") or relayout.get("yaxis.autorange"):
        return {"xrange": FULL_XRANGE, "yrange": FULL_YRANGE}

    return None


def make_placeholder_figure(title: str, viewport: dict | None = None) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=1100,
        margin=dict(l=40, r=90, t=70, b=40),
        dragmode="pan",
        uirevision="keep",
        annotations=[
            dict(
                text=title,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16),
            )
        ],
    )
    xr = (viewport or {}).get("xrange", FULL_XRANGE)
    yr = (viewport or {}).get("yrange", FULL_YRANGE)
    fig.update_xaxes(range=xr)
    fig.update_yaxes(range=yr)
    return fig


def build_figure(
    source_path: str,
    grid_id: int,
    show_lines: bool,
    circle_km_list,
    centroid_size: int,
    centroid_opacity: float,
    gauge_size: int,
    gauge_opacity: float,
    event_no: int | None,
    time_idx: int | None,
    viewport: dict | None = None,
    panel: str = "left",
    ):
    row = GRID_DF.loc[GRID_DF[GRID_ID_COL] == int(grid_id)].iloc[0]
    tx = float(row[GRID_LON_COL])
    ty = float(row[GRID_LAT_COL])

    source_label = get_source_label(source_path)
    selected_gauges = []
    event_station_ids = set()
    source_file = None

    if event_no is not None:
        if source_has_weights(source_path):
            try:
                weights_map = load_event_weights_map_for_source(source_path, int(event_no))
                selected_gauges = weights_map.get(int(grid_id), [])
            except Exception as e:
                print(f"[weights] load failed for source={source_path}, event={event_no}: {e}")

        try:
            event_station_ids = load_event_station_ids(int(event_no))
        except Exception as e:
            print(f"[event stations] load failed for event {event_no}: {e}")

    selected_ids = [d["gauge_id"] for d in selected_gauges]
    selected_weights_lookup = {d["gauge_id"]: d["weight"] for d in selected_gauges}

    fig = go.Figure()
    catchment_traces = []
    if not CATCH_GDF.empty:
        for geom in CATCH_GDF.geometry:
            catchment_traces.extend(polygon_to_traces(geom, line_width=1.5))

    station_rain = None
    common_vmin, common_vmax = 0.0, 1.0
    rain_note = f"Source: {source_label}"
    selected_grid_rain = np.nan
    missing_ids = [i for i in selected_ids if i not in STN_ID_SET]
    time_label = ""

    if event_no is not None:
        try:
            times, rain_mat, _, vmax_grid, source_file = load_event_matrix_from_source(source_path, int(event_no))

            if time_idx is None:
                time_idx = 0
            time_idx = int(np.clip(time_idx, 0, len(times) - 1))

            z = rain_mat[time_idx, :]
            lon_vals, lat_vals, Z2, ID2 = make_heatmap_arrays(
                GRID_LON, GRID_LAT, GRID_ID_ARR, z
            )

            customdata = np.empty(Z2.shape + (2,), dtype=object)
            customdata[:, :, 0] = ID2
            customdata[:, :, 1] = Z2
            try:
                selected_idx = np.where(GRID_ID_ARR == int(grid_id))[0][0]
                selected_grid_rain = z[selected_idx]
            except Exception:
                selected_grid_rain = np.nan

            if source_has_weights(source_path):
                try:
                    _, st_rain_mat, _, station_vmax = load_event_station_matrix(int(event_no))
                    station_rain = st_rain_mat[time_idx, :]
                    common_vmax = max(vmax_grid, station_vmax)
                except Exception as e:
                    print(f"[station-rain] load failed for event {event_no}: {e}")
                    common_vmax = max(vmax_grid, 1.0)
            else:
                common_vmax = max(vmax_grid, 1.0)

            if common_vmax <= 0:
                common_vmax = 1.0

            ts_label = times[time_idx].strftime("%Y-%m-%d %H:%M")
            rain_note = f"{source_label} | Event {event_no} | {ts_label}"
            time_label = ts_label

            fig.add_trace(go.Heatmap(
                x=lon_vals,
                y=lat_vals,
                z=Z2,
                customdata=customdata,
                colorscale=RAIN_COLORSCALE,
                zmin=common_vmin,
                zmax=common_vmax,
                showscale=(panel == "left"),
                colorbar=dict(
                    title="Rain (mm)",
                    x=1.08,
                ),
                hovertemplate=(
                    "Grid ID: %{customdata[0]}"
                    "<br>Rain: %{customdata[1]:.3f} mm"
                    "<br>Lon: %{x:.5f}"
                    "<br>Lat: %{y:.5f}"
                    "<extra></extra>"
                ),
                connectgaps=False,
                xgap=0,
                ygap=0,
            ))
        except Exception as e:
            print(f"[event load] fallback due to: {e}")
            fig.add_trace(go.Scatter(
                x=GRID_LON,
                y=GRID_LAT,
                mode="markers",
                name="Centroids",
                marker=dict(
                    symbol="square",
                    size=centroid_size,
                    opacity=centroid_opacity,
                    color="gray",
                    line=dict(width=0),
                ),
                customdata=np.column_stack([
                    GRID_ID_ARR,
                    np.repeat("grid", len(GRID_ID_ARR)),
                    np.repeat(np.nan, len(GRID_ID_ARR))
                ]),
                hovertemplate=(
                    "Grid ID: %{customdata[0]}"
                    "<br>Lon: %{x:.5f}"
                    "<br>Lat: %{y:.5f}"
                    "<extra></extra>"
                ),
            ))
            rain_note = f"{source_label} | Event load failed: {e}"
    else:
        fig.add_trace(go.Scattergl(
            x=GRID_LON,
            y=GRID_LAT,
            mode="markers",
            name="Centroids",
            marker=dict(
                symbol="square",
                size=centroid_size,
                opacity=centroid_opacity,
                color="gray",
            ),
            customdata=np.column_stack([GRID_ID_ARR, np.repeat("grid", len(GRID_ID_ARR))]),
            hovertemplate="Grid ID: %{customdata[0]}<br>Lon: %{x:.5f}<br>Lat: %{y:.5f}<extra></extra>",
        ))

    fig.add_trace(go.Scattergl(
        x=[tx],
        y=[ty],
        mode="markers",
        name="Selected grid",
        marker=dict(
            symbol="square-open",
            size=centroid_size + 10,
            color="black",
            line=dict(width=2, color="black")
        ),
        customdata=[[int(grid_id), selected_grid_rain, tx, ty]],
        hovertemplate=(
            "Grid ID: %{customdata[0]}"
            "<br>Rain: %{customdata[1]:.3f} mm"
            "<br>Lon: %{customdata[2]:.5f}"
            "<br>Lat: %{customdata[3]:.5f}"
            "<extra></extra>"
        ),
    ))

    if station_rain is not None:
        selected_mask = np.isin(STN_ID_ARR, np.array(selected_ids, dtype=int)) if selected_ids else np.zeros(len(STN_ID_ARR), dtype=bool)
        event_station_mask = np.isin(STN_ID_ARR, np.array(sorted(event_station_ids), dtype=int)) if event_station_ids else np.zeros(len(STN_ID_ARR), dtype=bool)
        used_mask = event_station_mask
        unused_mask = np.zeros(len(STN_ID_ARR), dtype=bool)

        rain_for_plot = station_rain.copy()
        rain_for_plot[selected_mask & ~np.isfinite(rain_for_plot)] = 0.0

        weight_for_plot = np.full(len(STN_ID_ARR), np.nan, dtype=float)
        for i, sid in enumerate(STN_ID_ARR):
            if int(sid) in selected_weights_lookup:
                weight_for_plot[i] = selected_weights_lookup[int(sid)]

        if unused_mask.any():
            fig.add_trace(go.Scattergl(
                x=STN_LON[unused_mask],
                y=STN_LAT[unused_mask],
                mode="markers",
                name="Unused gauges",
                marker=dict(
                    symbol="triangle-up",
                    size=gauge_size,
                    opacity=max(gauge_opacity, 0.35),
                    color="lightgray",
                    line=dict(width=0.8, color="black"),
                ),
                text=STN_ID_ARR[unused_mask].astype(str),
                hovertemplate="Gauge ID: %{text}<br>Not in this event station file<extra></extra>",
            ))

        if used_mask.any():
            fig.add_trace(go.Scattergl(
                x=STN_LON[used_mask],
                y=STN_LAT[used_mask],
                mode="markers",
                name="Gauge rain",
                marker=dict(
                    symbol="triangle-up",
                    size=gauge_size + 2,
                    opacity=0.95,
                    color=rain_for_plot[used_mask],
                    colorscale=RAIN_COLORSCALE,
                    cmin=common_vmin,
                    cmax=common_vmax,
                    line=dict(width=1, color="black"),
                    showscale=False,
                ),
                text=STN_ID_ARR[used_mask].astype(str),
                customdata=np.column_stack([
                    STN_ID_ARR[used_mask],
                    np.round(rain_for_plot[used_mask], 3),
                    STN_LON[used_mask],
                    STN_LAT[used_mask],
                    weight_for_plot[used_mask],
                ]),
                hovertemplate=(
                    "Gauge ID: %{customdata[0]}"
                    "<br>Rain: %{customdata[1]:.3f} mm"
                    "<br>Lon: %{customdata[2]:.5f}"
                    "<br>Lat: %{customdata[3]:.5f}"
                    "<br>Weight: %{customdata[4]:.4f}"
                    "<extra></extra>"
                ),
            ))
    else:
        fig.add_trace(go.Scattergl(
            x=STN_LON,
            y=STN_LAT,
            mode="markers",
            name="All gauges",
            marker=dict(
                symbol="triangle-up",
                size=gauge_size,
                opacity=gauge_opacity,
                color="dodgerblue",
                line=dict(width=1, color="black"),
            ),
            text=STN_ID_ARR.astype(str),
            hovertemplate="Gauge ID: %{text}<br>Lon: %{x:.5f}<br>Lat: %{y:.5f}<extra></extra>",
        ))

    positive_selected_ids = [
        d["gauge_id"] for d in selected_gauges
        if pd.notna(d.get("weight", np.nan)) and float(d["weight"]) > 0
    ]

    if positive_selected_ids:
        positive_mask = np.isin(STN_ID_ARR, np.array(positive_selected_ids, dtype=int))
        sel_idx = np.where(positive_mask)[0]
        sel_weights = np.array([selected_weights_lookup.get(int(gid), np.nan) for gid in STN_ID_ARR[sel_idx]], dtype=float)

        fig.add_trace(go.Scattergl(
            x=STN_LON[sel_idx],
            y=STN_LAT[sel_idx],
            mode="markers",
            name="Weighted gauges",
            marker=dict(
                symbol="triangle-up",
                size=gauge_size + 8,
                color="rgba(0,0,0,0)",
                line=dict(width=1.5, color="red"),
            ),
            text=STN_ID_ARR[sel_idx].astype(str),
            customdata=np.column_stack([STN_ID_ARR[sel_idx], sel_weights]),
            hovertemplate="Gauge ID: %{customdata[0]}<br>Weight: %{customdata[1]:.4f}<extra></extra>",
        ))

    for km in sorted(circle_km_list):
        xc, yc = circle_lonlat(tx, ty, float(km))
        fig.add_trace(go.Scatter(
            x=xc,
            y=yc,
            mode="lines",
            name=f"{km} km circle",
            hoverinfo="skip",
            line=dict(width=2, dash="dot", color="gray"),
        ))

    if show_lines and selected_ids and source_has_weights(source_path):
        for d in selected_gauges:
            sid = d["gauge_id"]
            w = d["weight"]
            hit = np.where(STN_ID_ARR == int(sid))[0]
            if len(hit) == 0:
                continue
            j = hit[0]

            if w == 0 or np.isclose(w, 0):
                dash_style = "dot"
                line_width = 2
                line_color = "gray"
            else:
                dash_style = "solid"
                line_width = 2
                line_color = "black"

            fig.add_trace(go.Scatter(
                x=[tx, STN_LON[j]],
                y=[ty, STN_LAT[j]],
                mode="lines",
                showlegend=False,
                line=dict(width=line_width, color=line_color, dash=dash_style),
                hoverinfo="skip",
            ))

    for tr in catchment_traces:
        fig.add_trace(tr)
    valid_mask = np.isfinite(z)
    click_x = GRID_LON[valid_mask]
    click_y = GRID_LAT[valid_mask]
    click_ids = GRID_ID_ARR[valid_mask]
    click_rain = z[valid_mask]
    click_mask = np.isfinite(z) if event_no is not None else np.ones(len(GRID_ID_ARR), dtype=bool)

    fig.add_trace(go.Scattergl(
        x=GRID_LON[click_mask],
        y=GRID_LAT[click_mask],
        mode="markers",
        name="centroid-click-layer",
        marker=dict(
            symbol="square",
            size=max(centroid_size + 8, 12),
            opacity=0.001,
            color="rgba(0,0,0,0.001)",
            line=dict(width=0),
        ),
        customdata=np.column_stack([
            GRID_ID_ARR[click_mask],
            np.repeat("grid", int(np.sum(click_mask))),
        ]),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.update_layout(
        title=f"Grid {int(grid_id)} | {rain_note}",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=800,
        margin=dict(l=40, r=90, t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        dragmode="pan",
        uirevision="keep",
    )

    xr = (viewport or {}).get("xrange", FULL_XRANGE)
    yr = (viewport or {}).get("yrange", FULL_YRANGE)
    fig.update_xaxes(range=xr)
    fig.update_yaxes(range=yr)

    info = {
        "selected_ids_count": len(selected_ids),
        "missing_ids": missing_ids,
        "source_label": source_label,
        "source_file": source_file,
        "time_label": time_label,
        "selected_grid_rain": selected_grid_rain,
        "has_weights": source_has_weights(source_path),
    }
    return fig, info


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = Dash(__name__)

ALL_DEFAULT_EVENTS = get_union_available_events(DEFAULT_LEFT_SOURCE, DEFAULT_RIGHT_SOURCE)
default_grid_id = int(GRID_IDS[0]) if GRID_IDS else None
default_event = ALL_DEFAULT_EVENTS[0] if ALL_DEFAULT_EVENTS else None

app.layout = html.Div(
    style={"display": "flex", "gap": "14px", "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"},
    children=[
        html.Div(
            style={"width": "280px", "padding": "14px", "borderRight": "1px solid #ddd", "overflowY": "auto", "maxHeight": "100vh"},
            children=[
                html.H3("Controls", style={"marginTop": "0px"}),
                html.Label("Left source"),
                dcc.Dropdown(
                    id="left-source",
                    options=[
                        {"label": "WGS_OK", "value": "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/03_Interpolated_Rain/"},
                        {"label": "WGS_IDW", "value": "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_IDW/03_Interpolated_Rain/"},
                        {"label": "Composite2", "value": "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/Radar_Event_TimeSeries/Composite_2/"},
                        {"label": "Composite3", "value": "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/Radar_Event_TimeSeries/Composite_3/"},
                        {"label": "RA", "value": "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/Radar_Event_TimeSeries/RA/"},
                        {"label": "RKDP Event 1", "value": "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/Radar_Event_TimeSeries/RKDP/Event_1/Event_1_grid_rain_hourly_mm_RKDP.csv"},
                        {"label": "RZ Event 1", "value": "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/Radar_Event_TimeSeries/RZ/Event_1/Event_1_grid_rain_hourly_mm_RZ.csv"},
                    ],
                    value=DEFAULT_LEFT_SOURCE,
                    clearable=False,
                    searchable=False,
                ),
                html.Div(style={"height": "8px"}),
                html.Label("Right source"),
                dcc.Dropdown(
                    id="right-source",
                    options=[
                        {"label": "WGS_OK", "value": "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/03_Interpolated_Rain/"},
                        {"label": "WGS_IDW", "value": "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_IDW/03_Interpolated_Rain/"},
                        {"label": "Composite2", "value": "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/Radar_Event_TimeSeries/Composite_2/"},
                        {"label": "Composite3", "value": "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/Radar_Event_TimeSeries/Composite_3/"},
                        {"label": "RA", "value": "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/Radar_Event_TimeSeries/RA/"},
                        {"label": "RKDP Event 1", "value": "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/Radar_Event_TimeSeries/RKDP/Event_1/Event_1_grid_rain_hourly_mm_RKDP.csv"},
                        {"label": "RZ Event 1", "value": "/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/Radar_Event_TimeSeries/RZ/Event_1/Event_1_grid_rain_hourly_mm_RZ.csv"},
                    ],
                    value=DEFAULT_RIGHT_SOURCE,
                    clearable=False,
                    searchable=False,
                ),
                html.Div(style={"height": "12px"}),
                html.Label("Event"),
                dcc.Dropdown(id="event-dropdown", options=[], value=default_event, searchable=True, clearable=True, placeholder="Select event"),
                html.Div(style={"height": "10px"}),
                html.Label("Time"),
                dcc.Slider(id="time-slider", min=0, max=0, step=1, value=0, tooltip={"placement": "bottom", "always_visible": False}),
                html.Div(style={"height": "6px"}),
                html.Div(id="time-label", style={"fontSize": "13px"}),
                html.Div(style={"height": "8px"}),
                html.Div(style={"display": "flex", "gap": "8px"}, children=[
                    html.Button("Play", id="play-btn", n_clicks=0),
                    html.Button("Pause", id="pause-btn", n_clicks=0),
                ]),
                dcc.Interval(id="anim-interval", interval=500, n_intervals=0, disabled=True),
                dcc.Store(id="anim-playing", data=False),
                html.Hr(),
                html.Label("Grid ID"),
                dcc.Dropdown(
                    id="grid-dropdown",
                    options=[{"label": str(g), "value": int(g)} for g in GRID_IDS],
                    value=default_grid_id,
                    searchable=True,
                    clearable=False,
                ),
                html.Hr(),
                dcc.Checklist(
                    id="show-lines",
                    options=[{"label": "Show lines to weighted gauges", "value": "on"}],
                    value=["on"],
                ),
                html.Div(style={"height": "8px"}),
                html.Label("Circles to draw (km)"),
                dcc.Dropdown(
                    id="circles",
                    options=[{"label": str(k), "value": int(k)} for k in [5, 6, 7, 8, 9, 10]],
                    value=[5, 10],
                    multi=True,
                    clearable=False,
                ),
                html.Hr(),
                html.Label("Centroid size"),
                dcc.Slider(id="centroid-size", min=3, max=12, step=1, value=10),
                html.Label("Centroid opacity"),
                dcc.Slider(id="centroid-opacity", min=0.1, max=1.0, step=0.05, value=0.8),
                html.Div(style={"height": "10px"}),
                html.Label("Gauge size"),
                dcc.Slider(id="gauge-size", min=3, max=12, step=1, value=9),
                html.Label("Gauge opacity"),
                dcc.Slider(id="gauge-opacity", min=0.05, max=1.0, step=0.05, value=0.65),
                html.Hr(),
                html.Button("Reset view to full domain", id="reset-view", n_clicks=0),
                html.Div(style={"height": "10px"}),
                html.Div(id="status-box", style={"fontSize": "13px", "whiteSpace": "pre-wrap"}),
                dcc.Store(id="selected-grid-store", data=default_grid_id),
                dcc.Store(id="viewport-store", data={"xrange": FULL_XRANGE, "yrange": FULL_YRANGE}),
            ],
        ),
        html.Div(
            style={"flex": "1", "padding": "0px"},
            children=[
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1.1fr 1fr", "gap": "1px"},
                    children=[
                        html.Div([
                            html.Div(id="left-title", style={"fontWeight": "600", "marginBottom": "6px"}),
                            dcc.Graph(id="map-left", figure=go.Figure(), config={"displayModeBar": True, "scrollZoom": True}),
                        ]),
                        html.Div([
                            html.Div(id="right-title", style={"fontWeight": "600", "marginBottom": "6px"}),
                            dcc.Graph(id="map-right", figure=go.Figure(), config={"displayModeBar": True, "scrollZoom": True}),
                        ]),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("event-dropdown", "options"),
    Output("event-dropdown", "value"),
    Input("left-source", "value"),
    Input("right-source", "value"),
    State("event-dropdown", "value"),
)
def update_event_options(left_source, right_source, current_event):
    events = get_union_available_events(left_source, right_source)
    options = [{"label": f"E{e}", "value": int(e)} for e in events]
    if not events:
        return options, None
    if current_event in events:
        return options, current_event
    return options, events[0]


@app.callback(
    Output("selected-grid-store", "data"),
    Output("grid-dropdown", "value"),
    Input("map-left", "clickData"),
    Input("map-right", "clickData"),
    Input("grid-dropdown", "value"),
    State("selected-grid-store", "data"),
    prevent_initial_call=False,
)
def update_selected_grid(click_left, click_right, dropdown_value, current):
    if current is None:
        if GRID_IDS:
            current = int(GRID_IDS[0])
        else:
            return None, None

    trig = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""

    if trig == "grid-dropdown.value":
        if dropdown_value is None:
            return current, current
        try:
            v = int(dropdown_value)
            return v, v
        except Exception:
            return current, current

    clickData = None
    if trig == "map-left.clickData":
        clickData = click_left
    elif trig == "map-right.clickData":
        clickData = click_right

    if not clickData or "points" not in clickData or len(clickData["points"]) == 0:
        return current, current

    pt = clickData["points"][0]

    try:
        x_clicked = float(pt["x"])
        y_clicked = float(pt["y"])

        d2 = (GRID_LON - x_clicked) ** 2 + (GRID_LAT - y_clicked) ** 2
        nearest_idx = int(np.argmin(d2))
        new_gid = int(GRID_ID_ARR[nearest_idx])

        return new_gid, new_gid
    except Exception:
        return current, current


@app.callback(
    Output("viewport-store", "data"),
    Input("map-left", "relayoutData"),
    Input("map-right", "relayoutData"),
    Input("reset-view", "n_clicks"),
    State("viewport-store", "data"),
    prevent_initial_call=True,
)
def sync_viewport(relayout_left, relayout_right, reset_clicks, current):
    trig = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""

    if trig == "reset-view.n_clicks":
        return {"xrange": FULL_XRANGE, "yrange": FULL_YRANGE}

    if trig == "map-left.relayoutData":
        out = extract_ranges(relayout_left)
        return out if out is not None else current

    if trig == "map-right.relayoutData":
        out = extract_ranges(relayout_right)
        return out if out is not None else current

    raise PreventUpdate


@app.callback(
    Output("anim-interval", "disabled"),
    Output("anim-playing", "data"),
    Input("play-btn", "n_clicks"),
    Input("pause-btn", "n_clicks"),
    State("anim-playing", "data"),
)
def toggle_animation(n_play, n_pause, playing):
    trig = callback_context.triggered[0]["prop_id"] if callback_context.triggered else None
    if trig == "play-btn.n_clicks":
        return False, True
    if trig == "pause-btn.n_clicks":
        return True, False
    return (not bool(playing)), bool(playing)


@app.callback(
    Output("time-slider", "max"),
    Output("time-slider", "value"),
    Output("time-label", "children"),
    Input("left-source", "value"),
    Input("right-source", "value"),
    Input("event-dropdown", "value"),
    Input("anim-interval", "n_intervals"),
    Input("time-slider", "value"),
    State("anim-playing", "data"),
    prevent_initial_call=False,
)
def sync_time_controls(left_source, right_source, event_no, n_intervals, slider_value, playing):
    trig = callback_context.triggered[0]["prop_id"] if callback_context.triggered else None

    if event_no is None:
        return 0, 0, "No event selected."

    candidate_sources = [left_source, right_source]
    times = None
    source_used = None
    errs = []

    for src in candidate_sources:
        try:
            ts, _, _, _, _ = load_event_matrix_from_source(src, int(event_no))
            times = ts
            source_used = get_source_label(src)
            break
        except Exception as e:
            errs.append(f"{get_source_label(src)}: {e}")

    if times is None:
        return 0, 0, "Event load failed. " + " | ".join(errs)

    if len(times) == 0:
        return 0, 0, f"Event {int(event_no)}: no timestamps."

    max_idx = len(times) - 1
    idx = int(slider_value or 0)
    idx = int(np.clip(idx, 0, max_idx))

    if trig in ("event-dropdown.value", "left-source.value", "right-source.value"):
        idx = 0
    elif trig == "anim-interval.n_intervals" and playing:
        idx = 0 if idx >= max_idx else idx + 1

    label = f"Event {int(event_no)} | {times[idx].strftime('%Y-%m-%d %H:%M')} | time base: {source_used}"
    return max_idx, idx, label


@app.callback(
    Output("map-left", "figure"),
    Output("map-right", "figure"),
    Output("status-box", "children"),
    Output("left-title", "children"),
    Output("right-title", "children"),
    Input("left-source", "value"),
    Input("right-source", "value"),
    Input("selected-grid-store", "data"),
    Input("show-lines", "value"),
    Input("circles", "value"),
    Input("centroid-size", "value"),
    Input("centroid-opacity", "value"),
    Input("gauge-size", "value"),
    Input("gauge-opacity", "value"),
    Input("event-dropdown", "value"),
    Input("time-slider", "value"),
    Input("viewport-store", "data"),
)
def redraw_both(
    left_source,
    right_source,
    grid_id,
    show_lines_value,
    circles_value,
    centroid_size,
    centroid_opacity,
    gauge_size,
    gauge_opacity,
    event_no,
    time_idx,
    viewport,
    ):
    if grid_id is None and GRID_IDS:
        grid_id = int(GRID_IDS[0])
    elif grid_id is None:
        empty = make_placeholder_figure("No grid ids available.", viewport)
        return empty, empty, "No grid ids available.", "Left", "Right"

    show_lines = "on" in (show_lines_value or [])
    circle_km_list = circles_value or []
    viewport = viewport or {"xrange": FULL_XRANGE, "yrange": FULL_YRANGE}

    try:
        fig_left, info_left = build_figure(
            source_path=left_source,
            grid_id=int(grid_id),
            show_lines=show_lines,
            circle_km_list=circle_km_list,
            centroid_size=int(centroid_size),
            centroid_opacity=float(centroid_opacity),
            gauge_size=int(gauge_size),
            gauge_opacity=float(gauge_opacity),
            event_no=(int(event_no) if event_no is not None else None),
            time_idx=(int(time_idx) if time_idx is not None else None),
            viewport=viewport,
            panel="left",
        )
    except Exception as e:
        fig_left = make_placeholder_figure(f"Left panel error: {e}", viewport)
        info_left = {"selected_ids_count": 0, "missing_ids": [], "source_label": get_source_label(left_source), "source_file": None, "time_label": "", "selected_grid_rain": np.nan, "has_weights": False}

    try:
        fig_right, info_right = build_figure(
            source_path=right_source,
            grid_id=int(grid_id),
            show_lines=show_lines,
            circle_km_list=circle_km_list,
            centroid_size=int(centroid_size),
            centroid_opacity=float(centroid_opacity),
            gauge_size=int(gauge_size),
            gauge_opacity=float(gauge_opacity),
            event_no=(int(event_no) if event_no is not None else None),
            time_idx=(int(time_idx) if time_idx is not None else None),
            viewport=viewport,
            panel="right",
        )
    except Exception as e:
        fig_right = make_placeholder_figure(f"Right panel error: {e}", viewport)
        info_right = {"selected_ids_count": 0, "missing_ids": [], "source_label": get_source_label(right_source), "source_file": None, "time_label": "", "selected_grid_rain": np.nan, "has_weights": False}

    status = (
        f"Selected grid: {int(grid_id)}\n"
        f"Event: {event_no if event_no is not None else 'none'}\n"
        f"Time index: {int(time_idx or 0)}\n\n"
        f"LEFT\n"
        f"Source: {normalize_path_str(left_source)}\n"
        f"Resolved file: {info_left.get('source_file')}\n"
        f"Selected grid rain: {info_left.get('selected_grid_rain')}\n"
        f"Weighted gauges count: {info_left['selected_ids_count']}\n"
        f"Missing weighted gauges in Stations_df: {info_left['missing_ids']}\n\n"
        f"RIGHT\n"
        f"Source: {normalize_path_str(right_source)}\n"
        f"Resolved file: {info_right.get('source_file')}\n"
        f"Selected grid rain: {info_right.get('selected_grid_rain')}\n"
        f"Weighted gauges count: {info_right['selected_ids_count']}\n"
        f"Missing weighted gauges in Stations_df: {info_right['missing_ids']}"
    )

    left_title = f"Left: {info_left['source_label']}"
    right_title = f"Right: {info_right['source_label']}"
    return fig_left, fig_right, status, left_title, right_title


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8890, debug=True)
