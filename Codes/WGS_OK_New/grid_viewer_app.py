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
from shapely.geometry import MultiPolygon, Polygon


BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK_New")

GRID_CSV = BASE_DIR / "dependent_files" / "grid_centers_wgs84.csv"
STATIONS_CSV = BASE_DIR / "dependent_files" / "Stations_df.csv"
WEIGHTS_DIR = BASE_DIR / "02_OK_Weights"
EVENT_DIR = BASE_DIR / "03_Interpolated_Rain"
EVENT_TS_DIR = BASE_DIR / "01_Event_TimeSeries"

EVENT_GLOB = "Event_*_grid_rain_hourly_mm.csv"

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


def discover_events():
    events = []
    pat = re.compile(r"Event_(\d+)_grid_rain_hourly_mm\.csv$")
    for p in sorted(EVENT_DIR.glob(EVENT_GLOB)):
        m = pat.search(p.name)
        if m:
            events.append(int(m.group(1)))
    return sorted(set(events))


AVAILABLE_EVENTS = discover_events()
if not AVAILABLE_EVENTS:
    print(f"[events] No event files found in {EVENT_DIR} matching {EVENT_GLOB}")


@lru_cache(maxsize=12)
def load_event_weights_map(event_no: int):
    f = WEIGHTS_DIR / f"Event_{int(event_no)}_weights.csv"
    if not f.exists():
        raise FileNotFoundError(f"Weights file not found: {f}")

    df = pd.read_csv(f)
    if "id" not in df.columns:
        raise ValueError(f"{f} must contain an 'id' column.")

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


@lru_cache(maxsize=12)
def load_event_matrix(event_no: int):
    f = EVENT_DIR / f"Event_{int(event_no)}_grid_rain_hourly_mm.csv"
    if not f.exists():
        raise FileNotFoundError(f"Event file not found: {f}")

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
    print("First 10 GRID_ID_ARR:", GRID_ID_ARR[:10].tolist())
    print("First 10 matched event columns:", list(col_lookup.keys())[:10])
    print("Matched grid columns count:", len(col_lookup))
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
    print("Any finite centroid rain?", np.isfinite(rain_mat).any())
    print("First row first 10 centroid values:", rain_mat[0, :10])
    finite = rain_mat[np.isfinite(rain_mat)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = 0.0, float(np.nanmax(finite))
        if vmax <= 0:
            vmax = 1.0

    return times, rain_mat, vmin, vmax


@lru_cache(maxsize=12)
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

@lru_cache(maxsize=12)
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

def build_figure(
    grid_id: int,
    show_lines: bool,
    circle_km_list,
    centroid_size: int,
    centroid_opacity: float,
    gauge_size: int,
    gauge_opacity: float,
    event_no: int | None,
    time_idx: int | None,
    ):
    row = GRID_DF.loc[GRID_DF[GRID_ID_COL] == int(grid_id)].iloc[0]
    tx = float(row[GRID_LON_COL])
    ty = float(row[GRID_LAT_COL])

    selected_gauges = []
    event_station_ids = set()

    if event_no is not None and event_no in AVAILABLE_EVENTS:
        try:
            weights_map = load_event_weights_map(int(event_no))
            selected_gauges = weights_map.get(int(grid_id), [])
        except Exception as e:
            print(f"[weights] load failed for event {event_no}: {e}")

        try:
            event_station_ids = load_event_station_ids(int(event_no))
        except Exception as e:
            print(f"[event stations] load failed for event {event_no}: {e}")

    selected_ids = [d["gauge_id"] for d in selected_gauges]
    selected_weights_lookup = {d["gauge_id"]: d["weight"] for d in selected_gauges}

    fig = go.Figure()
    mask_selected = np.isin(STN_ID_ARR, np.array(selected_ids, dtype=int)) if selected_ids else np.zeros(len(STN_ID_ARR), dtype=bool)
    missing_ids = [i for i in selected_ids if i not in STN_ID_SET]
    catchment_traces = []
    if not CATCH_GDF.empty:
        for geom in CATCH_GDF.geometry:
            catchment_traces.extend(polygon_to_traces(geom, line_width=1.5))

    station_rain = None
    common_vmin, common_vmax = 0.0, 1.0
    rain_note = "No event selected."
    selected_grid_rain = np.nan

    if event_no is not None and event_no in AVAILABLE_EVENTS:
        try:
            times, rain_mat, _, vmax_grid = load_event_matrix(int(event_no))
            _, st_rain_mat, _, station_vmax = load_event_station_matrix(int(event_no))

            if time_idx is None:
                time_idx = 0
            time_idx = int(np.clip(time_idx, 0, len(times) - 1))

            common_vmax = max(vmax_grid, station_vmax)
            if common_vmax <= 0:
                common_vmax = 1.0

            z = rain_mat[time_idx, :]
            try:
                selected_idx = np.where(GRID_ID_ARR == int(grid_id))[0][0]
                selected_grid_rain = z[selected_idx]
            except Exception:
                selected_grid_rain = np.nan
            station_rain = st_rain_mat[time_idx, :]
            ts_label = times[time_idx].strftime("%Y-%m-%d %H:%M")

            fig.add_trace(go.Scatter(
                x=GRID_LON,
                y=GRID_LAT,
                mode="markers",
                name="Rain (centroids)",
                marker=dict(
                    symbol="square",
                    size=centroid_size,
                    opacity=centroid_opacity,
                    color=z,
                    colorscale=RAIN_COLORSCALE,
                    cmin=common_vmin,
                    cmax=common_vmax,
                    colorbar=dict(title="Rain (mm)"),
                    line=dict(width=0),
                ),
                customdata=np.column_stack([
                    GRID_ID_ARR,
                    np.repeat("grid", len(GRID_ID_ARR)),
                    z
                ]),
                hovertemplate=(
                    "Grid ID: %{customdata[0]}"
                    "<br>Rain: %{customdata[2]:.3f} mm"
                    "<br>Lon: %{x:.5f}"
                    "<br>Lat: %{y:.5f}"
                    "<extra></extra>"
                ),
            ))
            rain_note = f"Event {event_no}, time: {ts_label}"
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
            rain_note = f"Event load failed: {e}"
    else:
        fig.add_trace(go.Scattergl(
            x=GRID_LON,
            y=GRID_LAT,
            mode="markers",
            name="Centroids",
            marker=dict(
            symbol="square",size=centroid_size, opacity=centroid_opacity, color="gray"),
            customdata=GRID_ID_ARR,
            hovertemplate="Grid ID: %{customdata}<br>Lon: %{x:.5f}<br>Lat: %{y:.5f}<extra></extra>",
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
        selected_mask = (
            np.isin(STN_ID_ARR, np.array(selected_ids, dtype=int))
            if selected_ids else
            np.zeros(len(STN_ID_ARR), dtype=bool)
        )

        event_station_mask = np.isin(STN_ID_ARR, np.array(sorted(event_station_ids), dtype=int))
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
        sel_weights = np.array(
            [selected_weights_lookup.get(int(gid), np.nan) for gid in STN_ID_ARR[sel_idx]],
            dtype=float
        )

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

    fig.update_layout(
        title=f"Grid {int(grid_id)} | {rain_note} | centroids={len(GRID_DF)} | gauges={len(STN_DF)}",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=900,
        margin=dict(l=40, r=20, t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        uirevision="keep",
        dragmode="pan",
    )
    fig.update_xaxes(range=FULL_XRANGE)
    fig.update_yaxes(range=FULL_YRANGE)

    info = {"selected_ids_count": len(selected_ids), "missing_ids": missing_ids}
    
    # draw selected-gauge connection lines above points if you want
    if show_lines and selected_ids:
        for d in selected_gauges:
            sid = d["gauge_id"]
            w = d["weight"]   # <-- important

            hit = np.where(STN_ID_ARR == int(sid))[0]
            if len(hit) == 0:
                continue
            j = hit[0]

            # choose style based on weight
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
                line=dict(
                    width=line_width,
                    color=line_color,
                    dash=dash_style
                ),
                hoverinfo="skip",
            ))
    
    # catchment boundaries on top of visual layers
    for tr in catchment_traces:
        fig.add_trace(tr)
        
    fig.add_trace(go.Scatter(
            x=GRID_LON,
            y=GRID_LAT,
            mode="markers",
            name="centroid-click-layer",
            marker=dict(
                symbol="square",
                size=max(centroid_size + 14, 20),
                opacity=0.01,
                color="rgba(0,0,0,0.01)"
            ),
            customdata=GRID_ID_ARR,
            hoverinfo="skip",
            showlegend=False,
        ))
    
    return fig, info


app = Dash(__name__)
app.title = "WGS84 Grid centroids + gauges viewer"

default_grid_id = int(GRID_IDS[0]) if GRID_IDS else None
default_event = AVAILABLE_EVENTS[0] if AVAILABLE_EVENTS else None

app.layout = html.Div(
    style={"display": "flex", "gap": "14px", "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"},
    children=[
        html.Div(
            style={"width": "360px", "padding": "14px", "borderRight": "1px solid #ddd"},
            children=[
                html.H3("Controls", style={"marginTop": "0px"}),
                html.Label("Event"),
                dcc.Dropdown(
                    id="event-dropdown",
                    options=[{"label": f"E{e}", "value": int(e)} for e in AVAILABLE_EVENTS],
                    value=default_event,
                    searchable=True,
                    clearable=True,
                    placeholder="Select event",
                ),
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
            ],
        ),
        html.Div(
            style={"flex": "1", "padding": "10px"},
            children=[
                html.H2("WGS84 viewer: click a centroid to select", style={"marginTop": "0px"}),
                dcc.Graph(id="map", figure=go.Figure(), config={"displayModeBar": True, "scrollZoom": True}),
            ],
        ),
    ],
)


@app.callback(
    Output("selected-grid-store", "data"),
    Output("grid-dropdown", "value"),
    Input("map", "clickData"),
    Input("grid-dropdown", "value"),
    State("selected-grid-store", "data"),
    prevent_initial_call=False,
)
def update_selected_grid(clickData, dropdown_value, current):
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

    if trig == "map.clickData":
        if not clickData or "points" not in clickData or len(clickData["points"]) == 0:
            return current, current

        pt = clickData["points"][0]
        cd = pt.get("customdata", None)

        if cd is not None:
            try:
                if isinstance(cd, (list, tuple, np.ndarray)) and len(cd) >= 2 and cd[1] == "grid":
                    new_gid = int(float(cd[0]))
                    return new_gid, new_gid
            except Exception:
                pass

        return current, current
    return current, current


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
    Input("event-dropdown", "value"),
    Input("anim-interval", "n_intervals"),
    Input("time-slider", "value"),
    State("anim-playing", "data"),
    prevent_initial_call=False,
)
def sync_time_controls(event_no, n_intervals, slider_value, playing):
    trig = callback_context.triggered[0]["prop_id"] if callback_context.triggered else None

    if event_no is None:
        return 0, 0, "No event selected."

    try:
        times, _, _, _ = load_event_matrix(int(event_no))
    except Exception as e:
        return 0, 0, f"Event load failed: {e}"

    if len(times) == 0:
        return 0, 0, f"Event {int(event_no)}: no timestamps."

    max_idx = len(times) - 1
    idx = int(slider_value or 0)
    idx = int(np.clip(idx, 0, max_idx))

    if trig == "event-dropdown.value":
        idx = 0
    elif trig == "anim-interval.n_intervals" and playing:
        idx = 0 if idx >= max_idx else idx + 1

    label = f"Event {int(event_no)}: {times[idx].strftime('%Y-%m-%d %H:%M')}"
    return max_idx, idx, label


@app.callback(
    Output("map", "figure"),
    Output("status-box", "children"),
    Input("selected-grid-store", "data"),
    Input("show-lines", "value"),
    Input("circles", "value"),
    Input("centroid-size", "value"),
    Input("centroid-opacity", "value"),
    Input("gauge-size", "value"),
    Input("gauge-opacity", "value"),
    Input("reset-view", "n_clicks"),
    Input("event-dropdown", "value"),
    Input("time-slider", "value"),
)
def redraw(
    grid_id,
    show_lines_value,
    circles_value,
    centroid_size,
    centroid_opacity,
    gauge_size,
    gauge_opacity,
    reset_clicks,
    event_no,
    time_idx,
):
    if grid_id is None and GRID_IDS:
        grid_id = int(GRID_IDS[0])

    show_lines = "on" in (show_lines_value or [])
    circle_km_list = circles_value or []

    fig, info = build_figure(
        int(grid_id),
        show_lines=show_lines,
        circle_km_list=circle_km_list,
        centroid_size=int(centroid_size),
        centroid_opacity=float(centroid_opacity),
        gauge_size=int(gauge_size),
        gauge_opacity=float(gauge_opacity),
        event_no=(int(event_no) if event_no is not None else None),
        time_idx=(int(time_idx) if time_idx is not None else None),
    )

    if reset_clicks:
        fig.update_xaxes(range=FULL_XRANGE)
        fig.update_yaxes(range=FULL_YRANGE)
    fig.update_layout(
    yaxis_scaleanchor="x",
    yaxis_scaleratio=1
)

    status = f"Selected grid: {int(grid_id)}\nWeighted gauges count: {info['selected_ids_count']}\n"
    if info["missing_ids"]:
        status += f"Missing weighted gauges in Stations_df: {info['missing_ids']}\n"
    else:
        status += "Weighted gauge IDs all found in Stations_df.\n"

    if event_no is None:
        status += "Event: none\n"
    else:
        status += f"Event: {int(event_no)} | time index: {int(time_idx or 0)}\n"

    return fig, status


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8890, debug=True)
