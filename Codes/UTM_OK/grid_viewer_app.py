import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

from pathlib import Path                 # ### NEW/CHANGED ###
import re                               # ### NEW/CHANGED ###
from functools import lru_cache          # ### NEW/CHANGED ###

# ------------------- EDIT THESE PATHS -------------------
GRID_CSV = "/mnt/12TB/Sujan/Spatial_correlation/Codes/01_Correlation_and_Variogram/grid_centers_full.csv"
STATIONS_CSV = "/mnt/12TB/Sujan/Spatial_correlation/Codes/01_Correlation_and_Variogram/Stations_df.csv"
WEIGHTS_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/02_OK_Weights")
# ### NEW/CHANGED ### Kriging event files
EVENT_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/03_Interpolated_Rain")
EVENT_GLOB = "Event_*_grid_rain_hourly_mm.csv"
# --------------------------------------------------------

# ------------------- COLUMN NAMES (EDIT IF NEEDED) -------------------
GRID_ID_COL = "id"
GRID_X_COL = "UTM_Easting"
GRID_Y_COL = "UTM_Northing"

STN_ID_COL = "ID"
STN_X_COL = "NAD83_15N_Long"
STN_Y_COL = "NAD83_15N_Lat"

CATCHMENT_SHP_PATHS = [
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6893390/6893390.shp",
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/06893080/6893080.shp",
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6892513/6892513.shp",
]

RAIN_COLORSCALE = [
    [0.00, "rgb(255,255,255)"],
    [0.20, "rgb(0,0,255)"],
    [0.40, "rgb(0,180,0)"],
    [0.70, "rgb(255,255,0)"],
    [1.00, "rgb(255,0,0)"],
]
# --------------------------------------------------------------------

def load_catchments(shp_paths, target_epsg=26915, simplify_tol_m=20):
    gdfs = []
    for p in shp_paths:
        try:
            g = gpd.read_file(p)
            if g.empty:
                continue

            g = g.to_crs(epsg=target_epsg)

            if simplify_tol_m is not None:
                g["geometry"] = g["geometry"].simplify(simplify_tol_m, preserve_topology=True)

            gdfs.append(g[["geometry"]].copy())
        except Exception as e:
            print(f"[catchments] Failed to read {p}: {e}")

    if not gdfs:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=f"EPSG:{target_epsg}")

    out = pd.concat(gdfs, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=f"EPSG:{target_epsg}")
    out = out[out.geometry.notnull()].copy()
    return out


def polygon_to_traces(geom, line_width=1, fill_opacity=0.05):
    traces = []

    def add_poly(poly: Polygon):
        x, y = poly.exterior.coords.xy
        traces.append(
            go.Scatter(
                x=list(x),
                y=list(y),
                mode="lines",
                line=dict(width=line_width, color="rgba(0,0,0,0.25)"),
                fill="toself",
                fillcolor=f"rgba(0,0,0,{fill_opacity})",
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


def circle_xy(cx, cy, r_m, n=360):
    t = np.linspace(0, 2 * np.pi, n)
    return cx + r_m * np.cos(t), cy + r_m * np.sin(t)


def load_data():
    grid_df = pd.read_csv(GRID_CSV)
    stn_df = pd.read_csv(STATIONS_CSV)

    # IDs
    grid_df[GRID_ID_COL] = pd.to_numeric(grid_df[GRID_ID_COL], errors="coerce").astype("Int64")
    stn_df[STN_ID_COL] = pd.to_numeric(stn_df[STN_ID_COL], errors="coerce").astype("Int64")

    # coords
    grid_df[GRID_X_COL] = pd.to_numeric(grid_df[GRID_X_COL], errors="coerce")
    grid_df[GRID_Y_COL] = pd.to_numeric(grid_df[GRID_Y_COL], errors="coerce")
    stn_df[STN_X_COL] = pd.to_numeric(stn_df[STN_X_COL], errors="coerce")
    stn_df[STN_Y_COL] = pd.to_numeric(stn_df[STN_Y_COL], errors="coerce")

    # drop bad rows
    grid_df = grid_df.dropna(subset=[GRID_ID_COL, GRID_X_COL, GRID_Y_COL]).copy()
    stn_df = stn_df.dropna(subset=[STN_ID_COL, STN_X_COL, STN_Y_COL]).copy()

    grid_df[GRID_ID_COL] = grid_df[GRID_ID_COL].astype(int)
    stn_df[STN_ID_COL] = stn_df[STN_ID_COL].astype(int)

   

    return grid_df, stn_df


GRID_DF, STN_DF = load_data()

CATCH_GDF = load_catchments(
    CATCHMENT_SHP_PATHS,
    target_epsg=26915,
    simplify_tol_m=20,
)

GRID_IDS = sorted(GRID_DF[GRID_ID_COL].unique().tolist())

# Precompute arrays for speed
GRID_X = GRID_DF[GRID_X_COL].to_numpy(float)
GRID_Y = GRID_DF[GRID_Y_COL].to_numpy(float)
GRID_ID_ARR = GRID_DF[GRID_ID_COL].to_numpy(int)

GX_ALL = STN_DF[STN_X_COL].to_numpy(float)
GY_ALL = STN_DF[STN_Y_COL].to_numpy(float)
GID_ALL = STN_DF[STN_ID_COL].to_numpy(int)
GID_SET = set(GID_ALL.tolist())

# Full domain ranges
xmin = min(GRID_X.min(), GX_ALL.min())
xmax = max(GRID_X.max(), GX_ALL.max())
ymin = min(GRID_Y.min(), GY_ALL.min())
ymax = max(GRID_Y.max(), GY_ALL.max())
padx = 0.03 * (xmax - xmin)
pady = 0.03 * (ymax - ymin)
FULL_XRANGE = [float(xmin - padx), float(xmax + padx)]
FULL_YRANGE = [float(ymin - pady), float(ymax + pady)]


# ------------------- NEW/CHANGED: Event discovery + loader -------------------

def discover_events():
    """
    Returns sorted list of available event numbers discovered from filenames.
    """
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

@lru_cache(maxsize=6)
def load_event_weights_map(event_no: int):
    """
    Reads Event_{n}_weights.csv and returns:
      weights_map[grid_id] = [
          {"gauge_id": ..., "weight": ...},
          ...
      ]
    Works for g1..gN / w1..wN depending on available columns.
    """
    f = WEIGHTS_DIR / f"Event_{int(event_no)}_weights.csv"
    if not f.exists():
        raise FileNotFoundError(f"Weights file not found: {f}")

    df = pd.read_csv(f)

    if "id" not in df.columns:
        raise ValueError(f"{f} must contain an 'id' column.")

    # find all g-columns dynamically: g1, g2, g3, g4, ...
    g_cols = []
    for c in df.columns:
        if re.fullmatch(r"g\d+", str(c)):
            g_cols.append(c)

    # sort numerically, so g1,g2,g3,... not g1,g10,g2
    g_cols = sorted(g_cols, key=lambda x: int(x[1:]))

    weights_map = {}

    for _, row in df.iterrows():
        try:
            gid = int(row["id"])
        except Exception:
            continue

        chosen = []
        for gcol in g_cols:
            idx = gcol[1:]          # "1", "2", ...
            wcol = f"w{idx}"

            if gcol not in row.index or wcol not in row.index:
                continue
            if pd.isna(row[gcol]):
                continue

            try:
                gauge_id = int(row[gcol])
            except Exception:
                continue

            try:
                weight = float(row[wcol]) if pd.notna(row[wcol]) else np.nan
            except Exception:
                weight = np.nan

            chosen.append({
                "gauge_id": gauge_id,
                "weight": weight
            })

        weights_map[gid] = chosen

    return weights_map

@lru_cache(maxsize=6)
def load_event_matrix(event_no: int):
    f = EVENT_DIR / f"Event_{int(event_no)}_grid_rain_hourly_mm.csv"
    if not f.exists():
        raise FileNotFoundError(f"Event file not found: {f}")

    df = pd.read_csv(f)

    if "time_local" not in df.columns:
        raise ValueError(f"{f} must contain a 'time_local' column.")

    # Parse time_local
    t = pd.to_datetime(df["time_local"], errors="coerce")
    if t.isna().all():
        raise ValueError(f"Could not parse time_local in {f}.")

    # Build times index
    times = pd.DatetimeIndex(t)

    # Extract grid-id columns from header (they are strings in CSV)
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
    ngrid = len(GRID_ID_ARR)
    rain_mat = np.full((ntime, ngrid), np.nan, dtype=float)

    # Fill matrix aligned to GRID_ID_ARR
    for j, gid in enumerate(GRID_ID_ARR):
        c = col_lookup.get(int(gid), None)
        if c is not None:
            rain_mat[:, j] = pd.to_numeric(df[c], errors="coerce").to_numpy(float)

    # Sort by time (keeps slider/animation consistent)
    order = np.argsort(times.values)
    times = times[order]
    rain_mat = rain_mat[order, :]

    # Fixed limits across all times (per event)
    finite = rain_mat[np.isfinite(rain_mat)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        # Option A (recommended for rainfall): clip negatives on the color scale
        vmin, vmax = 0.0, float(np.nanmax(finite))
        if vmax <= 0:
            vmax = 1.0

        # Option B (diagnostics): show negatives too
        # vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        # if vmin == vmax:
        #     vmax = vmin + 1.0

    return times, rain_mat, vmin, vmax

@lru_cache(maxsize=6)
def load_event_station_matrix(event_no: int):
    f = EVENT_DIR / f"Event_{int(event_no)}_station_rain_used_hourly_mm.csv"
    if not f.exists():
        raise FileNotFoundError(f"Station event file not found: {f}")

    df = pd.read_csv(f)

    if "time_local" not in df.columns:
        raise ValueError(f"{f} must contain a 'time_local' column.")

    # parse times
    t = pd.to_datetime(df["time_local"], errors="coerce")
    if t.isna().all():
        raise ValueError(f"Could not parse time_local in {f}.")

    times = pd.DatetimeIndex(t)

    # station columns
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
    nstn = len(GID_ALL)
    rain_mat = np.full((ntime, nstn), np.nan, dtype=float)

    # align to station arrays GID_ALL / GX_ALL / GY_ALL
    for j, sid in enumerate(GID_ALL):
        c = col_lookup.get(int(sid), None)
        if c is not None:
            rain_mat[:, j] = pd.to_numeric(df[c], errors="coerce").to_numpy(float)

    # sort by time
    order = np.argsort(times.values)
    times = times[order]
    rain_mat = rain_mat[order, :]

    finite = rain_mat[np.isfinite(rain_mat)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = 0.0, float(np.nanmax(finite))   # fixed across all times for the event
        if vmax <= 0:
            vmax = 1.0

    return times, rain_mat, vmin, vmax

def build_figure(
    grid_id: int,
    show_lines: bool,
    circle_km_list,
    centroid_size: int,
    centroid_opacity: float,
    gauge_size: int,
    gauge_opacity: float,
    event_no: int | None,          # ### NEW/CHANGED ###
    time_idx: int | None,          # ### NEW/CHANGED ###
):
    # selected grid coords
    
    row = GRID_DF.loc[GRID_DF[GRID_ID_COL] == int(grid_id)].iloc[0]
    tx = float(row[GRID_X_COL])
    ty = float(row[GRID_Y_COL])
    station_rain = None
    station_vmin, station_vmax = 0.0, 1.0
    # distances
    dx = GX_ALL - tx
    dy = GY_ALL - ty
    dist = np.sqrt(dx * dx + dy * dy)

    # weighted gages
    selected_gauges = []
    if event_no is not None:
        try:
            weights_map = load_event_weights_map(int(event_no))
            selected_gauges = weights_map.get(int(grid_id), [])
        except Exception as e:
            print(f"[weights] load failed for event {event_no}: {e}")
            selected_gauges = []

    selected_ids = [d["gauge_id"] for d in selected_gauges]
    selected_weights_lookup = {d["gauge_id"]: d["weight"] for d in selected_gauges}
    
    fig = go.Figure()
    
    mask_selected = np.isin(GID_ALL, np.array(selected_ids, dtype=int)) if selected_ids else np.zeros(len(GID_ALL), dtype=bool)
    missing_ids = [i for i in selected_ids if i not in GID_SET]
    
    #lines st to weighed gages
    
    if show_lines and selected_ids:
        for d in selected_gauges:
            sid = d["gauge_id"]
            w = d["weight"]

            hit = np.where(GID_ALL == int(sid))[0]
            if len(hit) == 0:
                continue
            j = hit[0]

            fig.add_trace(go.Scatter(
                x=[tx, GX_ALL[j]],
                y=[ty, GY_ALL[j]],
                mode="lines",
                showlegend=False,
                line=dict(width=1, color="black"),
                customdata=[[sid, w], [sid, w]],
                hovertemplate=(
                    "Gauge ID: %{customdata[0]}"
                    "<br>Weight: %{customdata[1]:.4f}"
                    "<extra></extra>"
                ),
            ))

    # Catchment polygons (background)
    if not CATCH_GDF.empty:
        for geom in CATCH_GDF.geometry:
            for tr in polygon_to_traces(geom, line_width=1, fill_opacity=0.10):
                fig.add_trace(tr)

    # ------------------- NEW/CHANGED: Rainfall colored centroids -------------------
    rain_note = "No event selected."
    if event_no is not None and event_no in AVAILABLE_EVENTS:
        try:
            times, rain_mat, vmin_grid, vmax_grid = load_event_matrix(int(event_no))
            st_times, st_rain_mat, station_vmin, station_vmax = load_event_station_matrix(int(event_no))
            common_vmin = 0.0
            common_vmax = max(vmax_grid, station_vmax)
            if common_vmax <= 0:
                common_vmax = 1.0
            if time_idx is None:
                time_idx = 0
            time_idx = int(np.clip(time_idx, 0, len(times) - 1))
            station_rain = st_rain_mat[time_idx, :]

            z = rain_mat[time_idx, :]
            ts_label = str(times[time_idx])

            fig.add_trace(go.Scatter(
                x=GRID_X, y=GRID_Y,
                mode="markers",
                name="Rain (centroids)",
                marker=dict(
                    size=centroid_size,
                    opacity=centroid_opacity,
                    color=z,
                    colorscale=RAIN_COLORSCALE,
                    cmin=common_vmin,
                    cmax=common_vmax,
                    colorbar=dict(title="Rain (mm)"),
                ),
                customdata=GRID_ID_ARR,
                hovertemplate="Grid ID: %{customdata}<br>Rain: %{marker.color:.2f} mm<br>E: %{x:.1f} m<br>N: %{y:.1f} m<extra></extra>",
            ))
            rain_note = f"Event {event_no}, time: {ts_label} (index {time_idx}/{len(times)-1})"
        except Exception as e:
            print(f"[station rain] load failed for event {event_no}: {e}")
            station_rain = None
            # fallback: plain centroids if event load fails
            fig.add_trace(go.Scatter(
                x=GRID_X, y=GRID_Y,
                mode="markers",
                name="CENTROIDS",
                marker=dict(size=centroid_size, opacity=centroid_opacity, color="gray"),
                customdata=GRID_ID_ARR,
                hovertemplate="Grid ID: %{customdata}<br>E: %{x:.1f} m<br>N: %{y:.1f} m<extra></extra>",
            ))
            rain_note = f"Event load failed: {e}"
    else:
        # No event chosen, show neutral centroids
        fig.add_trace(go.Scatter(
            x=GRID_X, y=GRID_Y,
            mode="markers",
            name="CENTROIDS",
            marker=dict(size=centroid_size, opacity=centroid_opacity, color="gray"),
            customdata=GRID_ID_ARR,
            hovertemplate="Grid ID: %{customdata}<br>E: %{x:.1f} m<br>N: %{y:.1f} m<extra></extra>",
        ))

    # Centroid hitbox layer (invisible but clickable)
    fig.add_trace(go.Scatter(
        x=GRID_X, y=GRID_Y,
        mode="markers",
        name="Centroid hitbox",
        marker=dict(size=max(centroid_size + 12, 18), opacity=0.0, color="rgba(0,0,0,0)"),
        customdata=GRID_ID_ARR,
        hoverinfo="skip",
        showlegend=False,
    ))
    # ------------------------------------------------------------------------------

    # Selected centroid
    fig.add_trace(go.Scatter(
        x=[tx], y=[ty],
        mode="markers",
        name="Selected grid",
        marker=dict( size=16, color="black"),
        hovertemplate=f"Selected Grid ID: {int(grid_id)}<br>E: {tx:.1f} m<br>N: {ty:.1f} m<extra></extra>",
    ))

    if station_rain is not None:
        used_mask = np.isfinite(station_rain)
        unused_mask = ~used_mask

        # Unused gauges in gray
        if unused_mask.any():
            fig.add_trace(go.Scatter(
                x=GX_ALL[unused_mask], y=GY_ALL[unused_mask],
                mode="markers",
                name="Unused gauges",
                marker=dict(
                    symbol="triangle-up",
                    size=gauge_size,
                    opacity=max(gauge_opacity, 0.35),
                    color="lightgray",
                    line=dict(width=0.8, color="black"),
                ),
                text=GID_ALL[unused_mask].astype(str),
                hovertemplate="Gauge ID: %{text}<br>Not used in this event<extra></extra>",
            ))

        # Used gauges colored by rainfall
        if used_mask.any():
            fig.add_trace(go.Scatter(
                x=GX_ALL[used_mask], y=GY_ALL[used_mask],
                mode="markers",
                name="Gauge rain",
                marker=dict(
                    symbol="triangle-up",
                    size=gauge_size + 2,
                    opacity=0.95,
                    color=station_rain[used_mask],
                    colorscale=RAIN_COLORSCALE,
                    cmin=common_vmin,
                    cmax=common_vmax,
                    line=dict(width=1, color="black"),
                    showscale=False,
                ),
                text=GID_ALL[used_mask].astype(str),
                customdata=np.round(station_rain[used_mask], 3),
                hovertemplate=(
                    "Gauge ID: %{text}"
                    "<br>Rain: %{customdata:.3f} mm"
                    "<br>E: %{x:.1f} m"
                    "<br>N: %{y:.1f} m"
                    "<extra></extra>"
                ),
            ))
    else:
        # fallback if station event file is missing
        fig.add_trace(go.Scatter(
            x=GX_ALL, y=GY_ALL,
            mode="markers",
            name="All gauges",
            marker=dict(
                symbol="triangle-up",
                size=gauge_size,
                opacity=gauge_opacity,
                color="dodgerblue",
                line=dict(width=1, color="black"),
            ),
            text=GID_ALL.astype(str),
            hovertemplate="Gauge ID: %{text}<br>E: %{x:.1f} m<br>N: %{y:.1f} m<extra></extra>",
        ))

        # Selected 7
    if selected_ids:
        sel_idx = np.where(mask_selected)[0]

        sel_weights = np.array([selected_weights_lookup.get(int(gid), np.nan) for gid in GID_ALL[sel_idx]], dtype=float)

        fig.add_trace(go.Scatter(
            x=GX_ALL[sel_idx],
            y=GY_ALL[sel_idx],
            mode="markers",
            name="Weighted gauges",
            marker=dict(
                symbol="triangle-up",
                size=gauge_size + 8,
                color="rgba(0,0,0,0)",   # transparent fill so rainfall color stays visible underneath
                line=dict(width=0.5, color="red"),
            ),
            text=GID_ALL[sel_idx].astype(str),
            customdata=np.column_stack([GID_ALL[sel_idx], sel_weights]),
            hovertemplate=(
                "Gauge ID: %{customdata[0]}"
                "<br>Weight: %{customdata[1]:.4f}"
                "<extra></extra>"
            ),
        ))
    
    # Circles
    for km in sorted(circle_km_list):
        xc, yc = circle_xy(tx, ty, float(km) * 1000.0)
        fig.add_trace(go.Scatter(
            x=xc, y=yc,
            mode="lines",
            name=f"{km} km circle",
            hoverinfo="skip",
            line=dict(width=2, dash="dot", color="gray"),
        ))

    fig.update_layout(
        title=f"Grid {int(grid_id)} | {rain_note} | centroids={len(GRID_DF)} | gauges={len(STN_DF)}",
        xaxis_title="UTM Easting (m)",
        yaxis_title="UTM Northing (m)",
        height=820,
        margin=dict(l=40, r=20, t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        uirevision="keep",
        dragmode="pan",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(range=FULL_XRANGE)
    fig.update_yaxes(range=FULL_YRANGE)

    info = {
        "selected_ids_count": len(selected_ids),
        "missing_ids": missing_ids,
    }
    return fig, info


app = Dash(__name__)
app.title = "Grid centroids + gauges viewer (Dash)"

default_grid_id = int(GRID_IDS[0])
default_event = AVAILABLE_EVENTS[0] if AVAILABLE_EVENTS else None

app.layout = html.Div(
    style={"display": "flex", "gap": "14px", "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"},
    children=[
        # Sidebar
        html.Div(
            style={"width": "360px", "padding": "14px", "borderRight": "1px solid #ddd"},
            children=[
                html.H3("Controls", style={"marginTop": "0px"}),

                # ### NEW/CHANGED: Event selector
                html.Label("Event (kriging result)"),
                dcc.Dropdown(
                    id="event-dropdown",
                    options=[{"label": f"E{e}", "value": int(e)} for e in AVAILABLE_EVENTS],
                    value=default_event,
                    searchable=True,
                    clearable=True,
                    placeholder="Select event",
                ),

                html.Div(style={"height": "10px"}),

                # ### NEW/CHANGED: Time slider + play/pause
                html.Label("Time (drag slider)"),
                dcc.Slider(
                    id="time-slider",
                    min=0,
                    max=0,
                    step=1,
                    value=0,
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
                html.Div(style={"height": "6px"}),
                html.Div(id="time-label", style={"fontSize": "13px"}),

                html.Div(style={"height": "8px"}),

                html.Div(
                    style={"display": "flex", "gap": "8px"},
                    children=[
                        html.Button("Play", id="play-btn", n_clicks=0),
                        html.Button("Pause", id="pause-btn", n_clicks=0),
                    ],
                ),
                dcc.Interval(id="anim-interval", interval=500, n_intervals=0, disabled=True),
                dcc.Store(id="anim-playing", data=False),

                html.Hr(),

                html.Label("Grid ID (search/dropdown)"),
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

                html.Div(style={"height": "8px"}),

                html.Hr(),

                html.Label("Centroid size"),
                dcc.Slider(id="centroid-size", min=3, max=12, step=1, value=6),

                html.Label("Centroid opacity"),
                dcc.Slider(id="centroid-opacity", min=0.1, max=1.0, step=0.05, value=0.8),

                html.Div(style={"height": "10px"}),

                html.Label("Gauge size"),
                dcc.Slider(id="gauge-size", min=3, max=12, step=1, value=6),

                html.Label("Gauge opacity"),
                dcc.Slider(id="gauge-opacity", min=0.05, max=1.0, step=0.05, value=0.45),

                html.Hr(),

                html.Button("Reset view to full domain", id="reset-view", n_clicks=0),

                html.Div(style={"height": "10px"}),

                html.Div(id="status-box", style={"fontSize": "13px", "whiteSpace": "pre-wrap"}),

                dcc.Store(id="selected-grid-store", data=default_grid_id),
            ],
        ),

        # Main panel
        html.Div(
            style={"flex": "1", "padding": "10px"},
            children=[
                html.H2("One map: click a centroid to select", style={"marginTop": "0px"}),
                dcc.Graph(
                    id="map",
                    figure=go.Figure(),
                    config={"displayModeBar": True, "scrollZoom": True},
                ),
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
        current = int(GRID_IDS[0])

    trig = callback_context.triggered[0]["prop_id"] if callback_context.triggered else None

    if trig == "grid-dropdown.value":
        if dropdown_value is None:
            return current, current
        v = int(dropdown_value)
        return v, v

    if trig == "map.clickData" and clickData:
        pt = clickData["points"][0]
        idx = pt.get("pointIndex", None)
        if idx is not None and 0 <= idx < len(GRID_ID_ARR):
            new_gid = int(GRID_ID_ARR[int(idx)])
            return new_gid, new_gid
        return current, current

    return current, current


# ### NEW/CHANGED: when event changes, reset slider bounds and label
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

# ### NEW/CHANGED: interval advances the time slider
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
    Input("event-dropdown", "value"),     # ### NEW/CHANGED ###
    Input("time-slider", "value"),        # ### NEW/CHANGED ###
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

    status = (
        f"Selected grid: {int(grid_id)}\n"
        f"Weighted gauges count: {info['selected_ids_count']}\n"
    )
    if info["missing_ids"]:
        status += f"Missing selected-7 in Stations_df: {info['missing_ids']}\n"
    else:
        status += "Selected-7 IDs all found in Stations_df.\n"

    if event_no is None:
        status += "Event: none\n"
    else:
        status += f"Event: {int(event_no)} | time index: {int(time_idx or 0)}\n"

    return fig, status
print(app.callback_map.keys())

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8890, debug=True)