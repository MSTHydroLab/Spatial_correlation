import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

# ------------------- EDIT THESE PATHS -------------------
GRID_CSV = "/mnt/12TB/Sujan/Spatial_correlation/Codes/01_Correlation_and_Variogram/grid_centers_full.csv"
STATIONS_CSV = "/mnt/12TB/Sujan/Spatial_correlation/Codes/01_Correlation_and_Variogram/Stations_df.csv"
NEAREST7_CSV = "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/grid_nearest7.csv"
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
# --------------------------------------------------------------------
def load_catchments(shp_paths, target_epsg=26915, simplify_tol_m=20):
    gdfs = []
    for p in shp_paths:
        try:
            g = gpd.read_file(p)
            if g.empty:
                continue

            # Your shapefiles are EPSG:4326, reproject to UTM 15N meters
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
    n7_df = pd.read_csv(NEAREST7_CSV)

    # IDs
    grid_df[GRID_ID_COL] = pd.to_numeric(grid_df[GRID_ID_COL], errors="coerce").astype("Int64")
    n7_df[GRID_ID_COL] = pd.to_numeric(n7_df[GRID_ID_COL], errors="coerce").astype("Int64")
    stn_df[STN_ID_COL] = pd.to_numeric(stn_df[STN_ID_COL], errors="coerce").astype("Int64")

    # coords
    grid_df[GRID_X_COL] = pd.to_numeric(grid_df[GRID_X_COL], errors="coerce")
    grid_df[GRID_Y_COL] = pd.to_numeric(grid_df[GRID_Y_COL], errors="coerce")
    stn_df[STN_X_COL] = pd.to_numeric(stn_df[STN_X_COL], errors="coerce")
    stn_df[STN_Y_COL] = pd.to_numeric(stn_df[STN_Y_COL], errors="coerce")

    # drop bad rows
    grid_df = grid_df.dropna(subset=[GRID_ID_COL, GRID_X_COL, GRID_Y_COL]).copy()
    stn_df = stn_df.dropna(subset=[STN_ID_COL, STN_X_COL, STN_Y_COL]).copy()
    n7_df = n7_df.dropna(subset=[GRID_ID_COL]).copy()

    grid_df[GRID_ID_COL] = grid_df[GRID_ID_COL].astype(int)
    n7_df[GRID_ID_COL] = n7_df[GRID_ID_COL].astype(int)
    stn_df[STN_ID_COL] = stn_df[STN_ID_COL].astype(int)

    # nearest7 mapping
    n7_map = {}
    for _, r in n7_df.iterrows():
        gid = int(r[GRID_ID_COL])
        ids = []
        for i in range(1, 8):
            c = f"g{i}"
            if c in r.index and pd.notna(r[c]):
                try:
                    ids.append(int(r[c]))
                except Exception:
                    pass
        n7_map[gid] = ids

    return grid_df, stn_df, n7_map


GRID_DF, STN_DF, N7_MAP = load_data()

CATCH_GDF = load_catchments(
    CATCHMENT_SHP_PATHS,
    target_epsg=26915,
    simplify_tol_m=20,   # optional: simplify by 20 m (speeds plotting). Use None to disable.
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


def build_figure(
    grid_id: int,
    show_lines: bool,
    circle_km_list,
    highlight10: bool,
    highlight5: bool,
    centroid_size: int,
    centroid_opacity: float,
    gauge_size: int,
    gauge_opacity: float,
):
    # selected grid coords
    # fast lookup by boolean mask (533 rows is tiny)
    row = GRID_DF.loc[GRID_DF[GRID_ID_COL] == int(grid_id)].iloc[0]
    tx = float(row[GRID_X_COL])
    ty = float(row[GRID_Y_COL])

    # distances
    dx = GX_ALL - tx
    dy = GY_ALL - ty
    dist = np.sqrt(dx * dx + dy * dy)
    mask10 = dist <= 10_000.0
    mask5 = dist <= 5_000.0

    # selected 7
    selected_ids = N7_MAP.get(int(grid_id), [])
    mask_sel7 = np.isin(GID_ALL, np.array(selected_ids, dtype=int))
    missing_ids = [i for i in selected_ids if i not in GID_SET]

    fig = go.Figure()
        # Catchment polygons (background)
    if not CATCH_GDF.empty:
        for geom in CATCH_GDF.geometry:
            for tr in polygon_to_traces(geom, line_width=1, fill_opacity=0.10):
                fig.add_trace(tr)

    # 0) All centroids (clickable)
    fig.add_trace(go.Scatter(
        x=GRID_X, y=GRID_Y,
        mode="markers",
        name="CENTROIDS",
        marker=dict(size=centroid_size, opacity=centroid_opacity, color="gray"),
        customdata=GRID_ID_ARR,
        hovertemplate="Grid ID: %{customdata}<br>E: %{x:.1f} m<br>N: %{y:.1f} m<extra></extra>",
    ))
    
    # 0b) Centroid hitbox layer (invisible but clickable)
    fig.add_trace(go.Scatter(
        x=GRID_X, y=GRID_Y,
        mode="markers",
        name="Centroid hitbox",
        marker=dict(size=max(centroid_size + 8, 14), opacity=0.0, color="rgba(0,0,0,0)"),
        customdata=GRID_ID_ARR,
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=GRID_X, y=GRID_Y,
        mode="markers",
        marker=dict(size=max(centroid_size + 10, 16), opacity=0.0, color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=False,
        name="Centroid hitbox",
    ))

    # 1) Selected centroid
    fig.add_trace(go.Scatter(
        x=[tx], y=[ty],
        mode="markers",
        name="Selected grid",
        marker=dict(symbol="triangle-up", size=16, color="black"),
        hovertemplate=f"Selected Grid ID: {int(grid_id)}<br>E: {tx:.1f} m<br>N: {ty:.1f} m<extra></extra>",
    ))

    # 2) All gauges
    fig.add_trace(go.Scatter(
        x=GX_ALL, y=GY_ALL,
        mode="markers",
        name="All gauges",
        marker=dict(size=gauge_size, opacity=gauge_opacity, color="dodgerblue"),
        text=GID_ALL.astype(str),
        hovertemplate="Gauge ID: %{text}<br>E: %{x:.1f} m<br>N: %{y:.1f} m<extra></extra>",
    ))

    # 3) Highlight ≤10 km
    if highlight10:
        fig.add_trace(go.Scatter(
            x=GX_ALL[mask10], y=GY_ALL[mask10],
            mode="markers",
            name="Gauges ≤10 km",
            marker=dict(size=max(gauge_size + 2, 8), opacity=0.95, color="blue"),
            text=GID_ALL[mask10].astype(str),
            hovertemplate="Gauge ID: %{text}<extra></extra>",
        ))

    # 4) Highlight ≤5 km
    if highlight5:
        fig.add_trace(go.Scatter(
            x=GX_ALL[mask5], y=GY_ALL[mask5],
            mode="markers",
            name="Gauges ≤5 km",
            marker=dict(size=max(gauge_size + 4, 10), opacity=1.0, color="navy"),
            text=GID_ALL[mask5].astype(str),
            hovertemplate="Gauge ID: %{text}<extra></extra>",
        ))

    # 5) Selected 7
    fig.add_trace(go.Scatter(
        x=GX_ALL[mask_sel7], y=GY_ALL[mask_sel7],
        mode="markers",
        name="Selected 7",
        marker=dict(size=14, color="red", line=dict(width=1.5, color="black")),
        text=GID_ALL[mask_sel7].astype(str),
        hovertemplate="Selected Gauge ID: %{text}<extra></extra>",
    ))
    

    # Lines to selected 7
    if show_lines and mask_sel7.any():
        idxs = np.where(mask_sel7)[0]
        for j in idxs:
            fig.add_trace(go.Scatter(
                x=[tx, GX_ALL[j]], y=[ty, GY_ALL[j]],
                mode="lines",
                showlegend=False,
                hoverinfo="skip",
                line=dict(width=1, color="black"),
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
        title=f"Grid {int(grid_id)} | centroids={len(GRID_DF)} | gauges={len(STN_DF)}",
        xaxis_title="UTM Easting (m)",
        yaxis_title="UTM Northing (m)",
        height=820,
        margin=dict(l=40, r=20, t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        uirevision="keep",  # keep zoom/pan
        dragmode="pan",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(range=FULL_XRANGE)
    fig.update_yaxes(range=FULL_YRANGE)

    # summary info for UI
    info = {
        "selected_ids_count": len(selected_ids),
        "missing_ids": missing_ids,
        "count10": int(mask10.sum()),
        "count5": int(mask5.sum()),
    }
    # Add invisible centroid hitbox LAST (always clickable)
    fig.add_trace(go.Scatter(
        x=GRID_X,
        y=GRID_Y,
        mode="markers",
        marker=dict(size=max(centroid_size + 12, 18),
                    opacity=0.0,
                    color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=False,
        name="CENTROID_HITBOX",
    ))
    return fig, info


app = Dash(__name__)
app.title = "Grid centroids + gauges viewer (Dash)"

default_grid_id = int(GRID_IDS[0])

app.layout = html.Div(
    style={"display": "flex", "gap": "14px", "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"},
    children=[
        # Sidebar
        html.Div(
            style={"width": "320px", "padding": "14px", "borderRight": "1px solid #ddd"},
            children=[
                html.H3("Controls", style={"marginTop": "0px"}),

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
                    options=[{"label": "Show lines to selected 7", "value": "on"}],
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

                dcc.Checklist(
                    id="highlight10",
                    options=[{"label": "Highlight gauges within 10 km", "value": "on"}],
                    value=["on"],
                ),
                dcc.Checklist(
                    id="highlight5",
                    options=[{"label": "Highlight gauges within 5 km", "value": "on"}],
                    value=["on"],
                ),

                html.Hr(),

                html.Label("Centroid size"),
                dcc.Slider(id="centroid-size", min=3, max=12, step=1, value=6),

                html.Label("Centroid opacity"),
                dcc.Slider(id="centroid-opacity", min=0.1, max=1.0, step=0.05, value=0.6),

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
                    config={
                        "displayModeBar": True,
                        "scrollZoom": True,     # mouse wheel zoom
                    },
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

    # Dropdown changed
    if trig == "grid-dropdown.value":
        if dropdown_value is None:
            return current, current
        v = int(dropdown_value)
        return v, v

    # Map clicked
    if trig == "map.clickData" and clickData:
        pt = clickData["points"][0]

        # Always interpret hitbox click using pointIndex
        idx = pt.get("pointIndex", None)
        if idx is not None and 0 <= idx < len(GRID_ID_ARR):
            new_gid = int(GRID_ID_ARR[int(idx)])
            return new_gid, new_gid

        return current, current

    # Default
    return current, current

@app.callback(
    Output("map", "figure"),
    Output("status-box", "children"),
    Input("selected-grid-store", "data"),
    Input("show-lines", "value"),
    Input("circles", "value"),
    Input("highlight10", "value"),
    Input("highlight5", "value"),
    Input("centroid-size", "value"),
    Input("centroid-opacity", "value"),
    Input("gauge-size", "value"),
    Input("gauge-opacity", "value"),
    Input("reset-view", "n_clicks"),
)
def redraw(
    grid_id,
    show_lines_value,
    circles_value,
    highlight10_value,
    highlight5_value,
    centroid_size,
    centroid_opacity,
    gauge_size,
    gauge_opacity,
    reset_clicks,
):
    show_lines = "on" in (show_lines_value or [])
    highlight10 = "on" in (highlight10_value or [])
    highlight5 = "on" in (highlight5_value or [])
    circle_km_list = circles_value or []

    fig, info = build_figure(
        int(grid_id),
        show_lines=show_lines,
        circle_km_list=circle_km_list,
        highlight10=highlight10,
        highlight5=highlight5,
        centroid_size=int(centroid_size),
        centroid_opacity=float(centroid_opacity),
        gauge_size=int(gauge_size),
        gauge_opacity=float(gauge_opacity),
    )

    # Reset view just forces full range again (it is already set, but keep this hook)
    if reset_clicks:
        fig.update_xaxes(range=FULL_XRANGE)
        fig.update_yaxes(range=FULL_YRANGE)

    status = (
        f"Selected grid: {int(grid_id)}\n"
        f"Selected-7 count: {info['selected_ids_count']}\n"
        f"Gauges ≤10 km: {info['count10']}\n"
        f"Gauges ≤5 km: {info['count5']}\n"
    )
    if info["missing_ids"]:
        status += f"Missing selected-7 in Stations_df: {info['missing_ids']}\n"
    else:
        status += "Selected-7 IDs all found in Stations_df.\n"

    return fig, status


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8890, debug=True)