import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
# --------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("Grid selection viewer (nearest-7 gauges around grid cell)")

@st.cache_data(show_spinner=False)
def load_data():
    grid_df = pd.read_csv(GRID_CSV)
    stn_df = pd.read_csv(STATIONS_CSV)
    n7_df = pd.read_csv(NEAREST7_CSV)

    # Normalize IDs
    grid_df[GRID_ID_COL] = pd.to_numeric(grid_df[GRID_ID_COL], errors="coerce").astype("Int64")
    n7_df[GRID_ID_COL] = pd.to_numeric(n7_df[GRID_ID_COL], errors="coerce").astype("Int64")
    stn_df[STN_ID_COL] = pd.to_numeric(stn_df[STN_ID_COL], errors="coerce").astype("Int64")

    # Coords numeric
    grid_df[GRID_X_COL] = pd.to_numeric(grid_df[GRID_X_COL], errors="coerce")
    grid_df[GRID_Y_COL] = pd.to_numeric(grid_df[GRID_Y_COL], errors="coerce")
    stn_df[STN_X_COL] = pd.to_numeric(stn_df[STN_X_COL], errors="coerce")
    stn_df[STN_Y_COL] = pd.to_numeric(stn_df[STN_Y_COL], errors="coerce")

    # Drop unusable rows
    grid_df = grid_df.dropna(subset=[GRID_ID_COL, GRID_X_COL, GRID_Y_COL])
    stn_df = stn_df.dropna(subset=[STN_ID_COL, STN_X_COL, STN_Y_COL])
    n7_df = n7_df.dropna(subset=[GRID_ID_COL])

    grid_df[GRID_ID_COL] = grid_df[GRID_ID_COL].astype(int)
    n7_df[GRID_ID_COL] = n7_df[GRID_ID_COL].astype(int)
    stn_df[STN_ID_COL] = stn_df[STN_ID_COL].astype(int)

    return grid_df, stn_df, n7_df

def circle_xy(cx, cy, r_m, n=240):
    t = np.linspace(0, 2*np.pi, n)
    return cx + r_m*np.cos(t), cy + r_m*np.sin(t)

grid_df, stations_df, nearest7_df = load_data()

grid_ids = sorted(grid_df[GRID_ID_COL].unique().tolist())

# ---- Sidebar controls ----
with st.sidebar:
    st.header("Controls")
    default_id = grid_ids[0]
    grid_id = st.selectbox("Grid ID", grid_ids, index=grid_ids.index(default_id))
    radius_km = st.slider("Radius (km)", 1.0, 30.0, 10.0, 0.5)

    show_all = st.checkbox("Show all gauges in radius", True)
    show_lines = st.checkbox("Show lines to selected", True)
    show_circles = st.checkbox("Show 5 km and 10 km circles", True)

    st.markdown("---")
    st.caption("Hover over points for ID + UTM coords.")

# ---- Extract grid center ----
row = grid_df.loc[grid_df[GRID_ID_COL] == int(grid_id)].iloc[0]
tx = float(row[GRID_X_COL])
ty = float(row[GRID_Y_COL])

# ---- Selected 7 IDs ----
sel = nearest7_df.loc[nearest7_df[GRID_ID_COL] == int(grid_id)]
selected_ids = []
if not sel.empty:
    sel_row = sel.iloc[0]
    for i in range(1, 8):
        c = f"g{i}"
        if c in sel_row.index and pd.notna(sel_row[c]):
            try:
                selected_ids.append(int(sel_row[c]))
            except Exception:
                pass

# ---- Station arrays ----
gid_all = stations_df[STN_ID_COL].to_numpy(dtype=int)
gx_all = stations_df[STN_X_COL].to_numpy(dtype=float)
gy_all = stations_df[STN_Y_COL].to_numpy(dtype=float)
gid_set = set(gid_all.tolist())

# ---- Distances ----
dx = gx_all - tx
dy = gy_all - ty
dist = np.sqrt(dx*dx + dy*dy)

r_m = radius_km * 1000.0
mask = dist <= r_m

sel_mask = np.isin(gid_all, np.array(selected_ids, dtype=int)) & mask
matched = int(sel_mask.sum())
expected = len(selected_ids)
missing_ids = [i for i in selected_ids if i not in gid_set]

# ---- Build figure ----
fig = go.Figure()

if show_all:
    fig.add_trace(go.Scattergl(
        x=gx_all[mask],
        y=gy_all[mask],
        mode="markers",
        name=f"All gauges ≤{radius_km:.1f} km",
        marker=dict(size=6, opacity=0.35),
        text=gid_all[mask].astype(str),
        hovertemplate="Gauge ID: %{text}<br>E: %{x:.1f} m<br>N: %{y:.1f} m<extra></extra>",
    ))

fig.add_trace(go.Scattergl(
    x=gx_all[sel_mask],
    y=gy_all[sel_mask],
    mode="markers",
    name="Selected 7",
    marker=dict(size=14, color="red", line=dict(width=1.5, color="black")),
    text=gid_all[sel_mask].astype(str),
    hovertemplate="Selected Gauge ID: %{text}<br>E: %{x:.1f} m<br>N: %{y:.1f} m<extra></extra>",
))

fig.add_trace(go.Scattergl(
    x=[tx], y=[ty],
    mode="markers",
    name="Grid center",
    marker=dict(symbol="triangle-up", size=14, color="black"),
    hovertemplate=f"Grid ID: {int(grid_id)}<br>E: {tx:.1f} m<br>N: {ty:.1f} m<extra></extra>",
))

if show_lines and matched > 0:
    idxs = np.where(sel_mask)[0]
    for j in idxs:
        fig.add_trace(go.Scattergl(
            x=[tx, gx_all[j]],
            y=[ty, gy_all[j]],
            mode="lines",
            showlegend=False,
            hoverinfo="skip",
            line=dict(width=1),
        ))

if show_circles:
    x5, y5 = circle_xy(tx, ty, 5000.0)
    x10, y10 = circle_xy(tx, ty, 10000.0)
    fig.add_trace(go.Scattergl(
        x=x5, y=y5, mode="lines", name="5 km",
        line=dict(width=2, dash="dash", color="gray"),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scattergl(
        x=x10, y=y10, mode="lines", name="10 km",
        line=dict(width=2, dash="dot", color="gray"),
        hoverinfo="skip",
    ))

pad = 0.08 * r_m
fig.update_layout(
    title=f"Grid ID {int(grid_id)} (Matched {matched} of {expected} selected within radius)",
    xaxis_title="UTM Easting (m)",
    yaxis_title="UTM Northing (m)",
    height=780,
    margin=dict(l=40, r=20, t=70, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(range=[tx - r_m - pad, tx + r_m + pad])
fig.update_yaxes(range=[ty - r_m - pad, ty + r_m + pad])

# ---- Status + plot ----
c1, c2, c3 = st.columns([1.2, 1.2, 2.6])
c1.metric("Matched", f"{matched} / {expected}")
c2.metric("Radius", f"{radius_km:.1f} km")
if missing_ids:
    c3.warning(f"Missing in Stations_df: {missing_ids}")
else:
    c3.success("All selected IDs exist in Stations_df.")

st.plotly_chart(fig, use_container_width=True)