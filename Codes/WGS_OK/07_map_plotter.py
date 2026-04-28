#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK")
EVENT_META_DIR = BASE_DIR / "01_Event_TimeSeries"
STATIONS_CSV = BASE_DIR / "dependent_files" / "Stations_df.csv"
OUT_DIR = BASE_DIR / "07_IDW_OK_Avg_method_results"

CATCHMENT_SHP_PATHS = [
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6893390/6893390.shp",
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/06893080/6893080.shp",
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6892513/6892513.shp",
]


def norm_station_id(x) -> str:
    s = str(x).strip().strip("'").strip('"')
    if s == "" or s.lower() == "nan":
        return ""
    try:
        return str(int(float(s)))
    except Exception:
        return s


def load_event_station_ids(event: int, event_meta_dir: Path) -> list[str]:
    fp = event_meta_dir / f"Event_{event}_all_used_station_timeseries.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing event station timeseries file: {fp}")

    df = pd.read_csv(fp)
    if "time_local" not in df.columns:
        raise ValueError(f"{fp} must contain time_local")

    station_ids = []
    for c in df.columns:
        if c == "time_local":
            continue
        sid = norm_station_id(c)
        if sid != "":
            station_ids.append(sid)

    return sorted(set(station_ids))


def load_stations(station_file: Path, station_ids: list[str]) -> gpd.GeoDataFrame:
    st = pd.read_csv(station_file)
    req = ["ID", "Latitude", "Longitude"]
    missing = [c for c in req if c not in st.columns]
    if missing:
        raise ValueError(f"{station_file} missing required columns: {missing}")

    st = st[req].copy()
    st["ID"] = st["ID"].apply(norm_station_id)
    st["Latitude"] = pd.to_numeric(st["Latitude"], errors="coerce")
    st["Longitude"] = pd.to_numeric(st["Longitude"], errors="coerce")
    st = st.dropna(subset=["ID", "Latitude", "Longitude"])
    st = st[st["ID"].isin(set(station_ids))].copy()
    st = st.drop_duplicates(subset=["ID"]).reset_index(drop=True)

    gdf = gpd.GeoDataFrame(
        st,
        geometry=gpd.points_from_xy(st["Longitude"], st["Latitude"]),
        crs="EPSG:4326",
    )
    return gdf


def load_catchments(shp_paths: list[str]) -> gpd.GeoDataFrame:
    gdfs = []
    for raw_path in shp_paths:
        p = Path(raw_path)
        if not p.exists():
            raise FileNotFoundError(f"Missing catchment shapefile: {p}")
        g = gpd.read_file(p)
        if g.empty or g.crs is None:
            continue
        gdfs.append(g)

    if not gdfs:
        raise ValueError("No valid catchment geometries were loaded")

    catch = pd.concat(gdfs, ignore_index=True)
    catch = gpd.GeoDataFrame(catch, geometry="geometry", crs=gdfs[0].crs)
    catch = catch.to_crs(epsg=4326)
    return catch


def classify_stations(
    stations_gdf: gpd.GeoDataFrame,
    catchments_gdf: gpd.GeoDataFrame,
    buffer_km: float,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoSeries]:
    # project to meters for buffering
    catch_m = catchments_gdf.to_crs(epsg=26915)
    st_m = stations_gdf.to_crs(epsg=26915)

    catch_union = catch_m.geometry.unary_union
    catch_buffer = catch_m.buffer(buffer_km * 1000.0).unary_union

    target_mask = st_m.geometry.within(catch_union) | st_m.geometry.touches(catch_union)
    donor_mask = st_m.geometry.within(catch_buffer) | st_m.geometry.touches(catch_buffer)

    targets = stations_gdf.loc[target_mask].copy().reset_index(drop=True)
    donors = stations_gdf.loc[donor_mask].copy().reset_index(drop=True)

    buffer_gs = gpd.GeoSeries([catch_buffer], crs="EPSG:26915").to_crs(epsg=4326)
    return targets, donors, buffer_gs


def plot_map(
    catchments_gdf: gpd.GeoDataFrame,
    buffer_gs: gpd.GeoSeries,
    donors_gdf: gpd.GeoDataFrame,
    targets_gdf: gpd.GeoDataFrame,
    out_png: Path,
    event: int,
    buffer_km: float,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))

    # catchment boundaries
    catchments_gdf.boundary.plot(ax=ax, linewidth=2, color="black", label="Catchments")

    # donor buffer boundary
    buffer_gs.boundary.plot(ax=ax, linewidth=1.5, color="gray", linestyle="--", label=f"{buffer_km:.1f} km donor buffer")

    # donors
    if not donors_gdf.empty:
        donors_gdf.plot(
            ax=ax,
            markersize=45,
            color="tab:orange",
            edgecolor="black",
            linewidth=0.4,
            label="Donor gauges",
            zorder=3,
        )

    # targets
    if not targets_gdf.empty:
        targets_gdf.plot(
            ax=ax,
            markersize=55,
            color="tab:blue",
            edgecolor="black",
            linewidth=0.5,
            label="Target gauges",
            zorder=4,
        )

    # label station IDs
    for _, row in donors_gdf.iterrows():
        ax.text(
            row.geometry.x,
            row.geometry.y,
            str(row["ID"]),
            fontsize=7,
            ha="left",
            va="bottom",
            color="black",
        )

    ax.set_title(f"Event {event}: catchments and donor gauges")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot catchments, donor gauges, and target gauges for one event.")
    ap.add_argument("--event", type=int, required=True)
    ap.add_argument("--event-meta-dir", type=Path, default=EVENT_META_DIR)
    ap.add_argument("--station-file", type=Path, default=STATIONS_CSV)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--buffer-km", type=float, default=5.0)
    ap.add_argument("--catchment-shp", nargs="*", default=CATCHMENT_SHP_PATHS)
    args = ap.parse_args()

    station_ids = load_event_station_ids(args.event, args.event_meta_dir)
    stations_gdf = load_stations(args.station_file, station_ids)
    catchments_gdf = load_catchments(list(args.catchment_shp))

    targets_gdf, donors_gdf, buffer_gs = classify_stations(
        stations_gdf=stations_gdf,
        catchments_gdf=catchments_gdf,
        buffer_km=float(args.buffer_km),
    )

    event_out_dir = args.out_dir / f"Event_{args.event}"
    event_out_dir.mkdir(parents=True, exist_ok=True)

    out_png = event_out_dir / f"Event_{args.event}_catchments_donor_gauges_{args.buffer_km:.1f}km.png"
    out_donor_csv = event_out_dir / f"Event_{args.event}_donor_gauges_{args.buffer_km:.1f}km.csv"
    out_target_csv = event_out_dir / f"Event_{args.event}_target_gauges.csv"

    donors_gdf.drop(columns="geometry").to_csv(out_donor_csv, index=False)
    targets_gdf.drop(columns="geometry").to_csv(out_target_csv, index=False)

    plot_map(
        catchments_gdf=catchments_gdf,
        buffer_gs=buffer_gs,
        donors_gdf=donors_gdf,
        targets_gdf=targets_gdf,
        out_png=out_png,
        event=int(args.event),
        buffer_km=float(args.buffer_km),
    )

    print(f"Total event-used stations: {len(stations_gdf)}")
    print(f"Target gauges (inside/touching catchments): {len(targets_gdf)}")
    print(f"Donor gauges (within {args.buffer_km:.1f} km buffer): {len(donors_gdf)}")
    print(f"Saved map: {out_png}")
    print(f"Saved donor list: {out_donor_csv}")
    print(f"Saved target list: {out_target_csv}")


if __name__ == "__main__":
    main()