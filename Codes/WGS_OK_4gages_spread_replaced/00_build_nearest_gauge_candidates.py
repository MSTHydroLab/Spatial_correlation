#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from geo_utils import haversine_km, initial_bearing_deg

BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK_4gages_spread_replaced")
DEP_DIR = BASE_DIR / "dependent_files"
GRID_CSV = DEP_DIR / "grid_centers_wgs84.csv"
STATIONS_CSV = DEP_DIR / "Stations_df.csv"
OUT_CSV = DEP_DIR / "grid_nearest_gauges_wgs84.csv"


def norm_station_id(x) -> str:
    s = str(x).strip().strip("'").strip('"')
    if s == "" or s.lower() == "nan":
        return ""
    try:
        return str(int(float(s)))
    except Exception:
        return s


def build_table(grid_df: pd.DataFrame,
                stn_df: pd.DataFrame,
                search_radius_km: float,
                keep_n: int,
                allow_fill_outside_radius: bool) -> pd.DataFrame:
    stn_ids = stn_df["ID"].apply(norm_station_id).to_numpy()
    stn_lat = stn_df["Latitude"].to_numpy(float)
    stn_lon = stn_df["Longitude"].to_numpy(float)

    rows: list[dict] = []
    for _, row in grid_df.iterrows():
        cid = str(row["id"])
        clat = float(row["Latitude"])
        clon = float(row["Longitude"])

        d_km = haversine_km(clat, clon, stn_lat, stn_lon)
        bears = initial_bearing_deg(clat, clon, stn_lat, stn_lon)
        order = np.argsort(d_km)

        inside = [int(i) for i in order if float(d_km[i]) <= float(search_radius_km)]
        chosen = inside[:keep_n]
        if allow_fill_outside_radius and len(chosen) < keep_n:
            for i in order:
                ii = int(i)
                if ii not in chosen:
                    chosen.append(ii)
                if len(chosen) >= keep_n:
                    break

        ids = [norm_station_id(stn_ids[i]) for i in chosen]
        dists_m = [float(d_km[i]) * 1000.0 for i in chosen]
        bears_deg = [float(bears[i]) for i in chosen]

        rec = {
            "id": cid,
            "Latitude": clat,
            "Longitude": clon,
            "search_radius_km": float(search_radius_km),
            "keep_n": int(keep_n),
            "n_inside_radius": int(len(inside)),
            "n_kept": int(len(ids)),
            "allow_fill_outside_radius": bool(allow_fill_outside_radius),
            "nearest_station_ids": json.dumps(ids),
            "nearest_station_dists_m": json.dumps(dists_m),
            "nearest_station_bears_deg": json.dumps(bears_deg),
        }
        for k in range(keep_n):
            rec[f"g{k+1}"] = ids[k] if k < len(ids) else ""
            rec[f"dist{k+1}_m"] = dists_m[k] if k < len(dists_m) else np.nan
            rec[f"bear{k+1}_deg"] = bears_deg[k] if k < len(bears_deg) else np.nan
        rows.append(rec)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build nearest-gauge candidate table for each WGS84 centroid.")
    ap.add_argument("--base-dir", default=str(BASE_DIR))
    ap.add_argument("--grid-csv", default="")
    ap.add_argument("--stations-csv", default="")
    ap.add_argument("--out-csv", default="")
    ap.add_argument("--search-radius-km", type=float, default=7.0)
    ap.add_argument("--keep-n", type=int, default=10)
    ap.add_argument("--no-fill-outside-radius", action="store_true")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    dep_dir = base_dir / "dependent_files"
    grid_csv = Path(args.grid_csv) if args.grid_csv else dep_dir / "grid_centers_wgs84.csv"
    stations_csv = Path(args.stations_csv) if args.stations_csv else dep_dir / "Stations_df.csv"
    out_csv = Path(args.out_csv) if args.out_csv else dep_dir / "grid_nearest_gauges_wgs84.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    grid_df = pd.read_csv(grid_csv)
    stn_df = pd.read_csv(stations_csv)

    req_grid = ["id", "Latitude", "Longitude"]
    req_stn = ["ID", "Latitude", "Longitude"]
    missing_grid = [c for c in req_grid if c not in grid_df.columns]
    missing_stn = [c for c in req_stn if c not in stn_df.columns]
    if missing_grid:
        raise ValueError(f"Grid CSV missing columns: {missing_grid}")
    if missing_stn:
        raise ValueError(f"Stations CSV missing columns: {missing_stn}")

    grid_df = grid_df[req_grid].copy()
    stn_df = stn_df[req_stn].copy()
    grid_df["id"] = grid_df["id"].astype(str)
    stn_df["ID"] = stn_df["ID"].apply(norm_station_id)

    out = build_table(
        grid_df=grid_df,
        stn_df=stn_df,
        search_radius_km=float(args.search_radius_km),
        keep_n=int(args.keep_n),
        allow_fill_outside_radius=not args.no_fill_outside_radius,
    )
    out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(f"Rows: {len(out)}")


if __name__ == "__main__":
    main()
