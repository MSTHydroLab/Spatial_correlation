#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geo_utils import build_regular_wgs84_grid, haversine_km, initial_bearing_deg, ang_sep_deg


@dataclass
class SelectionResult:
    gauge_ids: list[str]
    gauge_dists_m: list[float]
    gauge_bear_deg: list[float]
    radius_used_km: int


REQUIRED_STATION_COLS = ["ID", "Latitude", "Longitude"]


def select_spread_gauges_for_target(
    stn_lat: np.ndarray,
    stn_lon: np.ndarray,
    stn_id: np.ndarray,
    target_lat: float,
    target_lon: float,
    start_km: int = 5,
    end_km: int = 10,
    want_n: int = 10,
    min_ang_sep_deg: float = 30.0,
) -> SelectionResult:
    d_km = haversine_km(target_lat, target_lon, stn_lat, stn_lon)
    bearing = initial_bearing_deg(target_lat, target_lon, stn_lat, stn_lon)
    sort_idx = np.argsort(d_km)

    d_sorted = d_km[sort_idx]
    b_sorted = bearing[sort_idx]
    id_sorted = stn_id[sort_idx]

    for r_km in range(start_km, end_km + 1):
        cand_idx = np.where(d_sorted <= r_km)[0]
        if len(cand_idx) == 0:
            continue

        picked_ids: list[str] = []
        picked_d: list[float] = []
        picked_b: list[float] = []
        picked_set: set[str] = set()

        for ii in cand_idx:
            if len(picked_ids) >= want_n:
                break
            sid = str(id_sorted[ii])
            if sid in picked_set:
                continue

            bb = float(b_sorted[ii])
            if all(ang_sep_deg(bb, prev) >= min_ang_sep_deg for prev in picked_b):
                picked_ids.append(sid)
                picked_d.append(float(d_sorted[ii]) * 1000.0)
                picked_b.append(bb)
                picked_set.add(sid)

        if len(picked_ids) < want_n:
            for ii in cand_idx:
                if len(picked_ids) >= want_n:
                    break
                sid = str(id_sorted[ii])
                if sid in picked_set:
                    continue
                picked_ids.append(sid)
                picked_d.append(float(d_sorted[ii]) * 1000.0)
                picked_b.append(float(b_sorted[ii]))
                picked_set.add(sid)

        if len(picked_ids) >= want_n or r_km == end_km:
            picked_ids = picked_ids[:want_n]
            picked_d = picked_d[:want_n]
            picked_b = picked_b[:want_n]

            while len(picked_ids) < want_n:
                picked_ids.append("")
                picked_d.append(np.nan)
                picked_b.append(np.nan)

            return SelectionResult(picked_ids, picked_d, picked_b, r_km)

    return SelectionResult([""] * want_n, [np.nan] * want_n, [np.nan] * want_n, end_km)


def build_neighbor_table(
    grid_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    start_km: int,
    end_km: int,
    want_n: int,
    min_ang_sep_deg: float,
) -> pd.DataFrame:
    for col in REQUIRED_STATION_COLS:
        if col not in stations_df.columns:
            raise ValueError(f"Stations CSV missing required column: {col}")

    stn_lat = stations_df["Latitude"].to_numpy(dtype=float)
    stn_lon = stations_df["Longitude"].to_numpy(dtype=float)
    stn_id = stations_df["ID"].astype(str).to_numpy()

    out_rows = []
    for _, row in grid_df.iterrows():
        sel = select_spread_gauges_for_target(
            stn_lat=stn_lat,
            stn_lon=stn_lon,
            stn_id=stn_id,
            target_lat=float(row["Latitude"]),
            target_lon=float(row["Longitude"]),
            start_km=start_km,
            end_km=end_km,
            want_n=want_n,
            min_ang_sep_deg=min_ang_sep_deg,
        )

        rec = {
            "id": str(row["id"]),
            "Latitude": float(row["Latitude"]),
            "Longitude": float(row["Longitude"]),
            "radius_used_km": sel.radius_used_km,
        }
        for k in range(want_n):
            rec[f"g{k+1}"] = sel.gauge_ids[k]
            rec[f"d{k+1}_m"] = sel.gauge_dists_m[k]
            rec[f"b{k+1}_deg"] = sel.gauge_bear_deg[k]
        out_rows.append(rec)

    return pd.DataFrame(out_rows)


def main():
    base_dir = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK")
    dep_dir = base_dir / "dependent_files"

    ap = argparse.ArgumentParser(description="Build WGS84 grid and spread-based neighbor list for kriging.")
    ap.add_argument("--stations-csv", default=str(dep_dir / "Stations_df.csv"))
    ap.add_argument("--grid-csv", default="", help="Optional prebuilt WGS84 grid CSV with columns id, Latitude, Longitude.")
    ap.add_argument("--grid-out-csv", default=str(dep_dir / "grid_centers_wgs84.csv"))
    ap.add_argument("--neighbors-out-csv", default=str(dep_dir / "grid_nearest10_spread_wgs84.csv"))
    ap.add_argument("--start-lat",type=float, default=38.8447, required=False)
    ap.add_argument("--end-lat", type=float,default=39.0218, required=False)
    ap.add_argument("--start-lon",type=float, default=-94.8653, required=False)
    ap.add_argument("--end-lon", type=float,default=-94.5838, required=False)
    ap.add_argument("--delta",type=float, default=0.004167, required=False)
    ap.add_argument("--lon-major", action="store_true", help="Assign ids with longitude as outer loop. Default is latitude-major.")
    ap.add_argument("--start-km", type=int, default=5)
    ap.add_argument("--end-km", type=int, default=10)
    ap.add_argument("--want-n", type=int, default=10)
    ap.add_argument("--min-ang-sep-deg", type=float, default=45.0)
    args = ap.parse_args()

    stations = pd.read_csv(args.stations_csv)

    if args.grid_csv:
        grid = pd.read_csv(args.grid_csv)
        req = ["id", "Latitude", "Longitude"]
        missing = [c for c in req if c not in grid.columns]
        if missing:
            raise ValueError(f"Grid CSV missing required columns: {missing}")
        grid = grid[req].copy()
    else:
        needed = [args.start_lat, args.end_lat, args.start_lon, args.end_lon, args.delta]
        if any(v is None for v in needed):
            raise ValueError("Provide either --grid-csv or all of --start-lat --end-lat --start-lon --end-lon --delta")
        grid = build_regular_wgs84_grid(
            start_lat=args.start_lat,
            end_lat=args.end_lat,
            start_lon=args.start_lon,
            end_lon=args.end_lon,
            delta=args.delta,
            lat_major=not args.lon_major,
        )

    Path(args.grid_out_csv).parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.grid_out_csv, index=False)

    neighbors = build_neighbor_table(
        grid_df=grid,
        stations_df=stations,
        start_km=args.start_km,
        end_km=args.end_km,
        want_n=args.want_n,
        min_ang_sep_deg=args.min_ang_sep_deg,
    )
    Path(args.neighbors_out_csv).parent.mkdir(parents=True, exist_ok=True)
    neighbors.to_csv(args.neighbors_out_csv, index=False)

    print(f"Saved grid: {args.grid_out_csv}")
    print(f"Saved neighbors: {args.neighbors_out_csv}")
    print(f"Grid cells: {len(grid)}")


if __name__ == "__main__":
    main()
