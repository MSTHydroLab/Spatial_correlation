#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
import pandas as pd
import numpy as np
import itertools
import numpy as np

from geo_utils import build_regular_wgs84_grid, haversine_km, initial_bearing_deg, ang_sep_deg

REQUIRED_STATION_COLS = ["ID", "Latitude", "Longitude"]


def norm_station_id(x):
    s = str(x).strip().strip("'").strip('"')
    if s == "" or s.lower() == "nan":
        return ""
    try:
        return str(int(float(s)))
    except Exception:
        return s

def circular_gaps_deg(bearings_deg):
    b = np.sort(np.mod(np.asarray(bearings_deg, dtype=float), 360.0))
    if len(b) == 0:
        return np.array([], dtype=float)
    wrap = np.r_[b, b[0] + 360.0]
    return np.diff(wrap)


def min_pairwise_sep_deg(bearings_deg):
    vals = list(bearings_deg)
    if len(vals) < 2:
        return 360.0
    seps = []
    for i, j in itertools.combinations(range(len(vals)), 2):
        a = abs(vals[i] - vals[j]) % 360.0
        seps.append(min(a, 360.0 - a))
    return float(min(seps))


def max_circular_gap_deg(bearings_deg):
    gaps = circular_gaps_deg(bearings_deg)
    if len(gaps) == 0:
        return 360.0
    return float(np.max(gaps))


def group_passes_geometry(
    bearings_deg,
    n_group,
    min_sep_3,
    min_sep_4,
    max_gap_3,
    max_gap_4,
):
    bearings_deg = [float(x) for x in bearings_deg]

    min_sep = min_pairwise_sep_deg(bearings_deg)
    max_pair_sep = max_pairwise_sep_deg(bearings_deg)

    if n_group == 3:
        return (min_sep >= float(min_sep_3)) and (max_pair_sep <= float(max_gap_3))

    if n_group == 4:
        return (min_sep >= float(min_sep_4)) and (max_pair_sep <= float(max_gap_4))

    return False

def sector_index(bearing_deg: float, n_sectors: int) -> int:
    width = 360.0 / float(n_sectors)
    idx = int(np.floor((bearing_deg % 360.0) / width))
    return min(idx, n_sectors - 1)


def build_full_groups_from_sectors(
    candidates,
    n_sectors: int,
    max_per_sector: int,
    max_groups: int,
    inner_radius_km: float | None = None,
    min_within_inner_radius: int = 0,
    ):
    sector_bins = {k: [] for k in range(n_sectors)}
    for cand in candidates:
        sector_bins[sector_index(cand["bearing_deg"], n_sectors)].append(cand)

    if any(len(v) == 0 for v in sector_bins.values()):
        return []

    trimmed = []
    for k in range(n_sectors):
        vals = sorted(sector_bins[k], key=lambda x: x["dist_m"])
        trimmed.append(vals[:max_per_sector])

    groups = []
    seen = set()
    for combo in product(*trimmed):
        ids = [c["id"] for c in combo]
        if len(set(ids)) != len(ids):
            continue

        dists = [float(c["dist_m"]) for c in combo]
        bears = [float(c["bearing_deg"]) for c in combo]
        seps = [ang_sep_deg(bears[i], bears[j]) for i in range(len(bears)) for j in range(i + 1, len(bears))]
        inner_count = 0
        if inner_radius_km is not None:
            inner_count = int(np.sum(np.asarray(dists, dtype=float) <= float(inner_radius_km) * 1000.0))
            if inner_count < int(min_within_inner_radius):
                continue

        rec = {
            "ids": ids,
            "dists_m": dists,
            "bears_deg": bears,
            "sum_dist_m": float(np.sum(dists)),
            "min_ang_sep_deg": float(min(seps)) if seps else 360.0,
            "inner_radius_km": float(inner_radius_km) if inner_radius_km is not None else np.nan,
            "n_within_inner_radius": int(inner_count),
        }
        key = tuple(ids)
        if key in seen:
            continue
        seen.add(key)
        groups.append(rec)

    groups.sort(key=lambda g: (g["sum_dist_m"], -g["min_ang_sep_deg"]))
    return groups[:max_groups]

def max_pairwise_sep_deg(bearings_deg):
    vals = list(bearings_deg)
    if len(vals) < 2:
        return 0.0
    seps = []
    for i, j in itertools.combinations(range(len(vals)), 2):
        a = abs(vals[i] - vals[j]) % 360.0
        seps.append(min(a, 360.0 - a))
    return float(max(seps))

def select_candidates_and_groups_for_target(
    stn_lat: np.ndarray,
    stn_lon: np.ndarray,
    stn_id: np.ndarray,
    target_lat: float,
    target_lon: float,
    radius_km: float,
    preview_n: int,
    max_per_sector: int,
    max_groups_per_type: int,
    inner_radius_km: float,
    min_within_inner_radius: int,
    ):
    d_km = haversine_km(target_lat, target_lon, stn_lat, stn_lon)
    bearing = initial_bearing_deg(target_lat, target_lon, stn_lat, stn_lon)
    order = np.argsort(d_km)

    candidates = []
    for ii in order:
        if float(d_km[ii]) <= float(radius_km):
            candidates.append({
                "id": norm_station_id(stn_id[ii]),
                "dist_m": float(d_km[ii]) * 1000.0,
                "bearing_deg": float(bearing[ii]),
            })

    preview = candidates[:preview_n]

    candidate_ids = [c["id"] for c in candidates]
    candidate_dists_m = [c["dist_m"] for c in candidates]
    candidate_bears_deg = [c["bearing_deg"] for c in candidates]

    groups3, groups4 = build_angle_based_groups(
        candidate_ids=candidate_ids,
        candidate_dists_m=candidate_dists_m,
        candidate_bears_deg=candidate_bears_deg,
        min_sep_3=60,
        min_sep_4=45,
        max_gap_3=180,
        max_gap_4=160,
        max_groups_per_type=max_groups_per_type,
        inner_radius_km=inner_radius_km,
        min_within_inner_radius=min_within_inner_radius,
    )

    quadrant_groups = groups4
    sector3_groups = groups3

    return candidates, preview, quadrant_groups, sector3_groups

def build_neighbor_table(
    grid_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    radius_km: float,
    preview_n: int,
    max_per_sector: int,
    max_groups_per_type: int,
    inner_radius_km: float,
    min_within_inner_radius: int,
    ) -> pd.DataFrame:
    for col in REQUIRED_STATION_COLS:
        if col not in stations_df.columns:
            raise ValueError(f"Stations CSV missing required column: {col}")

    stn_lat = stations_df["Latitude"].to_numpy(dtype=float)
    stn_lon = stations_df["Longitude"].to_numpy(dtype=float)
    stn_id = stations_df["ID"].astype(str).to_numpy()
    rows = []
    for _, row in grid_df.iterrows():
        candidates, preview, quadrant_groups, sector3_groups = select_candidates_and_groups_for_target(
            stn_lat=stn_lat,
            stn_lon=stn_lon,
            stn_id=stn_id,
            target_lat=float(row["Latitude"]),
            target_lon=float(row["Longitude"]),
            radius_km=radius_km,
            preview_n=preview_n,
            max_per_sector=max_per_sector,
            max_groups_per_type=max_groups_per_type,
            inner_radius_km=inner_radius_km,
            min_within_inner_radius=min_within_inner_radius,
        )
        rec = {
            "id": str(row["id"]),
            "Latitude": float(row["Latitude"]),
            "Longitude": float(row["Longitude"]),
            "search_radius_km": float(radius_km),
            "candidate_count": int(len(candidates)),
            "candidate_ids": json.dumps([c["id"] for c in candidates]),
            "candidate_dists_m": json.dumps([c["dist_m"] for c in candidates]),
            "candidate_bears_deg": json.dumps([c["bearing_deg"] for c in candidates]),
            "quadrant_group_count": int(len(quadrant_groups)),
            "quadrant_groups": json.dumps(quadrant_groups),
            "sector3_group_count": int(len(sector3_groups)),
            "sector3_groups": json.dumps(sector3_groups),
            "group_inner_radius_km": float(inner_radius_km),
            "group_min_within_inner_radius": int(min_within_inner_radius),
        }
        print(rec["id"])
        rows.append(rec)

    return pd.DataFrame(rows)

def build_angle_based_groups(
    candidate_ids,
    candidate_dists_m,
    candidate_bears_deg,
    min_sep_3=60.0,
    min_sep_4=50.0,
    max_gap_3=210.0,
    max_gap_4=1700.0,
    max_groups_per_type=50,
    inner_radius_km=None,
    min_within_inner_radius=0,
    ):
    groups3 = []
    groups4 = []

    items = list(zip(candidate_ids, candidate_dists_m, candidate_bears_deg))

    for comb in itertools.combinations(items, 3):
        ids = [x[0] for x in comb]
        dists = [float(x[1]) for x in comb]
        bears = [float(x[2]) for x in comb]

        if inner_radius_km is not None:
            inner_count = int(np.sum(np.asarray(dists, dtype=float) <= float(inner_radius_km) * 1000.0))
            if inner_count < int(min_within_inner_radius):
                continue
        else:
            inner_count = 0

        if group_passes_geometry(
            bears, 3,
            min_sep_3=min_sep_3,
            min_sep_4=min_sep_4,
            max_gap_3=max_gap_3,
            max_gap_4=max_gap_4,
        ):
            groups3.append({
                "ids": ids,
                "dists_m": dists,
                "bears_deg": bears,
                "sum_dist_m": float(np.sum(dists)),
                "min_ang_sep_deg": float(min_pairwise_sep_deg(bears)),
                "max_gap_deg": float(max_circular_gap_deg(bears)),
                "inner_radius_km": float(inner_radius_km) if inner_radius_km is not None else np.nan,
                "n_within_inner_radius": int(inner_count),
            })

    for comb in itertools.combinations(items, 4):
        ids = [x[0] for x in comb]
        dists = [float(x[1]) for x in comb]
        bears = [float(x[2]) for x in comb]

        if inner_radius_km is not None:
            inner_count = int(np.sum(np.asarray(dists, dtype=float) <= float(inner_radius_km) * 1000.0))
            if inner_count < int(min_within_inner_radius):
                continue
        else:
            inner_count = 0

        if group_passes_geometry(
            bears, 4,
            min_sep_3=min_sep_3,
            min_sep_4=min_sep_4,
            max_gap_3=max_gap_3,
            max_gap_4=max_gap_4,
        ):
            groups4.append({
                "ids": ids,
                "dists_m": dists,
                "bears_deg": bears,
                "sum_dist_m": float(np.sum(dists)),
                "min_ang_sep_deg": float(min_pairwise_sep_deg(bears)),
                "max_gap_deg": float(max_circular_gap_deg(bears)),
                "inner_radius_km": float(inner_radius_km) if inner_radius_km is not None else np.nan,
                "n_within_inner_radius": int(inner_count),
            })

    groups3 = sorted(
        groups3,
        key=lambda g: (g["sum_dist_m"], -g["min_ang_sep_deg"], g["max_gap_deg"])
    )[:max_groups_per_type]

    groups4 = sorted(
        groups4,
        key=lambda g: (g["sum_dist_m"], -g["min_ang_sep_deg"], g["max_gap_deg"])
    )[:max_groups_per_type]

    return groups3, groups4

def main():
    base_dir = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK")
    dep_dir = base_dir / "dependent_files"

    ap = argparse.ArgumentParser(description="Build WGS84 grid and grouped station candidates for kriging.")
    ap.add_argument("--stations-csv", default=str(dep_dir / "Stations_df.csv"))
    ap.add_argument("--grid-csv", default="", help="Optional prebuilt WGS84 grid CSV with columns id, Latitude, Longitude.")
    ap.add_argument("--grid-out-csv", default=str(dep_dir / "grid_centers_wgs84.csv"))
    ap.add_argument("--neighbors-out-csv", default=str(dep_dir / "grid_grouped_candidates_wgs84.csv"))
    ap.add_argument("--start-lat", type=float, default=38.7314084)  #grid center start
    ap.add_argument("--end-lat", type=float, default=39.0397664) #grid center end
    ap.add_argument("--start-lon", type=float, default=-94.9019185) #grid center start
    ap.add_argument("--end-lon", type=float, default=-94.5935605) #grid center end
    ap.add_argument("--delta", type=float, default=0.004167)
    ap.add_argument("--lon-major", action="store_true", help="Assign ids with longitude as outer loop. Default is latitude-major.")
    ap.add_argument("--radius-km", type=float, default=10.0, help="All stations within this radius are considered.")
    ap.add_argument("--preview-n", type=int, default=10, help="Also save the first N nearest stations for inspection.")
    ap.add_argument("--max-per-sector", type=int, default=1, help="Use up to this many nearest candidates from each sector when forming groups.")
    ap.add_argument("--max-groups-per-type", type=int, default=50, help="Maximum quadrant groups and sector-3 groups saved per grid.")
    ap.add_argument("--inner-radius-km", type=float, default=6.0, help="At least this many stations in a group must fall inside this radius.")
    ap.add_argument("--min-within-inner-radius", type=int, default=2, help="Minimum number of stations in each group that must lie within --inner-radius-km.")
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
        radius_km=args.radius_km,
        preview_n=args.preview_n,
        max_per_sector=args.max_per_sector,
        max_groups_per_type=args.max_groups_per_type,
        inner_radius_km=args.inner_radius_km,
        min_within_inner_radius=args.min_within_inner_radius,
    )
    Path(args.neighbors_out_csv).parent.mkdir(parents=True, exist_ok=True)
    neighbors.to_csv(args.neighbors_out_csv, index=False)

    print(f"Saved grid: {args.grid_out_csv}")
    print(f"Saved grouped candidates: {args.neighbors_out_csv}")
    print(f"Grid cells: {len(grid)}")


if __name__ == "__main__":
    main()
