#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import pandas as pd


# -----------------------------
# Geometry helpers (UTM plane)
# -----------------------------
def bearing_deg_from_north(dx: float, dy: float) -> float:
    """
    Bearing clockwise from North in degrees [0, 360).
    dx = x_gauge - x_target (Easting)
    dy = y_gauge - y_target (Northing)
    """
    ang = math.degrees(math.atan2(dx, dy))  # atan2(E, N)
    return (ang + 360.0) % 360.0


def ang_sep_deg(a: float, b: float) -> float:
    """Smallest angular separation (degrees) between bearings a and b."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def sector4(bearing_deg: float) -> int:
    """4 quadrants (90° bins), returns 0..3"""
    return int(bearing_deg // 90.0)


def sector8(bearing_deg: float) -> int:
    """8 octants (45° bins), returns 0..7"""
    return int(bearing_deg // 45.0)


# -----------------------------
# Selection core
# -----------------------------
@dataclass
class SelectionResult:
    gauge_ids: list[str]
    gauge_dists_m: list[float]
    gauge_bear_deg: list[float]
    radius_used_km: int


def select_spread_gauges_for_target(
    gx: np.ndarray,
    gy: np.ndarray,
    gid: np.ndarray,
    tx: float,
    ty: float,
    start_km: int = 5,
    end_km: int = 10,
    want_n: int = 7,
    min_ang_sep_deg: float = 30.0,
) -> SelectionResult:
    """
    Expanding radius (5..10 km). Select want_n gauges with:
      - First pass: 1 per quadrant (4 bins)
      - Second pass: prefer unused octants (8 bins)
      - Final pass: nearest remaining
    Enforce minimum angular separation between selected gauges where possible.
    """

    dx = gx - tx
    dy = gy - ty
    dist = np.sqrt(dx * dx + dy * dy)
    bear = np.array([bearing_deg_from_north(float(dx[i]), float(dy[i])) for i in range(len(dist))], dtype=float)

    # Sort everything by distance (closest first)
    order = np.argsort(dist)
    dist_s = dist[order]
    gid_s  = gid[order].astype(str)
    bear_s = bear[order]
    q4_s   = (bear_s // 90.0).astype(int)
    o8_s   = (bear_s // 45.0).astype(int)

    def ok_angle(b: float, selected_bearings: list[float]) -> bool:
        if not selected_bearings:
            return True
        return all(ang_sep_deg(b, bb) >= min_ang_sep_deg for bb in selected_bearings)

    # Try each radius until success (or end_km)
    for r_km in range(start_km, end_km + 1):
        r_m = r_km * 1000.0
        cand_idx = np.where(dist_s <= r_m)[0]
        if cand_idx.size == 0:
            continue

        picked_ids: list[str] = []
        picked_d: list[float] = []
        picked_b: list[float] = []
        picked_set = set()

        used_octants = set()
        used_quadrants = set()

        def try_pick(ii: int) -> bool:
            g = gid_s[ii]
            if g in picked_set:
                return False
            b = float(bear_s[ii])
            if not ok_angle(b, picked_b):
                return False
            picked_ids.append(g)
            picked_d.append(float(dist_s[ii]))
            picked_b.append(b)
            picked_set.add(g)
            used_octants.add(int(o8_s[ii]))
            used_quadrants.add(int(q4_s[ii]))
            return True

        # -------------------------
        # Step A: 1 per quadrant
        # -------------------------
        for q in range(4):
            # candidates in this quadrant, in distance order
            qi = cand_idx[q4_s[cand_idx] == q]
            if qi.size == 0:
                continue
            # try nearest first, but respect 30° constraint
            for ii in qi:
                if try_pick(int(ii)):
                    break

        # -------------------------
        # Step B: prefer unused octants (fill the "other 4" as much as possible)
        # -------------------------
        if len(picked_ids) < want_n:
            # Iterate octants in distance order, pick from octants not yet used
            for ii in cand_idx:
                if len(picked_ids) >= want_n:
                    break
                if int(o8_s[ii]) in used_octants:
                    continue
                try_pick(int(ii))

        # -------------------------
        # Step C: fill remaining with nearest
        # (still respects 30° if possible; if blocked, relax at very end)
        # -------------------------
        if len(picked_ids) < want_n:
            for ii in cand_idx:
                if len(picked_ids) >= want_n:
                    break
                try_pick(int(ii))

        # If still short due to strict 30°, relax angle constraint for final fill
        if len(picked_ids) < want_n:
            for ii in cand_idx:
                if len(picked_ids) >= want_n:
                    break
                g = gid_s[ii]
                if g in picked_set:
                    continue
                # no angle check now
                picked_ids.append(g)
                picked_d.append(float(dist_s[ii]))
                picked_b.append(float(bear_s[ii]))
                picked_set.add(g)

        # Success condition: got want_n OR reached end radius
        if len(picked_ids) >= want_n or r_km == end_km:
            picked_ids = picked_ids[:want_n]
            picked_d   = picked_d[:want_n]
            picked_b   = picked_b[:want_n]

            # pad if still short at 10 km
            while len(picked_ids) < want_n:
                picked_ids.append("")
                picked_d.append(np.nan)
                picked_b.append(np.nan)

            return SelectionResult(picked_ids, picked_d, picked_b, r_km)

    # Fallback (should not happen unless no gauges exist)
    return SelectionResult([""] * want_n, [np.nan] * want_n, [np.nan] * want_n, end_km)


def build_nearest7_for_grid(
    grid_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    grid_id_col: str,
    grid_x_col: str,
    grid_y_col: str,
    stn_id_col: str = "ID",
    stn_x_col: str = "NAD83_15N_Long",
    stn_y_col: str = "NAD83_15N_Lat",
    start_km: int = 5,
    end_km: int = 10,
    want_n: int = 7,
    min_ang_sep_deg: float = 30.0,
) -> pd.DataFrame:

    gx = stations_df[stn_x_col].to_numpy(dtype=float)
    gy = stations_df[stn_y_col].to_numpy(dtype=float)
    gid = stations_df[stn_id_col].astype(str).to_numpy()

    out_rows = []
    for _, r in grid_df.iterrows():
        cell_id = str(r[grid_id_col])
        tx = float(r[grid_x_col])
        ty = float(r[grid_y_col])

        sel = select_spread_gauges_for_target(
            gx=gx, gy=gy, gid=gid,
            tx=tx, ty=ty,
            start_km=start_km, end_km=end_km,
            want_n=want_n,
            min_ang_sep_deg=min_ang_sep_deg,
        )

        rec = {grid_id_col: cell_id, "radius_used_km": sel.radius_used_km}
        for k in range(want_n):
            rec[f"g{k+1}"] = sel.gauge_ids[k]
            rec[f"d{k+1}_m"] = sel.gauge_dists_m[k]
            rec[f"b{k+1}_deg"] = sel.gauge_bear_deg[k]
        out_rows.append(rec)

    return pd.DataFrame(out_rows)


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Select nearest 7 stations per grid cell with spread + expanding buffer (5..10 km)."
    )
    ap.add_argument("--stations-csv",
                    default="/mnt/12TB/Sujan/Spatial_correlation/Codes/01_Correlation_and_Variogram/Stations_df.csv",
                    help="Stations CSV with columns ID, NAD83_15N_Long, NAD83_15N_Lat.")
    ap.add_argument("--grid-csv", required=True,
                    help="Grid cell centers CSV (UTM 15N meters).")
    ap.add_argument("--out-csv", required=True,
                    help="Output CSV path.")

    ap.add_argument("--grid-id-col", default="cell_id",
                    help="ID column for grid cells.")
    ap.add_argument("--grid-x-col", default="NAD83_15N_Long",
                    help="Grid easting column (m).")
    ap.add_argument("--grid-y-col", default="NAD83_15N_Lat",
                    help="Grid northing column (m).")

    ap.add_argument("--start-km", type=int, default=5)
    ap.add_argument("--end-km", type=int, default=10)
    ap.add_argument("--want-n", type=int, default=7)
    ap.add_argument("--min-ang-sep-deg", type=float, default=30.0)

    args = ap.parse_args()

    stations = pd.read_csv(args.stations_csv)
    grid = pd.read_csv(args.grid_csv)

    out = build_nearest7_for_grid(
        grid_df=grid,
        stations_df=stations,
        grid_id_col=args.grid_id_col,
        grid_x_col=args.grid_x_col,
        grid_y_col=args.grid_y_col,
        start_km=args.start_km,
        end_km=args.end_km,
        want_n=args.want_n,
        min_ang_sep_deg=args.min_ang_sep_deg,
    )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} (rows={len(out)})")


if __name__ == "__main__":
    main()