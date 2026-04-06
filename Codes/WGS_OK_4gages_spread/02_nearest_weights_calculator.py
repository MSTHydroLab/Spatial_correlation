#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import ast
import itertools
import json

import numpy as np
import pandas as pd

from geo_utils import haversine_km, initial_bearing_deg, ang_sep_deg

BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK_4gages_spread")
DEP_DIR = BASE_DIR / "dependent_files"
OUT_DIR = BASE_DIR / "02_OK_Weights"
CORRELATION_DIR = BASE_DIR / "01_Event_TimeSeries"

GRID_CSV = DEP_DIR / "grid_centers_wgs84.csv"
STATIONS_CSV = DEP_DIR / "Stations_df.csv"
NEAREST_CSV = DEP_DIR / "grid_nearest_gauges_wgs84.csv"


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def norm_station_id(x) -> str:
    s = str(x).strip().strip("'").strip('"')
    if s == "" or s.lower() == "nan":
        return ""
    try:
        return str(int(float(s)))
    except Exception:
        return s


def safe_list(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if s == "":
        return []
    try:
        out = json.loads(s)
        return out if isinstance(out, list) else []
    except Exception:
        try:
            out = ast.literal_eval(s)
            return list(out) if isinstance(out, (list, tuple)) else []
        except Exception:
            return []


def rho_powerexp(d_km, a_km, b):
    d = np.asarray(d_km, dtype=float)
    return np.exp(-((d / float(a_km)) ** float(b)))


def compute_ok_weights(stations_df, target_lat, target_lon, ids, a_km, b, nugget=0.0):
    st = stations_df.set_index("ID")
    ids = [str(i) for i in ids]

    xs_lat = np.array([float(st.loc[sid, "Latitude"]) for sid in ids], dtype=float)
    xs_lon = np.array([float(st.loc[sid, "Longitude"]) for sid in ids], dtype=float)
    n = len(ids)

    dij_km = np.zeros((n, n), dtype=float)
    for i in range(n):
        dij_km[i, :] = haversine_km(xs_lat[i], xs_lon[i], xs_lat, xs_lon)

    d0_km = haversine_km(target_lat, target_lon, xs_lat, xs_lon)
    C = rho_powerexp(dij_km, a_km, b)
    c0 = rho_powerexp(d0_km, a_km, b)

    if nugget > 0:
        C += np.eye(n) * float(nugget)

    A = np.zeros((n + 1, n + 1), dtype=float)
    A[:n, :n] = C
    A[:n, n] = 1.0
    A[n, :n] = 1.0

    rhs = np.zeros(n + 1, dtype=float)
    rhs[:n] = c0
    rhs[n] = 1.0

    solver = "solve"
    try:
        sol = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        solver = "lstsq"

    return sol[:n], solver


def min_pairwise_sep_deg(bears):
    bears = [float(x) for x in bears]
    if len(bears) < 2:
        return 360.0
    return min(
        ang_sep_deg(bears[i], bears[j])
        for i, j in itertools.combinations(range(len(bears)), 2)
    )


# ---------------------------------------------------------------------
# Candidate loading
# ---------------------------------------------------------------------
def build_nearest_table_from_scratch(grid_df, stations_df, search_radius_km: float, keep_n: int) -> pd.DataFrame:
    stn_lat = stations_df["Latitude"].to_numpy(dtype=float)
    stn_lon = stations_df["Longitude"].to_numpy(dtype=float)
    stn_id = stations_df["ID"].astype(str).to_numpy()

    rows = []
    for _, row in grid_df.iterrows():
        target_lat = float(row["Latitude"])
        target_lon = float(row["Longitude"])

        d_km = haversine_km(target_lat, target_lon, stn_lat, stn_lon)
        bears = initial_bearing_deg(target_lat, target_lon, stn_lat, stn_lon)
        order = np.argsort(d_km)

        inside = [i for i in order if float(d_km[i]) <= float(search_radius_km)]
        chosen = inside[:keep_n]
        if len(chosen) < keep_n:
            for i in order:
                if i not in chosen:
                    chosen.append(i)
                if len(chosen) >= keep_n:
                    break

        ids = [norm_station_id(stn_id[i]) for i in chosen]
        dists_m = [float(d_km[i]) * 1000.0 for i in chosen]
        bears_deg = [float(bears[i]) for i in chosen]

        rows.append({
            "id": str(row["id"]),
            "Latitude": target_lat,
            "Longitude": target_lon,
            "search_radius_km": float(search_radius_km),
            "candidate_count": int(len(ids)),
            "candidate_ids": json.dumps(ids),
            "candidate_dists_m": json.dumps(dists_m),
            "candidate_bears_deg": json.dumps(bears_deg),
        })

    return pd.DataFrame(rows)


def load_or_build_candidates(nearest_file: Path, grid_file: Path, station_file: Path,
                             search_radius_km: float, keep_n: int) -> pd.DataFrame:
    grid_df = pd.read_csv(grid_file)
    stations_df = pd.read_csv(station_file)
    grid_df["id"] = pd.to_numeric(grid_df["id"], errors="coerce").astype("Int64")
    stations_df["ID"] = stations_df["ID"].apply(norm_station_id)

    if nearest_file.exists():
        nei = pd.read_csv(nearest_file)
        req = ["id", "Latitude", "Longitude", "candidate_ids", "candidate_dists_m", "candidate_bears_deg"]
        missing = [c for c in req if c not in nei.columns]
        if not missing:
            nei["id"] = nei["id"].astype(str)
            return nei, stations_df

    nei = build_nearest_table_from_scratch(grid_df, stations_df, search_radius_km, keep_n)
    return nei, stations_df


# ---------------------------------------------------------------------
# Geometry selection
# ---------------------------------------------------------------------
def quadrant_idx(bearing_deg: float) -> int:
    return int(np.floor((bearing_deg % 360.0) / 90.0))


def sector3_idx(bearing_deg: float) -> int:
    return int(np.floor((bearing_deg % 360.0) / 120.0))


def make_candidates_from_row(row) -> list[dict]:
    ids = [norm_station_id(x) for x in safe_list(row.get("candidate_ids", []))]
    dists = [float(x) for x in safe_list(row.get("candidate_dists_m", []))]
    bears = [float(x) for x in safe_list(row.get("candidate_bears_deg", []))]

    n = min(len(ids), len(dists), len(bears))
    out = []
    for i in range(n):
        sid = ids[i]
        if sid == "":
            continue
        out.append({
            "id": sid,
            "dist_m": float(dists[i]),
            "bearing_deg": float(bears[i]),
        })
    out.sort(key=lambda x: x["dist_m"])
    return out


def choose_spread_group(candidates: list[dict], n_gauges: int):
    if len(candidates) < n_gauges:
        return None

    forced = candidates[0]
    forced_id = forced["id"]
    forced_bearing = float(forced["bearing_deg"])

    if n_gauges == 4:
        bin_fn = quadrant_idx
        label = "quadrant"
        n_bins = 4
    elif n_gauges == 3:
        bin_fn = sector3_idx
        label = "sector3"
        n_bins = 3
    else:
        raise ValueError("n_gauges must be 3 or 4")

    for cand in candidates:
        cand["bin_idx"] = int(bin_fn(float(cand["bearing_deg"])))

    forced_bin = int(bin_fn(forced_bearing))
    others = [c for c in candidates[1:] if c["id"] != forced_id]

    combos = []
    for combo in itertools.combinations(others, n_gauges - 1):
        group = [forced] + list(combo)
        ids = [g["id"] for g in group]
        if len(set(ids)) != len(ids):
            continue

        bins = [int(g["bin_idx"]) for g in group]
        n_unique_bins = len(set(bins))
        bears = [float(g["bearing_deg"]) for g in group]
        dists = [float(g["dist_m"]) for g in group]

        combos.append({
            "ids": ids,
            "dists": dists,
            "bears": bears,
            "bins": bins,
            "bin_count": n_unique_bins,
            "min_sep_deg": float(min_pairwise_sep_deg(bears)),
            "sum_dist_m": float(np.sum(dists)),
            "mean_dist_m": float(np.mean(dists)),
            "method_grouping": label,
            "forced_near_sid": forced_id,
            "forced_near_dist_m": float(forced["dist_m"]),
            "forced_near_bin": forced_bin,
        })

    if not combos:
        return None

    # First try full spread: all bins represented.
    full = [c for c in combos if c["bin_count"] == n_bins]
    pool = full if full else combos

    # Ranking:
    # 1) maximize represented bins
    # 2) maximize minimum angular separation
    # 3) minimize total distance
    # 4) minimize mean distance
    pool_sorted = sorted(
        pool,
        key=lambda r: (
            r["bin_count"],
            r["min_sep_deg"],
            -r["sum_dist_m"],
            -r["mean_dist_m"],
        ),
        reverse=True,
    )
    best = dict(pool_sorted[0])
    best["spread_complete"] = bool(best["bin_count"] == n_bins)
    return best


# ---------------------------------------------------------------------
# Main event runner
# ---------------------------------------------------------------------
def run_event(event_number: int, event_meta_dir: Path, out_dir: Path, nearest_file: Path,
              grid_file: Path, station_file: Path, n_gauges: int,
              nugget: float, search_radius_km: float, keep_n: int):
    event_file = event_meta_dir / f"Event_{event_number}_Stations_correlation.csv"
    if not event_file.exists():
        raise FileNotFoundError(f"Missing event metadata: {event_file}")

    event_df = pd.read_csv(event_file)
    a_km = float(event_df["corr_a_km"].iloc[0])
    b = float(event_df["corr_b"].iloc[0])

    nei, stations = load_or_build_candidates(
        nearest_file=nearest_file,
        grid_file=grid_file,
        station_file=station_file,
        search_radius_km=search_radius_km,
        keep_n=keep_n,
    )

    stations = stations.copy()
    stations["ID"] = stations["ID"].apply(norm_station_id)

    results = []
    for _, row in nei.iterrows():
        cid = str(row["id"])
        target_lat = float(row["Latitude"])
        target_lon = float(row["Longitude"])

        candidates = make_candidates_from_row(row)
        choice = choose_spread_group(candidates, n_gauges=n_gauges)

        if choice is None:
            results.append({
                "id": cid,
                "Latitude": target_lat,
                "Longitude": target_lon,
                "event": int(event_number),
                "n_gauges_requested": int(n_gauges),
                "candidate_count": int(len(candidates)),
                "spread_type": "quadrant" if n_gauges == 4 else "sector3",
                "spread_complete": False,
                "remarks": "not_enough_candidates_for_requested_group",
            })
            continue

        weights, solver = compute_ok_weights(
            stations_df=stations,
            target_lat=target_lat,
            target_lon=target_lon,
            ids=choice["ids"],
            a_km=a_km,
            b=b,
            nugget=nugget,
        )
        weights = np.asarray(weights, dtype=float)

        rec = {
            "id": cid,
            "Latitude": target_lat,
            "Longitude": target_lon,
            "event": int(event_number),
            "corr_a_km": float(a_km),
            "corr_b": float(b),
            "n_gauges_requested": int(n_gauges),
            "candidate_count": int(len(candidates)),
            "spread_type": str(choice["method_grouping"]),
            "spread_complete": bool(choice["spread_complete"]),
            "represented_bins": int(choice["bin_count"]),
            "forced_near_sid": str(choice["forced_near_sid"]),
            "forced_near_dist_m": float(choice["forced_near_dist_m"]),
            "min_ang_sep_deg": float(choice["min_sep_deg"]),
            "sum_dist_m": float(choice["sum_dist_m"]),
            "mean_dist_m": float(choice["mean_dist_m"]),
            "solver": solver,
            "n_negative_weights": int(np.sum(weights < 0)),
            "min_weight": float(np.min(weights)),
            "max_weight": float(np.max(weights)),
            "sum_weights": float(np.sum(weights)),
            "remarks": (
                f"nearest_forced_{choice['method_grouping']}_complete"
                if choice["spread_complete"] else
                f"nearest_forced_{choice['method_grouping']}_best_available"
            ),
        }

        for k in range(1, n_gauges + 1):
            rec[f"g{k}"] = str(choice["ids"][k - 1])
            rec[f"d{k}_m"] = float(choice["dists"][k - 1])
            rec[f"b{k}_deg"] = float(choice["bears"][k - 1])
            rec[f"w{k}"] = float(weights[k - 1])
            rec[f"bin{k}"] = int(choice["bins"][k - 1])

        results.append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"Event_{event_number}_nearest{n_gauges}_weights.csv"
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"Saved: {out_file}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Compute ordinary-kriging weights using nearest-gauge candidates, "
            "forcing the nearest gauge into the selected group and choosing the "
            "best spread by quadrants (4 gauges) or 3 sectors (3 gauges)."
        )
    )
    ap.add_argument("--event", type=int, nargs="+", required=True, help="One or more event numbers")
    ap.add_argument("--n-gauges", type=int, default=4, choices=[3, 4], help="Use 3 gauges (sector3) or 4 gauges (quadrants)")
    ap.add_argument("--base-dir", type=Path, default=BASE_DIR)
    ap.add_argument("--event-meta-dir", type=Path, default=CORRELATION_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--nearest-file", type=Path, default=NEAREST_CSV)
    ap.add_argument("--grid-file", type=Path, default=GRID_CSV)
    ap.add_argument("--station-file", type=Path, default=STATIONS_CSV)
    ap.add_argument("--nugget", type=float, default=0.0)
    ap.add_argument("--search-radius-km", type=float, default=7.0, help="Used only if nearest-file is missing or incomplete")
    ap.add_argument("--keep-n", type=int, default=10, help="Used only if nearest-file is missing or incomplete")
    args = ap.parse_args()

    for ev in args.event:
        print("=" * 80)
        print(f"Running event {ev} with nearest {args.n_gauges} gauges")
        print("=" * 80)
        run_event(
            event_number=int(ev),
            event_meta_dir=Path(args.event_meta_dir),
            out_dir=Path(args.out_dir),
            nearest_file=Path(args.nearest_file),
            grid_file=Path(args.grid_file),
            station_file=Path(args.station_file),
            n_gauges=int(args.n_gauges),
            nugget=float(args.nugget),
            search_radius_km=float(args.search_radius_km),
            keep_n=int(args.keep_n),
        )


if __name__ == "__main__":
    main()
