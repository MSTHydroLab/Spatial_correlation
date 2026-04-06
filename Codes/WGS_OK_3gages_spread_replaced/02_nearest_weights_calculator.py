#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import ast
import json

import numpy as np
import pandas as pd

from geo_utils import haversine_km, initial_bearing_deg, ang_sep_deg

BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK_3gages_spread_replaced")
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

    return np.asarray(sol[:n], dtype=float), solver



def min_pairwise_sep_deg(bears):
    bears = [float(x) for x in bears]
    if len(bears) < 2:
        return 360.0
    vals = []
    for i in range(len(bears)):
        for j in range(i + 1, len(bears)):
            vals.append(ang_sep_deg(bears[i], bears[j]))
    return float(min(vals)) if vals else 360.0


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
                             search_radius_km: float, keep_n: int):
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
# Nearest-n and negative-weight refit logic
# ---------------------------------------------------------------------
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



def choose_nearest_n(candidates: list[dict], n_gauges: int):
    if len(candidates) < n_gauges:
        return None
    group = candidates[:n_gauges]
    ids = [g["id"] for g in group]
    dists = [float(g["dist_m"]) for g in group]
    bears = [float(g["bearing_deg"]) for g in group]
    return {
        "ids": ids,
        "dists": dists,
        "bears": bears,
        "forced_near_sid": ids[0],
        "forced_near_dist_m": float(dists[0]),
        "min_sep_deg": float(min_pairwise_sep_deg(bears)),
        "sum_dist_m": float(np.sum(dists)),
        "mean_dist_m": float(np.mean(dists)),
    }



def refit_negative_weights(stations_df, target_lat, target_lon, ids, a_km, b, nugget=0.0):
    ids_initial = [str(x) for x in ids]

    if len(ids_initial) == 0:
        return {
            "final_ids": [],
            "final_weights": np.array([], dtype=float),
            "raw_initial_weights": np.array([], dtype=float),
            "solver": "none",
            "solver_history": "",
            "dropped_negative_ids": "",
            "n_negative_initial": 0,
            "n_negative_final": 0,
            "refit_performed": False,
            "all_negative_removed": False,
        }

    # -------------------------------------------------
    # First solve with all initially selected gauges
    # -------------------------------------------------
    w_initial, solver_initial = compute_ok_weights(
        stations_df=stations_df,
        target_lat=target_lat,
        target_lon=target_lon,
        ids=ids_initial,
        a_km=a_km,
        b=b,
        nugget=nugget,
    )

    neg_mask_initial = w_initial < 0
    n_negative_initial = int(np.sum(neg_mask_initial))

    # If no negatives, keep initial result and stop
    if n_negative_initial == 0:
        return {
            "final_ids": ids_initial,
            "final_weights": np.asarray(w_initial, dtype=float),
            "raw_initial_weights": np.asarray(w_initial, dtype=float),
            "solver": solver_initial,
            "solver_history": f"n={len(ids_initial)}:{solver_initial}",
            "dropped_negative_ids": "",
            "n_negative_initial": 0,
            "n_negative_final": 0,
            "refit_performed": False,
            "all_negative_removed": False,
        }

    # -------------------------------------------------
    # Remove negative-weight gauges from FIRST solve only
    # -------------------------------------------------
    dropped_ids = [sid for sid, is_neg in zip(ids_initial, neg_mask_initial) if is_neg]
    ids_remaining = [sid for sid, is_neg in zip(ids_initial, neg_mask_initial) if not is_neg]

    # If everything was negative, return empty final set
    if len(ids_remaining) == 0:
        return {
            "final_ids": [],
            "final_weights": np.array([], dtype=float),
            "raw_initial_weights": np.asarray(w_initial, dtype=float),
            "solver": "none",
            "solver_history": f"n={len(ids_initial)}:{solver_initial} | n=0:none",
            "dropped_negative_ids": ",".join(dropped_ids),
            "n_negative_initial": n_negative_initial,
            "n_negative_final": 0,
            "refit_performed": True,
            "all_negative_removed": True,
        }

    # -------------------------------------------------
    # Refit ONCE using remaining gauges, then stop
    # -------------------------------------------------
    w_final, solver_final = compute_ok_weights(
        stations_df=stations_df,
        target_lat=target_lat,
        target_lon=target_lon,
        ids=ids_remaining,
        a_km=a_km,
        b=b,
        nugget=nugget,
    )

    return {
        "final_ids": ids_remaining,
        "final_weights": np.asarray(w_final, dtype=float),
        "raw_initial_weights": np.asarray(w_initial, dtype=float),
        "solver": solver_final,
        "solver_history": f"n={len(ids_initial)}:{solver_initial} | n={len(ids_remaining)}:{solver_final}",
        "dropped_negative_ids": ",".join(dropped_ids),
        "n_negative_initial": n_negative_initial,
        "n_negative_final": int(np.sum(np.asarray(w_final) < 0)),
        "refit_performed": True,
        "all_negative_removed": False,
    }

def pad_output(final_ids, final_weights, original_choice, n_gauges):
    out_ids = []
    out_dists = []
    out_bears = []
    out_weights = []

    orig_lookup = {
        str(sid): (float(dist), float(bear))
        for sid, dist, bear in zip(original_choice["ids"], original_choice["dists"], original_choice["bears"])
    }

    for sid, w in zip(final_ids, final_weights):
        dist, bear = orig_lookup[str(sid)]
        out_ids.append(str(sid))
        out_dists.append(dist)
        out_bears.append(bear)
        out_weights.append(float(w))

    dropped_ids = [sid for sid in original_choice["ids"] if sid not in set(final_ids)]
    for sid in dropped_ids:
        if len(out_ids) >= n_gauges:
            break
        dist, bear = orig_lookup[str(sid)]
        out_ids.append(str(sid))
        out_dists.append(dist)
        out_bears.append(bear)
        out_weights.append(0.0)

    while len(out_ids) < n_gauges:
        out_ids.append("")
        out_dists.append(np.nan)
        out_bears.append(np.nan)
        out_weights.append(0.0)

    return out_ids, out_dists, out_bears, np.asarray(out_weights, dtype=float)


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
        choice = choose_nearest_n(candidates, n_gauges=n_gauges)

        if choice is None:
            results.append({
                "id": cid,
                "Latitude": target_lat,
                "Longitude": target_lon,
                "event": int(event_number),
                "n_gauges_requested": int(n_gauges),
                "candidate_count": int(len(candidates)),
                "spread_type": "nearest_only",
                "spread_complete": False,
                "represented_bins": np.nan,
                "forced_near_sid": "",
                "forced_near_dist_m": np.nan,
                "min_ang_sep_deg": np.nan,
                "sum_dist_m": np.nan,
                "mean_dist_m": np.nan,
                "solver": "none",
                "solver_history": "",
                "n_negative_weights": np.nan,
                "n_negative_initial": np.nan,
                "n_negative_final": np.nan,
                "min_weight": np.nan,
                "max_weight": np.nan,
                "sum_weights": np.nan,
                "refit_performed": False,
                "dropped_negative_ids": "",
                "n_gauges_final": 0,
                "remarks": "not_enough_candidates_for_requested_group",
            })
            continue

        refit = refit_negative_weights(
            stations_df=stations,
            target_lat=target_lat,
            target_lon=target_lon,
            ids=choice["ids"],
            a_km=a_km,
            b=b,
            nugget=nugget,
        )

        padded_ids, padded_dists, padded_bears, padded_weights = pad_output(
            final_ids=refit["final_ids"],
            final_weights=refit["final_weights"],
            original_choice=choice,
            n_gauges=n_gauges,
        )

        raw_initial = np.asarray(refit["raw_initial_weights"], dtype=float)
        final_weights = np.asarray(refit["final_weights"], dtype=float)

        if raw_initial.size == 0:
            min_w = np.nan
            max_w = np.nan
            sum_w = np.nan
        else:
            min_w = float(np.min(padded_weights))
            max_w = float(np.max(padded_weights))
            sum_w = float(np.sum(padded_weights))

        remarks = "nearest_n_all_nonnegative"
        if refit["refit_performed"]:
            remarks = "nearest_n_negative_refit"
        if len(refit["final_ids"]) == 0:
            remarks = "nearest_n_all_negative_removed"

        rec = {
            "id": cid,
            "Latitude": target_lat,
            "Longitude": target_lon,
            "event": int(event_number),
            "corr_a_km": float(a_km),
            "corr_b": float(b),
            "n_gauges_requested": int(n_gauges),
            "n_gauges_final": int(len(refit["final_ids"])),
            "candidate_count": int(len(candidates)),
            "spread_type": "nearest_only",
            "spread_complete": False,
            "represented_bins": np.nan,
            "forced_near_sid": str(choice["forced_near_sid"]),
            "forced_near_dist_m": float(choice["forced_near_dist_m"]),
            "min_ang_sep_deg": float(choice["min_sep_deg"]),
            "sum_dist_m": float(choice["sum_dist_m"]),
            "mean_dist_m": float(choice["mean_dist_m"]),
            "solver": str(refit["solver"]),
            "solver_history": str(refit["solver_history"]),
            "n_negative_weights": int(np.sum(padded_weights < 0)),
            "n_negative_initial": int(refit["n_negative_initial"]),
            "n_negative_final": int(refit["n_negative_final"]),
            "min_weight": min_w,
            "max_weight": max_w,
            "sum_weights": sum_w,
            "refit_performed": bool(refit["refit_performed"]),
            "dropped_negative_ids": str(refit["dropped_negative_ids"]),
            "remarks": remarks,
        }

        for k in range(1, n_gauges + 1):
            rec[f"g{k}"] = str(padded_ids[k - 1])
            rec[f"d{k}_m"] = float(padded_dists[k - 1]) if pd.notna(padded_dists[k - 1]) else np.nan
            rec[f"b{k}_deg"] = float(padded_bears[k - 1]) if pd.notna(padded_bears[k - 1]) else np.nan
            rec[f"w{k}"] = float(padded_weights[k - 1])

        for k in range(1, n_gauges + 1):
            rec[f"raw_w{k}"] = float(raw_initial[k - 1]) if k <= len(raw_initial) else np.nan

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
            "Compute ordinary-kriging weights using the nearest n gauges only. "
            "If any weights are negative, remove those gauges, refit on the remaining gauges, "
            "and write the output in the same g/w column format with removed gauges padded as zero weights."
        )
    )
    ap.add_argument("--event", type=int, nargs="+", required=True, help="One or more event numbers")
    ap.add_argument("--n-gauges", type=int, default=4, choices=[3, 4], help="Use nearest 3 or nearest 4 gauges")
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
