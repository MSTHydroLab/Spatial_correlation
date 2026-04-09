#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import ast
import json

import numpy as np
import pandas as pd

from geo_utils import haversine_km, initial_bearing_deg, ang_sep_deg

BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK")
DEP_DIR = BASE_DIR / "dependent_files"
OUT_DIR = BASE_DIR / "02_OK_Weights"
CORRELATION_DIR = BASE_DIR / "01_Event_TimeSeries"

GRID_CSV = DEP_DIR / "grid_centers_wgs84.csv"
STATIONS_CSV = DEP_DIR / "Stations_df.csv"
NEAREST_CSV = DEP_DIR / "grid_nearest_gauges_wgs84.csv"

POOL1=5
POOL2=9
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
def filter_candidates_by_radius(candidates: list[dict], radius_km: float) -> list[dict]:
    limit_m = float(radius_km) * 1000.0
    return [c for c in candidates if float(c["dist_m"]) <= limit_m]


def build_choice_from_selected(selected: list[dict]):
    ids = [g["id"] for g in selected]
    dists = [float(g["dist_m"]) for g in selected]
    bears = [float(g["bearing_deg"]) for g in selected]
    return {
        "ids": ids,
        "dists": dists,
        "bears": bears,
        "forced_near_sid": ids[0] if ids else "",
        "forced_near_dist_m": float(dists[0]) if dists else np.nan,
        "min_sep_deg": float(min_pairwise_sep_deg(bears)) if bears else np.nan,
        "sum_dist_m": float(np.sum(dists)) if dists else np.nan,
        "mean_dist_m": float(np.mean(dists)) if dists else np.nan,
    }

def make_candidates_from_row(row, valid_station_set=None) -> list[dict]:
    ids = [norm_station_id(x) for x in safe_list(row.get("candidate_ids", []))]
    dists = [float(x) for x in safe_list(row.get("candidate_dists_m", []))]
    bears = [float(x) for x in safe_list(row.get("candidate_bears_deg", []))]

    n = min(len(ids), len(dists), len(bears))
    out = []
    for i in range(n):
        sid = ids[i]
        if sid == "":
            continue
        if valid_station_set is not None and sid not in valid_station_set:
            continue
        out.append({
            "id": sid,
            "dist_m": float(dists[i]),
            "bearing_deg": float(bears[i]),
        })

    out.sort(key=lambda x: x["dist_m"])
    return out

def compute_idw_weights(dists_m, power=2.0):
    d = np.asarray(dists_m, dtype=float)

    if len(d) == 0:
        return np.array([], dtype=float)

    if np.any(d <= 0):
        w = np.zeros_like(d, dtype=float)
        w[np.argmin(d)] = 1.0
        return w

    inv = 1.0 / np.power(d, power)
    s = np.sum(inv)
    if s <= 0:
        return np.zeros_like(d, dtype=float)

    return inv / s

def iterative_replace_until_positive(
    stations_df,
    target_lat,
    target_lon,
    pool: list[dict],
    n_select: int,
    a_km: float,
    b: float,
    nugget: float = 0.0,
):
    """
    Start with the n_select nearest gauges from pool.
    Keep the absolute nearest gauge fixed at all times.
    If negative weights appear, replace the most-negative selected gauge
    EXCEPT the forced nearest gauge.
    """
    if len(pool) < n_select:
        return None

    selected = [dict(x) for x in pool[:n_select]]
    next_idx = n_select

    forced_near_id = str(pool[0]["id"])   # absolute nearest in this pool
    solver_history = []
    replacement_history = []
    seen_states = set()

    while True:
        state_key = tuple(x["id"] for x in selected)
        if state_key in seen_states:
            break
        seen_states.add(state_key)

        choice = build_choice_from_selected(selected)
        ids_now = choice["ids"]

        w, solver = compute_ok_weights(
            stations_df=stations_df,
            target_lat=target_lat,
            target_lon=target_lon,
            ids=ids_now,
            a_km=a_km,
            b=b,
            nugget=nugget,
        )
        w = np.asarray(w, dtype=float)
        solver_history.append(f"{ids_now}:{solver}")

        neg_idx = np.where(w < 0)[0]
        if len(neg_idx) == 0:
            return {
                "success": True,
                "choice": choice,
                "weights": w,
                "solver": solver,
                "solver_history": " | ".join(solver_history),
                "replacement_history": " | ".join(replacement_history),
                "n_negative_final": 0,
                "forced_near_id": forced_near_id,
            }

        # choose the most negative selected gauge, but NEVER drop forced nearest
        candidate_drop_idx = np.argsort(w)  # most negative first
        worst_local_idx = None
        for idx in candidate_drop_idx:
            if str(selected[int(idx)]["id"]) != forced_near_id:
                worst_local_idx = int(idx)
                break

        # if the only negative gauge is the forced nearest, this stage cannot be repaired by replacement
        if worst_local_idx is None:
            return {
                "success": False,
                "choice": choice,
                "weights": w,
                "solver": solver,
                "solver_history": " | ".join(solver_history),
                "replacement_history": " | ".join(replacement_history),
                "n_negative_final": int(np.sum(w < 0)),
                "forced_near_id": forced_near_id,
            }

        replacement = None
        while next_idx < len(pool):
            cand = pool[next_idx]
            next_idx += 1
            cand_id = str(cand["id"])
            if cand_id not in [str(x["id"]) for x in selected]:
                replacement = dict(cand)
                break

        if replacement is None:
            return {
                "success": False,
                "choice": choice,
                "weights": w,
                "solver": solver,
                "solver_history": " | ".join(solver_history),
                "replacement_history": " | ".join(replacement_history),
                "n_negative_final": int(np.sum(w < 0)),
                "forced_near_id": forced_near_id,
            }

        removed = str(selected[worst_local_idx]["id"])
        selected[worst_local_idx] = replacement
        replacement_history.append(f"drop {removed} -> add {replacement['id']}")

def build_idw_case(choice, stage_name):
    weights = compute_idw_weights(choice["dists"], power=2.0)

    padded_ids, padded_dists, padded_bears, padded_weights = pad_output(
        final_ids=choice["ids"],
        final_weights=weights,
        original_choice=choice,
        n_gauges=4,
    )

    return {
        "stage_used": f"{stage_name}_idw",
        "choice_used": choice,
        "final_ids": choice["ids"],
        "final_weights": np.asarray(weights, dtype=float),
        "padded_ids": padded_ids,
        "padded_dists": padded_dists,
        "padded_bears": padded_bears,
        "padded_weights": padded_weights,
        "solver": "idw",
        "solver_history": "idw_fallback",
        "replacement_history": "",
        "n_negative_final": 0,
        "forced_near_id": str(choice["ids"][0]) if len(choice["ids"]) > 0 else "",
        "remarks": f"{stage_name}_idw_fallback",
    }
    
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
    """
    One-stage refit only:
    - solve with given ids
    - if any negatives, remove those negative gauges
    - refit once on remaining gauges
    - stop there
    """
    ids_initial = [str(x) for x in ids]

    if len(ids_initial) == 0:
        return {
            "initial_ids": [],
            "final_ids": [],
            "final_weights": np.array([], dtype=float),
            "raw_initial_weights": np.array([], dtype=float),
            "solver_initial": "none",
            "solver_final": "none",
            "solver_history": "",
            "dropped_negative_ids": "",
            "n_negative_initial": 0,
            "n_negative_final": 0,
            "refit_performed": False,
            "all_negative_removed": False,
        }

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

    if n_negative_initial == 0:
        return {
            "initial_ids": ids_initial,
            "final_ids": ids_initial,
            "final_weights": np.asarray(w_initial, dtype=float),
            "raw_initial_weights": np.asarray(w_initial, dtype=float),
            "solver_initial": solver_initial,
            "solver_final": solver_initial,
            "solver_history": f"n={len(ids_initial)}:{solver_initial}",
            "dropped_negative_ids": "",
            "n_negative_initial": 0,
            "n_negative_final": 0,
            "refit_performed": False,
            "all_negative_removed": False,
        }

    dropped_ids = [sid for sid, is_neg in zip(ids_initial, neg_mask_initial) if is_neg]
    ids_remaining = [sid for sid, is_neg in zip(ids_initial, neg_mask_initial) if not is_neg]

    if len(ids_remaining) == 0:
        return {
            "initial_ids": ids_initial,
            "final_ids": [],
            "final_weights": np.array([], dtype=float),
            "raw_initial_weights": np.asarray(w_initial, dtype=float),
            "solver_initial": solver_initial,
            "solver_final": "none",
            "solver_history": f"n={len(ids_initial)}:{solver_initial} | n=0:none",
            "dropped_negative_ids": ",".join(dropped_ids),
            "n_negative_initial": n_negative_initial,
            "n_negative_final": 0,
            "refit_performed": True,
            "all_negative_removed": True,
        }

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
        "initial_ids": ids_initial,
        "final_ids": ids_remaining,
        "final_weights": np.asarray(w_final, dtype=float),
        "raw_initial_weights": np.asarray(w_initial, dtype=float),
        "solver_initial": solver_initial,
        "solver_final": solver_final,
        "solver_history": f"n={len(ids_initial)}:{solver_initial} | n={len(ids_remaining)}:{solver_final}",
        "dropped_negative_ids": ",".join(dropped_ids),
        "n_negative_initial": n_negative_initial,
        "n_negative_final": int(np.sum(np.asarray(w_final) < 0)),
        "refit_performed": True,
        "all_negative_removed": False,
    }
    


def run_stagewise_search(stations_df, target_lat, target_lon, candidates, a_km, b, nugget=0.0):
    """
    Search order:
      1) nearest 4 within POOL1 using OK + replacement
      2) nearest 3 within POOL1 using OK + replacement
      3) nearest 4 within POOL2 using OK + replacement
      4) nearest 3 within POOL2 using OK + replacement

    If all OK stages fail, use IDW fallback with:
      - nearest 4 if available
      - else nearest 3
      - else nearest 2
    """
    first_pool = filter_candidates_by_radius(candidates, POOL1)
    second_pool = filter_candidates_by_radius(candidates, POOL2)

    stages = [
        ("nearest4_within5km", first_pool, 4),
        ("nearest3_within5km", first_pool, 3),
        ("nearest4_within7km", second_pool, 4),
        ("nearest3_within7km", second_pool, 3),
    ]

    last_result = None

    for stage_name, pool, n_select in stages:
        result = iterative_replace_until_positive(
            stations_df=stations_df,
            target_lat=target_lat,
            target_lon=target_lon,
            pool=pool,
            n_select=n_select,
            a_km=a_km,
            b=b,
            nugget=nugget,
        )

        if result is None:
            continue

        last_result = (stage_name, n_select, result)

        if result["success"]:
            choice = result["choice"]
            padded_ids, padded_dists, padded_bears, padded_weights = pad_output(
                final_ids=choice["ids"],
                final_weights=result["weights"],
                original_choice=choice,
                n_gauges=4,
            )

            return {
                "stage_used": stage_name,
                "choice_used": choice,
                "final_ids": choice["ids"],
                "final_weights": np.asarray(result["weights"], dtype=float),
                "padded_ids": padded_ids,
                "padded_dists": padded_dists,
                "padded_bears": padded_bears,
                "padded_weights": padded_weights,
                "solver": result["solver"],
                "solver_history": result["solver_history"],
                "replacement_history": result["replacement_history"],
                "n_negative_final": 0,
                "remarks": f"{stage_name}_success",
            }

        # ------------------------------------------------------------
    # Final fallback: prefer local IDW first
    # Rule:
    # - if at least 3 gauges exist within POOL1, use nearest 3 within POOL1
    # - else if at least 2 gauges exist within POOL1, use nearest 2 within POOL1
    # - else expand to POOL2 with 4, then 3, then 2
    # ------------------------------------------------------------
    if len(first_pool) >= 3:
        choice = build_choice_from_selected(first_pool[:3])
        return build_idw_case(choice, "nearest3_within5km")

    if len(first_pool) >= 2:
        choice = build_choice_from_selected(first_pool[:2])
        return build_idw_case(choice, "nearest2_within5km")

    idw_options = [
        ("nearest4_within7km", second_pool, 4),
        ("nearest3_within7km", second_pool, 3),
        ("nearest2_within7km", second_pool, 2),
    ]

    for stage_name, pool, n_select in idw_options:
        if len(pool) >= n_select:
            choice = build_choice_from_selected(pool[:n_select])
            return build_idw_case(choice, stage_name)

    # If absolutely nothing usable
    if last_result is not None:
        stage_name, _, result = last_result
        choice = result["choice"]
        padded_ids, padded_dists, padded_bears, padded_weights = pad_output(
            final_ids=choice["ids"],
            final_weights=result["weights"],
            original_choice=choice,
            n_gauges=4,
        )

        return {
            "stage_used": stage_name,
            "choice_used": choice,
            "final_ids": choice["ids"],
            "final_weights": np.asarray(result["weights"], dtype=float),
            "padded_ids": padded_ids,
            "padded_dists": padded_dists,
            "padded_bears": padded_bears,
            "padded_weights": padded_weights,
            "solver": result["solver"],
            "solver_history": result["solver_history"],
            "replacement_history": result["replacement_history"],
            "n_negative_final": int(result["n_negative_final"]),
            "remarks": f"{stage_name}_failed_still_negative",
        }

    return None

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
    valid_station_list = event_df["stations_selected"].iloc[0].split(",")
    valid_station_set = set(valid_station_list)
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
    stations = stations[stations["ID"].isin(valid_station_set)].copy()

    results = []
    for _, row in nei.iterrows():
        cid = str(row["id"])
        target_lat = float(row["Latitude"])
        target_lon = float(row["Longitude"])

        candidates = make_candidates_from_row(row, valid_station_set)

        first_pool = filter_candidates_by_radius(candidates, POOL1)
        second_pool = filter_candidates_by_radius(candidates, POOL2)

        if len(second_pool) < 2:
            results.append({
                "id": cid,
                "Latitude": target_lat,
                "Longitude": target_lon,
                "event": int(event_number),
                "n_gauges_requested": 4,
                "candidate_count": int(len(candidates)),
                "stage_used": "none",
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
                "remarks": "not_enough_candidates_for_nearest3",
                "g1": "", "g2": "", "g3": "", "g4": "",
                "d1_m": np.nan, "d2_m": np.nan, "d3_m": np.nan, "d4_m": np.nan,
                "b1_deg": np.nan, "b2_deg": np.nan, "b3_deg": np.nan, "b4_deg": np.nan,
                "w1": 0.0, "w2": 0.0, "w3": 0.0, "w4": 0.0,
                "raw_w1": np.nan, "raw_w2": np.nan, "raw_w3": np.nan, "raw_w4": np.nan,
            })
            continue

        case = run_stagewise_search(
            stations_df=stations,
            target_lat=target_lat,
            target_lon=target_lon,
            candidates=candidates,
            a_km=a_km,
            b=b,
            nugget=nugget,
        )

        if case is None:
            results.append({
                "id": cid,
                "Latitude": target_lat,
                "Longitude": target_lon,
                "event": int(event_number),
                "n_gauges_requested": 4,
                "candidate_count": int(len(candidates)),
                "stage_used": "none",
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
                "remarks": "hierarchical_case_failed",
                "g1": "", "g2": "", "g3": "", "g4": "",
                "d1_m": np.nan, "d2_m": np.nan, "d3_m": np.nan, "d4_m": np.nan,
                "b1_deg": np.nan, "b2_deg": np.nan, "b3_deg": np.nan, "b4_deg": np.nan,
                "w1": 0.0, "w2": 0.0, "w3": 0.0, "w4": 0.0,
                "raw_w1": np.nan, "raw_w2": np.nan, "raw_w3": np.nan, "raw_w4": np.nan,
            })
            continue

        choice = case["choice_used"]
        padded_ids = case["padded_ids"]
        padded_dists = case["padded_dists"]
        padded_bears = case["padded_bears"]
        padded_weights = np.asarray(case["padded_weights"], dtype=float)
        final_weights = np.asarray(case["final_weights"], dtype=float)

        if padded_weights.size == 0:
            min_w = np.nan
            max_w = np.nan
            sum_w = np.nan
        else:
            min_w = float(np.min(padded_weights))
            max_w = float(np.max(padded_weights))
            sum_w = float(np.sum(padded_weights))

        rec = {
            "id": cid,
            "Latitude": target_lat,
            "Longitude": target_lon,
            "event": int(event_number),
            "corr_a_km": float(a_km),
            "corr_b": float(b),
            "n_gauges_requested": 4,
            "n_gauges_final": int(len(case["final_ids"])),
            "candidate_count": int(len(candidates)),
            "stage_used": str(case["stage_used"]),
            "forced_near_dist_m": float(choice["forced_near_dist_m"]),
            "min_ang_sep_deg": float(choice["min_sep_deg"]),
            "sum_dist_m": float(choice["sum_dist_m"]),
            "mean_dist_m": float(choice["mean_dist_m"]),
            "solver": str(case["solver"]),
            "solver_history": str(case["solver_history"]),
            "replacement_history": str(case["replacement_history"]),
            "n_negative_initial": np.nan,
            "n_negative_final": int(case["n_negative_final"]),
            "n_negative_weights": int(np.sum(padded_weights < 0)),
            "min_weight": min_w,
            "max_weight": max_w,
            "sum_weights": sum_w,
            "refit_performed": True,
            "dropped_negative_ids": "",
            "remarks": str(case["remarks"]),
            "forced_near_sid": str(case["choice_used"]["ids"][0]) if len(case["choice_used"]["ids"]) > 0 else "",
        }

        for k in range(1, 5):
            rec[f"g{k}"] = str(padded_ids[k - 1]) if k <= len(padded_ids) else ""
            rec[f"d{k}_m"] = float(padded_dists[k - 1]) if k <= len(padded_dists) and pd.notna(padded_dists[k - 1]) else np.nan
            rec[f"b{k}_deg"] = float(padded_bears[k - 1]) if k <= len(padded_bears) and pd.notna(padded_bears[k - 1]) else np.nan
            rec[f"w{k}"] = float(padded_weights[k - 1]) if k <= len(padded_weights) else 0.0

        for k in range(1, 5):
            rec[f"raw_w{k}"] = np.nan

        results.append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"Event_{event_number}_weights.csv"
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
    ap.add_argument("--n-gauges", type=int, default=4, choices=[3, 4], help="Kept for compatibility, but hierarchical mode always starts from nearest 4 then falls back to nearest 3")
    ap.add_argument("--base-dir", type=Path, default=BASE_DIR)
    ap.add_argument("--event-meta-dir", type=Path, default=CORRELATION_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--nearest-file", type=Path, default=NEAREST_CSV)
    ap.add_argument("--grid-file", type=Path, default=GRID_CSV)
    ap.add_argument("--station-file", type=Path, default=STATIONS_CSV)
    ap.add_argument("--nugget", type=float, default=0.0)
    ap.add_argument("--search-radius-km", type=float, default=POOL2, help="Used only if nearest-file is missing or incomplete")
    ap.add_argument("--keep-n", type=int, default=10, help="Used only if nearest-file is missing or incomplete")
    args = ap.parse_args()

    for ev in args.event:
        print("=" * 80)
        print(f"Running event {ev} with staged OK search and final IDW fallback")
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