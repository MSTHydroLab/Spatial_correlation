#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import ast
import itertools
import json

import numpy as np
import pandas as pd

from geo_utils import haversine_km, ang_sep_deg

BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK")
DEP_DIR = BASE_DIR / "dependent_files"
OUT_DIR = BASE_DIR / "02_OK_Weights"
CORRELATION_DIR = BASE_DIR / "01_Event_TimeSeries"


def norm_station_id(x):
    s = str(x).strip().strip("'").strip('"')
    if s == "" or s.lower() == "nan":
        return ""
    try:
        return str(int(float(s)))
    except Exception:
        return s


def rho_powerexp(d_km, a_km, b):
    return np.exp(-((np.asarray(d_km, dtype=float) / float(a_km)) ** float(b)))


def compute_idw_weights(dists_m, power=2.0):
    d = np.asarray(dists_m, dtype=float)
    if np.any(d <= 0):
        w = np.zeros_like(d, dtype=float)
        w[np.argmin(d)] = 1.0
        return w
    inv = 1.0 / np.power(d, power)
    return inv / np.sum(inv)

def fix_small_negative_weights(weights, tol):
    w = np.asarray(weights, dtype=float).copy()

    # Identify small negatives
    small_neg = (w < 0) & (w >= -tol)

    # If there are any large negatives, do NOT fix here
    if np.any(w < -tol):
        return w, False  # not acceptable group

    # Clip small negatives to zero
    w[small_neg] = 0.0

    # Renormalize (only if sum > 0)
    s = w.sum()
    if s > 0:
        w = w / s

    return w, True

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

    try:
        sol = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)

    return sol[:n]


def parse_selected_station_ids(raw_value):
    raw = str(raw_value).strip()
    if raw.startswith("[") or raw.startswith("("):
        parsed = ast.literal_eval(raw)
        return {norm_station_id(x) for x in parsed}
    return {norm_station_id(x) for x in raw.split(",") if norm_station_id(x) != ""}


def safe_json_loads(x, default):
    if pd.isna(x):
        return default
    s = str(x).strip()
    if s == "":
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


def group_score(group):
    bears = [float(x) for x in group["bears"]]
    dists = [float(x) for x in group["dists"]]
    weights = np.asarray(group.get("weights", []), dtype=float)

    seps = [
        ang_sep_deg(bears[i], bears[j])
        for i, j in itertools.combinations(range(len(bears)), 2)
    ]
    min_sep = float(min(seps)) if seps else 360.0

    min_dist = float(np.min(dists)) if dists else np.inf
    mean_dist = float(np.mean(dists)) if dists else np.inf
    sum_dist = float(np.sum(dists)) if dists else np.inf
    n_used = int(group["n_used"])

    has_forced_near = bool(group.get("has_forced_near", False))

    # distinguish fully positive from corrected-small-negative
    status = str(group.get("remarks", ""))
    clean_positive = status.startswith("ok_allpositive")
    corrected_small_negative = status.startswith("ok_corrected")

    # New priority:
    # 1) forced near gauge included
    # 2) more gauges
    # 3) closer total distance
    # 4) closer nearest gauge
    # 5) smaller mean distance
    # 6) better spread
    # 7) prefer clean positive over corrected only as a weak tie-breaker
    return (
        has_forced_near,
        clean_positive,
        corrected_small_negative,
        -sum_dist,
        -min_dist,
        n_used,
        -mean_dist,
        min_sep,
    )

def convert_groups_from_row(row, selected_station_ids, inner_radius_km=5.0, min_within_inner_radius=2):
    groups = []
    for raw_col, gtype in [("quadrant_groups", "quadrant"), ("sector3_groups", "sector3")]:
        raw_groups = safe_json_loads(row.get(raw_col, ""), [])
        for g in raw_groups:
            ids = [norm_station_id(x) for x in g.get("ids", [])]
            dists = [float(x) for x in g.get("dists_m", [])]
            bears = [float(x) for x in g.get("bears_deg", [])]
            if not ids or len(ids) != len(dists) or len(ids) != len(bears):
                continue
            if any(sid == "" for sid in ids):
                continue
            if not all(sid in selected_station_ids for sid in ids):
                continue
            inner_count = int(np.sum(np.asarray(dists, dtype=float) <= float(inner_radius_km) * 1000.0))
            if inner_count < int(min_within_inner_radius):
                continue
            groups.append({
                "group_type": gtype,
                "ids": ids,
                "dists": dists,
                "bears": bears,
                "n_used": len(ids),
                "n_within_inner_radius": inner_count,
            })

    dedup = []
    seen = set()
    for g in groups:
        key = (g["group_type"], tuple(g["ids"]))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(g)
    return dedup


def choose_best_group_with_fallback(
    stations_df,
    target_lat,
    target_lon,
    groups,
    a_km,
    b,
    nugget=0.0,
):
    if not groups:
        return None

    # ---------------------------------------------------------
    # Always force the absolute nearest gauge among all candidate groups
    # ---------------------------------------------------------
    forced_near_sid = None
    forced_near_dist = None

    for g in groups:
        for sid, dist in zip(g["ids"], g["dists"]):
            dist = float(dist)
            if (forced_near_dist is None) or (dist < forced_near_dist):
                forced_near_dist = dist
                forced_near_sid = sid

    # Keep only groups containing the absolute nearest gauge
    if forced_near_sid is not None:
        filtered_groups = [g for g in groups if forced_near_sid in g["ids"]]
        if filtered_groups:
            groups = filtered_groups

    positive = []
    

    for g in groups:
        try:
            w = compute_ok_weights(
                stations_df, target_lat, target_lon, g["ids"], a_km, b, nugget=nugget
            )
        except Exception:
            continue
        neg_mask = w < 0
        neg_sum = float(np.sum(np.abs(w[neg_mask])))
        
        NEG_SUM_TOL = 0.1
        if np.any(w < -0.1):
            continue

        if neg_sum > 0.1:
            continue
        had_small_neg = np.any((w < 0) & (w >= -0.1))
        w_fixed, ok = fix_small_negative_weights(w, tol=0.09)

        if ok:
            w = w_fixed

            rec = dict(g)
            rec["weights"] = np.asarray(w, dtype=float)
            rec["weight_method"] = "ordinary_kriging"
            rec["has_forced_near"] = bool(forced_near_sid is not None and forced_near_sid in g["ids"])
            rec["forced_near_sid"] = forced_near_sid if forced_near_sid is not None else ""
            rec["forced_near_dist_m"] = float(forced_near_dist) if forced_near_dist is not None else np.nan

            if np.all(w > 0):
                rec["remarks"] = f"ok_allpositive_{g['group_type']}"
            elif had_small_neg:
                rec["remarks"] = f"ok_corrected_{g['group_type']}"
            else:
                rec["remarks"] = f"ok_with_zero_{g['group_type']}"

            positive.append(rec)

    if positive:
        return max(positive, key=group_score)

    # ---------------------------------------------------------
    # Fallback to IDW on the best geometry group
    # ---------------------------------------------------------
    idw_groups = []
    for g in groups:
        rec = dict(g)
        rec["weights"] = compute_idw_weights(g["dists"], power=2.0)
        rec["weight_method"] = "idw"
        rec["remarks"] = f"idw_fallback_{g['group_type']}"
        rec["has_forced_near"] = bool(forced_near_sid is not None and forced_near_sid in g["ids"])
        rec["forced_near_sid"] = forced_near_sid if forced_near_sid is not None else ""
        rec["forced_near_dist_m"] = float(forced_near_dist) if forced_near_dist is not None else np.nan
        idw_groups.append(rec)

    if idw_groups:
        return max(idw_groups, key=group_score)

    return None


def pad_to_four(choice, candidate_groups):
    ids = list(choice["ids"])
    dists = list(choice["dists"])
    bears = list(choice["bears"])
    weights = list(np.asarray(choice["weights"], dtype=float))

    if len(ids) == 4:
        return ids, dists, bears, np.asarray(weights, dtype=float), 4

    extra = None
    for g in candidate_groups:
        for sid, dist, bear in zip(g["ids"], g["dists"], g["bears"]):
            if sid not in ids:
                extra = (sid, dist, bear)
                break
        if extra is not None:
            break

    if extra is None:
        extra = (ids[0], dists[0], bears[0])

    ids.append(extra[0])
    dists.append(float(extra[1]))
    bears.append(float(extra[2]))
    weights.append(0.0)
    return ids, dists, bears, np.asarray(weights, dtype=float), len(choice["ids"])


def run_event(event_number: int, event_meta_dir: Path, neighbor_file: Path, station_file: Path, out_dir: Path, nugget: float, inner_radius_km: float, min_within_inner_radius: int):
    event_file = event_meta_dir / f"Event_{event_number}_Stations_correlation.csv"
    if not event_file.exists():
        raise FileNotFoundError(f"Missing event metadata: {event_file}")
    if not neighbor_file.exists():
        raise FileNotFoundError(f"Missing grouped candidate file: {neighbor_file}")
    if not station_file.exists():
        raise FileNotFoundError(f"Missing stations file: {station_file}")

    event_df = pd.read_csv(event_file)
    a_km = float(event_df["corr_a_km"].iloc[0])
    b = float(event_df["corr_b"].iloc[0])
    selected = parse_selected_station_ids(event_df["stations_selected"].iloc[0])

    nei = pd.read_csv(neighbor_file)
    stations = pd.read_csv(station_file)
    stations["ID"] = stations["ID"].apply(norm_station_id)

    results = []
    for _, row in nei.iterrows():
        target_id = str(row["id"])
        target_lat = float(row["Latitude"])
        target_lon = float(row["Longitude"])
        
        raw_candidate_ids = safe_json_loads(row.get("candidate_ids", ""), [])
        raw_candidate_dists = safe_json_loads(row.get("candidate_dists_m", ""), [])

        raw_pairs = []
        for sid, dist in zip(raw_candidate_ids, raw_candidate_dists):
            sid = norm_station_id(sid)
            try:
                dist = float(dist)
            except Exception:
                continue
            if sid != "":
                raw_pairs.append((sid, dist))

        if raw_pairs:
            true_nearest_sid, true_nearest_dist = min(raw_pairs, key=lambda x: x[1])
        else:
            true_nearest_sid, true_nearest_dist = "", np.nan
        groups = convert_groups_from_row(row, selected_station_ids=selected, inner_radius_km=inner_radius_km, min_within_inner_radius=min_within_inner_radius)
        if not groups:
            continue

        choice = choose_best_group_with_fallback(
            stations_df=stations,
            target_lat=target_lat,
            target_lon=target_lon,
            groups=groups,
            a_km=a_km,
            b=b,
            nugget=nugget,
        )
        if choice is None:
            continue
        print(
            f"centroid={target_id} "
            f"true_nearest={true_nearest_sid} ({true_nearest_dist:.2f} m), "
            f"forced_near={choice.get('forced_near_sid', '')} ({choice.get('forced_near_dist_m', np.nan):.2f} m), "
            f"chosen={choice['ids']}"
        )
        chosen_ids, chosen_d, chosen_b, weights, n_used = pad_to_four(choice, groups)
        print(target_id, "and", chosen_ids)
        rec = {
            "id": target_id,
            "Latitude": target_lat,
            "Longitude": target_lon,
            "group_type": choice["group_type"],
            "candidate_group_count": int(len(groups)),
            "group_inner_radius_km": float(inner_radius_km),
            "group_min_within_inner_radius": int(min_within_inner_radius),
            "has_forced_near": bool(choice.get("has_forced_near", False)),
            "forced_near_sid": str(choice.get("forced_near_sid", "")),
            "forced_near_dist_m": float(choice.get("forced_near_dist_m", np.nan)),
        }

        for k in range(4):
            rec[f"g{k+1}"] = chosen_ids[k]
            rec[f"d{k+1}_m"] = float(chosen_d[k])
            rec[f"b{k+1}_deg"] = float(chosen_b[k])
            rec[f"w{k+1}"] = float(weights[k])

        rec["sum_w"] = float(np.sum(weights))
        rec["n_gauges_used"] = int(n_used)
        rec["all_nonnegative"] = bool(np.all(weights >= 0))
        rec["n_negative_weights"] = int(np.sum(weights < 0))
        rec["neg_penalty"] = float(np.sum(np.abs(weights[weights < 0])))
        rec["min_weight"] = float(np.min(weights))
        rec["remarks"] = str(choice.get("remarks", ""))
        rec["weight_method"] = str(choice.get("weight_method", ""))

        results.append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"Event_{event_number}_weights.csv"
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute ordinary kriging weights using grouped station candidates.")
    parser.add_argument("--event", type=int, required=True)
    parser.add_argument("--event-meta-dir", default=str(CORRELATION_DIR))
    parser.add_argument("--neighbor-file", default=str(DEP_DIR / "grid_grouped_candidates_wgs84.csv"))
    parser.add_argument("--station-file", default=str(DEP_DIR / "Stations_df.csv"))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--nugget", type=float, default=0.0)
    parser.add_argument("--inner-radius-km", type=float, default=7.0)
    parser.add_argument("--min-within-inner-radius", type=int, default=2)
    args = parser.parse_args()

    run_event(
        event_number=args.event,
        event_meta_dir=Path(args.event_meta_dir),
        neighbor_file=Path(args.neighbor_file),
        station_file=Path(args.station_file),
        out_dir=Path(args.out_dir),
        nugget=args.nugget,
        inner_radius_km=args.inner_radius_km,
        min_within_inner_radius=args.min_within_inner_radius,
    )
