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


# -----------------------------
# Utilities
# -----------------------------

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


def fix_small_negative_weights(weights, tol=0.1):
    w = np.asarray(weights, dtype=float).copy()
    small_neg = (w < 0) & (w >= -tol)
    if np.any(w < -tol):
        return w, False
    w[small_neg] = 0.0
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


def min_pairwise_sep_from_bears(bears):
    bears = [float(x) for x in bears]
    if len(bears) < 2:
        return 360.0
    return float(min(
        ang_sep_deg(bears[i], bears[j])
        for i, j in itertools.combinations(range(len(bears)), 2)
    ))


def convert_groups_from_row(row, selected_station_ids, inner_radius_km=7.0, min_within_inner_radius=2):
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
                "sum_dist_m": float(np.sum(dists)),
                "min_dist_m": float(np.min(dists)),
                "mean_dist_m": float(np.mean(dists)),
                "min_sep_deg": float(min_pairwise_sep_from_bears(bears)),
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


# -----------------------------
# Candidate evaluation
# -----------------------------

def evaluate_candidate_groups(stations_df, target_lat, target_lon, groups, a_km, b, nugget=0.0, negative_tol=0.1, include_idw=True):
    if not groups:
        return []

    forced_near_sid = None
    forced_near_dist = None
    for g in groups:
        for sid, dist in zip(g["ids"], g["dists"]):
            dist = float(dist)
            if (forced_near_dist is None) or (dist < forced_near_dist):
                forced_near_dist = dist
                forced_near_sid = sid

    if forced_near_sid is not None:
        filtered = [g for g in groups if forced_near_sid in g["ids"]]
        if filtered:
            groups = filtered

    accepted = []
    for g in groups:
        try:
            w_raw = compute_ok_weights(stations_df, target_lat, target_lon, g["ids"], a_km, b, nugget=nugget)
        except Exception:
            continue

        had_small_neg = np.any((w_raw < 0) & (w_raw >= -negative_tol))
        w_fixed, ok = fix_small_negative_weights(w_raw, tol=negative_tol)
        if not ok:
            continue

        rec = dict(g)
        rec["weights"] = np.asarray(w_fixed, dtype=float)
        rec["raw_weights"] = np.asarray(w_raw, dtype=float)
        rec["weight_method"] = "ordinary_kriging"
        rec["has_forced_near"] = bool(forced_near_sid is not None and forced_near_sid in g["ids"])
        rec["forced_near_sid"] = forced_near_sid if forced_near_sid is not None else ""
        rec["forced_near_dist_m"] = float(forced_near_dist) if forced_near_dist is not None else np.nan
        rec["clean_positive"] = bool(np.all(w_fixed > 0))
        rec["corrected_small_negative"] = bool(had_small_neg)
        rec["all_nonnegative"] = bool(np.all(w_fixed >= 0))
        rec["n_negative_raw"] = int(np.sum(w_raw < 0))
        rec["min_raw_weight"] = float(np.min(w_raw))
        rec["sum_w"] = float(np.sum(w_fixed))
        if np.all(w_fixed > 0):
            rec["remarks"] = f"ok_allpositive_{g['group_type']}"
        elif had_small_neg:
            rec["remarks"] = f"ok_corrected_{g['group_type']}"
        else:
            rec["remarks"] = f"ok_with_zero_{g['group_type']}"
        accepted.append(rec)

    if include_idw:
        for g in groups:
            rec = dict(g)
            rec["weights"] = compute_idw_weights(g["dists"], power=2.0)
            rec["raw_weights"] = rec["weights"].copy()
            rec["weight_method"] = "idw"
            rec["has_forced_near"] = bool(forced_near_sid is not None and forced_near_sid in g["ids"])
            rec["forced_near_sid"] = forced_near_sid if forced_near_sid is not None else ""
            rec["forced_near_dist_m"] = float(forced_near_dist) if forced_near_dist is not None else np.nan
            rec["clean_positive"] = bool(np.all(rec["weights"] > 0))
            rec["corrected_small_negative"] = False
            rec["all_nonnegative"] = True
            rec["n_negative_raw"] = 0
            rec["min_raw_weight"] = float(np.min(rec["weights"]))
            rec["sum_w"] = float(np.sum(rec["weights"]))
            rec["remarks"] = f"idw_fallback_{g['group_type']}"
            accepted.append(rec)

    dedup = []
    seen = set()
    for rec in accepted:
        key = tuple(rec["ids"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(rec)
    return dedup


# -----------------------------
# Ranking rules
# -----------------------------

def rank_default(rec):
    return (
        bool(rec["has_forced_near"]),
        bool(rec["clean_positive"]),
        bool(rec["corrected_small_negative"]),
        -float(rec["sum_dist_m"]),
        -float(rec["min_dist_m"]),
        -float(rec["mean_dist_m"]),
        int(rec["n_used"]),
        float(rec["min_sep_deg"]),
        rec["weight_method"] == "ordinary_kriging",
    )


def rank_positive_first(rec):
    return (
        bool(rec["clean_positive"]),
        bool(rec["corrected_small_negative"]),
        bool(rec["has_forced_near"]),
        rec["weight_method"] == "ordinary_kriging",
        -float(rec["sum_dist_m"]),
        -float(rec["min_dist_m"]),
        -float(rec["mean_dist_m"]),
        float(rec["min_sep_deg"]),
        int(rec["n_used"]),
    )


def rank_corrected_first(rec):
    return (
        bool(rec["corrected_small_negative"]),
        bool(rec["clean_positive"]),
        bool(rec["has_forced_near"]),
        rec["weight_method"] == "ordinary_kriging",
        -float(rec["sum_dist_m"]),
        -float(rec["min_dist_m"]),
        -float(rec["mean_dist_m"]),
        float(rec["min_sep_deg"]),
        int(rec["n_used"]),
    )


def rank_sumdist_first(rec):
    return (
        -float(rec["sum_dist_m"]),
        bool(rec["clean_positive"]),
        bool(rec["corrected_small_negative"]),
        bool(rec["has_forced_near"]),
        -float(rec["min_dist_m"]),
        -float(rec["mean_dist_m"]),
        float(rec["min_sep_deg"]),
        int(rec["n_used"]),
        rec["weight_method"] == "ordinary_kriging",
    )


def rank_minsep_first(rec):
    return (
        float(rec["min_sep_deg"]),
        bool(rec["clean_positive"]),
        bool(rec["corrected_small_negative"]),
        bool(rec["has_forced_near"]),
        -float(rec["sum_dist_m"]),
        -float(rec["min_dist_m"]),
        -float(rec["mean_dist_m"]),
        int(rec["n_used"]),
        rec["weight_method"] == "ordinary_kriging",
    )


RANKING_RULES = [
    ("default", rank_default),
    ("positive_first", rank_positive_first),
    ("corrected_first", rank_corrected_first),
    ("sumdist_first", rank_sumdist_first),
    ("minsep_first", rank_minsep_first),
]


def choose_option_set(candidates, max_options):
    chosen = []
    used_group_keys = set()

    for rule_name, rule_fn in RANKING_RULES:
        ordered = sorted(candidates, key=rule_fn, reverse=True)
        for rec in ordered:
            group_key = tuple(rec["ids"])
            if group_key in used_group_keys:
                continue
            out = dict(rec)
            out["ranking_rule"] = rule_name
            chosen.append(out)
            used_group_keys.add(group_key)
            break
        if len(chosen) >= max_options:
            return chosen[:max_options]

    if len(chosen) < max_options:
        ordered = sorted(candidates, key=rank_default, reverse=True)
        for rec in ordered:
            group_key = tuple(rec["ids"])
            if group_key in used_group_keys:
                continue
            out = dict(rec)
            out["ranking_rule"] = "default_extra"
            chosen.append(out)
            used_group_keys.add(group_key)
            if len(chosen) >= max_options:
                break

    return chosen[:max_options]


# -----------------------------
# Main
# -----------------------------

def run_event(event_number: int, event_meta_dir: Path, neighbor_file: Path, station_file: Path, out_dir: Path,
              nugget: float, inner_radius_km: float, min_within_inner_radius: int,
              max_options: int, negative_tol: float, include_idw: bool):
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

    rows = []
    for _, row in nei.iterrows():
        target_id = str(row["id"])
        target_lat = float(row["Latitude"])
        target_lon = float(row["Longitude"])

        groups = convert_groups_from_row(
            row,
            selected_station_ids=selected,
            inner_radius_km=inner_radius_km,
            min_within_inner_radius=min_within_inner_radius,
        )
        if not groups:
            continue

        candidates = evaluate_candidate_groups(
            stations_df=stations,
            target_lat=target_lat,
            target_lon=target_lon,
            groups=groups,
            a_km=a_km,
            b=b,
            nugget=nugget,
            negative_tol=negative_tol,
            include_idw=include_idw,
        )
        if not candidates:
            continue

        selected_options = choose_option_set(candidates, max_options=max_options)
        for i, rec in enumerate(selected_options, start=1):
            out = {
                "id": target_id,
                "Latitude": target_lat,
                "Longitude": target_lon,
                "option_rank": i,
                "ranking_rule": rec["ranking_rule"],
                "group_type": rec["group_type"],
                "weight_method": rec["weight_method"],
                "remarks": rec["remarks"],
                "has_forced_near": bool(rec["has_forced_near"]),
                "forced_near_sid": rec["forced_near_sid"],
                "forced_near_dist_m": float(rec["forced_near_dist_m"]),
                "n_gauges_used": int(rec["n_used"]),
                "sum_dist_m": float(rec["sum_dist_m"]),
                "min_dist_m": float(rec["min_dist_m"]),
                "mean_dist_m": float(rec["mean_dist_m"]),
                "min_sep_deg": float(rec["min_sep_deg"]),
                "all_nonnegative": bool(rec["all_nonnegative"]),
                "clean_positive": bool(rec["clean_positive"]),
                "corrected_small_negative": bool(rec["corrected_small_negative"]),
                "n_negative_raw": int(rec["n_negative_raw"]),
                "min_raw_weight": float(rec["min_raw_weight"]),
                "sum_w": float(rec["sum_w"]),
            }
            ids = list(rec["ids"])
            dists = list(rec["dists"])
            bears = list(rec["bears"])
            weights = list(np.asarray(rec["weights"], dtype=float))
            while len(ids) < 4:
                ids.append("")
                dists.append(np.nan)
                bears.append(np.nan)
                weights.append(0.0)
            for k in range(4):
                out[f"g{k+1}"] = ids[k]
                out[f"d{k+1}_m"] = float(dists[k]) if pd.notna(dists[k]) else np.nan
                out[f"b{k+1}_deg"] = float(bears[k]) if pd.notna(bears[k]) else np.nan
                out[f"w{k+1}"] = float(weights[k])
            rows.append(out)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"Event_{event_number}_weight_options.csv"
    pd.DataFrame(rows).to_csv(out_file, index=False)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build fallback weight options for each centroid.")
    parser.add_argument("--event", type=int, required=True)
    parser.add_argument("--event-meta-dir", default=str(CORRELATION_DIR))
    parser.add_argument("--neighbor-file", default=str(DEP_DIR / "grid_grouped_candidates_wgs84.csv"))
    parser.add_argument("--station-file", default=str(DEP_DIR / "Stations_df.csv"))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--nugget", type=float, default=0.0)
    parser.add_argument("--inner-radius-km", type=float, default=7.0)
    parser.add_argument("--min-within-inner-radius", type=int, default=2)
    parser.add_argument("--max-options", type=int, default=5)
    parser.add_argument("--negative-tol", type=float, default=0.1)
    parser.add_argument("--no-idw", action="store_true")
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
        max_options=args.max_options,
        negative_tol=args.negative_tol,
        include_idw=not args.no_idw,
    )
