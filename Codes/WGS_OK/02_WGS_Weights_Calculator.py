#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import ast
import itertools

import numpy as np
import pandas as pd

from geo_utils import haversine_km, ang_sep_deg


BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK")
DEP_DIR = BASE_DIR / "dependent_files"
OUT_DIR = BASE_DIR / "02_OK_Weights"
correlation_dir = BASE_DIR / "01_Event_TimeSeries"

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



def pick4_max_spread(ids, dists, bears):
    valid = [
        i for i in range(len(ids))
        if pd.notna(ids[i]) and pd.notna(dists[i]) and pd.notna(bears[i]) and str(ids[i]).strip() != ""
    ]
    if len(valid) < 4:
        return []
    if len(valid) == 4:
        return valid

    best = None
    best_score = (-1.0, np.inf)
    for combo in itertools.combinations(valid, 4):
        seps = [ang_sep_deg(bears[i], bears[j]) for i, j in itertools.combinations(combo, 2)]
        score = (min(seps), sum(float(dists[i]) for i in combo))
        if (score[0] > best_score[0]) or (score[0] == best_score[0] and score[1] < best_score[1]):
            best_score = score
            best = list(combo)
    return best or []

def choose_nonnegative_weights_with_fallback(
    stations_df,
    target_lat,
    target_lon,
    ids_f,
    dists_f,
    bears_f,
    a_km,
    b,
    nugget=0.0,
):
    """
    Heuristic:
    - start from spread-based 4-gauge choice
    - if any negative weights, drop the most negative station
      and replace it with the next available candidate
    - repeat until all nonnegative or no more replacements
    - if still negative, try 3-gauge combinations from filtered list
    - return always as 4 slots, padding slot 4 with blank/zero if 3-gauge fallback is used
    """

    n_all = len(ids_f)
    if n_all < 3:
        return None

    # ---------- try 4 gauges first ----------
    if n_all >= 4:
        chosen_idx = pick4_max_spread(ids_f, dists_f, bears_f)

        if len(chosen_idx) == 4:
            tried_sets = set()

            while True:
                chosen_tuple = tuple(sorted(chosen_idx))
                if chosen_tuple in tried_sets:
                    break
                tried_sets.add(chosen_tuple)

                chosen_ids = [ids_f[i] for i in chosen_idx]
                chosen_d = [dists_f[i] for i in chosen_idx]
                chosen_b = [bears_f[i] for i in chosen_idx]

                try:
                    weights = compute_ok_weights(
                        stations_df, target_lat, target_lon, chosen_ids, a_km, b, nugget=nugget
                    )
                except Exception:
                    break

                if np.all(weights >= 0):
                    return {
                        "ids": chosen_ids,
                        "dists": chosen_d,
                        "bears": chosen_b,
                        "weights": weights,
                        "n_used": 4,
                        "all_nonnegative": True,
                    }

                # drop the most negative station
                worst_local = int(np.argmin(weights))
                worst_global = chosen_idx[worst_local]

                # replacement candidates = filtered stations not already chosen
                remaining = [i for i in range(n_all) if i not in chosen_idx]

                # keep original filtered order, pick the next available one
                replacement = None
                for cand in remaining:
                    replacement = cand
                    break

                if replacement is None:
                    break

                # replace worst with next candidate
                chosen_idx = [replacement if i == worst_global else i for i in chosen_idx]

                # ensure uniqueness
                if len(set(chosen_idx)) < 4:
                    break

    # ---------- fallback: try 3 gauges ----------
    best3 = None
    best3_score = None

    for combo in itertools.combinations(range(n_all), 3):
        combo_ids = [ids_f[i] for i in combo]
        combo_d = [dists_f[i] for i in combo]
        combo_b = [bears_f[i] for i in combo]

        try:
            w3 = compute_ok_weights(
                stations_df, target_lat, target_lon, combo_ids, a_km, b, nugget=nugget
            )
        except Exception:
            continue

        neg_penalty = float(np.sum(np.abs(w3[w3 < 0]))) if np.any(w3 < 0) else 0.0
        min_w = float(np.min(w3))
        score = (-neg_penalty, min_w, -float(np.sum(combo_d)))

        if (best3 is None) or (score > best3_score):
            best3 = (combo_ids, combo_d, combo_b, w3)
            best3_score = score

            if np.all(w3 >= 0):
                break

    if best3 is None:
        return None

    combo_ids, combo_d, combo_b, w3 = best3

    # pad to 4 slots for downstream compatibility
    # choose a real 4th gauge with zero weight so downstream code still finds a station
    extra_idx = None
    for i in range(n_all):
        if ids_f[i] not in combo_ids:
            extra_idx = i
            break

    if extra_idx is not None:
        extra_id = ids_f[extra_idx]
        extra_d = dists_f[extra_idx]
        extra_b = bears_f[extra_idx]
    else:
        # fallback: reuse the first chosen gauge with zero weight
        extra_id = combo_ids[0]
        extra_d = combo_d[0]
        extra_b = combo_b[0]

    ids_out = combo_ids + [extra_id]
    d_out = combo_d + [extra_d]
    b_out = combo_b + [extra_b]
    w_out = np.array(list(w3) + [0.0], dtype=float)
    return {
        "ids": ids_out,
        "dists": d_out,
        "bears": b_out,
        "weights": w_out,
        "n_used": 3,
        "all_nonnegative": bool(np.all(w3 >= 0)),
    }

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



def run_event(event_number: int, event_meta_dir: Path, neighbor_file: Path, station_file: Path, out_dir: Path, nugget: float):
    event_file = event_meta_dir / f"Event_{event_number}_Stations_correlation.csv"
    if not event_file.exists():
        raise FileNotFoundError(f"Missing event metadata: {event_file}")
    if not neighbor_file.exists():
        raise FileNotFoundError(f"Missing neighbor file: {neighbor_file}")
    if not station_file.exists():
        raise FileNotFoundError(f"Missing stations file: {station_file}")

    event_df = pd.read_csv(event_file)
    a_km = float(event_df["corr_a_km"].iloc[0])
    b = float(event_df["corr_b"].iloc[0])
    selected = parse_selected_station_ids(event_df["stations_selected"].iloc[0])

    print("corr_a_km:", a_km)
    print("corr_b:", b)

    print("Number of selected stations:", len(selected))
    print("First 20 selected stations:", list(selected)[:20])
        
    nei = pd.read_csv(neighbor_file)
    stations = pd.read_csv(station_file)
    stations["ID"] = stations["ID"].apply(norm_station_id)
    

    results = []
    for _, row in nei.iterrows():
        target_id = str(row["id"])
        target_lat = float(row["Latitude"])
        target_lon = float(row["Longitude"])

        ids = [row.get(f"g{i}", "") for i in range(1, 11)]
        dists = [row.get(f"d{i}_m", np.nan) for i in range(1, 11)]
        bears = [row.get(f"b{i}_deg", np.nan) for i in range(1, 11)]

        filtered = []
        for i in range(len(ids)):
            sid = norm_station_id(ids[i])
            if sid == "":
                continue
            if sid not in selected:
                continue
            filtered.append((i, sid, float(dists[i]), float(bears[i])))

        if len(filtered) < 4:
            continue

        ids_f = [t[1] for t in filtered]
        dists_f = [t[2] for t in filtered]
        bears_f = [t[3] for t in filtered]

        choice = choose_nonnegative_weights_with_fallback(
            stations_df=stations,
            target_lat=target_lat,
            target_lon=target_lon,
            ids_f=ids_f,
            dists_f=dists_f,
            bears_f=bears_f,
            a_km=a_km,
            b=b,
            nugget=nugget,
        )

        if choice is None:
            continue

        chosen_ids = choice["ids"]
        chosen_d = choice["dists"]
        chosen_b = choice["bears"]
        weights = choice["weights"]

        rec = {"id": target_id, "Latitude": target_lat, "Longitude": target_lon}
        for k in range(4):
            rec[f"g{k+1}"] = chosen_ids[k]
            rec[f"d{k+1}_m"] = chosen_d[k]
            rec[f"b{k+1}_deg"] = chosen_b[k]
            rec[f"w{k+1}"] = float(weights[k])

        rec["sum_w"] = float(np.sum(weights))
        rec["n_gauges_used"] = int(choice["n_used"])
        rec["all_nonnegative"] = bool(choice["all_nonnegative"])
        rec["min_weight"] = float(np.min(weights))
        results.append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"Event_{event_number}_weights.csv"
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute ordinary kriging weights on a WGS84 grid.")
    parser.add_argument("--event", type=int, required=True)
    parser.add_argument("--event-meta-dir", default=str(correlation_dir))
    parser.add_argument("--neighbor-file", default=str(DEP_DIR / "grid_nearest10_spread_wgs84.csv"))
    parser.add_argument("--station-file", default=str(DEP_DIR / "Stations_df.csv"))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--nugget", type=float, default=0.0)
    args = parser.parse_args()

    run_event(
        event_number=args.event,
        event_meta_dir=Path(args.event_meta_dir),
        neighbor_file=Path(args.neighbor_file),
        station_file=Path(args.station_file),
        out_dir=Path(args.out_dir),
        nugget=args.nugget,
    )
