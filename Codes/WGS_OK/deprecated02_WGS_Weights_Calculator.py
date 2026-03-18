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

def compute_idw_weights(dists_m, power=2.0):
    d = np.asarray(dists_m, dtype=float)

    if np.any(d <= 0):
        w = np.zeros_like(d, dtype=float)
        w[np.argmin(d)] = 1.0
        return w

    inv = 1.0 / np.power(d, power)
    return inv / np.sum(inv)

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
    Rule:
    - For 4 gauges: choose the all-nonnegative OK solution with best spread.
    - If none exists, for 3 gauges: choose the all-nonnegative OK solution with best spread.
    - If still none exists, use IDW on the best-spread set:
        * best-spread 4 gauges if available
        * otherwise best-spread 3 gauges padded with a real 4th gauge with zero weight
    """

    n_all = len(ids_f)
    if n_all < 3:
        return None
    candidates = []
    nearest_idx = int(np.argmin(dists_f))
    # ---------------------------------------------------------
    # best-spread combinations (used also for IDW fallback)
    # ---------------------------------------------------------
    best_spread_4 = choose_best_spread_combo(ids_f, dists_f, bears_f, 4) if n_all >= 4 else None
    best_spread_3 = choose_best_spread_combo(ids_f, dists_f, bears_f, 3)

    # ---------------------------------------------------------
    # Try all 4-gauge combinations: keep nonnegative one with best spread
    # ---------------------------------------------------------
    best4_positive = None
    best4_positive_score = None

    if n_all >= 4:
        for combo in itertools.combinations(range(n_all), 4):
            if nearest_idx not in combo:
                continue
            chosen_ids = [ids_f[i] for i in combo]
            chosen_d = [dists_f[i] for i in combo]
            chosen_b = [bears_f[i] for i in combo]

            try:
                weights = compute_ok_weights(
                    stations_df, target_lat, target_lon, chosen_ids, a_km, b, nugget=nugget
                )
            except Exception:
                continue

            if np.all(weights >= 0):
                seps = [
                    ang_sep_deg(chosen_b[i], chosen_b[j])
                    for i, j in itertools.combinations(range(4), 2)
                ]
                min_sep = float(min(seps))
                sum_dist = float(np.sum(chosen_d))
                min_weight = float(np.min(weights))

                score = (-sum_dist, min_sep)

                if (best4_positive is None) or (score > best4_positive_score):
                    best4_positive = {
                        "ids": chosen_ids,
                        "dists": chosen_d,
                        "bears": chosen_b,
                        "weights": np.array(weights, dtype=float),
                        "n_used": 4,
                        "all_nonnegative": True,
                        "n_negative_weights": 0,
                        "neg_penalty": 0.0,
                        "min_weight": min_weight,
                        "remarks": "ok_4gauges_nonnegative_bestspread",
                        "weight_method": "ordinary_kriging",
                    }
                    best4_positive_score = score
        

        if best4_positive is not None:
            candidates.append(best4_positive)

    # ---------------------------------------------------------
    # Try all 3-gauge combinations: keep nonnegative one with best spread
    # ---------------------------------------------------------
    best3_positive = None
    best3_positive_score = None

    for combo in itertools.combinations(range(n_all), 3):
        if nearest_idx not in combo:
            continue
        combo_ids = [ids_f[i] for i in combo]
        combo_d = [dists_f[i] for i in combo]
        combo_b = [bears_f[i] for i in combo]

        try:
            w3 = compute_ok_weights(
                stations_df, target_lat, target_lon, combo_ids, a_km, b, nugget=nugget
            )
        except Exception:
            continue

        if np.all(w3 >= 0):
            seps = [
                ang_sep_deg(combo_b[i], combo_b[j])
                for i, j in itertools.combinations(range(3), 2)
            ]
            min_sep = float(min(seps))
            sum_dist = float(np.sum(combo_d))
            min_weight = float(np.min(w3))

            score = (min_sep, -sum_dist)

            if (best3_positive is None) or (score > best3_positive_score):
                # pad with a real 4th gauge with zero weight
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
                    extra_id = combo_ids[0]
                    extra_d = combo_d[0]
                    extra_b = combo_b[0]

                ids_out = combo_ids + [extra_id]
                d_out = combo_d + [extra_d]
                b_out = combo_b + [extra_b]
                w_out = np.array(list(w3) + [0.0], dtype=float)

                best3_positive = {
                    "ids": ids_out,
                    "dists": d_out,
                    "bears": b_out,
                    "weights": w_out,
                    "n_used": 3,
                    "all_nonnegative": True,
                    "n_negative_weights": 0,
                    "neg_penalty": 0.0,
                    "min_weight": min_weight,
                    "remarks": "ok_3gauges_nonnegative_bestspread",
                    "weight_method": "ordinary_kriging",
                }
                best3_positive_score = score

    if best3_positive is not None:
        candidates.append(best3_positive)
    
    # ---------------------------------------------------------
    # FINAL CHOICE: compare 3 vs 4 gauges
    # ---------------------------------------------------------
    if len(candidates) > 0:

        def final_score(c):
            return (
                -np.sum(c["dists"]),   # prioritize closer gauges
                min(
                    ang_sep_deg(c["bears"][i], c["bears"][j])
                    for i, j in itertools.combinations(range(c["n_used"]), 2)
                ),
                c["n_used"]  # slight preference for 4 over 3
            )

        return max(candidates, key=final_score)
    # ---------------------------------------------------------
    # Final fallback: IDW using best-spread stations
    # ---------------------------------------------------------
    if best_spread_4 is not None:
        w_idw = compute_idw_weights(best_spread_4["dists"], power=2.0)

        return {
            "ids": best_spread_4["ids"],
            "dists": best_spread_4["dists"],
            "bears": best_spread_4["bears"],
            "weights": np.array(w_idw, dtype=float),
            "n_used": 4,
            "all_nonnegative": True,
            "n_negative_weights": 0,
            "neg_penalty": 0.0,
            "min_weight": float(np.min(w_idw)),
            "remarks": "idw_fallback_4gauges_bestspread",
            "weight_method": "idw",
        }

    if best_spread_3 is not None:
        extra_idx = None
        for i in range(n_all):
            if ids_f[i] not in best_spread_3["ids"]:
                extra_idx = i
                break

        if extra_idx is not None:
            extra_id = ids_f[extra_idx]
            extra_d = dists_f[extra_idx]
            extra_b = bears_f[extra_idx]
        else:
            extra_id = best_spread_3["ids"][0]
            extra_d = best_spread_3["dists"][0]
            extra_b = best_spread_3["bears"][0]

        w3_idw = compute_idw_weights(best_spread_3["dists"], power=2.0)

        return {
            "ids": best_spread_3["ids"] + [extra_id],
            "dists": best_spread_3["dists"] + [extra_d],
            "bears": best_spread_3["bears"] + [extra_b],
            "weights": np.array(list(w3_idw) + [0.0], dtype=float),
            "n_used": 3,
            "all_nonnegative": True,
            "n_negative_weights": 0,
            "neg_penalty": 0.0,
            "min_weight": float(np.min(w3_idw)),
            "remarks": "idw_fallback_3gauges_bestspread",
            "weight_method": "idw",
        }

    return None

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

def choose_best_spread_combo(ids_f, dists_f, bears_f, n_choose):
    """
    Choose the combination with:
    1) minimum total distance
    2) then maximum minimum angular separation
    """
    if len(ids_f) < n_choose:
        return None

    best = None
    best_score = None

    for combo in itertools.combinations(range(len(ids_f)), n_choose):
        combo_ids = [ids_f[i] for i in combo]
        combo_d = [dists_f[i] for i in combo]
        combo_b = [bears_f[i] for i in combo]

        seps = [
            ang_sep_deg(combo_b[i], combo_b[j])
            for i, j in itertools.combinations(range(n_choose), 2)
        ]
        min_sep = float(min(seps))
        sum_dist = float(np.sum(combo_d))

        score = (-sum_dist, min_sep)

        if (best is None) or (score > best_score):
            best = {
                "ids": combo_ids,
                "dists": combo_d,
                "bears": combo_b,
                "min_sep": min_sep,
                "sum_dist": sum_dist,
            }
            best_score = score

    return best

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

        if len(filtered) < 3:
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
        weight_method = "ordinary_kriging"
        remarks = choice["remarks"]

        if np.any(weights < 0):
            weights = compute_idw_weights(choice["dists"], power=2.0)
            weight_method = "idw_fallback_due_to_negative_ok_weights"
            remarks = "negative_ok_weights_replaced_with_idw"
            
        rec = {
            "id": target_id,
            "Latitude": target_lat,
            "Longitude": target_lon,
        }

        for k in range(4):
            rec[f"g{k+1}"] = chosen_ids[k]
            rec[f"d{k+1}_m"] = chosen_d[k]
            rec[f"b{k+1}_deg"] = chosen_b[k]
            rec[f"w{k+1}"] = float(weights[k])

        rec["sum_w"] = float(np.sum(weights))
        rec["n_gauges_used"] = int(choice["n_used"])
        rec["all_nonnegative"] = bool(choice["all_nonnegative"])
        rec["n_negative_weights"] = int(choice["n_negative_weights"])
        rec["neg_penalty"] = float(choice["neg_penalty"])
        rec["min_weight"] = float(choice["min_weight"])
        rec["remarks"] = str(remarks)
        rec["weight_method"] = str(weight_method)

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
