#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import ast
import json

import numpy as np
import pandas as pd

BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_IDW")
DEP_DIR = BASE_DIR / "dependent_files"
OUT_DIR = BASE_DIR / "02_IDW_Weights"
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


def parse_selected_station_ids(raw_value):
    raw = str(raw_value).strip()
    if raw.startswith("[") or raw.startswith("("):
        parsed = ast.literal_eval(raw)
        return {norm_station_id(x) for x in parsed if norm_station_id(x) != ""}
    return {norm_station_id(x) for x in raw.split(",") if norm_station_id(x) != ""}


def compute_idw_weights(dists_m, power=2.0):
    d = np.asarray(dists_m, dtype=float)
    if d.size == 0:
        return np.array([], dtype=float)
    if np.any(d <= 0):
        w = np.zeros_like(d, dtype=float)
        w[np.argmin(d)] = 1.0
        return w
    inv = 1.0 / np.power(d, power)
    return inv / np.sum(inv)


def pad_to_four(ids, dists_m, bears_deg, weights):
    ids = list(ids)
    dists_m = list(dists_m)
    bears_deg = list(bears_deg)
    weights = list(np.asarray(weights, dtype=float))

    n_used = len(ids)
    while len(ids) < 4:
        ids.append("")
        dists_m.append(np.nan)
        bears_deg.append(np.nan)
        weights.append(0.0)

    return ids[:4], dists_m[:4], bears_deg[:4], np.asarray(weights[:4], dtype=float), n_used


def extract_candidates_from_row(row, selected_station_ids):
    cand_ids = [norm_station_id(x) for x in safe_json_loads(row.get("candidate_ids", ""), [])]
    cand_dists = safe_json_loads(row.get("candidate_dists_m", ""), [])
    cand_bears = safe_json_loads(row.get("candidate_bears_deg", ""), [])

    candidates = []
    for sid, dist, bear in zip(cand_ids, cand_dists, cand_bears):
        if sid == "":
            continue
        if sid not in selected_station_ids:
            continue
        try:
            dist_f = float(dist)
            bear_f = float(bear)
        except Exception:
            continue
        if not np.isfinite(dist_f):
            continue
        candidates.append((sid, dist_f, bear_f))

    # de-duplicate while preserving nearest occurrence
    dedup = {}
    for sid, dist_f, bear_f in candidates:
        if sid not in dedup or dist_f < dedup[sid][0]:
            dedup[sid] = (dist_f, bear_f)

    out = [(sid, vals[0], vals[1]) for sid, vals in dedup.items()]
    out.sort(key=lambda x: x[1])
    return out


# -----------------------------
# Main logic
# -----------------------------

def run_event(event_number: int, event_meta_dir: Path, neighbor_file: Path, station_file: Path, out_dir: Path,
              nugget: float, inner_radius_km: float, min_within_inner_radius: int):
    event_file = event_meta_dir / f"Event_{event_number}_Stations_correlation.csv"
    if not event_file.exists():
        raise FileNotFoundError(f"Missing event metadata: {event_file}")
    if not neighbor_file.exists():
        raise FileNotFoundError(f"Missing grouped candidate file: {neighbor_file}")
    if not station_file.exists():
        raise FileNotFoundError(f"Missing stations file: {station_file}")

    event_df = pd.read_csv(event_file)
    if "stations_selected" not in event_df.columns:
        raise ValueError(f"Missing stations_selected column in {event_file}")
    selected = parse_selected_station_ids(event_df["stations_selected"].iloc[0])

    nei = pd.read_csv(neighbor_file)
    stations = pd.read_csv(station_file)
    stations["ID"] = stations["ID"].apply(norm_station_id)
    available_station_ids = set(stations["ID"].tolist())
    selected &= available_station_ids

    results = []
    for _, row in nei.iterrows():
        target_id = str(row["id"])
        target_lat = float(row["Latitude"])
        target_lon = float(row["Longitude"])

        candidates = extract_candidates_from_row(row, selected)
        if len(candidates) == 0:
            continue

        # Use 4 nearest if available, otherwise 3 if available, otherwise keep whatever exists.
        if len(candidates) >= 4:
            chosen = candidates[:4]
        elif len(candidates) >= 3:
            chosen = candidates[:3]
        else:
            chosen = candidates[:len(candidates)]

        chosen_ids = [x[0] for x in chosen]
        chosen_dists = [x[1] for x in chosen]
        chosen_bears = [x[2] for x in chosen]
        weights = compute_idw_weights(chosen_dists, power=2.0)

        chosen_ids4, chosen_d4, chosen_b4, weights4, n_used = pad_to_four(
            chosen_ids, chosen_dists, chosen_bears, weights
        )

        true_nearest_sid = chosen_ids[0] if len(chosen_ids) > 0 else ""
        true_nearest_dist = chosen_dists[0] if len(chosen_dists) > 0 else np.nan

        rec = {
            "id": target_id,
            "Latitude": target_lat,
            "Longitude": target_lon,
            "group_type": f"nearest_{n_used}_idw",
            "candidate_group_count": int(len(candidates)),
            "group_inner_radius_km": float(inner_radius_km),
            "group_min_within_inner_radius": int(min_within_inner_radius),
            "has_forced_near": True if n_used > 0 else False,
            "forced_near_sid": str(true_nearest_sid),
            "forced_near_dist_m": float(true_nearest_dist) if np.isfinite(true_nearest_dist) else np.nan,
        }

        for k in range(4):
            rec[f"g{k+1}"] = chosen_ids4[k]
            rec[f"d{k+1}_m"] = float(chosen_d4[k]) if pd.notna(chosen_d4[k]) else np.nan
            rec[f"b{k+1}_deg"] = float(chosen_b4[k]) if pd.notna(chosen_b4[k]) else np.nan
            rec[f"w{k+1}"] = float(weights4[k])

        rec["sum_w"] = float(np.sum(weights4))
        rec["n_gauges_used"] = int(n_used)
        rec["all_nonnegative"] = bool(np.all(weights4 >= 0))
        rec["n_negative_weights"] = int(np.sum(weights4 < 0))
        rec["neg_penalty"] = float(np.sum(np.abs(weights4[weights4 < 0])))
        rec["min_weight"] = float(np.min(weights4)) if len(weights4) else np.nan
        rec["remarks"] = f"idw_nearest_{n_used}"
        rec["weight_method"] = "idw"

        results.append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"Event_{event_number}_weights.csv"
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute IDW weights using the nearest available event-selected gauges.")
    parser.add_argument("--event", type=int, nargs="+", required=True)
    parser.add_argument("--event-meta-dir", default=str(CORRELATION_DIR))
    parser.add_argument("--neighbor-file", default=str(DEP_DIR / "grid_grouped_candidates_wgs84.csv"))
    parser.add_argument("--station-file", default=str(DEP_DIR / "Stations_df.csv"))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--nugget", type=float, default=0.0, help="Accepted for interface compatibility. Not used by IDW.")
    parser.add_argument("--inner-radius-km", type=float, default=7.0, help="Accepted for interface compatibility.")
    parser.add_argument("--min-within-inner-radius", type=int, default=2, help="Accepted for interface compatibility.")
    args = parser.parse_args()

    for ev in args.event:
        run_event(
            event_number=ev,
            event_meta_dir=Path(args.event_meta_dir),
            neighbor_file=Path(args.neighbor_file),
            station_file=Path(args.station_file),
            out_dir=Path(args.out_dir),
            nugget=args.nugget,
            inner_radius_km=args.inner_radius_km,
            min_within_inner_radius=args.min_within_inner_radius,
        )
