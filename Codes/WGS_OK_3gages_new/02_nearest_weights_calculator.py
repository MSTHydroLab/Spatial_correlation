#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from geo_utils import haversine_km

BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK_3gages_new")
DEP_DIR = BASE_DIR / "dependent_files"
EVENT_DIR = BASE_DIR / "01_Event_TimeSeries"
OUT_DIR = BASE_DIR / "02_OK_Weights"

GRID_NEAREST_CSV = DEP_DIR / "grid_nearest_gauges_wgs84.csv"
STATIONS_CSV = DEP_DIR / "Stations_df.csv"


def norm_station_id(x) -> str:
    s = str(x).strip().strip("'").strip('"')
    if s == "" or s.lower() == "nan":
        return ""
    try:
        return str(int(float(s)))
    except Exception:
        return s


def rho_powerexp(d_km, a_km, b):
    return np.exp(-((np.asarray(d_km, dtype=float) / float(a_km)) ** float(b)))


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

    return sol[:n], solver, float(np.linalg.cond(A))


def parse_listlike(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if s == "":
        return []
    try:
        vals = json.loads(s)
        return list(vals)
    except Exception:
        return [v.strip() for v in s.split(",") if v.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute ordinary kriging weights using nearest 3 or 4 gauges.")
    ap.add_argument("--event", type=int, required=True)
    ap.add_argument("--n-gauges", type=int, choices=[3, 4], default=4)
    ap.add_argument("--base-dir", default=str(BASE_DIR))
    ap.add_argument("--nearest-csv", default="")
    ap.add_argument("--stations-csv", default="")
    ap.add_argument("--event-meta-dir", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--nugget", type=float, default=0.0)
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    dep_dir = base_dir / "dependent_files"
    event_meta_dir = Path(args.event_meta_dir) if args.event_meta_dir else base_dir / "01_Event_TimeSeries"
    nearest_csv = Path(args.nearest_csv) if args.nearest_csv else dep_dir / "grid_nearest_gauges_wgs84.csv"
    stations_csv = Path(args.stations_csv) if args.stations_csv else dep_dir / "Stations_df.csv"
    out_dir = Path(args.out_dir) if args.out_dir else base_dir / "02_OK_Weights"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_fp = event_meta_dir / f"Event_{args.event}_Stations_correlation.csv"
    meta = pd.read_csv(meta_fp)
    a_km = float(meta["corr_a_km"].iloc[0])
    b = float(meta["corr_b"].iloc[0])

    nearest = pd.read_csv(nearest_csv)
    stn = pd.read_csv(stations_csv)
    stn["ID"] = stn["ID"].apply(norm_station_id)
    nearest["id"] = nearest["id"].astype(str)

    rows = []
    for _, row in nearest.iterrows():
        cid = str(row["id"])
        clat = float(row["Latitude"])
        clon = float(row["Longitude"])

        ids_all = [norm_station_id(row.get(f"g{k}", "")) for k in range(1, 11)]
        ids = [sid for sid in ids_all if sid != ""][:args.n_gauges]
        if len(ids) < args.n_gauges:
            continue

        weights, solver, condA = compute_ok_weights(
            stations_df=stn,
            target_lat=clat,
            target_lon=clon,
            ids=ids,
            a_km=a_km,
            b=b,
            nugget=float(args.nugget),
        )
        d_km = []
        bears = []
        st = stn.set_index("ID")
        for sid in ids:
            slat = float(st.loc[sid, "Latitude"])
            slon = float(st.loc[sid, "Longitude"])
            d_km.append(float(haversine_km(clat, clon, slat, slon)))
            # bearing not needed here; keep NaN-compatible output style simple
            bears.append(np.nan)

        rec = {
            "id": cid,
            "Latitude": clat,
            "Longitude": clon,
            "event": int(args.event),
            "n_gauges_used": int(args.n_gauges),
            "corr_a_km": a_km,
            "corr_b": b,
            "nugget": float(args.nugget),
            "solver": solver,
            "condA": condA,
            "n_negative_weights": int(np.sum(np.asarray(weights) < 0.0)),
            "min_weight": float(np.min(weights)),
            "has_negative_weight": bool(np.any(np.asarray(weights) < 0.0)),
            "gauge_ids_json": json.dumps(ids),
            "weights_json": json.dumps([float(x) for x in weights]),
        }
        for k in range(1, 5):
            rec[f"g{k}"] = ids[k - 1] if k <= len(ids) else ""
            rec[f"w{k}"] = float(weights[k - 1]) if k <= len(ids) else 0.0
            rec[f"dist{k}_km"] = float(d_km[k - 1]) if k <= len(d_km) else np.nan
        rows.append(rec)

    out = pd.DataFrame(rows)
    out_fp = out_dir / f"Event_{args.event}_nearest{args.n_gauges}_weights.csv"
    out.to_csv(out_fp, index=False)
    print(f"Saved: {out_fp}")
    print(f"Rows: {len(out)}")


if __name__ == "__main__":
    main()
