#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import ast

import numpy as np
import pandas as pd

from geo_utils import haversine_km, initial_bearing_deg, ang_sep_deg

BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK")
DEP_DIR = BASE_DIR / "dependent_files"
CORRELATION_DIR = BASE_DIR / "01_Event_TimeSeries"

GRID_CSV = DEP_DIR / "grid_centers_wgs84.csv"
STATIONS_CSV = DEP_DIR / "Stations_df.csv"


def norm_station_id(x):
    s = str(x).strip().strip("'").strip('"')
    if s == "" or s.lower() == "nan":
        return ""
    try:
        return str(int(float(s)))
    except Exception:
        return s


def parse_station_list(raw: str) -> list[str]:
    raw = str(raw).strip()
    if raw.startswith("[") or raw.startswith("("):
        vals = ast.literal_eval(raw)
        return [norm_station_id(v) for v in vals if norm_station_id(v) != ""]
    out = []
    for p in raw.split(","):
        sid = norm_station_id(p)
        if sid != "":
            out.append(sid)
    return out


def rho_powerexp(d_km, a_km, b):
    return np.exp(-((np.asarray(d_km, dtype=float) / float(a_km)) ** float(b)))


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


def compute_ok_details(stations_df, target_lat, target_lon, ids, a_km, b, nugget=0.0):
    st = stations_df.set_index("ID")
    ids = [str(i) for i in ids]

    xs_lat = np.array([float(st.loc[sid, "Latitude"]) for sid in ids], dtype=float)
    xs_lon = np.array([float(st.loc[sid, "Longitude"]) for sid in ids], dtype=float)
    n = len(ids)

    dij_km = np.zeros((n, n), dtype=float)
    for i in range(n):
        dij_km[i, :] = haversine_km(xs_lat[i], xs_lon[i], xs_lat, xs_lon)

    d0_km = haversine_km(target_lat, target_lon, xs_lat, xs_lon)
    bears = initial_bearing_deg(target_lat, target_lon, xs_lat, xs_lon)

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

    w = sol[:n]
    lam = float(sol[n])
    condA = float(np.linalg.cond(A))

    return {
        "ids": ids,
        "station_lats": xs_lat,
        "station_lons": xs_lon,
        "d0_km": d0_km,
        "bears_deg": bears,
        "C": C,
        "c0": c0,
        "A": A,
        "rhs": rhs,
        "weights": w,
        "lambda": lam,
        "condA": condA,
        "solver": solver,
    }


def min_angular_separation(bears_deg):
    bears = list(map(float, bears_deg))
    if len(bears) < 2:
        return 360.0
    seps = [ang_sep_deg(bears[i], bears[j]) for i in range(len(bears)) for j in range(i + 1, len(bears))]
    return float(min(seps))


def load_event_params(event_no: int, event_meta_dir: Path):
    f = event_meta_dir / f"Event_{int(event_no)}_Stations_correlation.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing event metadata: {f}")
    df = pd.read_csv(f)
    return float(df["corr_a_km"].iloc[0]), float(df["corr_b"].iloc[0])


def main():
    parser = argparse.ArgumentParser(description="Diagnose ordinary kriging weights for one centroid and a chosen set of 3 to 4 stations.")
    parser.add_argument("--centroid-id", required=True, help="Grid centroid id")
    parser.add_argument("--stations", required=True, help='Comma list or Python-style list, e.g. "16018,16046,16032" or "[16018,16046,16032,16086]"')
    parser.add_argument("--event", type=int, default=None, help="Event number to read corr_a_km and corr_b from Event_X_Stations_correlation.csv")
    parser.add_argument("--a-km", type=float, default=None, help="Correlation range parameter a in km, if not using --event")
    parser.add_argument("--b", type=float, default=None, help="Correlation shape parameter b, if not using --event")
    parser.add_argument("--nugget", type=float, default=0.0)
    parser.add_argument("--neg-tol", type=float, default=0.1, help="Small negative tolerance. Weights in [-tol,0) will be clipped to 0 and renormalized.")
    parser.add_argument("--grid-file", type=Path, default=GRID_CSV)
    parser.add_argument("--station-file", type=Path, default=STATIONS_CSV)
    parser.add_argument("--event-meta-dir", type=Path, default=CORRELATION_DIR)
    parser.add_argument("--show-matrix", action="store_true", help="Also print kriging matrix A and rhs.")
    args = parser.parse_args()

    if args.event is None and (args.a_km is None or args.b is None):
        raise SystemExit("Provide either --event or both --a-km and --b.")

    station_ids = parse_station_list(args.stations)
    if len(station_ids) not in (3, 4):
        raise SystemExit(f"You gave {len(station_ids)} stations. This tool expects 3 or 4 stations.")

    grid_df = pd.read_csv(args.grid_file)
    stn_df = pd.read_csv(args.station_file)
    grid_df["id"] = pd.to_numeric(grid_df["id"], errors="coerce").astype("Int64")
    stn_df["ID"] = stn_df["ID"].apply(norm_station_id)

    centroid_id = int(float(args.centroid_id))
    row = grid_df.loc[grid_df["id"] == centroid_id]
    if row.empty:
        raise SystemExit(f"Centroid {centroid_id} not found in {args.grid_file}")
    row = row.iloc[0]
    target_lat = float(row["Latitude"])
    target_lon = float(row["Longitude"])

    missing = [sid for sid in station_ids if sid not in set(stn_df["ID"].tolist())]
    if missing:
        raise SystemExit(f"These stations were not found in {args.station_file}: {missing}")

    if args.event is not None:
        a_km, b = load_event_params(args.event, args.event_meta_dir)
        param_source = f"event {args.event}"
    else:
        a_km = float(args.a_km)
        b = float(args.b)
        param_source = "manual"

    d = compute_ok_details(
        stations_df=stn_df,
        target_lat=target_lat,
        target_lon=target_lon,
        ids=station_ids,
        a_km=a_km,
        b=b,
        nugget=args.nugget,
    )

    w_raw = d["weights"]
    w_fix, ok_fix = fix_small_negative_weights(w_raw, tol=args.neg_tol)
    min_sep = min_angular_separation(d["bears_deg"])

    print("=" * 80)
    print(f"Centroid: {centroid_id}")
    print(f"Location : lat={target_lat:.6f}, lon={target_lon:.6f}")
    print(f"Stations : {', '.join(station_ids)}")
    print(f"Params   : a_km={a_km:.6f}, b={b:.6f}, nugget={args.nugget:.6f} ({param_source})")
    print(f"Solver   : {d['solver']}")
    print(f"cond(A)  : {d['condA']:.6f}")
    print(f"min ang sep among chosen stations: {min_sep:.3f} deg")
    print("=" * 80)

    out = pd.DataFrame({
        "station_id": d["ids"],
        "lat": d["station_lats"],
        "lon": d["station_lons"],
        "dist_to_centroid_km": d["d0_km"],
        "bearing_deg": d["bears_deg"],
        "raw_weight": w_raw,
        "fixed_weight": w_fix,
    })
    pd.set_option("display.float_format", lambda x: f"{x:0.6f}")
    print(out.to_string(index=False))

    print("-" * 80)
    print(f"sum(raw weights)   = {np.sum(w_raw):.12f}")
    print(f"sum(fixed weights) = {np.sum(w_fix):.12f}")
    print(f"lambda             = {d['lambda']:.12f}")
    print(f"n negative raw     = {int(np.sum(w_raw < 0))}")
    print(f"min raw weight     = {np.min(w_raw):.12f}")

    if np.all(w_raw >= 0):
        print("status             = clean non-negative ordinary kriging")
    elif ok_fix:
        print(f"status             = small negatives corrected with tol={args.neg_tol}")
    else:
        print(f"status             = rejected by correction rule, at least one weight < -{args.neg_tol}")

    if args.show_matrix:
        print("-" * 80)
        print("C matrix:")
        print(np.array2string(d["C"], precision=6, suppress_small=False))
        print("-" * 80)
        print("c0 vector:")
        print(np.array2string(d["c0"], precision=6, suppress_small=False))
        print("-" * 80)
        print("A matrix:")
        print(np.array2string(d["A"], precision=6, suppress_small=False))
        print("-" * 80)
        print("rhs:")
        print(np.array2string(d["rhs"], precision=6, suppress_small=False))


if __name__ == "__main__":
    main()
