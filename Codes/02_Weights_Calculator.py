#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import ast
import itertools

# ---------------------------------------------------
# Default paths
# ---------------------------------------------------
BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/01_Correlation_and_Variogram")
NEIGHBOR_FILE = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/01_Correlation_and_Variogram/nearest7_spread_grid.csv")
STATION_FILE = BASE_DIR / "Stations_df.csv"
OUT_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/02_OK_Weights")

# ---------------------------------------------------
# Correlation model
# ---------------------------------------------------
def rho_powerexp(d_km, a_km, b):
    return np.exp(-(d_km / a_km) ** b)

# ---------------------------------------------------
# Angle helpers
# ---------------------------------------------------
def ang_sep_deg(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)

def pick_k_max_spread(ids, dists, bears, k: int):
    """
    Choose k indices that:
      1) maximize the minimum pairwise angular separation (spread)
      2) tie-break: minimize total distance
    Returns list of k indices into ids/dists/bears.

    If <k valid gauges exist, returns all valid indices.
    """
    valid = [
        i for i in range(len(ids))
        if pd.notna(ids[i]) and pd.notna(dists[i]) and pd.notna(bears[i])
    ]
    if len(valid) <= k:
        return valid

    best = None
    best_score = (-1.0, np.inf)  # (min angular sep, total distance)

    for comb in itertools.combinations(valid, k):
        # minimum pairwise angular separation among chosen
        minsep = np.inf
        for i, j in itertools.combinations(comb, 2):
            minsep = min(minsep, ang_sep_deg(float(bears[i]), float(bears[j])))

        # tie-breaker: total distance
        tot = float(np.sum([float(dists[i]) for i in comb]))

        score = (minsep, tot)
        if (score[0] > best_score[0]) or (score[0] == best_score[0] and score[1] < best_score[1]):
            best_score = score
            best = list(comb)

    return best

# ---------------------------------------------------
# Ordinary Kriging weights
# ---------------------------------------------------
def compute_ok_weights(stations_df, ids, dists_m, a_km, b, nugget=0.0):
    """
    Uses:
      - station coords from Stations_df.csv (UTM meters)
      - target->station distances from dists_m (already computed)
      - powered-exponential correlation rho(d)=exp(-(d/a)^b)

    Returns weights array length n.
    """
    st = stations_df.set_index("ID")

    xs = np.array([st.loc[s, "NAD83_15N_Long"] for s in ids], dtype=float)
    ys = np.array([st.loc[s, "NAD83_15N_Lat"]  for s in ids], dtype=float)
    n = len(ids)

    # gauge-gauge distances (km)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dij_km = np.sqrt(dx**2 + dy**2) / 1000.0

    # target-gauge distances (km) from table
    d0_km = np.array(dists_m, dtype=float) / 1000.0

    C  = rho_powerexp(dij_km, a_km, b)
    c0 = rho_powerexp(d0_km,  a_km, b)

    if nugget > 0:
        C = C + np.eye(n) * nugget

    # OK system
    A = np.zeros((n + 1, n + 1), dtype=float)
    A[:n, :n] = C
    A[:n,  n] = 1.0
    A[n,  :n] = 1.0
    A[n,  n]  = 0.0

    rhs = np.zeros(n + 1, dtype=float)
    rhs[:n] = c0
    rhs[n]  = 1.0

    try:
        sol = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)

    return sol[:n]  # weights

# ---------------------------------------------------
# Main Event Runner
# ---------------------------------------------------
def run_event(event_number: int, n_gauges: int, nugget: float = 0.0):
    if n_gauges not in (3, 4):
        raise ValueError("--n-gauges must be 3 or 4")

    event_file = BASE_DIR / f"Event_{event_number}_Stations_correlation.csv"
    if not event_file.exists():
        raise FileNotFoundError(f"{event_file} not found")

    event_df = pd.read_csv(event_file)

    a_km = float(event_df["corr_a_km"].iloc[0])
    b    = float(event_df["corr_b"].iloc[0])

    selected = ast.literal_eval(event_df["stations_selected"].iloc[0])
    selected = set(int(s) for s in selected)

    print(f"Running Event {event_number}")
    print(f"a = {a_km}, b = {b}, n_gauges = {n_gauges}, nugget = {nugget}")
    print(f"Allowed stations: {len(selected)}")

    nei = pd.read_csv(NEIGHBOR_FILE)
    stations = pd.read_csv(STATION_FILE)
    stations["ID"] = stations["ID"].astype(int)

    results = []

    for _, row in nei.iterrows():
        grid_id = int(row["id"])

        ids   = [row.get(f"g{i}", np.nan)     for i in range(1, 8)]
        dists = [row.get(f"d{i}_m", np.nan)   for i in range(1, 8)]
        bears = [row.get(f"b{i}_deg", np.nan) for i in range(1, 8)]

        # Keep only stations allowed for this event
        filtered = []
        for i in range(7):
            if pd.isna(ids[i]):
                continue
            sid = int(float(ids[i]))
            if sid in selected and pd.notna(dists[i]) and pd.notna(bears[i]):
                filtered.append((sid, float(dists[i]), float(bears[i])))

        if len(filtered) < n_gauges:
            continue

        ids_f   = [x[0] for x in filtered]
        dists_f = [x[1] for x in filtered]
        bears_f = [x[2] for x in filtered]

        chosen_idx = pick_k_max_spread(ids_f, dists_f, bears_f, k=n_gauges)

        chosen_ids = [ids_f[i]   for i in chosen_idx]
        chosen_d   = [dists_f[i] for i in chosen_idx]
        chosen_b   = [bears_f[i] for i in chosen_idx]

        weights = compute_ok_weights(stations, chosen_ids, chosen_d, a_km, b, nugget=nugget)

        rec = {"id": grid_id, "n_gauges": n_gauges, "a_km": a_km, "b": b, "nugget": nugget}
        for k in range(n_gauges):
            rec[f"g{k+1}"]     = chosen_ids[k]
            rec[f"d{k+1}_m"]   = chosen_d[k]
            rec[f"b{k+1}_deg"] = chosen_b[k]
            rec[f"w{k+1}"]     = float(weights[k])

        results.append(rec)

    out_df = pd.DataFrame(results)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"Event_{event_number}_weights_{n_gauges}gauges.csv"
    out_df.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")

# ---------------------------------------------------
# CLI
# ---------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", type=int, required=True)
    parser.add_argument("--n-gauges", type=int, default=4, choices=[3, 4],
                        help="How many gauges to use per grid cell (3 or 4). Default=4.")
    parser.add_argument("--nugget", type=float, default=0.0,
                        help="Diagonal stabilization nugget added to correlation matrix. Default=0.")
    args = parser.parse_args()

    run_event(args.event, n_gauges=args.n_gauges, nugget=args.nugget)