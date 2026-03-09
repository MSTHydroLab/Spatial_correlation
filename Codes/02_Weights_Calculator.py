#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import ast

# ---------------------------------------------------
# Default paths
# ---------------------------------------------------
BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/01_Correlation_and_Variogram")
NEIGHBOR_FILE = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/grid_nearest7.csv")
STATION_FILE = BASE_DIR / "Stations_df.csv"
out_dir= Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/02_OK_Weights")


# ---------------------------------------------------
# Correlation model
# ---------------------------------------------------
def rho_powerexp(d_km, a_km, b):
    return np.exp(-(d_km / a_km) ** b)


# ---------------------------------------------------
# Angle helpers
# ---------------------------------------------------
def ang_sep_deg(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def pick4_max_spread(ids, dists, bears):
    """
    Choose 4 indices that:
      1) maximize the minimum pairwise angular separation (spread)
      2) tie-break: minimize total distance
    Returns list of 4 indices into ids/dists/bears.
    """
    valid = [
        i for i in range(len(ids))
        if pd.notna(ids[i]) and pd.notna(dists[i]) and pd.notna(bears[i])
    ]
    if len(valid) <= 4:
        return valid

    best = None
    best_score = (-1.0, np.inf)  # (min angular sep, total distance)

    for a in range(len(valid)):
        for b in range(a + 1, len(valid)):
            for c in range(b + 1, len(valid)):
                for d in range(c + 1, len(valid)):
                    i, j, k, m = valid[a], valid[b], valid[c], valid[d]

                    seps = [
                        ang_sep_deg(bears[i], bears[j]),
                        ang_sep_deg(bears[i], bears[k]),
                        ang_sep_deg(bears[i], bears[m]),
                        ang_sep_deg(bears[j], bears[k]),
                        ang_sep_deg(bears[j], bears[m]),
                        ang_sep_deg(bears[k], bears[m]),
                    ]
                    minsep = min(seps)
                    tot = dists[i] + dists[j] + dists[k] + dists[m]

                    score = (minsep, tot)
                    if (score[0] > best_score[0]) or (score[0] == best_score[0] and score[1] < best_score[1]):
                        best_score = score
                        best = [i, j, k, m]

    return best


# ---------------------------------------------------
# Ordinary Kriging weights
# ---------------------------------------------------
def compute_ok_weights(stations_df, ids, dists_m, a_km, b, nugget=0.0):
    st = stations_df.set_index("ID")

    xs = np.array([st.loc[s, "NAD83_15N_Long"] for s in ids])
    ys = np.array([st.loc[s, "NAD83_15N_Lat"] for s in ids])
    n = len(ids)

    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dij_km = np.sqrt(dx**2 + dy**2) / 1000.0

    d0_km = np.array(dists_m) / 1000.0

    C = rho_powerexp(dij_km, a_km, b)
    c0 = rho_powerexp(d0_km, a_km, b)

    if nugget > 0:
        C += np.eye(n) * nugget

    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = C
    A[:n, n] = 1
    A[n, :n] = 1

    rhs = np.zeros(n + 1)
    rhs[:n] = c0
    rhs[n] = 1

    try:
        sol = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)

    return sol[:n]


# ---------------------------------------------------
# Main Event Runner
# ---------------------------------------------------
def run_event(event_number):

    event_file = BASE_DIR / f"Event_{event_number}_Stations_correlation.csv"
    if not event_file.exists():
        raise FileNotFoundError(f"{event_file} not found")

    event_df = pd.read_csv(event_file)

    a_km = float(event_df["corr_a_km"].iloc[0])
    b = float(event_df["corr_b"].iloc[0])
    selected = ast.literal_eval(event_df["stations_selected"].iloc[0])
    selected = set(int(s) for s in selected)

    print(f"Running Event {event_number}")
    print(f"a={a_km}, b={b}")
    print(f"Allowed stations: {len(selected)}")

    nei = pd.read_csv(NEIGHBOR_FILE)
    stations = pd.read_csv(STATION_FILE)
    stations["ID"] = stations["ID"].astype(int)

    results = []

    for _, row in nei.iterrows():

        grid_id = int(row["id"])

        ids = [row[f"g{i}"] for i in range(1, 8)]
        dists = [row[f"d{i}_m"] for i in range(1, 8)]
        bears = [row[f"b{i}_deg"] for i in range(1, 8)]

        filtered = [
            (i, int(float(ids[i])), dists[i], bears[i])
            for i in range(7)
            if pd.notna(ids[i]) and int(float(ids[i])) in selected
        ]

        if len(filtered) < 4:
            continue

        ids_f = [f[1] for f in filtered]
        dists_f = [f[2] for f in filtered]
        bears_f = [f[3] for f in filtered]

        chosen_idx = pick4_max_spread(ids_f, dists_f, bears_f)

        chosen_ids = [ids_f[i] for i in chosen_idx]
        chosen_d = [dists_f[i] for i in chosen_idx]
        chosen_b = [bears_f[i] for i in chosen_idx]

        weights = compute_ok_weights(stations, chosen_ids, chosen_d, a_km, b)

        rec = {"id": grid_id}

        for k in range(4):
            rec[f"g{k+1}"] = chosen_ids[k]
            rec[f"d{k+1}_m"] = chosen_d[k]
            rec[f"b{k+1}_deg"] = chosen_b[k]
            rec[f"w{k+1}"] = weights[k]

        results.append(rec)

    out_df = pd.DataFrame(results)

    out_file = out_dir / f"Event_{event_number}_weights.csv"
    out_df.to_csv(out_file, index=False)

    print(f"Saved: {out_file}")


# ---------------------------------------------------
# CLI
# ---------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", type=int, required=True)
    args = parser.parse_args()

    run_event(args.event)
