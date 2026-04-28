#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import math
import random
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

from geo_utils import haversine_km

BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK")
EVENT_META_DIR = BASE_DIR / "01_Event_TimeSeries"
STATIONS_CSV = BASE_DIR / "dependent_files" / "Stations_df.csv"
OUT_DIR = BASE_DIR / "07_IDW_OK_Avg_method_results"

CATCHMENT_SHP_PATHS = [
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6893390/6893390.shp",
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/06893080/6893080.shp",
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6892513/6892513.shp",
]

LOCAL_TZ = "America/Chicago"


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------
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


def compute_idw_weights(dists_km: np.ndarray, power: float = 2.0) -> np.ndarray:
    d = np.asarray(dists_km, dtype=float)
    if d.size == 0:
        return np.array([], dtype=float)
    if np.any(d <= 0):
        w = np.zeros_like(d, dtype=float)
        w[np.argmin(d)] = 1.0
        return w
    inv = 1.0 / np.power(d, power)
    return inv / inv.sum()


def compute_ok_weights(
    stations_df: pd.DataFrame,
    target_sid: str,
    donor_ids: list[str],
    a_km: float,
    b: float,
    nugget: float = 0.0,
) -> np.ndarray:
    st = stations_df.set_index("ID")
    donor_ids = [str(x) for x in donor_ids]

    target_lat = float(st.loc[target_sid, "Latitude"])
    target_lon = float(st.loc[target_sid, "Longitude"])

    xs_lat = np.array([float(st.loc[sid, "Latitude"]) for sid in donor_ids], dtype=float)
    xs_lon = np.array([float(st.loc[sid, "Longitude"]) for sid in donor_ids], dtype=float)
    n = len(donor_ids)

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


def compute_metrics(obs: np.ndarray, est: np.ndarray) -> dict[str, float]:
    obs = np.asarray(obs, dtype=float)
    est = np.asarray(est, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(est)
    n = int(mask.sum())

    if n == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "mse": np.nan,
            "rmse": np.nan,
            "bias_mean": np.nan,
            "pbias_pct": np.nan,
            "cc": np.nan,
            "kge": np.nan,
        }

    o = obs[mask]
    e = est[mask]
    diff = e - o
    so = float(np.sum(o))

    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    bias_mean = float(np.mean(diff))
    pbias_pct = float(100.0 * np.sum(diff) / so) if so != 0 else np.nan
    cc = float(np.corrcoef(o, e)[0, 1]) if n >= 2 else np.nan

    mu_o = float(np.mean(o))
    mu_e = float(np.mean(e))
    sd_o = float(np.std(o, ddof=0))
    sd_e = float(np.std(e, ddof=0))

    if n < 2 or not np.isfinite(cc) or mu_o == 0 or sd_o == 0:
        kge = np.nan
    else:
        alpha = sd_e / sd_o
        beta = mu_e / mu_o
        kge = float(1.0 - np.sqrt((cc - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))

    return {
        "n": n,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "bias_mean": bias_mean,
        "pbias_pct": pbias_pct,
        "cc": cc,
        "kge": kge,
    }


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def load_catchment_union(shp_paths: list[str]) -> object:
    geoms = []
    for raw_path in shp_paths:
        p = Path(raw_path)
        if not p.exists():
            raise FileNotFoundError(f"Missing catchment shapefile: {p}")
        gdf = gpd.read_file(p)
        if gdf.empty or gdf.crs is None:
            continue
        gdf = gdf.to_crs(epsg=4326)
        geoms.extend([geom for geom in gdf.geometry if geom is not None and not geom.is_empty])

    if not geoms:
        raise ValueError("No valid catchment geometries were loaded")
    return unary_union(geoms)


def filter_stations_to_catchments(stations_df: pd.DataFrame, catchment_union) -> pd.DataFrame:
    gdf = gpd.GeoDataFrame(
        stations_df.copy(),
        geometry=gpd.points_from_xy(stations_df["Longitude"], stations_df["Latitude"]),
        crs="EPSG:4326",
    )
    mask = gdf.geometry.within(catchment_union) | gdf.geometry.touches(catchment_union)
    out = pd.DataFrame(gdf.loc[mask].drop(columns="geometry")).reset_index(drop=True)
    return out


def load_event_meta(event: int, event_meta_dir: Path) -> tuple[float, float, str, str]:
    fp = event_meta_dir / f"Event_{event}_Stations_correlation.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing event metadata file: {fp}")
    df = pd.read_csv(fp)

    required = ["corr_a_km", "corr_b", "event_start", "event_end"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{fp} missing required columns: {missing}")

    a_km = float(df["corr_a_km"].iloc[0])
    b = float(df["corr_b"].iloc[0])
    start = str(df["event_start"].iloc[0])
    end = str(df["event_end"].iloc[0])
    return a_km, b, start, end


def load_event_station_timeseries(event: int, event_meta_dir: Path) -> pd.DataFrame:
    fp = event_meta_dir / f"Event_{event}_all_used_station_timeseries.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing event station timeseries file: {fp}")

    df = pd.read_csv(fp)
    if "time_local" not in df.columns:
        raise ValueError(f"{fp} must contain time_local")

    df["time_local"] = pd.to_datetime(df["time_local"], errors="coerce")
    bad = int(df["time_local"].isna().sum())
    if bad > 0:
        raise ValueError(f"{fp} contains {bad} bad time_local rows")

    value_cols = []
    rename_map = {}
    for c in df.columns:
        if c == "time_local":
            continue
        sid = norm_station_id(c)
        if sid != "":
            value_cols.append(c)
            rename_map[c] = sid

    out = df[["time_local", *value_cols]].copy().rename(columns=rename_map)
    out = out.set_index("time_local").sort_index()
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.loc[:, ~out.columns.duplicated()].copy()
    return out


def load_stations(station_file: Path, station_ids: list[str]) -> pd.DataFrame:
    st = pd.read_csv(station_file)
    req = ["ID", "Latitude", "Longitude"]
    missing = [c for c in req if c not in st.columns]
    if missing:
        raise ValueError(f"{station_file} missing required columns: {missing}")

    st = st[req].copy()
    st["ID"] = st["ID"].apply(norm_station_id)
    st["Latitude"] = pd.to_numeric(st["Latitude"], errors="coerce")
    st["Longitude"] = pd.to_numeric(st["Longitude"], errors="coerce")
    st = st.dropna(subset=["ID", "Latitude", "Longitude"])
    st = st[st["ID"].isin(set(station_ids))].copy()
    st = st.drop_duplicates(subset=["ID"]).reset_index(drop=True)

    missing_ids = sorted(set(station_ids) - set(st["ID"]))
    if missing_ids:
        raise ValueError(f"Missing station coordinates for event stations: {missing_ids[:10]}")

    return st


def build_distance_lookup(stations_df: pd.DataFrame) -> dict[tuple[str, str], float]:
    st = stations_df.set_index("ID")
    ids = st.index.tolist()
    out: dict[tuple[str, str], float] = {}
    for sid in ids:
        dists = haversine_km(
            float(st.loc[sid, "Latitude"]),
            float(st.loc[sid, "Longitude"]),
            st["Latitude"].to_numpy(float),
            st["Longitude"].to_numpy(float),
        )
        for oid, d in zip(ids, dists):
            out[(sid, oid)] = float(d)
    return out


# -----------------------------------------------------------------------------
# Combination logic
# -----------------------------------------------------------------------------
def donor_pool_within_radius(
    target_id: str,
    all_station_ids: list[str],
    distance_lookup: dict[tuple[str, str], float],
    max_distance_km: float,
) -> list[str]:
    donors = [
        sid for sid in all_station_ids
        if sid != target_id and distance_lookup[(target_id, sid)] <= max_distance_km
    ]
    donors.sort(key=lambda sid: (distance_lookup[(target_id, sid)], sid))
    return donors


def sample_random_combinations(
    donor_ids: list[str],
    n_gauges: int,
    n_combos: int,
    rng: random.Random,
) -> list[tuple[str, ...]]:
    if len(donor_ids) < n_gauges:
        return []

    all_count = math.comb(len(donor_ids), n_gauges)
    if all_count <= n_combos:
        return list(itertools.combinations(donor_ids, n_gauges))

    seen: set[tuple[str, ...]] = set()
    out: list[tuple[str, ...]] = []

    donor_ids = list(donor_ids)
    while len(out) < n_combos:
        combo = tuple(sorted(rng.sample(donor_ids, n_gauges)))
        if combo not in seen:
            seen.add(combo)
            out.append(combo)

    return out


def evaluate_idw_combo(
    target_id: str,
    donor_ids: list[str],
    rain_df: pd.DataFrame,
    distance_lookup: dict[tuple[str, str], float],
) -> dict:
    dists_km = np.array([distance_lookup[(target_id, sid)] for sid in donor_ids], dtype=float)
    w = compute_idw_weights(dists_km, power=2.0)

    obs = rain_df[target_id].to_numpy(dtype=float)
    est = rain_df[donor_ids].to_numpy(dtype=float) @ w
    m = compute_metrics(obs, est)

    return {
        "method": "IDW",
        "target_id": target_id,
        "donor_ids": ",".join(donor_ids),
        "n_used": len(donor_ids),
        "avg_distance_km": float(np.mean(dists_km)),
        "min_distance_km": float(np.min(dists_km)),
        "max_distance_km": float(np.max(dists_km)),
        "remarks": "ok",
        **m,
    }


def evaluate_ok_combo(
    target_id: str,
    donor_ids: list[str],
    rain_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    distance_lookup: dict[tuple[str, str], float],
    a_km: float,
    b: float,
    nugget: float,
) -> dict | None:
    w = compute_ok_weights(
        stations_df=stations_df,
        target_sid=target_id,
        donor_ids=list(donor_ids),
        a_km=a_km,
        b=b,
        nugget=nugget,
    )

    if np.any(w < 0):
        return None

    dists_km = np.array([distance_lookup[(target_id, sid)] for sid in donor_ids], dtype=float)
    obs = rain_df[target_id].to_numpy(dtype=float)
    est = rain_df[list(donor_ids)].to_numpy(dtype=float) @ w
    m = compute_metrics(obs, est)

    return {
        "method": "OK",
        "target_id": target_id,
        "donor_ids": ",".join(donor_ids),
        "n_used": len(donor_ids),
        "avg_distance_km": float(np.mean(dists_km)),
        "min_distance_km": float(np.min(dists_km)),
        "max_distance_km": float(np.max(dists_km)),
        "remarks": "ok",
        **m,
    }


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------
def run_event(
    event: int,
    event_meta_dir: Path,
    station_file: Path,
    out_dir: Path,
    nugget: float,
    catchment_shp_paths: list[str],
    n_gauges: int,
    max_distance_km: float,
    n_combos: int,
    random_seed: int,
) -> None:
    a_km, b, event_start, event_end = load_event_meta(event, event_meta_dir)
    rain_df = load_event_station_timeseries(event, event_meta_dir)
    station_ids_all = [norm_station_id(c) for c in rain_df.columns]
    stations_df_all = load_stations(station_file, station_ids_all)
    catchment_union = load_catchment_union(catchment_shp_paths)

    target_stations_df = filter_stations_to_catchments(stations_df_all, catchment_union)
    target_station_ids = target_stations_df["ID"].astype(str).tolist()

    if len(target_station_ids) < 1:
        raise ValueError("No event stations inside catchments were found for LOO analysis")

    donor_station_ids = stations_df_all["ID"].astype(str).tolist()
    rain_df = rain_df.loc[:, donor_station_ids].copy()
    distance_lookup = build_distance_lookup(stations_df_all)

    event_out_dir = out_dir / f"Event_{event}"
    event_out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(random_seed)
    combo_rows: list[dict] = []

    for i, target_id in enumerate(target_station_ids, start=1):
        print(f"[{i}/{len(target_station_ids)}] target station {target_id}")

        donor_pool = donor_pool_within_radius(
            target_id=target_id,
            all_station_ids=donor_station_ids,
            distance_lookup=distance_lookup,
            max_distance_km=max_distance_km,
        )

        combos = sample_random_combinations(
            donor_ids=donor_pool,
            n_gauges=n_gauges,
            n_combos=n_combos,
            rng=rng,
        )

        print(f"    donors within {max_distance_km:.1f} km: {len(donor_pool)}")
        print(f"    sampled combinations: {len(combos)}")
        if len(combos) == 0:
            print(f"    WARNING: no valid combinations for target {target_id}")
            continue

        for combo_index, combo in enumerate(combos, start=1):
            combo = list(combo)

            row_idw = evaluate_idw_combo(
                target_id=target_id,
                donor_ids=combo,
                rain_df=rain_df,
                distance_lookup=distance_lookup,
            )
            row_idw["combo_index"] = combo_index
            combo_rows.append(row_idw)

            row_ok = evaluate_ok_combo(
                target_id=target_id,
                donor_ids=combo,
                rain_df=rain_df,
                stations_df=stations_df_all,
                distance_lookup=distance_lookup,
                a_km=a_km,
                b=b,
                nugget=nugget,
            )
            if row_ok is not None:
                row_ok["combo_index"] = combo_index
                combo_rows.append(row_ok)

    combo_df = pd.DataFrame(combo_rows)

    if combo_df.empty:
        raise ValueError("No valid combination results were produced")

    station_summary = (
        combo_df.groupby(["method", "target_id"], as_index=False)
        .agg(
            n_combos=("combo_index", "count"),
            mean_avg_distance_km=("avg_distance_km", "mean"),
            mean_min_distance_km=("min_distance_km", "mean"),
            mean_max_distance_km=("max_distance_km", "mean"),
            mean_mae=("mae", "mean"),
            median_mae=("mae", "median"),
            mean_mse=("mse", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_bias_mean=("bias_mean", "mean"),
            mean_pbias_pct=("pbias_pct", "mean"),
            mean_cc=("cc", "mean"),
            median_cc=("cc", "median"),
            mean_kge=("kge", "mean"),
            median_kge=("kge", "median"),
        )
        .sort_values(["method", "target_id"])
        .reset_index(drop=True)
    )

    overall_summary = (
        combo_df.groupby("method", as_index=False)
        .agg(
            n_combos=("combo_index", "count"),
            mean_mae=("mae", "mean"),
            median_mae=("mae", "median"),
            mean_mse=("mse", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_bias_mean=("bias_mean", "mean"),
            mean_pbias_pct=("pbias_pct", "mean"),
            mean_cc=("cc", "mean"),
            median_cc=("cc", "median"),
            mean_kge=("kge", "mean"),
            median_kge=("kge", "median"),
        )
        .sort_values("method")
        .reset_index(drop=True)
    )

    meta_df = pd.DataFrame([{
        "event": event,
        "event_start": event_start,
        "event_end": event_end,
        "corr_a_km": a_km,
        "corr_b": b,
        "n_event_stations_total": len(station_ids_all),
        "n_event_stations_in_catchments": len(target_station_ids),
        "nugget": nugget,
        "n_gauges": n_gauges,
        "max_distance_km": max_distance_km,
        "n_combos_requested_per_target": n_combos,
        "random_seed": random_seed,
        "sampling_rule": f"Random combinations of exactly {n_gauges} donors within {max_distance_km} km",
        "ok_negative_rule": "Any OK combination with any negative kriging weight is skipped",
        "outputs_saved": "combination-level metrics, station-level summary, overall summary",
    }])

    meta_df.to_csv(event_out_dir / f"Event_{event}_loo_run_metadata.csv", index=False)
    combo_df.to_csv(event_out_dir / f"Event_{event}_loo_metrics_by_combo.csv", index=False)
    station_summary.to_csv(event_out_dir / f"Event_{event}_loo_metrics_by_station.csv", index=False)
    overall_summary.to_csv(event_out_dir / f"Event_{event}_loo_metrics_overall.csv", index=False)

    print("\nSaved:")
    print(f"  {event_out_dir / f'Event_{event}_loo_run_metadata.csv'}")
    print(f"  {event_out_dir / f'Event_{event}_loo_metrics_by_combo.csv'}")
    print(f"  {event_out_dir / f'Event_{event}_loo_metrics_by_station.csv'}")
    print(f"  {event_out_dir / f'Event_{event}_loo_metrics_overall.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="LOOCV for IDW and OK using random donor combinations within a distance threshold."
    )
    ap.add_argument("--event", type=int, required=True, help="Event number")
    ap.add_argument("--event-meta-dir", type=Path, default=EVENT_META_DIR)
    ap.add_argument("--station-file", type=Path, default=STATIONS_CSV)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--nugget", type=float, default=0.0)
    ap.add_argument(
        "--catchment-shp",
        nargs="*",
        default=CATCHMENT_SHP_PATHS,
        help="Catchment shapefiles used to restrict target stations",
    )
    ap.add_argument(
        "--n-gauges",
        type=int,
        choices=[3, 4],
        default=4,
        help="Number of donor gages in each sampled combination",
    )
    ap.add_argument(
        "--max-distance-km",
        type=float,
        default=30.0,
        help="Only donor gages within this distance of target are eligible",
    )
    ap.add_argument(
        "--n-combos",
        type=int,
        default=500,
        help="Number of random combinations to sample per target",
    )
    ap.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampled combinations",
    )
    args = ap.parse_args()

    run_event(
        event=int(args.event),
        event_meta_dir=args.event_meta_dir,
        station_file=args.station_file,
        out_dir=args.out_dir,
        nugget=float(args.nugget),
        catchment_shp_paths=list(args.catchment_shp),
        n_gauges=int(args.n_gauges),
        max_distance_km=float(args.max_distance_km),
        n_combos=int(args.n_combos),
        random_seed=int(args.random_seed),
    )


if __name__ == "__main__":
    main()