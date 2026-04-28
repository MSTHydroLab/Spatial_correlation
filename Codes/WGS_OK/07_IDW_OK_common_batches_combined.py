#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
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

MAX_GAUGES = 4
METHOD_ORDER = ["IDW", "OK"]


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

    return np.asarray(sol[:n], dtype=float)


def compute_metrics(obs: np.ndarray, est: np.ndarray) -> dict[str, float]:
    obs = np.asarray(obs, dtype=float)
    est = np.asarray(est, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(est)
    n = int(mask.sum())

    if n == 0:
        return {"n": 0, "mae": np.nan, "mse": np.nan, "rmse": np.nan, "bias_mean": np.nan, "pbias_pct": np.nan, "cc": np.nan, "kge": np.nan}

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

    return {"n": n, "mae": mae, "mse": mse, "rmse": rmse, "bias_mean": bias_mean, "pbias_pct": pbias_pct, "cc": cc, "kge": kge}


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
    return pd.DataFrame(gdf.loc[mask].drop(columns="geometry")).reset_index(drop=True)


def filter_stations_to_catchment_buffer(stations_df: pd.DataFrame, shp_paths: list[str], buffer_km: float = 5.0) -> pd.DataFrame:
    catch_gdfs = []
    for raw_path in shp_paths:
        p = Path(raw_path)
        if not p.exists():
            raise FileNotFoundError(f"Missing catchment shapefile: {p}")
        gdf = gpd.read_file(p)
        if gdf.empty or gdf.crs is None:
            continue
        catch_gdfs.append(gdf)
    if not catch_gdfs:
        raise ValueError("No valid catchment geometries were loaded")

    catch = pd.concat(catch_gdfs, ignore_index=True)
    catch = gpd.GeoDataFrame(catch, geometry="geometry", crs=catch_gdfs[0].crs)
    catch_m = catch.to_crs(epsg=26915)
    catch_buffer_union = catch_m.buffer(buffer_km * 1000.0).union_all() if hasattr(catch_m.buffer(buffer_km * 1000.0), "union_all") else catch_m.buffer(buffer_km * 1000.0).unary_union

    stations_gdf = gpd.GeoDataFrame(
        stations_df.copy(),
        geometry=gpd.points_from_xy(stations_df["Longitude"], stations_df["Latitude"]),
        crs="EPSG:4326",
    )
    stations_m = stations_gdf.to_crs(epsg=26915)
    mask = stations_m.geometry.within(catch_buffer_union) | stations_m.geometry.touches(catch_buffer_union)
    return pd.DataFrame(stations_m.loc[mask].drop(columns="geometry")).reset_index(drop=True)


def load_event_meta(event: int, event_meta_dir: Path) -> tuple[float, float, str, str]:
    fp = event_meta_dir / f"Event_{event}_Stations_correlation.csv"
    df = pd.read_csv(fp)
    return float(df["corr_a_km"].iloc[0]), float(df["corr_b"].iloc[0]), str(df["event_start"].iloc[0]), str(df["event_end"].iloc[0])


def load_event_station_timeseries(event: int, event_meta_dir: Path) -> pd.DataFrame:
    fp = event_meta_dir / f"Event_{event}_all_used_station_timeseries.csv"
    df = pd.read_csv(fp)
    df["time_local"] = pd.to_datetime(df["time_local"], errors="coerce")
    if int(df["time_local"].isna().sum()) > 0:
        raise ValueError(f"{fp} contains bad time_local rows")

    value_cols, rename_map = [], {}
    for c in df.columns:
        if c == "time_local":
            continue
        sid = norm_station_id(c)
        if sid:
            value_cols.append(c)
            rename_map[c] = sid

    out = df[["time_local", *value_cols]].copy().rename(columns=rename_map)
    out = out.set_index("time_local").sort_index()
    out = out.apply(pd.to_numeric, errors="coerce")
    return out.loc[:, ~out.columns.duplicated()].copy()


def load_stations(station_file: Path, station_ids: list[str]) -> pd.DataFrame:
    st = pd.read_csv(station_file)[["ID", "Latitude", "Longitude"]].copy()
    st["ID"] = st["ID"].apply(norm_station_id)
    st["Latitude"] = pd.to_numeric(st["Latitude"], errors="coerce")
    st["Longitude"] = pd.to_numeric(st["Longitude"], errors="coerce")
    st = st.dropna(subset=["ID", "Latitude", "Longitude"])
    st = st[st["ID"].isin(set(station_ids))].drop_duplicates(subset=["ID"]).reset_index(drop=True)
    return st


def build_distance_lookup(stations_df: pd.DataFrame) -> dict[tuple[str, str], float]:
    st = stations_df.set_index("ID")
    ids = st.index.tolist()
    out = {}
    for sid in ids:
        dists = haversine_km(float(st.loc[sid, "Latitude"]), float(st.loc[sid, "Longitude"]), st["Latitude"].to_numpy(float), st["Longitude"].to_numpy(float))
        for oid, d in zip(ids, dists):
            out[(sid, oid)] = float(d)
    return out


@dataclass
class BatchResult:
    method: str
    target_id: str
    batch_index: int
    candidate_ids: list[str]
    used_ids: list[str]
    dropped_negative_ids: list[str]
    avg_distance_km: float
    min_distance_km: float
    max_distance_km: float
    n_candidates: int
    n_used: int
    prediction: pd.Series
    remarks: str


def batch_rows_from_results(results: list[BatchResult]) -> list[dict]:
    return [{
        "method": r.method,
        "target_id": r.target_id,
        "batch_index": r.batch_index,
        "candidate_ids": ",".join(r.candidate_ids),
        "used_ids": ",".join(r.used_ids),
        "dropped_negative_ids": ",".join(r.dropped_negative_ids),
        "n_candidates": r.n_candidates,
        "n_used": r.n_used,
        "avg_distance_km": r.avg_distance_km,
        "min_distance_km": r.min_distance_km,
        "max_distance_km": r.max_distance_km,
        "remarks": r.remarks,
    } for r in results]


def build_long_prediction_rows(results: list[BatchResult], obs_df: pd.DataFrame) -> list[dict]:
    rows = []
    for r in results:
        obs_s = obs_df[r.target_id]
        for ts, obs_val, est_val in zip(obs_s.index, obs_s.to_numpy(dtype=float), r.prediction.to_numpy(dtype=float)):
            rows.append({
                "time_local": ts,
                "method": r.method,
                "target_id": r.target_id,
                "batch_index": r.batch_index,
                "observed_mm": obs_val,
                "predicted_mm": est_val,
                "error_mm": est_val - obs_val if np.isfinite(obs_val) and np.isfinite(est_val) else np.nan,
                "abs_error_mm": abs(est_val - obs_val) if np.isfinite(obs_val) and np.isfinite(est_val) else np.nan,
                "avg_distance_km": r.avg_distance_km,
                "n_used": r.n_used,
                "remarks": r.remarks,
            })
    return rows


def summarize_metrics(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    station_batch_rows, station_rows, overall_rows = [], [], []
    for (method, target_id, batch_index), g in long_df.groupby(["method", "target_id", "batch_index"], dropna=False):
        m = compute_metrics(g["observed_mm"].to_numpy(), g["predicted_mm"].to_numpy())
        station_batch_rows.append({
            "method": method,
            "target_id": target_id,
            "batch_index": batch_index,
            "avg_distance_km": float(g["avg_distance_km"].dropna().iloc[0]) if g["avg_distance_km"].notna().any() else np.nan,
            "n_used": int(g["n_used"].dropna().iloc[0]) if g["n_used"].notna().any() else np.nan,
            **m,
        })
    for (method, target_id), g in long_df.groupby(["method", "target_id"], dropna=False):
        m = compute_metrics(g["observed_mm"].to_numpy(), g["predicted_mm"].to_numpy())
        station_rows.append({"method": method, "target_id": target_id, **m})
    for method, g in long_df.groupby("method", dropna=False):
        m = compute_metrics(g["observed_mm"].to_numpy(), g["predicted_mm"].to_numpy())
        overall_rows.append({"method": method, **m})
    return (
        pd.DataFrame(station_batch_rows).sort_values(["method", "target_id", "batch_index"]).reset_index(drop=True),
        pd.DataFrame(station_rows).sort_values(["method", "target_id"]).reset_index(drop=True),
        pd.DataFrame(overall_rows).sort_values(["method"]).reset_index(drop=True),
    )


def summarize_selection_stats(results: list[BatchResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame([{"common_batches_accepted": 0, "ok_selected": 0, "idw_selected": 0, "failed_anchor_batches": 0}])
    df = pd.DataFrame(batch_rows_from_results(results))
    failed = int(df.loc[(df["method"] == "OK") & (df["remarks"].str.contains("failed", na=False))].shape[0])
    ok_ok = int(df.loc[(df["method"] == "OK") & (~df["remarks"].str.contains("failed", na=False))].shape[0])
    idw_ok = int(df.loc[df["method"] == "IDW"].shape[0])
    return pd.DataFrame([{
        "common_batches_accepted": ok_ok,
        "ok_selected": ok_ok,
        "idw_selected": idw_ok,
        "failed_anchor_batches": failed,
    }])


def sort_pool_by_distance(target_id: str, station_ids: list[str], distance_lookup: dict[tuple[str, str], float]) -> list[str]:
    pool = [sid for sid in station_ids if sid != target_id and (target_id, sid) in distance_lookup]
    pool.sort(key=lambda sid: (distance_lookup[(target_id, sid)], sid))
    return pool


def choose_replace_index(working_ids: list[str], weights: np.ndarray, target_id: str, distance_lookup: dict[tuple[str, str], float]) -> int:
    neg_idx = np.where(weights < 0)[0]
    if neg_idx.size == 0:
        return -1
    neg_idx = neg_idx.tolist()
    # replace most negative first, tie-break by farthest distance
    return min(
        neg_idx,
        key=lambda i: (weights[i], -distance_lookup[(target_id, working_ids[i])])
    )


def try_outward_substitution(
    target_id: str,
    anchor_ids: list[str],
    outward_ids: list[str],
    stations_df: pd.DataFrame,
    distance_lookup: dict[tuple[str, str], float],
    a_km: float,
    b: float,
    nugget: float,
) -> tuple[list[str] | None, np.ndarray | None, list[str], str]:
    working_ids = list(anchor_ids)
    rejected_ids: list[str] = []
    outward_ptr = 0
    history = []

    max_attempts = len(outward_ids) + MAX_GAUGES + 5
    attempts = 0

    while attempts < max_attempts:
        attempts += 1
        w_ok = compute_ok_weights(stations_df, target_id, working_ids, a_km, b, nugget)
        neg_idx = np.where(w_ok < 0)[0]
        if neg_idx.size == 0:
            remark = "ok_4_common_outward_sub" if rejected_ids else "ok_4_common_no_sub"
            return working_ids, w_ok, rejected_ids, remark

        repl_idx = choose_replace_index(working_ids, w_ok, target_id, distance_lookup)
        if repl_idx < 0 or outward_ptr >= len(outward_ids):
            history.append("failed_no_more_outward")
            break

        outgoing = working_ids[repl_idx]
        incoming = outward_ids[outward_ptr]
        outward_ptr += 1

        rejected_ids.append(outgoing)
        history.append(f"{outgoing}->{incoming}")
        working_ids[repl_idx] = incoming

    return None, None, rejected_ids, "failed_after_outward_sub"


def run_common_batches_with_substitution_for_target(
    target_id: str,
    station_ids: list[str],
    rain_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    distance_lookup: dict[tuple[str, str], float],
    a_km: float,
    b: float,
    nugget: float,
) -> list[BatchResult]:
    ordered = sort_pool_by_distance(target_id, station_ids, distance_lookup)
    available = list(ordered)
    results: list[BatchResult] = []
    batch_index = 0

    while len(available) >= MAX_GAUGES:
        batch_index += 1
        anchor_ids = available[:MAX_GAUGES]
        outward_ids = available[MAX_GAUGES:]

        used_ids, w_ok, rejected_ids, remark = try_outward_substitution(
            target_id=target_id,
            anchor_ids=anchor_ids,
            outward_ids=outward_ids,
            stations_df=stations_df,
            distance_lookup=distance_lookup,
            a_km=a_km,
            b=b,
            nugget=nugget,
        )

        if used_ids is None or w_ok is None:
            results.append(BatchResult(
                method="OK",
                target_id=target_id,
                batch_index=batch_index,
                candidate_ids=list(anchor_ids),
                used_ids=[],
                dropped_negative_ids=list(rejected_ids),
                avg_distance_km=np.nan,
                min_distance_km=np.nan,
                max_distance_km=np.nan,
                n_candidates=MAX_GAUGES,
                n_used=0,
                prediction=pd.Series(np.nan, index=rain_df.index),
                remarks=remark,
            ))
            # advance by dropping the original anchor block
            for sid in anchor_ids:
                if sid in available:
                    available.remove(sid)
            continue

        sel_dists = np.array([distance_lookup[(target_id, sid)] for sid in used_ids], dtype=float)
        w_idw = compute_idw_weights(sel_dists)
        pred_ok = (rain_df[used_ids] * w_ok).sum(axis=1)
        pred_idw = (rain_df[used_ids] * w_idw).sum(axis=1)
        avg_dist = float(np.mean(sel_dists))
        min_dist = float(np.min(sel_dists))
        max_dist = float(np.max(sel_dists))

        results.append(BatchResult("OK", target_id, batch_index, list(anchor_ids), list(used_ids), list(rejected_ids), avg_dist, min_dist, max_dist, 4, 4, pred_ok, remark))
        results.append(BatchResult("IDW", target_id, batch_index, list(anchor_ids), list(used_ids), list(rejected_ids), avg_dist, min_dist, max_dist, 4, 4, pred_idw, "idw_same_as_ok_batch"))

        accepted_remove = set(used_ids)
        available = [sid for sid in available if sid not in accepted_remove]
        # rejected_ids were never removed from available, so they remain for later evaluation automatically

    return results


def resolve_metric_column(df: pd.DataFrame, metric: str) -> str:
    aliases = {
        "mae": "mae", "rmse": "rmse", "mse": "mse", "cc": "cc", "kge": "kge", "n": "n",
        "bias": "bias_mean", "mean_error": "bias_mean", "mean_diff": "bias_mean", "abs_bias": "bias_mean",
    }
    metric = metric.lower().strip()
    col = aliases.get(metric)
    if col is None or col not in df.columns:
        raise ValueError(f"Metric '{metric}' not available. Available columns include: {', '.join(df.columns)}")
    return col


def prepare_metric_series(df: pd.DataFrame, metric_col: str, requested_metric: str) -> pd.DataFrame:
    out = df.copy()
    out["plot_metric"] = out[metric_col].abs() if requested_metric.lower().strip() == "abs_bias" else out[metric_col]
    return out[np.isfinite(out["avg_distance_km"]) & np.isfinite(out["plot_metric"])].copy()


def plot_scatter(df: pd.DataFrame, metric_label: str, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for method in METHOD_ORDER:
        g = df[df["method"] == method].copy()
        if g.empty:
            continue
        x = g["avg_distance_km"].to_numpy(dtype=float)
        y = g["plot_metric"].to_numpy(dtype=float)
        color = "tab:blue" if method == "IDW" else "tab:orange"
        line_color = "tab:blue" if method == "IDW" else "#d95f02"
        ax.scatter(x, y, s=24, alpha=0.40, color=color, label=method)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) >= 2:
            m, c = np.polyfit(x[mask], y[mask], 1)
            x_line = np.linspace(np.min(x[mask]), np.max(x[mask]), 200)
            ax.plot(x_line, m * x_line + c, linewidth=2, color=line_color)
    ax.set_xlabel("Inter-station distance (km)", fontweight="bold", fontsize=18)
    ax.axvline(x=2.68, color="black", alpha=0.5, linewidth=1.5)
    ax.set_ylabel(metric_label, fontweight="bold", fontsize=18)
    ax.set_xlim(0, 20)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False, prop={"size": 17, "weight": "bold"}, facecolor="white")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)




def run_event(event: int, event_meta_dir: Path, station_file: Path, out_dir: Path, nugget: float,
              catchment_shp_paths: list[str], buffer_km: float, make_plots: bool, metric: str, bin_width_km: float) -> None:
    a_km, b, event_start, event_end = load_event_meta(event, event_meta_dir)
    rain_df = load_event_station_timeseries(event, event_meta_dir)
    station_ids_all = [norm_station_id(c) for c in rain_df.columns]
    stations_df_all = load_stations(station_file, station_ids_all)
    catchment_union = load_catchment_union(catchment_shp_paths)

    target_station_ids = filter_stations_to_catchments(stations_df_all, catchment_union)["ID"].astype(str).tolist()
    donor_station_ids = filter_stations_to_catchment_buffer(stations_df_all, catchment_shp_paths, buffer_km=buffer_km)["ID"].astype(str).tolist()

    if len(target_station_ids) < 1:
        raise ValueError("No event stations inside catchments were found for LOO analysis")
    if len(donor_station_ids) < MAX_GAUGES:
        raise ValueError(f"Only {len(donor_station_ids)} donor stations in buffer, need at least {MAX_GAUGES}")

    rain_df = rain_df.loc[:, donor_station_ids].copy()
    distance_lookup = build_distance_lookup(stations_df_all)

    event_out_dir = out_dir / f"Event_{event}"
    event_out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for i, target_id in enumerate(target_station_ids, start=1):
        print(f"[{i}/{len(target_station_ids)}] target station {target_id}")
        all_results.extend(run_common_batches_with_substitution_for_target(
            target_id=target_id,
            station_ids=donor_station_ids,
            rain_df=rain_df,
            stations_df=stations_df_all,
            distance_lookup=distance_lookup,
            a_km=a_km,
            b=b,
            nugget=nugget,
        ))

    batch_df = pd.DataFrame(batch_rows_from_results(all_results))
    stats_df = summarize_selection_stats(all_results)
    long_df = pd.DataFrame(build_long_prediction_rows(all_results, rain_df))
    station_batch_metrics_df, station_metrics_df, overall_metrics_df = summarize_metrics(long_df)

    meta_df = pd.DataFrame([{
        "event": event,
        "event_start": event_start,
        "event_end": event_end,
        "corr_a_km": a_km,
        "corr_b": b,
        "n_event_stations_total": len(station_ids_all),
        "n_event_stations_in_catchments": len(target_station_ids),
        "n_event_stations_in_donor_buffer": len(donor_station_ids),
        "nugget": nugget,
        "buffer_km": buffer_km,
        "batch_rule": "Anchor on nearest 4. If OK has negative weights, replace one negative station with the next outward donor until all-positive OK is found. Accepted OK combination is used for IDW too. Replaced-out stations remain eligible for later batches.",
    }])

    meta_df.to_csv(event_out_dir / f"Event_{event}_loo_run_metadata.csv", index=False)
    batch_df.to_csv(event_out_dir / f"Event_{event}_loo_batch_summary.csv", index=False)
    long_df.to_csv(event_out_dir / f"Event_{event}_loo_predictions_long.csv", index=False)
    station_batch_metrics_df.to_csv(event_out_dir / f"Event_{event}_loo_metrics_by_station_batch.csv", index=False)
    station_metrics_df.to_csv(event_out_dir / f"Event_{event}_loo_metrics_by_station.csv", index=False)
    overall_metrics_df.to_csv(event_out_dir / f"Event_{event}_loo_metrics_overall.csv", index=False)
    stats_file = event_out_dir / f"Event_{event}_selection_stats.csv"
    stats_df.to_csv(stats_file, index=False)

    print("\nSelection stats:")
    print(stats_df.to_string(index=False))

    if make_plots and not station_batch_metrics_df.empty:
        metric_col = resolve_metric_column(station_batch_metrics_df, metric)
        plot_df = prepare_metric_series(station_batch_metrics_df, metric_col, metric)
        metric_label_map = {
            "mae": "MAE (mm)", "rmse": "RMSE (mm)", "mse": "MSE (mm²)",
            "bias": "Mean bias (mm)", "abs_bias": "Absolute mean bias (mm)",
            "mean_error": "Mean error (mm)", "cc": "Correlation coefficient",
            "kge": "KGE (-)", "n": "Sample count",
        }
        metric_label = metric_label_map.get(metric.lower().strip(), metric)
        scatter_png = event_out_dir / f"Event_{event}_distance_{metric.lower()}_scatter.png"
        plot_scatter(plot_df, metric_label, scatter_png)
        print(f"Saved: {scatter_png}")

    print("\nSaved:")
    for p in [
        event_out_dir / f"Event_{event}_loo_run_metadata.csv",
        event_out_dir / f"Event_{event}_loo_batch_summary.csv",
        event_out_dir / f"Event_{event}_loo_predictions_long.csv",
        event_out_dir / f"Event_{event}_loo_metrics_by_station_batch.csv",
        event_out_dir / f"Event_{event}_loo_metrics_by_station.csv",
        event_out_dir / f"Event_{event}_loo_metrics_overall.csv",
        stats_file,
    ]:
        print(f"  {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Combined LOO + plotting with outward substitution and common OK/IDW batches.")
    ap.add_argument("--event", type=int, required=True)
    ap.add_argument("--event-meta-dir", type=Path, default=EVENT_META_DIR)
    ap.add_argument("--station-file", type=Path, default=STATIONS_CSV)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--nugget", type=float, default=0.0)
    ap.add_argument("--buffer-km", type=float, default=5.0)
    ap.add_argument("--metric", type=str, default="mae")
    ap.add_argument("--bin-width-km", type=float, default=1.0)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--catchment-shp", nargs="*", default=CATCHMENT_SHP_PATHS)
    args = ap.parse_args()

    run_event(
        event=int(args.event),
        event_meta_dir=args.event_meta_dir,
        station_file=args.station_file,
        out_dir=args.out_dir,
        nugget=float(args.nugget),
        catchment_shp_paths=list(args.catchment_shp),
        buffer_km=float(args.buffer_km),
        make_plots=not args.no_plots,
        metric=str(args.metric),
        bin_width_km=float(args.bin_width_km),
    )


if __name__ == "__main__":
    main()
