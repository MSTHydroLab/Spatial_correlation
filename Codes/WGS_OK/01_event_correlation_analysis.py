from pathlib import Path
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.optimize import curve_fit
from pyproj import Transformer
import argparse
import matplotlib.pyplot as plt
import re
import math

BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK")
DEPENDENT = BASE_DIR / "dependent_files"
STATION_META = DEPENDENT / "Stations_df.csv"

RAIN_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/Compiled_rain/compiled_rawstyle_hourly_mm/per_station_hourly/")

EVENT_TS_DIR = BASE_DIR / "01_Event_TimeSeries"
EVENT_TS_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_TZ = "America/Chicago"

PLOT_DIR = BASE_DIR / "01_Correlation_Plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# QC settings
# ------------------------------------------------------------
POSITIVE_COUNT_PERCENTILE = 25.0   # event-wise percentile of positive-rain hours
MIN_POSITIVE_COUNT_FLOOR = 1       # optional hard floor
MAX_MISSING_FRACTION = 0.7         # reject if >50% of event hours are missing
MIN_PAIR_OVERLAP = 3               # minimum overlap after pairwise dropna

EVENT_WINDOWS = {
    1: ("2013-05-30 12:00:00", "2013-05-31 12:00:00"),
    2: ("2014-08-06 12:00:00", "2014-08-07 12:00:00"),
    3: ("2015-09-10 11:00:00", "2015-09-11 12:00:00"),
    4: ("2016-04-26 07:00:00", "2016-04-27 12:00:00"),
    5: ("2016-08-26 06:00:00", "2016-08-27 12:00:00"),
    6: ("2017-07-23 00:00:00", "2017-07-23 12:00:00"),
    7: ("2017-07-26 17:00:00", "2017-07-27 12:00:00"),
    8: ("2017-08-21 12:00:00", "2017-08-22 12:00:00"),
    9: ("2018-07-17 21:00:00", "2018-07-18 16:00:00"),
    11: ("2019-08-25 11:00:00", "2019-08-26 17:00:00"),
    10: ("2019-06-23 01:00:00", "2019-06-23 20:00:00"),
    12: ("2020-05-28 01:00:00", "2020-05-29 00:00:00"),
    13: ("2020-07-03 19:00:00", "2020-07-04 03:00:00"),
    14: ("2021-08-12 21:00:00", "2021-08-13 15:00:00"),
    15: ("2022-03-30 01:00:00", "2022-03-30 11:00:00"),
}

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def save_correlation_plot(event, dists, corrs, a_km, b, n_stations_used=None, required_positive_count=None):
    if len(dists) == 0 or len(corrs) == 0:
        print(f"No correlation plot saved for event {event}, no valid data.")
        return

    xfit = np.linspace(0, max(dists) * 1.05, 400)
    yfit = corr_model(xfit, a_km, b)

    plt.figure(figsize=(8, 6))
    plt.scatter(dists, corrs, s=5, alpha=0.6, label="Station pairs")
    plt.plot(
        xfit,
        yfit,
        linewidth=2,
        color="red",
        label=f"Fit: a={a_km:.2f} km, b={b:.2f}"
    )

    title = f"Event {event}: correlation decay"
    if n_stations_used is not None:
        title += f"\nStations used: {n_stations_used}"
    if required_positive_count is not None:
        title += f" | Required rain hours > 0: {required_positive_count}"
    plt.title(title)

    plt.xlabel("Distance between stations (km)")
    plt.ylabel("Correlation")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plot_path = PLOT_DIR / f"Event_{event}_correlation_fit.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Correlation plot saved:", plot_path)

def save_positive_hours_plot(event, positive_counts_by_station, required_positive_count=None):
    if len(positive_counts_by_station) == 0:
        print(f"No positive-hours plot saved for event {event}, no station counts available.")
        return

    s = pd.Series(positive_counts_by_station, dtype=float).sort_values()

    q1 = float(s.quantile(0.25))
    q2 = float(s.quantile(0.50))
    q3 = float(s.quantile(0.75))

    plt.figure(figsize=(9, 5))
    x = np.arange(1, len(s) + 1)

    plt.plot(x, s.values, marker="o", linewidth=1, markersize=3, label="Stations")
    plt.axhline(q1, linestyle="--", linewidth=1, label=f"Q1 = {q1:.1f}")
    plt.axhline(q2, linestyle="--", linewidth=1, label=f"Median = {q2:.1f}")
    plt.axhline(q3, linestyle="--", linewidth=1, label=f"Q3 = {q3:.1f}")

    if required_positive_count is not None:
        plt.axhline(
            required_positive_count,
            linestyle="-",
            linewidth=2,
            label=f"Required threshold = {required_positive_count}"
        )

    plt.xlabel("Usable stations (sorted)")
    plt.ylabel("Hours with rainfall > 0 mm")
    plt.title(f"Event {event}: positive-rain-hour distribution across usable stations")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out_path = PLOT_DIR / f"Event_{event}_positive_hours_distribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Positive-hours distribution plot saved:", out_path)
    
def norm_station_id(x):
    s = str(x).strip()
    try:
        return str(int(float(s)))
    except Exception:
        return s


def parse_local_series_time(col):
    t = pd.to_datetime(col, errors="coerce", utc=True)
    return t.dt.tz_convert(LOCAL_TZ)


def build_rain_file_map(rain_dir: Path):
    file_map = {}

    for f in rain_dir.glob("*.csv"):
        stem = f.stem.strip()
        candidates = {stem}

        try:
            candidates.add(str(int(float(stem))))
        except Exception:
            pass

        m = re.search(r"(\d+)", stem)
        if m:
            candidates.add(str(int(m.group(1))))

        for key in candidates:
            if key not in file_map:
                file_map[key] = f

    return file_map


# ------------------------------------------------------------
# Coordinate conversion
# ------------------------------------------------------------
transformer = Transformer.from_crs("EPSG:4326", "EPSG:26915", always_xy=True)


def latlon_to_utm(lon, lat):
    return transformer.transform(lon, lat)


# ------------------------------------------------------------
# Correlation model
# ------------------------------------------------------------
def corr_model(d, a, b):
    return np.exp(-((d / a) ** b))


# ------------------------------------------------------------
# Distance calculation
# ------------------------------------------------------------
def pairwise_distances(coords):
    dist = {}
    for (s1, (x1, y1)), (s2, (x2, y2)) in combinations(coords.items(), 2):
        d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / 1000.0
        dist[(s1, s2)] = d
        dist[(s2, s1)] = d
    return dist


# ------------------------------------------------------------
# Load rainfall series
# ------------------------------------------------------------
def load_station_series(station, start, end, rain_file_map, event_idx=None):
    sid = norm_station_id(station)
    f = rain_file_map.get(sid, None)
    if f is None:
        return None

    df = pd.read_csv(f)

    if "time_local" not in df.columns or "rain_mm" not in df.columns:
        return None

    df["time_local"] = parse_local_series_time(df["time_local"])
    df["rain_mm"] = pd.to_numeric(df["rain_mm"], errors="coerce")

    df = df.dropna(subset=["time_local"])
    df = df[(df["time_local"] >= start) & (df["time_local"] <= end)]

    if len(df) == 0:
        return None

    s = df.set_index("time_local")["rain_mm"]

    # collapse duplicate timestamps if any
    s = s.groupby(level=0).mean().sort_index()

    # reindex to full event timeline if provided
    if event_idx is not None:
        s = s.reindex(event_idx)

    # if everything is missing after reindex, treat as unusable
    if s.notna().sum() < 2:
        return None

    print(f"Using file {f.name} for station {sid}")
    return s


def get_event_window_local(event):
    if event not in EVENT_WINDOWS:
        raise ValueError(f"Event {event} not defined")

    start_str, end_str = EVENT_WINDOWS[event]
    start = pd.Timestamp(start_str).tz_localize(LOCAL_TZ)
    end = pd.Timestamp(end_str).tz_localize(LOCAL_TZ)
    return start, end


# ------------------------------------------------------------
# Save event time series
# ------------------------------------------------------------
def save_event_station_timeseries(event, series_dict, full_event_index=None, out_name_suffix="all_used_station_timeseries"):
    if len(series_dict) == 0:
        print(f"No event rainfall time series saved for event {event}, no usable stations.")
        return

    event_df = pd.concat(series_dict, axis=1)
    event_df.columns = [str(c) for c in event_df.columns]

    if full_event_index is not None:
        event_df = event_df.reindex(full_event_index)

    event_df = event_df.sort_index().reset_index()
    event_df = event_df.rename(columns={event_df.columns[0]: "time_local"})

    out_ts = EVENT_TS_DIR / f"Event_{event}_{out_name_suffix}.csv"
    event_df.to_csv(out_ts, index=False)

    print("Event rainfall time series saved:", out_ts)


# ------------------------------------------------------------
# Main event correlation calculation
# ------------------------------------------------------------
def run_event(
    event,
    start,
    end,
    positive_count_percentile=POSITIVE_COUNT_PERCENTILE,
    min_positive_count_floor=MIN_POSITIVE_COUNT_FLOOR,
    max_missing_fraction=MAX_MISSING_FRACTION,
    min_pair_overlap=MIN_PAIR_OVERLAP,
):
    stations = pd.read_csv(STATION_META)
    stations["ID_norm"] = stations["ID"].apply(norm_station_id)

    rain_file_map = build_rain_file_map(RAIN_DIR)

    print(f"Event {event}")
    print(f"Start: {start}, End: {end}")
    print(f"Rain files found: {len(rain_file_map)}")

    event_idx = pd.date_range(start=start, end=end, freq="1h")
    full_event_index = pd.date_range(start, end, freq="1h")

    coords = {}
    for _, r in stations.iterrows():
        sid = r["ID_norm"]
        x, y = latlon_to_utm(r["Longitude"], r["Latitude"])
        coords[sid] = (x, y)

    distances = pairwise_distances(coords)

    series = {}
    missing_files = 0
    no_window_data = 0

    for sid in stations["ID_norm"]:
        s = load_station_series(sid, start, end, rain_file_map, event_idx=event_idx)

        if s is None:
            if sid not in rain_file_map:
                missing_files += 1
            else:
                no_window_data += 1
            continue

        series[sid] = s

    save_event_station_timeseries(
        event,
        series,
        full_event_index=full_event_index,
        out_name_suffix="all_usable_station_timeseries"
    )

    station_ids = list(series.keys())
    corrs = []
    dists = []
    stations_in_valid_pairs = set()
    pairs_info = []

    # ------------------------------------------------------------
    # Event-wise required positive-rain-hour threshold
    # based on percentile of usable stations
    # ------------------------------------------------------------
    positive_counts_by_station = {
        sid: int((series[sid] > 0).sum())
        for sid in station_ids
    }

    positive_counts = np.array(
        list(positive_counts_by_station.values()),
        dtype=float
    )

    if len(positive_counts) == 0:
        print(f"Event {event}: no usable station series found. Skipping.")
        return False

    required_positive_count = int(
        max(
            int(min_positive_count_floor),
            math.ceil(np.percentile(positive_counts, float(positive_count_percentile)))
        )
    )

    print(
        f"Event {event}: required positive-rain hours = {required_positive_count} "
        f"(from {positive_count_percentile}th percentile of usable stations)"
    )
    
    save_positive_hours_plot(
        event=event,
        positive_counts_by_station=positive_counts_by_station,
        required_positive_count=required_positive_count,
    )
    reject_poscount = 0
    reject_missing = 0
    reject_overlap = 0
    reject_constant = 0
    reject_nan_corr = 0

    for s1, s2 in combinations(station_ids, 2):
        s1_full = series[s1].reindex(event_idx)
        s2_full = series[s2].reindex(event_idx)

        # ------------------------------------------------------------
        # QC 1: event-wise required minimum positive-rain hours
        # ------------------------------------------------------------
        pos_count_1 = int((s1_full > 0).sum())
        pos_count_2 = int((s2_full > 0).sum())

        if pos_count_1 < required_positive_count:
            reject_poscount += 1
            continue
        if pos_count_2 < required_positive_count:
            reject_poscount += 1
            continue

        # ------------------------------------------------------------
        # QC 2: maximum missing-data fraction per station
        # ------------------------------------------------------------
        miss_frac_1 = float(s1_full.isna().sum()) / float(len(event_idx))
        miss_frac_2 = float(s2_full.isna().sum()) / float(len(event_idx))

        if miss_frac_1 > float(max_missing_fraction):
            reject_missing += 1
            continue
        if miss_frac_2 > float(max_missing_fraction):
            reject_missing += 1
            continue

        # ------------------------------------------------------------
        # QC 3: pair-level overlap on usable data
        # MATLAB-style pairwise valid timestamps
        # ------------------------------------------------------------
        pair = pd.concat([s1_full, s2_full], axis=1)
        pair.columns = ["s1", "s2"]
        pair = pair.dropna()

        if len(pair) < int(min_pair_overlap):
            reject_overlap += 1
            continue

        if pair["s1"].nunique() <= 1:
            reject_constant += 1
            continue
        if pair["s2"].nunique() <= 1:
            reject_constant += 1
            continue

        c = pair["s1"].corr(pair["s2"])
        if pd.isna(c):
            reject_nan_corr += 1
            continue

        d = distances[(s1, s2)]

        corrs.append(c)
        dists.append(d)

        stations_in_valid_pairs.add(s1)
        stations_in_valid_pairs.add(s2)

        pairs_info.append({
            "station_1": s1,
            "station_2": s2,
            "distance_km": d,
            "correlation": c,
            "n_overlap": int(len(pair)),
            "pos_count_1": pos_count_1,
            "pos_count_2": pos_count_2,
            "miss_frac_1": miss_frac_1,
            "miss_frac_2": miss_frac_2,
            "required_positive_count": required_positive_count,
        })

    pairs_df = pd.DataFrame(pairs_info)
    pairs_df.to_csv(EVENT_TS_DIR / f"Event_{event}_pair_correlations.csv", index=False)

    series_valid = {sid: series[sid] for sid in sorted(stations_in_valid_pairs)}
    save_event_station_timeseries(
        event,
        series_valid,
        full_event_index=full_event_index,
        out_name_suffix="all_used_station_timeseries"
    )

    corrs = np.array(corrs, dtype=float)
    dists = np.array(dists, dtype=float)

    print(f"Total stations in metadata: {len(stations)}")
    print(f"Stations with usable series: {len(series)}")
    print(f"Missing station files: {missing_files}")
    print(f"Files found but no usable rows in event window: {no_window_data}")
    print(f"Required positive-rain hours for this event: {required_positive_count}")
    print(f"Number of pair correlations: {len(corrs)}")
    print(f"Loaded station IDs: {station_ids[:20]}")
    print(f"Rejected by positive-count QC: {reject_poscount}")
    print(f"Rejected by missing-fraction QC: {reject_missing}")
    print(f"Rejected by overlap QC: {reject_overlap}")
    print(f"Rejected by constant-series QC: {reject_constant}")
    print(f"Rejected by NaN correlation: {reject_nan_corr}")

    if len(series) < 2:
        print(f"Event {event}: fewer than 2 stations had usable rainfall series. Skipping.")
        return False

    if len(corrs) == 0:
        print(f"Event {event}: no valid overlapping station-pair correlations found. Skipping.")
        return False

    try:
        popt, _ = curve_fit(corr_model, dists, corrs, bounds=(0, [500, 5]))
        a_km, b = popt
    except Exception as e:
        print(f"Event {event}: curve_fit failed. Skipping. Error: {e}")
        return False

    save_correlation_plot(
        event,
        dists,
        corrs,
        a_km,
        b,
        n_stations_used=len(stations_in_valid_pairs),
        required_positive_count=required_positive_count,
    )

    result = pd.DataFrame({
        "event_start": [str(start)],
        "event_end": [str(end)],
        "corr_a_km": [a_km],
        "corr_b": [b],
        "stations_selected": [",".join(sorted(stations_in_valid_pairs))],
        "required_positive_count": [required_positive_count],
        "positive_count_percentile": [float(positive_count_percentile)],
        "min_positive_count_floor": [int(min_positive_count_floor)],
        "max_missing_fraction": [float(max_missing_fraction)],
        "min_pair_overlap": [int(min_pair_overlap)],
    })
    result.to_csv(EVENT_TS_DIR / f"Event_{event}_Stations_correlation.csv", index=False)

    print(f"a_km={a_km:.4f}, b={b:.4f}")
    return True


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", type=int, nargs="+", required=True, help="One or more event numbers")
    parser.add_argument("--positive-count-percentile", type=float, default=POSITIVE_COUNT_PERCENTILE)
    parser.add_argument("--min-positive-count-floor", type=int, default=MIN_POSITIVE_COUNT_FLOOR)
    parser.add_argument("--max-missing-fraction", type=float, default=MAX_MISSING_FRACTION)
    parser.add_argument("--min-pair-overlap", type=int, default=MIN_PAIR_OVERLAP)
    args = parser.parse_args()

    failed_events = []

    for event in args.event:
        try:
            start, end = get_event_window_local(event)

            print("=" * 80)
            print(f"Running event {event}")
            print(f"Local start: {start}")
            print(f"Local end  : {end}")
            print("=" * 80)

            ok = run_event(
                event,
                start,
                end,
                positive_count_percentile=args.positive_count_percentile,
                min_positive_count_floor=args.min_positive_count_floor,
                max_missing_fraction=args.max_missing_fraction,
                min_pair_overlap=args.min_pair_overlap,
            )

            if not ok:
                failed_events.append((event, "run_event returned False"))

        except Exception as e:
            print("=" * 80)
            print(f"Skipping event {event} due to error:")
            print(str(e))
            print("=" * 80)
            failed_events.append((event, str(e)))
            continue

    if failed_events:
        print("\nSummary of skipped events:")
        for ev, msg in failed_events:
            print(f"  Event {ev}: {msg}")