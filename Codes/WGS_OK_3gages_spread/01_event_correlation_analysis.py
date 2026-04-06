from pathlib import Path
import pandas as pd
import numpy as np
from itertools import combinations
from scipy.optimize import curve_fit
from pyproj import Transformer
import argparse
import matplotlib.pyplot as plt
import re

BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK_3gages_spread")
DEPENDENT = BASE_DIR / "dependent_files"
STATION_META = DEPENDENT / "Stations_df.csv"

RAIN_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/Compiled_rain/compiled_rawstyle_hourly_mm/per_station_hourly/")

EVENT_TS_DIR = BASE_DIR / "01_Event_TimeSeries"
EVENT_TS_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_TZ = "America/Chicago"
PLOT_DIR = BASE_DIR / "01_Correlation_Plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

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
    10: ("2019-06-23 01:00:00", "2019-06-23 20:00:00"),
    11: ("2019-08-25 11:00:00", "2019-08-26 17:00:00"),
    12: ("2020-05-28 01:00:00", "2020-05-29 00:00:00"),
    13: ("2020-07-03 19:00:00", "2020-07-04 03:00:00"),
    14: ("2021-08-12 21:00:00", "2021-08-13 15:00:00"),
    15: ("2022-03-30 01:00:00", "2022-03-30 11:00:00"),
}

#Normalize station IDs consistently
def save_correlation_plot(event, dists, corrs, a_km, b):
    if len(dists) == 0 or len(corrs) == 0:
        print(f"No correlation plot saved for event {event}, no valid data.")
        return

    xfit = np.linspace(0, max(dists) * 1.05, 400)
    yfit = corr_model(xfit, a_km, b)

    plt.figure(figsize=(8, 6))
    plt.scatter(dists, corrs, s=5, alpha=0.6, label="Station pairs")
    plt.plot(xfit, yfit, linewidth=2, label=f"Fit: a={a_km:.2f} km, b={b:.2f}",color="red")

    plt.xlabel("Distance between stations (km)")
    plt.ylabel("Correlation")
    plt.title(f"Event {event}: correlation decay")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plot_path = PLOT_DIR / f"Event_{event}_correlation_fit.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Correlation plot saved:", plot_path)
    
def norm_station_id(x):
    s = str(x).strip()
    try:
        return str(int(float(s)))
    except Exception:
        return s
# ----------------------------------------------------------
# 
def parse_local_series_time(col):
    # Parse safely even if offsets vary across DST periods
    t = pd.to_datetime(col, errors="coerce", utc=True)

    # Convert back to local timezone for local-time filtering
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

        # also extract numeric token if filename has extra text
        m = re.search(r"(\d+)", stem)
        if m:
            candidates.add(str(int(m.group(1))))

        for key in candidates:
            if key not in file_map:
                file_map[key] = f

    return file_map
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

def load_station_series(station, start, end, rain_file_map):
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

    s = df.set_index("time_local")["rain_mm"].dropna()

    if len(s) < 2:
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
# Main event correlation calculation
# ------------------------------------------------------------
def save_event_station_timeseries(event, series_dict, full_event_index=None, out_name_suffix="all_used_station_timeseries"):
    if len(series_dict) == 0:
        print(f"No event rainfall time series saved for event {event}, no usable stations.")
        return

    # combine all station series on a common timestamp index
    event_df = pd.concat(series_dict, axis=1)

    # flatten column index if needed
    event_df.columns = [str(c) for c in event_df.columns]

    # force full event window if provided
    if full_event_index is not None:
        event_df = event_df.reindex(full_event_index)

    # keep timestamp as a normal column
    event_df = event_df.sort_index().reset_index()
    event_df = event_df.rename(columns={event_df.columns[0]: "time_local"})

    out_ts = EVENT_TS_DIR / f"Event_{event}_{out_name_suffix}.csv"
    event_df.to_csv(out_ts, index=False)

    print("Event rainfall time series saved:", out_ts)
    
def run_event(event, start, end):
    stations = pd.read_csv(STATION_META)
    stations["ID_norm"] = stations["ID"].apply(norm_station_id)

    rain_file_map = build_rain_file_map(RAIN_DIR)

    print(f"Event {event}")
    print(f"Start: {start}, End: {end}")
    print(f"Rain files found: {len(rain_file_map)}")
    
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
        s = load_station_series(sid, start, end, rain_file_map)

        if s is None:
            # distinguish likely reason
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

    for s1, s2 in combinations(station_ids, 2):

        pair = pd.concat([series[s1], series[s2]], axis=1, join="inner")
        pair.columns = ["s1", "s2"]
        pair = pair.dropna()

        if len(pair) < 2:
            continue

        if pair["s1"].nunique() <= 1 or pair["s2"].nunique() <= 1:
            continue

        c = pair["s1"].corr(pair["s2"])

        if pd.isna(c):
            continue

        d = distances[(s1, s2)]

        corrs.append(c)
        dists.append(d)

        pairs_info.append((s1, s2, d, c))
        stations_in_valid_pairs.add(s1)
        stations_in_valid_pairs.add(s2)
    pairs_df = pd.DataFrame(pairs_info,columns=["station1", "station2", "distance_km", "correlation"])
    pairs_df.to_csv(EVENT_TS_DIR/f"{event}_correlation.csv",index=False)
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
    print(f"Number of pair correlations: {len(corrs)}")
    print(f"Loaded station IDs: {station_ids[:20]}")

    if len(series) < 2:
        raise ValueError(
            "Fewer than 2 stations had usable local-time rainfall series. "
            "This strongly suggests file matching or rainfall-source mismatch."
        )

    if len(corrs) == 0:
        raise ValueError(
            "Stations were loaded, but no valid overlapping station-pair correlations were found."
        )

    popt, _ = curve_fit(corr_model, dists, corrs, bounds=(0, [500, 5]))
    a_km, b = popt
    save_correlation_plot(event, dists, corrs, a_km, b)

    result = pd.DataFrame({
        "event_start": [str(start)],
        "event_end": [str(end)],
        "corr_a_km": [a_km],
        "corr_b": [b],
        "stations_selected": [",".join(station_ids)]
    })
    result.to_csv(EVENT_TS_DIR/(f"Event_{event}_Stations_correlation.csv"), index=False)

    print(f"a_km={a_km:.4f}, b={b:.4f}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", type=int, required=True)
    args = parser.parse_args()

    start, end = get_event_window_local(args.event)

    print(f"Running event {args.event}")
    print(f"Local start: {start}")
    print(f"Local end  : {end}")

    run_event(args.event, start, end)