#!/usr/bin/env python3
from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pyproj import Transformer


# ============================================================
# Paths from the Python workflow
# ============================================================

BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_IDW")
DEPENDENT = BASE_DIR / "dependent_files"
STATION_META = DEPENDENT / "Stations_df.csv"

RAIN_DIR = Path(
    "/mnt/12TB/Sujan/Spatial_correlation/Codes/Compiled_rain/"
    "compiled_rawstyle_hourly_mm/per_station_hourly/"
)

# Use Python-side directories, but keep MATLAB-like filenames
PARAM_DIR = BASE_DIR / "Parameters"
PARAM_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = BASE_DIR / "01_Correlation_Plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_TZ = "America/Chicago"


# ============================================================
# MATLAB-style parameters
# ============================================================

FRACTION_NODATA = 0.5
NGTHRESH = 8

# MATLAB script has 10 events
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


# ============================================================
# Helpers
# ============================================================

def norm_station_id(x):
    s = str(x).strip()
    try:
        return str(int(float(s)))
    except Exception:
        return s


def parse_local_series_time(col):
    # Same approach you used in Python:
    # parse with UTC awareness, then convert to local timezone
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


transformer = Transformer.from_crs("EPSG:4326", "EPSG:26915", always_xy=True)


def latlon_to_utm(lon, lat):
    return transformer.transform(lon, lat)


def corr_model(d, a, b):
    return np.exp(-((d / a) ** b))


def get_event_window_local(event):
    if event not in EVENT_WINDOWS:
        raise ValueError(f"Event {event} not defined in MATLAB-style event list.")
    start_str, end_str = EVENT_WINDOWS[event]
    start = pd.Timestamp(start_str).tz_localize(LOCAL_TZ)
    end = pd.Timestamp(end_str).tz_localize(LOCAL_TZ)
    return start, end


def load_station_event_series_as_matlab_style(station_id, start, end, event_idx, rain_file_map):
    """
    Returns a numpy array of event rainfall values aligned to the full event index.
    Missing/unavailable values are filled with -99, matching MATLAB nodata logic.
    """
    sid = norm_station_id(station_id)
    f = rain_file_map.get(sid, None)

    # If file missing, return full -99
    if f is None:
        return np.full(len(event_idx), -99.0, dtype=float)

    try:
        df = pd.read_csv(f)
    except Exception:
        return np.full(len(event_idx), -99.0, dtype=float)

    if "time_local" not in df.columns or "rain_mm" not in df.columns:
        return np.full(len(event_idx), -99.0, dtype=float)

    df["time_local"] = parse_local_series_time(df["time_local"])
    df["rain_mm"] = pd.to_numeric(df["rain_mm"], errors="coerce")

    df = df.dropna(subset=["time_local"])
    df = df[(df["time_local"] >= start) & (df["time_local"] <= end)]

    # If no rows in event window, return full -99
    if len(df) == 0:
        return np.full(len(event_idx), -99.0, dtype=float)

    s = df.set_index("time_local")["rain_mm"]

    # Collapse duplicate timestamps as in your Python workflow
    s = s.groupby(level=0).mean().sort_index()

    # Reindex to full event timeline, then fill missing with -99
    s = s.reindex(event_idx)
    s = s.fillna(-99.0)

    return s.to_numpy(dtype=float)


def apply_matlab_bad_gauge_mask(event_year, G_event):
    """
    Replicates the intended MATLAB bad-gauge masking by column position.
    MATLAB used 1-based indices:
      2016: [48, 144]
      2017: [24]

    This only matches exactly if STATION_META row order matches MATLAB matrix/Gauge_latlon order.
    """
    if event_year == 2016:
        badgauge_1based = [48, 144]
    elif event_year == 2017:
        badgauge_1based = [24]
    else:
        badgauge_1based = []

    for idx1 in badgauge_1based:
        idx0 = idx1 - 1
        if 0 <= idx0 < G_event.shape[1]:
            G_event[:, idx0] = -99.0

    return G_event


def matlab_pair_correlation(G1, G2):
    """
    Replicates MATLAB QC sequence:

    1) require at least NGTHRESH values > 0 for each station
    2) if nodata count in G1 > 50%, reject
       else remove G1 nodata positions from both series
    3) if nodata count in shortened G2 > 50%, reject
       else remove G2 nodata positions from both series
    4) compute correlation on remaining values

    Returns:
        valid_pair (bool), corr_value (float or -99)
    """
    ok1 = np.where(G1 > 0)[0]
    ok2 = np.where(G2 > 0)[0]

    if len(ok1) < NGTHRESH or len(ok2) < NGTHRESH:
        return False, -99.0

    noG1 = np.where(G1 == -99)[0]
    if len(noG1) > FRACTION_NODATA * len(G1):
        return False, -99.0
    else:
        G1 = np.delete(G1, noG1)
        G2 = np.delete(G2, noG1)

    noG2 = np.where(G2 == -99)[0]
    if len(noG2) > FRACTION_NODATA * len(G2):
        return False, -99.0
    else:
        G2 = np.delete(G2, noG2)
        G1 = np.delete(G1, noG2)

    # MATLAB then directly does corr(G1,G2)
    # In Python, if too short or constant, corrcoef gives nan.
    # We keep that behavior as closely as possible.
    if len(G1) == 0 or len(G2) == 0:
        return True, np.nan

    if len(G1) == 1 or len(G2) == 1:
        return True, np.nan

    try:
        c = np.corrcoef(G1, G2)[0, 1]
    except Exception:
        c = np.nan

    return True, c


def run_event(event):
    start, end = get_event_window_local(event)
    year = start.year

    print("=" * 80)
    print(f"Running MATLAB-style event {event}")
    print(f"Local start: {start}")
    print(f"Local end  : {end}")
    print("=" * 80)

    stations = pd.read_csv(STATION_META).copy()
    stations["ID_norm"] = stations["ID"].apply(norm_station_id)

    ng = len(stations)
    event_idx = pd.date_range(start=start, end=end, freq="1h")
    rain_file_map = build_rain_file_map(RAIN_DIR)

    # Build MATLAB-like event matrix G_event: [time, station]
    G_cols = []
    for _, r in stations.iterrows():
        sid = r["ID_norm"]
        arr = load_station_event_series_as_matlab_style(
            sid, start, end, event_idx, rain_file_map
        )
        G_cols.append(arr)

    G_event = np.column_stack(G_cols)

    # Apply MATLAB bad-gauge masking by event year
    G_event = apply_matlab_bad_gauge_mask(year, G_event)

    # Coordinates / distances
    xs = []
    ys = []
    for _, r in stations.iterrows():
        x, y = latlon_to_utm(r["Longitude"], r["Latitude"])
        xs.append(x)
        ys.append(y)
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)

    # MATLAB-style matrices
    scorr = np.full((ng, ng), -99.0, dtype=float)
    sdist = np.zeros((ng, ng), dtype=float)

    for i in range(ng):
        for j in range(i + 1, ng):
            G1 = G_event[:, i].copy()
            G2 = G_event[:, j].copy()

            valid_pair, c = matlab_pair_correlation(G1, G2)
            if not valid_pair:
                sdist[i, j] = -99.0
                scorr[i, j] = -99.0
                continue

            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_km = np.sqrt(dx * dx + dy * dy) / 1000.0

            sdist[i, j] = dist_km
            scorr[i, j] = c

    # MATLAB equivalent:
    # i=find(scorr~=-99);
    # scorr2=scorr(i);
    # sdist2=sdist(i);
    valid_idx = np.where(scorr != -99.0)
    scorr2 = scorr[valid_idx]
    sdist2 = sdist[valid_idx]

    # Save MATLAB-style sample correlation table
    table = np.column_stack([sdist2, scorr2])
    samplecorr_path = PARAM_DIR / f"SampleCorr_E{event}.out"
    np.savetxt(samplecorr_path, table, fmt="%.10f")
    print(f"Saved: {samplecorr_path}")

    # MATLAB-style fitting
    x0 = np.array([30.0, 1.0], dtype=float)

    # MATLAB can fail here if scorr2 contains nan or is empty.
    # To keep the workflow usable, we skip failed events instead of crashing.
    try:
        finite = np.isfinite(sdist2) & np.isfinite(scorr2)
        sdist_fit = sdist2[finite]
        scorr_fit = scorr2[finite]

        if len(sdist_fit) == 0 or len(scorr_fit) == 0:
            raise ValueError("No finite pairs remain for fitting.")

        popt, _ = curve_fit(
            corr_model,
            sdist_fit,
            scorr_fit,
            p0=x0,
            maxfev=10000
        )
        a_km, b = popt

    except Exception as e:
        print(f"Skipping fit/plot for event {event}: {e}")
        return

    # Plot MATLAB-style
    plt.figure(figsize=(8, 6))
    plt.plot(sdist2, scorr2, ".", label="Station pairs")
    plt.grid(True)
    plt.xlabel("Distance (km)")
    plt.ylabel("Correlation")
    plt.title(f"Event {event}")

    dx = np.arange(0, 81, 1, dtype=float)
    ecorr = np.exp(-((dx / a_km) ** b))
    plt.plot(dx, ecorr, "r-", label=f"Fit: a={a_km:.3f}, b={b:.3f}")
    plt.legend()

    plot_path = PLOT_DIR / f"Event_{event}_correlation_fit_matlab_style.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path}")

    # Save MATLAB-style parameters
    para_path = PARAM_DIR / f"Para_E{event}.out"
    np.savetxt(para_path, np.array([a_km, b]), fmt="%.10f")
    print(f"Saved: {para_path}")
    print(f"a_km={a_km:.6f}, b={b:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="MATLAB-style sample correlation and exponential fit using Python directories."
    )
    parser.add_argument(
        "--event",
        type=int,
        nargs="+",
        required=True,
        help="One or more MATLAB-style event numbers (1-10)"
    )
    args = parser.parse_args()

    failed = []
    for event in args.event:
        try:
            run_event(event)
        except Exception as e:
            print("=" * 80)
            print(f"Skipping event {event} due to error:")
            print(str(e))
            print("=" * 80)
            failed.append((event, str(e)))

    if failed:
        print("\nSummary of skipped events:")
        for ev, msg in failed:
            print(f"  Event {ev}: {msg}")


if __name__ == "__main__":
    main()