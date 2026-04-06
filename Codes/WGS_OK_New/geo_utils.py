from __future__ import annotations

import math
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Accepts scalars or numpy arrays."""
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def initial_bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing clockwise from North in degrees [0, 360)."""
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dlmb = np.radians(lon2 - lon1)

    y = np.sin(dlmb) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlmb)
    ang = np.degrees(np.arctan2(y, x))
    return (ang + 360.0) % 360.0


def ang_sep_deg(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def inclusive_range(start: float, end: float, step: float, decimals: int = 10) -> np.ndarray:
    if step <= 0:
        raise ValueError("step must be > 0")
    if end < start:
        raise ValueError("end must be >= start")

    vals = []
    cur = float(start)
    tol = step * 1e-9
    while cur <= end + tol:
        vals.append(round(cur, decimals))
        cur += step
    return np.array(vals, dtype=float)


def build_regular_wgs84_grid(
    start_lat: float,
    end_lat: float,
    start_lon: float,
    end_lon: float,
    delta: float,
    lat_major: bool = True,
) -> pd.DataFrame:
    lats = inclusive_range(start_lat, end_lat, delta)
    lons = inclusive_range(start_lon, end_lon, delta)

    rows = []
    cell_id = 1
    if lat_major:
        outer_vals = lats
        inner_vals = lons
        outer_name = "Latitude"
        inner_name = "Longitude"
    else:
        outer_vals = lons
        inner_vals = lats
        outer_name = "Longitude"
        inner_name = "Latitude"

    for outer in outer_vals:
        for inner in inner_vals:
            if outer_name == "Latitude":
                lat = float(outer)
                lon = float(inner)
            else:
                lon = float(outer)
                lat = float(inner)
            rows.append({"id": cell_id, "Latitude": lat, "Longitude": lon})
            cell_id += 1

    return pd.DataFrame(rows)
