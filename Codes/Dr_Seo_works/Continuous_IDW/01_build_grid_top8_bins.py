#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import re
from bisect import bisect_right

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088
BASE_DIR = Path('/mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Continuous_IDW')
AVAIL_DIR = BASE_DIR / '00_station_availability'
GRID_INFO_DIR = BASE_DIR / '01_grid_information'
TOP8_DIR = BASE_DIR / '02_top8_available'
DEFAULT_GRID_CSV = Path('/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/dependent_files/grid_centers_wgs84.csv')
DEFAULT_STATIONS_CSV = Path('/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/dependent_files/Stations_df.csv')
DEFAULT_START = None
DEFAULT_END = None
DEFAULT_TOPK = 8


def norm_station_id(x) -> str:
    s = str(x).strip().strip("'").strip('"')
    if s == '' or s.lower() == 'nan':
        return ''
    try:
        return str(int(float(s)))
    except Exception:
        return s


def haversine_km(lat1, lon1, lat2, lon2):
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


def make_window(start_str: str | None, end_str: str | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start = pd.to_datetime(start_str, errors='raise') if start_str else None
    end = pd.to_datetime(end_str, errors='raise') if end_str else None
    if start is not None and end is not None and end < start:
        raise ValueError('end must be >= start')
    return start, end


def parse_station_id_from_availability_file(path: Path) -> str:
    m = re.search(r'(\d+)', path.stem)
    return norm_station_id(m.group(1)) if m else norm_station_id(path.stem)


def load_station_periods(avail_dir: Path, start: pd.Timestamp | None, end: pd.Timestamp | None) -> tuple[dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]], pd.DataFrame]:
    periods_by_station: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    summary_rows = []
    files = sorted(avail_dir.glob('*.csv'))
    files = [p for p in files if p.name != 'station_availability_summary.csv']
    if not files:
        raise FileNotFoundError(f'No per-station availability CSVs found in {avail_dir}')

    print(f'[1/4] Reading station availability from {avail_dir}')
    print(f'      Found {len(files)} station availability files')

    for i, fp in enumerate(files, start=1):
        sid = parse_station_id_from_availability_file(fp)
        df = pd.read_csv(fp)
        req = {'start_time', 'end_time', 'status'}
        if not req.issubset(df.columns):
            print(f'      Skipping {fp.name}: missing required columns {sorted(req - set(df.columns))}')
            continue
        df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
        df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')
        df = df.dropna(subset=['start_time', 'end_time'])
        df = df[df['status'].astype(str).str.lower() == 'available'].copy()
        if start is not None:
            df['start_time'] = df['start_time'].clip(lower=start)
        if end is not None:
            df['end_time'] = df['end_time'].clip(upper=end)
        df = df[df['end_time'] >= df['start_time']].copy()
        periods = list(zip(df['start_time'].tolist(), df['end_time'].tolist()))
        periods.sort(key=lambda x: x[0])
        periods_by_station[sid] = periods

        summary_rows.append({
            'station_id': sid,
            'availability_file': fp.name,
            'n_available_periods_in_window': len(periods),
            'first_available_time': periods[0][0] if periods else pd.NaT,
            'last_available_time': periods[-1][1] if periods else pd.NaT,
        })
        if i <= 5 or i == len(files) or i % 25 == 0:
            print(f'      [{i}/{len(files)}] station {sid}: {len(periods)} available periods in requested window')

    return periods_by_station, pd.DataFrame(summary_rows)


def collect_change_times(periods_by_station: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]], start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    change_times = {start, end + pd.Timedelta(hours=1)}
    for periods in periods_by_station.values():
        for s, e in periods:
            if s <= end and e >= start:
                ss = max(s, start)
                ee = min(e, end)
                change_times.add(ss)
                after = ee + pd.Timedelta(hours=1)
                if after <= end + pd.Timedelta(hours=1):
                    change_times.add(after)
    out = sorted(change_times)
    return out


def is_active_at(periods: list[tuple[pd.Timestamp, pd.Timestamp]], t: pd.Timestamp) -> bool:
    # Periods are sorted and non-overlapping enough for linear scan to be fine at station scale
    for s, e in periods:
        if s <= t <= e:
            return True
        if s > t:
            return False
    return False


def build_bins(periods_by_station: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    print('[2/4] Building availability bins from station periods')
    change_times = collect_change_times(periods_by_station, start, end)
    print(f'      Candidate change times: {len(change_times)}')

    bins = []
    prev_sig = None
    current_start = None
    bin_id = 0

    for idx in range(len(change_times) - 1):
        t0 = change_times[idx]
        t1 = change_times[idx + 1] - pd.Timedelta(hours=1)
        if t0 > end or t1 < start or t1 < t0:
            continue
        active = [sid for sid, periods in periods_by_station.items() if is_active_at(periods, t0)]
        active.sort()
        sig = tuple(active)

        if prev_sig is None:
            current_start = t0
            prev_sig = sig
        elif sig != prev_sig:
            bin_id += 1
            bins.append({
                'bin_id': bin_id,
                'start_time': current_start,
                'end_time': t0 - pd.Timedelta(hours=1),
                'n_active_stations': len(prev_sig),
                'active_station_ids': '|'.join(prev_sig),
            })
            current_start = t0
            prev_sig = sig

    if prev_sig is not None and current_start is not None:
        bin_id += 1
        bins.append({
            'bin_id': bin_id,
            'start_time': current_start,
            'end_time': end,
            'n_active_stations': len(prev_sig),
            'active_station_ids': '|'.join(prev_sig),
        })

    bins_df = pd.DataFrame(bins)
    print(f'      Final bins written in memory: {len(bins_df)}')
    return bins_df


def build_grid_topk_for_bin(grid_df: pd.DataFrame, stations_df: pd.DataFrame, active_station_ids: set[str], topk: int) -> pd.DataFrame:
    active = stations_df[stations_df['ID'].isin(active_station_ids)].copy()
    out_rows = []
    if active.empty:
        for _, row in grid_df.iterrows():
            rec = {'id': str(row['id']), 'Latitude': float(row['Latitude']), 'Longitude': float(row['Longitude']), 'n_active_candidates': 0}
            for k in range(1, topk + 1):
                rec[f'g{k}'] = ''
                rec[f'd{k}_m'] = np.nan
                rec[f'b{k}_deg'] = np.nan
            out_rows.append(rec)
        return pd.DataFrame(out_rows)

    stn_lat = active['Latitude'].to_numpy(dtype=float)
    stn_lon = active['Longitude'].to_numpy(dtype=float)
    stn_id = active['ID'].astype(str).to_numpy()

    for _, row in grid_df.iterrows():
        glat = float(row['Latitude'])
        glon = float(row['Longitude'])
        d_km = haversine_km(glat, glon, stn_lat, stn_lon)
        b_deg = initial_bearing_deg(glat, glon, stn_lat, stn_lon)
        order = np.argsort(d_km)
        order = order[:topk]

        rec = {
            'id': str(row['id']),
            'Latitude': glat,
            'Longitude': glon,
            'n_active_candidates': int(len(stn_id)),
        }
        for pos in range(topk):
            if pos < len(order):
                ii = int(order[pos])
                rec[f'g{pos+1}'] = str(stn_id[ii])
                rec[f'd{pos+1}_m'] = float(d_km[ii] * 1000.0)
                rec[f'b{pos+1}_deg'] = float(b_deg[ii])
            else:
                rec[f'g{pos+1}'] = ''
                rec[f'd{pos+1}_m'] = np.nan
                rec[f'b{pos+1}_deg'] = np.nan
        out_rows.append(rec)
    return pd.DataFrame(out_rows)


def main():
    ap = argparse.ArgumentParser(description='Build compact availability bins and top-k available gauges for each grid.')
    ap.add_argument('--availability-dir', type=Path, default=AVAIL_DIR)
    ap.add_argument('--grid-csv', type=Path, default=DEFAULT_GRID_CSV)
    ap.add_argument('--stations-csv', type=Path, default=DEFAULT_STATIONS_CSV)
    ap.add_argument('--grid-info-dir', type=Path, default=GRID_INFO_DIR)
    ap.add_argument('--top8-dir', type=Path, default=TOP8_DIR)
    ap.add_argument('--start', type=str, default=DEFAULT_START)
    ap.add_argument('--end', type=str, default=DEFAULT_END)
    ap.add_argument('--topk', type=int, default=DEFAULT_TOPK)
    args = ap.parse_args()

    start, end = make_window(args.start, args.end)
    if start is None or end is None:
        raise SystemExit('This script requires --start and --end so bins are created for a specific period.')

    args.grid_info_dir.mkdir(parents=True, exist_ok=True)
    args.top8_dir.mkdir(parents=True, exist_ok=True)

    periods_by_station, station_summary_df = load_station_periods(args.availability_dir, start, end)

    grid_df = pd.read_csv(args.grid_csv)
    stations_df = pd.read_csv(args.stations_csv)
    grid_df['id'] = grid_df['id'].astype(str)
    stations_df['ID'] = stations_df['ID'].apply(norm_station_id)
    grid_df['Latitude'] = pd.to_numeric(grid_df['Latitude'], errors='coerce')
    grid_df['Longitude'] = pd.to_numeric(grid_df['Longitude'], errors='coerce')
    stations_df['Latitude'] = pd.to_numeric(stations_df['Latitude'], errors='coerce')
    stations_df['Longitude'] = pd.to_numeric(stations_df['Longitude'], errors='coerce')
    grid_df = grid_df.dropna(subset=['id', 'Latitude', 'Longitude']).copy()
    stations_df = stations_df.dropna(subset=['ID', 'Latitude', 'Longitude']).copy()

    bins_df = build_bins(periods_by_station, start, end)
    if bins_df.empty:
        raise SystemExit('No bins were created for the requested window.')

    print('[3/4] Writing bin metadata and station window summary')
    bins_meta_csv = args.grid_info_dir / f'availability_bins_{start:%Y%m%d%H}_{end:%Y%m%d%H}.csv'
    station_window_csv = args.grid_info_dir / f'station_window_summary_{start:%Y%m%d%H}_{end:%Y%m%d%H}.csv'
    bins_df.to_csv(bins_meta_csv, index=False)
    station_summary_df.to_csv(station_window_csv, index=False)
    print(f'      Saved bin metadata: {bins_meta_csv}')
    print(f'      Saved station summary: {station_window_csv}')

    print('[4/4] Building per-bin grid top-k available gauge CSVs')
    bin_manifest_rows = []
    for i, row in bins_df.iterrows():
        bin_id = int(row['bin_id'])
        active_ids = set(x for x in str(row['active_station_ids']).split('|') if x != '')
        bin_grid_df = build_grid_topk_for_bin(grid_df, stations_df, active_ids, args.topk)
        out_csv = args.top8_dir / f'bin_{bin_id:04d}_grid_top{args.topk}.csv'
        bin_grid_df.to_csv(out_csv, index=False)
        bin_manifest_rows.append({
            'bin_id': bin_id,
            'start_time': row['start_time'],
            'end_time': row['end_time'],
            'n_active_stations': int(row['n_active_stations']),
            'topk': int(args.topk),
            'grid_csv': out_csv.name,
        })
        if i < 5 or i == len(bins_df) - 1 or (i + 1) % 10 == 0:
            print(f'      [{i+1}/{len(bins_df)}] bin {bin_id:04d}: {len(active_ids)} active stations -> {out_csv.name}')

    manifest_df = pd.DataFrame(bin_manifest_rows)
    manifest_csv = args.grid_info_dir / f'bin_grid_manifest_{start:%Y%m%d%H}_{end:%Y%m%d%H}.csv'
    manifest_df.to_csv(manifest_csv, index=False)
    print(f'      Saved manifest: {manifest_csv}')
    print('Done.')


if __name__ == '__main__':
    main()
