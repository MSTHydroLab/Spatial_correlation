#!/usr/bin/env python3
"""
stormwatch_api_to_csv_station_chunked.py

USES YOUR WORKING APPROACH (api/graph/flot sensor_details) and keeps filenames like before:
  <station_no>_dev<device_id>.csv   (preferred, if station number can be scraped)
  site<site_id>_dev<device_id>.csv  (fallback)

Key points
- Chunked requests (weekly by default) to avoid memory errors.
- Tries BOTH UUID and numeric forms for site/device params.
- Chooses a NON-empty series (prefers Rain Increment, else max data length).
- Writes ONLY timestamps returned by Stormwatch (no synthetic timestamps).
- Verbose progress printing.
- Append/merge-safe: you can re-run and it will de-duplicate by timestamp.

pip install requests pandas python-dateutil
"""

import csv
import re
import time
import pathlib
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import pandas as pd
from dateutil.relativedelta import relativedelta


# ----------------------------
# CONFIG DEFAULTS (override via CLI)
# ----------------------------
BASE = "https://www.stormwatch.com/api/graph/flot/"
SITE_PAGE_URL = "https://www.stormwatch.com/site/"

DEFAULT_SITES_CSV = "/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/stormwatch_sensor_info.csv"
DEFAULT_OUT_DIR = "/mnt/12TB/Sujan/Spatial_correlation/Codes/SW_Downloads/data"

DEFAULT_START = "2012-01-01 00:00:00"
DEFAULT_END = "2026-02-01 23:59:59"
DEFAULT_LOCAL_TZ = "US/Central"

DEFAULT_BIN_SECONDS = 3600          # set 60 to match your sensor page examples
DEFAULT_CHUNK_DAYS = 7            # weekly chunks
DEFAULT_REQUEST_PAUSE = 0.6
DEFAULT_REQUEST_RETRIES = 3
DEFAULT_TIMEOUT_SEC = 90

LABEL_ID_RE = re.compile(r"\((\d+)\)")
MEM_ERR_RE = re.compile(r"allowed memory size.*exhausted", re.IGNORECASE)
H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class SensorRow:
    site_id: str
    device_id: str
    site_uuid: str = ""
    device_uuid: str = ""


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--in-csv", default=DEFAULT_SITES_CSV, help="stormwatch_sensor_info.csv")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output data folder")
    p.add_argument("--cookie", required=True, help="WEBAPP_SESSION cookie value")

    p.add_argument("--start", default=DEFAULT_START, help="Local start datetime YYYY-mm-dd HH:MM:SS")
    p.add_argument("--end", default=DEFAULT_END, help="Local end datetime YYYY-mm-dd HH:MM:SS")
    p.add_argument("--time-zone", default=DEFAULT_LOCAL_TZ, help="Stormwatch time_zone param, e.g. US/Central")
    p.add_argument("--bin", type=int, default=DEFAULT_BIN_SECONDS, help="bin seconds (60, 300, 3600 etc.)")

    p.add_argument("--chunk-days", type=int, default=DEFAULT_CHUNK_DAYS, help="Days per request chunk")
    p.add_argument("--sleep", type=float, default=DEFAULT_REQUEST_PAUSE, help="Sleep between requests (sec)")
    p.add_argument("--retries", type=int, default=DEFAULT_REQUEST_RETRIES, help="Retries per request")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SEC, help="HTTP timeout (sec)")

    p.add_argument("--overwrite", action="store_true", help="Overwrite output files instead of appending/merging")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


def log(msg: str, verbose: bool):
    if verbose:
        print(msg, flush=True)


# ----------------------------
# Time helpers
# ----------------------------
def parse_local_dt(s: str, tz: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo(tz))


def fmt_local_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def iter_day_chunks(start_dt: datetime, end_dt: datetime, days: int):
    t0 = start_dt
    while t0 <= end_dt:
        t1 = min(t0 + relativedelta(days=days) - relativedelta(seconds=1), end_dt)
        yield t0, t1
        t0 = t1 + relativedelta(seconds=1)


# ----------------------------
# Station number scrape (for filenames)
# ----------------------------
def extract_station_number(text: str) -> str | None:
    m = LABEL_ID_RE.search(text or "")
    return m.group(1) if m else None


def fetch_station_no_from_site_page(
    session: requests.Session,
    site_id: str,
    site_uuid: str,
    timeout: int,
) -> str | None:
    if not site_uuid:
        return None

    params = {"site_id": site_id, "site": site_uuid}
    r = session.get(SITE_PAGE_URL, params=params, timeout=timeout)
    r.raise_for_status()
    html = r.text or ""

    m = H1_RE.search(html)
    if m:
        h1_text = TAG_RE.sub("", m.group(1)).strip()
        st = extract_station_number(h1_text)
        if st:
            return st

    m = TITLE_RE.search(html)
    if m:
        title_text = TAG_RE.sub("", m.group(1)).strip()
        st = extract_station_number(title_text)
        if st:
            return st

    return extract_station_number(html)


# ----------------------------
# API call logic (your working approach)
# ----------------------------
def parse_label_numeric_id(label_html: str) -> str | None:
    m = LABEL_ID_RE.search(label_html or "")
    return m.group(1) if m else None


def is_memory_error_html(text: str) -> bool:
    return bool(MEM_ERR_RE.search(text or ""))


def build_param_variants(row: SensorRow, start_str: str, end_str: str, bin_seconds: int, tz: str) -> list[dict]:
    """
    Build TWO variants:
    - Variant A: site/device = UUIDs (if present)
    - Variant B: site/device = numeric ids (site_id/device_id)
    """
    site_id = str(row.site_id).strip()
    device_id = str(row.device_id).strip()
    site_uuid = str(row.site_uuid or "").strip()
    device_uuid = str(row.device_uuid or "").strip()

    common = {
        "method": "sensor_details",
        "site_id": site_id,
        "device_id": device_id,
        "data_start": start_str,
        "data_end": end_str,
        "forecast_time": "",
        "range": "",
        "bin": str(bin_seconds),
        "time_zone": tz,
        "show_assoc_sensors": "1",
    }

    variants = []

    # UUID variant first (if available)
    if site_uuid and device_uuid:
        v = common.copy()
        v["site"] = site_uuid
        v["device"] = device_uuid
        variants.append(v)

    # numeric fallback (matches what you pasted from DevTools for flot)
    v = common.copy()
    v["site"] = site_id
    v["device"] = device_id
    variants.append(v)

    return variants


def make_referer(params: dict) -> str:
    return (
        "https://www.stormwatch.com/sensor/"
        f"?site_id={params['site_id']}&site={params['site']}"
        f"&device_id={params['device_id']}&device={params['device']}"
        f"&data_start={params['data_start']}&data_end={params['data_end']}"
        f"&bin={params['bin']}&time_zone={params['time_zone']}"
        f"&show_raw=true&show_quality=true"
    )


def fetch_sensor_details(
    session: requests.Session,
    params: dict,
    retries: int,
    pause: float,
    timeout: int,
) -> list[dict]:
    headers = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": make_referer(params),
        "User-Agent": "Mozilla/5.0",
    }

    last_err = None
    for _ in range(retries):
        try:
            r = session.get(BASE, params=params, headers=headers, timeout=timeout, allow_redirects=True)
            ct = (r.headers.get("Content-Type") or "").lower()

            if "json" not in ct:
                txt = r.text or ""
                snippet = txt[:250].replace("\n", "\\n")
                if is_memory_error_html(txt):
                    raise RuntimeError(f"StormWatch memory error | {snippet}")
                raise RuntimeError(f"Non-JSON response: status={r.status_code}, ct={ct}, url={r.url} | {snippet}")

            js = r.json()
            if not isinstance(js, list):
                raise RuntimeError(f"Unexpected JSON root type: {type(js)} url={r.url}")
            return js

        except Exception as e:
            last_err = e
            time.sleep(pause)

    raise last_err


def best_series(js_list: list[dict]) -> dict | None:
    """
    Pick the best series:
    1) Rain Increment with non-empty data (case-insensitive contains)
    2) otherwise any series with max len(data)
    """
    if not js_list:
        return None

    rain_candidates = []
    for s in js_list:
        lbl = (s.get("label") or "")
        data = s.get("data") or []
        if ("rain increment" in lbl.lower()) and len(data) > 0:
            rain_candidates.append(s)
    if rain_candidates:
        return max(rain_candidates, key=lambda s: len(s.get("data") or []))

    return max(js_list, key=lambda s: len(s.get("data") or []))


def series_to_dataframe(series_obj: dict, tz: str) -> pd.DataFrame:
    data = series_obj.get("data") or []
    if not data:
        return pd.DataFrame(columns=["timestamp", "value", "units"])

    df = pd.DataFrame(data, columns=["epoch_ms", "value"])
    df["timestamp_utc"] = pd.to_datetime(df["epoch_ms"], unit="ms", utc=True)
    df["timestamp"] = df["timestamp_utc"].dt.tz_convert(ZoneInfo(tz))
    df["units"] = series_obj.get("units", "")
    return df[["timestamp", "value", "units"]].copy()


def load_sites_csv(path: str) -> list[SensorRow]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("No rows found in sites CSV.")

    out: list[SensorRow] = []
    for r in rows:
        site_id = str(r.get("site_id", "")).strip()
        device_id = str(r.get("device_id", "")).strip()
        if not site_id or not device_id:
            continue
        out.append(
            SensorRow(
                site_id=site_id,
                device_id=device_id,
                site_uuid=str(r.get("site", "")).strip(),
                device_uuid=str(r.get("device", "")).strip(),
            )
        )
    return out


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sensors = load_sites_csv(args.in_csv)

    sess = requests.Session()
    sess.cookies.set("WEBAPP_SESSION", args.cookie, domain="www.stormwatch.com", path="/")

    # Build site_id -> station_no for filenames (best-effort)
    site_to_station: dict[str, str | None] = {}
    for sr in sensors:
        if sr.site_id in site_to_station:
            continue
        try:
            st = fetch_station_no_from_site_page(sess, sr.site_id, sr.site_uuid, timeout=args.timeout)
        except Exception:
            st = None
        site_to_station[sr.site_id] = st
        log(f"site_id={sr.site_id} -> station_no={st}", args.verbose)
        time.sleep(args.sleep)

    start_dt = parse_local_dt(args.start, args.time_zone)
    end_dt = parse_local_dt(args.end, args.time_zone)

    n = len(sensors)
    for i, sr in enumerate(sensors, 1):
        site_id = sr.site_id
        device_id = sr.device_id

        station_no = site_to_station.get(site_id)
        if station_no:
            out_path = out_dir / f"{station_no}_dev{device_id}.csv"
        else:
            out_path = out_dir / f"site{site_id}_dev{device_id}.csv"

        print(f"[{i}/{n}] site_id={site_id} device_id={device_id} -> {out_path.name}", flush=True)

        chunks: list[pd.DataFrame] = []
        total_rows = 0

        for t0, t1 in iter_day_chunks(start_dt, end_dt, days=args.chunk_days):
            start_str = fmt_local_dt(t0)
            end_str = fmt_local_dt(t1)

            js_list = []
            chosen_params = None
            
            # 1. Try to find WHICH ID variant works (UUID or Numeric)
            for p in build_param_variants(sr, start_str, end_str, args.bin, args.time_zone):
                js_try = fetch_sensor_details(
                    sess, p, retries=args.retries, pause=args.sleep, timeout=args.timeout
                )
                s_try = best_series(js_try)
                
                # If we got ANY response (even empty), we know this ID format is valid
                if js_try is not None:
                    js_list = js_try
                    chosen_params = p
                    
                # If we actually found data, stop looking for variants
                if s_try and len(s_try.get("data", [])) > 0:
                    break 

            s_obj = best_series(js_list)
            # ... (rest of your logic)

            for p in build_param_variants(sr, start_str, end_str, args.bin, args.time_zone):
                js_try = fetch_sensor_details(
                    sess, p, retries=args.retries, pause=args.sleep, timeout=args.timeout
                )
                s_try = best_series(js_try)
                n_try = len((s_try or {}).get("data") or [])
                if n_try > 0:
                    js_list = js_try
                    chosen_params = p
                    break
                js_list = js_try
                chosen_params = p

            s_obj = best_series(js_list or [])
            npts = len((s_obj or {}).get("data") or [])
            used_site = chosen_params["site"] if chosen_params else "?"
            used_dev = chosen_params["device"] if chosen_params else "?"
            print(
                f"   {t0.date()}→{t1.date()} | using site={used_site} device={used_dev} | pts={npts}",
                flush=True,
            )

            if not s_obj or npts == 0:
                time.sleep(args.sleep)
                continue

            df = series_to_dataframe(s_obj, tz=args.time_zone)
            if not df.empty:
                chunks.append(df)
                total_rows += len(df)

            time.sleep(args.sleep)

        if total_rows == 0:
            print("   ! no data points returned for this sensor; no file written", flush=True)
            continue

        out_df = pd.concat(chunks, ignore_index=True)

        # If re-running, merge with existing (unless overwrite)
        if out_path.exists() and not args.overwrite:
            try:
                old = pd.read_csv(out_path, parse_dates=["timestamp"])
                out_df = pd.concat([old, out_df], ignore_index=True)
            except Exception:
                # if old format differs, still proceed with new
                pass

        # De-dupe by timestamp (exact readings only, no synthetic timestamps are created)
        out_df = out_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        out_df.to_csv(out_path, index=False)
        print(f"   ✓ saved -> {out_path} (rows={len(out_df)})", flush=True)

    print(f"\nDone. Output dir: {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
