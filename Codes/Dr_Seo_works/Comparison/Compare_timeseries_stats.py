#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def normalize_grid_col(col) -> str:
    s = str(col).strip()
    if s.lower().startswith("unnamed"):
        return s
    try:
        f = float(s)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return s
    except Exception:
        return s


def load_rain_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if df.shape[1] < 2:
        raise ValueError(f"{path} has fewer than 2 columns")

    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    bad_time = int(df[time_col].isna().sum())
    if bad_time > 0:
        raise ValueError(f"{path} has {bad_time} unparseable timestamps in column '{time_col}'")

    df = df.set_index(time_col)

    df.columns = [normalize_grid_col(c) for c in df.columns]

    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_index()
    df = df.sort_index(axis=1)
    return df


def append_domain_sum_row(
    df: pd.DataFrame,
    appended_time: pd.Timestamp,
    label_col_name: str | None = None,
) -> pd.DataFrame:
    """
    Appends one extra row where each grid column contains the column sum over time.
    This lets downstream plotters show a 'sum over whole event' snapshot.
    """
    if appended_time in df.index:
        raise ValueError(
            f"Chosen appended timestamp {appended_time} already exists in the data. "
            "Choose another timestamp with --append-time."
        )

    sum_row = pd.DataFrame([df.sum(axis=0, skipna=True)], index=[appended_time])
    out = pd.concat([df, sum_row], axis=0)

    if label_col_name is not None:
        out = out.copy()
        out[label_col_name] = ""
        out.loc[out.index[:-1], label_col_name] = "observed"
        out.loc[appended_time, label_col_name] = "domain_sum_over_time"

    return out


def overall_metrics(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    aval = a.to_numpy(dtype=float)
    bval = b.to_numpy(dtype=float)

    mask = np.isfinite(aval) & np.isfinite(bval)
    n = int(mask.sum())

    if n == 0:
        return {
            "n_values_compared": 0,
            "input1_total_mm": np.nan,
            "input2_total_mm": np.nan,
            "difference_total_mm": np.nan,
            "mean_difference_mm": np.nan,
            "mae_mm": np.nan,
            "rmse_mm": np.nan,
        }

    x = aval[mask]
    y = bval[mask]
    d = y - x

    return {
        "n_values_compared": n,
        "input1_total_mm": float(np.sum(x)),
        "input2_total_mm": float(np.sum(y)),
        "difference_total_mm": float(np.sum(d)),
        "mean_difference_mm": float(np.mean(d)),
        "mae_mm": float(np.mean(np.abs(d))),
        "rmse_mm": float(np.sqrt(np.mean(d ** 2))),
    }


def per_grid_metrics(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for gid in a.columns:
        x = a[gid].to_numpy(dtype=float)
        y = b[gid].to_numpy(dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        n = int(mask.sum())

        if n == 0:
            rows.append({
                "grid_id": gid,
                "n": 0,
                "input1_total_mm": np.nan,
                "input2_total_mm": np.nan,
                "difference_total_mm": np.nan,
                "mean_difference_mm": np.nan,
                "mae_mm": np.nan,
                "rmse_mm": np.nan,
            })
            continue

        xv = x[mask]
        yv = y[mask]
        dv = yv - xv

        rows.append({
            "grid_id": gid,
            "n": n,
            "input1_total_mm": float(np.sum(xv)),
            "input2_total_mm": float(np.sum(yv)),
            "difference_total_mm": float(np.sum(dv)),
            "mean_difference_mm": float(np.mean(dv)),
            "mae_mm": float(np.mean(np.abs(dv))),
            "rmse_mm": float(np.sqrt(np.mean(dv ** 2))),
        })

    return pd.DataFrame(rows)


def build_domain_timeseries(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    ts = pd.DataFrame(index=a.index)
    ts["input1_domain_sum_mm"] = a.sum(axis=1, skipna=True)
    ts["input2_domain_sum_mm"] = b.sum(axis=1, skipna=True)
    ts["difference_domain_sum_mm"] = ts["input2_domain_sum_mm"] - ts["input1_domain_sum_mm"]
    ts = ts.reset_index().rename(columns={ts.index.name if ts.index.name else "index": "time_local"})
    return ts


def main():
    ap = argparse.ArgumentParser(
        description="Compare two rainfall grid time series files over a specified date range."
    )
    ap.add_argument("--input1", required=True, help="First rainfall time series CSV")
    ap.add_argument("--input2", required=True, help="Second rainfall time series CSV")
    ap.add_argument("--start", required=True, help="Start datetime, e.g. 2017-07-23 00:00:00")
    ap.add_argument("--end", required=True, help="End datetime, e.g. 2017-07-23 12:00:00")
    ap.add_argument("--out-dir", required=True, help="Output folder")
    ap.add_argument(
        "--append-time",
        default="2100-01-01 00:00:00",
        help="Arbitrary time to append at end for domain/grid sums"
    )
    ap.add_argument(
        "--append-label-column",
        action="store_true",
        help="Add a helper text column marking normal rows vs appended sum row"
    )
    args = ap.parse_args()

    input1 = Path(args.input1)
    input2 = Path(args.input2)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = pd.to_datetime(args.start, errors="raise")
    end = pd.to_datetime(args.end, errors="raise")
    append_time = pd.to_datetime(args.append_time, errors="raise")

    if end < start:
        raise ValueError("--end must be >= --start")

    df1 = load_rain_df(input1)
    df2 = load_rain_df(input2)

    df1 = df1.loc[(df1.index >= start) & (df1.index <= end)].copy()
    df2 = df2.loc[(df2.index >= start) & (df2.index <= end)].copy()

    if df1.empty:
        raise ValueError(f"No rows found in input1 within {start} to {end}")
    if df2.empty:
        raise ValueError(f"No rows found in input2 within {start} to {end}")

    common_time = df1.index.intersection(df2.index)
    common_cols = df1.columns.intersection(df2.columns)

    if len(common_time) == 0:
        raise ValueError("No overlapping timestamps between the two input files in the requested period")
    if len(common_cols) == 0:
        raise ValueError("No overlapping grid columns between the two input files")

    a = df1.loc[common_time, common_cols].copy()
    b = df2.loc[common_time, common_cols].copy()

    diff = b - a

    metrics_overall = overall_metrics(a, b)
    metrics_overall.update({
        "input1_file": str(input1),
        "input2_file": str(input2),
        "start_used": str(common_time.min()),
        "end_used": str(common_time.max()),
        "n_timesteps": int(len(common_time)),
        "n_common_grids": int(len(common_cols)),
    })

    df_overall = pd.DataFrame([metrics_overall])
    df_grid_metrics = per_grid_metrics(a, b)
    df_domain_ts = build_domain_timeseries(a, b)

    label_col = "row_type" if args.append_label_column else None

    a_out = append_domain_sum_row(a, append_time, label_col_name=label_col)
    b_out = append_domain_sum_row(b, append_time, label_col_name=label_col)
    diff_out = append_domain_sum_row(diff, append_time, label_col_name=label_col)

    a_out = a_out.reset_index().rename(columns={"index": "time_local"})
    b_out = b_out.reset_index().rename(columns={"index": "time_local"})
    diff_out = diff_out.reset_index().rename(columns={"index": "time_local"})

    df_overall.to_csv(out_dir / "comparison_overall_stats.csv", index=False)
    df_grid_metrics.to_csv(out_dir / "comparison_per_grid_stats.csv", index=False)
    df_domain_ts.to_csv(out_dir / "comparison_domain_timeseries.csv", index=False)

    a_out.to_csv(out_dir / "input1_clipped_with_sumrow.csv", index=False)
    b_out.to_csv(out_dir / "input2_clipped_with_sumrow.csv", index=False)
    diff_out.to_csv(out_dir / "difference_input2_minus_input1_with_sumrow.csv", index=False)

    print("=" * 80)
    print("Comparison finished")
    print(f"Input1: {input1}")
    print(f"Input2: {input2}")
    print(f"Output folder: {out_dir}")
    print(f"Common timesteps: {len(common_time)}")
    print(f"Common grids: {len(common_cols)}")
    print(f"Total rainfall input1: {metrics_overall['input1_total_mm']:.3f} mm")
    print(f"Total rainfall input2: {metrics_overall['input2_total_mm']:.3f} mm")
    print(f"Total difference (input2 - input1): {metrics_overall['difference_total_mm']:.3f} mm")
    print("=" * 80)


if __name__ == "__main__":
    main()