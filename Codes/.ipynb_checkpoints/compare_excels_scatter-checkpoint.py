#!/usr/bin/env python3
"""
Compare two Excel timeseries files with hourly "Date" column and station columns.

Outputs (per station):
  - scatter plot PNG with correlation value
Also outputs:
  - summary CSV with correlation and basic error stats

Usage:
  python compare_excels_scatter.py --file_a A.xlsx --file_b B.xlsx --out_dir out
Optional:
  --sheet_a Sheet1 --sheet_b Sheet1
  --date_col Date
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# EDIT THIS LIST ONLY
# Put your station column names exactly as they appear in Excel header
# Example: ["06893000", "06934500", ...]
# -----------------------------
STATIONS = ["3620","3600","7010","2540","2520","3810","3770","16048","16035","16040","16049",
    "2800","2320","2700","2710","2420","5110","2440","2460","4080","4060","4040","4160","2500","16093",
    "5600","16094","4920","16089","5700","5010","4910","4900","5800","5900","5170","5100","5200","5400",
    "6000","16087","16084","16085","16088","16086","16068","16073","16071","16069","16070","16072","16074",
    "16037","16038","16039","3640","2610","3960","3720","3860","16018","16007","16014","16028","16017","16016",
    "16015","16098","16013","16011","16006","16096","16029","16010","16008","16095","2790","1720","16105","16083",
    "16080","16076","16077","16082","16078","16081","16079","2730","2720","2890","2740","2840","4200","4000","16100",
    "16101","16102","16104","16103","4955","4960","4965","4970","4975","4935","4930","4940","4945","4950","3920","16067",
    "16055","16066","16057","16061","16075","16065","16064","16062","16059","16060","16058","16063","3700","2620","2600",
    "2650","3980","2630","2640","16030","16043","16042","16044","16045","3820","3800","4150","3740","3900","3840","3680",
    "3880","7020","7000","7060","7030","7080","7070","7050","3940","3760","3660","3690","16025","16023","16024","16021",
    "16022","16026","16027","16020","16019","16012","16053","16090","16052","16097","16054","2980","16051","3000","2900",
    "1010","16032","16033","16031","16046","16036","16047","16034"
]


def load_timeseries(path: Path, sheet: str | None, date_col: str) -> pd.DataFrame:
    """
    Load Excel or CSV time series.
    - If Excel has multiple sheets and sheet=None, uses the first sheet.
    """
    suffix = path.suffix.lower()

    if suffix in [".xls", ".xlsx"]:
        if sheet is None:
            dfs = pd.read_excel(path, sheet_name=None)
            df = next(iter(dfs.values()))
        else:
            df = pd.read_excel(path, sheet_name=sheet)

    elif suffix == ".csv":
        df = pd.read_csv(path)

    else:
        raise ValueError(f"Unsupported file type: {path}")

    # ---- from here on, df is GUARANTEED to exist ----
    df = df.copy()

    # force column names to strings for station matching
    df.columns = [str(c).strip() for c in df.columns]

    df = df.replace(-99, np.nan)

    if date_col not in df.columns:
        raise ValueError(
            f"{path.name}: date column '{date_col}' not found.\n"
            f"Columns found: {list(df.columns)[:20]}"
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
    df = df.set_index(date_col)

    # average duplicate timestamps safely
    if df.index.duplicated().any():
        df = df.groupby(level=0).mean(numeric_only=True)
    # Round all numeric station columns to 2 decimals to remove precision mismatch
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].round(2)

    return df




def safe_numeric(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def compute_stats(a: np.ndarray, b: np.ndarray):
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n < 2:
        return {
            "N": n,
            "r": np.nan,
            "bias_mean": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "max_pos_err": np.nan,
            "max_neg_err": np.nan,
            "mean_ratio_pointwise": np.nan,
            "ratio_of_means": np.nan,
        }

    aa = a[mask]
    bb = b[mask]
    diff = bb - aa

    r = float(np.corrcoef(aa, bb)[0, 1])

    ratio = np.full(diff.shape, np.nan, dtype=float)
    nz = aa != 0.0
    ratio[nz] = bb[nz] / aa[nz]

    meanA = float(np.mean(aa))
    meanB = float(np.mean(bb))
    rom = (meanB / meanA) if (np.isfinite(meanA) and meanA != 0.0) else np.nan

    return {
        "N": n,
        "r": r,
        "bias_mean": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "max_pos_err": float(np.max(diff)),
        "max_neg_err": float(np.min(diff)),
        "mean_ratio_pointwise": float(np.nanmean(ratio)),
        "ratio_of_means": float(rom),
    }


def build_bias_timeseries(joined: pd.DataFrame, st: str) -> pd.DataFrame:
    """
    Returns a dataframe indexed by Date with:
      A, B, bias (B-A), ratio (B/A), pct_bias
    """
    a = pd.to_numeric(joined[f"{st}_A"], errors="coerce")
    b = pd.to_numeric(joined[f"{st}_B"], errors="coerce")

    out = pd.DataFrame(index=joined.index)
    out["A"] = a
    out["B"] = b
    out["bias"] = out["B"] - out["A"]

    # ratio and percent bias, avoid dividing by zero
    denom = out["A"].to_numpy(dtype=float)
    out["ratio"] = np.where(np.isfinite(denom) & (denom != 0), out["B"].to_numpy(dtype=float) / denom, np.nan)
    out["pct_bias"] = np.where(np.isfinite(denom) & (denom != 0), 100.0 * out["bias"].to_numpy(dtype=float) / denom, np.nan)

    return out


def yearly_station_summary(ts: pd.DataFrame) -> pd.DataFrame:
    """
    Year-to-year summary for one station based on the bias time series.
    """
    df = ts.copy()
    df["year"] = df.index.year

    def agg(g):
        A = g["A"].to_numpy(dtype=float)
        B = g["B"].to_numpy(dtype=float)
        mask = np.isfinite(A) & np.isfinite(B)
        A = A[mask]
        B = B[mask]

        if A.size < 2:
            return pd.Series({
                "N": int(A.size),
                "bias_mean": np.nan,
                "bias_median": np.nan,
                "rmse": np.nan,
                "mean_ratio_pointwise": np.nan,
                "ratio_of_means": np.nan,
            })

        diff = B - A

        # mean of pointwise ratios B/A (avoid A=0)
        ratio = np.full(diff.shape, np.nan, dtype=float)
        nz = A != 0.0
        ratio[nz] = B[nz] / A[nz]

        meanA = float(np.mean(A))
        meanB = float(np.mean(B))
        rom = (meanB / meanA) if (np.isfinite(meanA) and meanA != 0.0) else np.nan

        return pd.Series({
            "N": int(A.size),
            "bias_mean": float(np.mean(diff)),
            "bias_median": float(np.median(diff)),
            "rmse": float(np.sqrt(np.mean(diff**2))),
            "mean_ratio_pointwise": float(np.nanmean(ratio)),
            "ratio_of_means": float(rom),
        })

    return df.groupby("year", dropna=True).apply(agg).reset_index()


def scatter_plot(station: str, a: np.ndarray, b: np.ndarray, stats: dict, out_png: Path):
    mask = np.isfinite(a) & np.isfinite(b)
    aa = a[mask]
    bb = b[mask]

    plt.figure()
    plt.scatter(aa, bb, s=10)

    # 1:1 line
    if aa.size > 0:
        lo = float(np.nanmin([aa.min(), bb.min()]))
        hi = float(np.nanmax([aa.max(), bb.max()]))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            plt.plot([lo, hi], [lo, hi])

    r_txt = "nan" if not np.isfinite(stats["r"]) else f"{stats['r']:.3f}"
    plt.title(f"Station {station} | r = {r_txt} | N = {stats['N']}")
    plt.xlabel("File A")
    plt.ylabel("File B")
    plt.tight_layout()
    plt.savefig(out_png, dpi=args.dpi)
    plt.close()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--file_a", required=True, help="First file (Excel or CSV)")
    p.add_argument("--file_b", required=True, help="Second file (Excel or CSV)")
    p.add_argument("--sheet_a", default=None, help="Sheet name for file A (Excel only)")
    p.add_argument("--sheet_b", default=None, help="Sheet name for file B (Excel only)")
    p.add_argument("--date_col", default="Date", help="Datetime column name")
    p.add_argument("--out_dir", default="excel_compare_out", help="Output directory")
    p.add_argument("--min_points", type=int, default=10, help="Min paired points for plotting")
    p.add_argument("--dpi", type=int, default=180, help="Plot DPI")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    ratio_hist_dir = out_dir / "ratio_bias_histograms"
    ratio_hist_dir.mkdir(parents=True, exist_ok=True)   
    ratio_ts_dir = out_dir / "ratio_bias_timeseries"
    ratio_ts_dir.mkdir(parents=True, exist_ok=True)

    bias_dir = out_dir / "bias_timeseries_excels"
    bias_dir.mkdir(parents=True, exist_ok=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_a = load_timeseries(Path(args.file_a), args.sheet_a, args.date_col)
    df_b = load_timeseries(Path(args.file_b), args.sheet_b, args.date_col)

    # inner-join on Date index (keeps only datetimes present in both)
    joined = df_a.join(df_b, how="inner", lsuffix="_A", rsuffix="_B")
    
    # -------------------------------------------------
    # Histogram of RATIO BIAS across ALL stations & times
    # ratio_bias = (B / A) - 1
    # -------------------------------------------------

    all_ratio_bias = []

    for st in STATIONS:
        col_a = f"{st}_A"
        col_b = f"{st}_B"

        if col_a not in joined.columns or col_b not in joined.columns:
            continue

        # -------------------------------------------------
        # Ratio-bias histogram for this station
        # ratio_bias = (B / A) - 1
        # -------------------------------------------------

        A = pd.to_numeric(joined[col_a], errors="coerce").to_numpy(dtype=float)
        B = pd.to_numeric(joined[col_b], errors="coerce").to_numpy(dtype=float)

        mask = np.isfinite(A) & np.isfinite(B) & (A != 0.0)
        ratio_bias = (B[mask] / A[mask]) - 1.0
        bins = np.arange(-1.0, 1.01, 0.01)  # from −100% to +100%, step = 1%

        if ratio_bias.size >= args.min_points:
            plt.figure(figsize=(6, 4))
            plt.hist(ratio_bias, bins=bins, edgecolor="black", log=True)
            #plt.axvline(0, linestyle="--")
            plt.xlabel("Ratio Bias  (Hari / Princeton)")
            plt.ylabel("Frequency (log scale)")
            plt.title(f"Station {st} | Ratio Bias Histogram")
            plt.tight_layout()
            plt.grid(True, which="both", axis="both", linestyle=":", linewidth=0.5)
            plt.savefig(ratio_hist_dir / f"ratio_bias_hist_{st}.png", dpi=args.dpi)
            plt.close()

        A = pd.to_numeric(joined[col_a], errors="coerce")
        B = pd.to_numeric(joined[col_b], errors="coerce")

        mask = A.notna() & B.notna() & (A != 0)
        rb = (B[mask] / A[mask])   # ratio bias series with datetime index

        if rb.size >= args.min_points:
            rb_m = rb.resample("MS").mean()   # monthly start
            plt.figure(figsize=(10, 3.5))
            plt.plot(rb_m.index, rb_m.values, marker="o", linewidth=1.2)
            plt.grid(True, which="both", axis="both", linestyle=":", linewidth=0.8)
            plt.xlabel("Time")
            plt.ylabel("Mean Ratio Bias (Hari/Princeton) (monthly)")
            plt.ylim(0, 3)
            plt.title(f"Station {st} | Monthly Mean Ratio Bias")
            plt.tight_layout()
            plt.savefig(ratio_ts_dir / f"ratio_bias_monthly_{st}.png", dpi=args.dpi)
            plt.close()
  
    results = []
    missing = []

    for st in STATIONS:
        col_a = f"{st}_A"
        col_b = f"{st}_B"

        if col_a not in joined.columns or col_b not in joined.columns:
            missing.append(st)
            continue
        # ---- Export bias time series for this station ----
        ts = build_bias_timeseries(joined, st)

        # Save each station to its own Excel file with 2 sheets:
        # 1) timeseries  2) yearly_stats
        ts_out = bias_dir / f"bias_timeseries_station_{st}.xlsx"
        with pd.ExcelWriter(ts_out, engine="openpyxl") as writer:
            ts.to_excel(writer, sheet_name="timeseries")
            yearly_station_summary(ts).to_excel(writer, sheet_name="yearly_stats", index=False)


        a = safe_numeric(joined[col_a])
        b = safe_numeric(joined[col_b])

        stats = compute_stats(a, b)
        stats_row = {"station": st, **stats}
        results.append(stats_row)

        if stats["N"] >= args.min_points:
            out_png = plots_dir / f"scatter_{st}.png"
            # note: scatter_plot uses args.dpi; keep it simple
            mask = np.isfinite(a) & np.isfinite(b)
            aa = a[mask]
            bb = b[mask]

            plt.figure()
            plt.scatter(aa, bb, s=10)
            text = (
                f"Max +err: {stats['max_pos_err']:.2f}\n"
                f"Max -err: {stats['max_neg_err']:.2f}"
            )

            plt.text(
                0.05, 0.95,
                text,
                transform=plt.gca().transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
            )

            if aa.size > 0:
                lo = float(np.nanmin([aa.min(), bb.min()]))
                hi = float(np.nanmax([aa.max(), bb.max()]))
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    plt.plot([lo, hi], [lo, hi])

            r_txt = "nan" if not np.isfinite(stats["r"]) else f"{stats['r']:.3f}"
            plt.title(f"Station {st} | r = {r_txt} | N = {stats['N']}")
            plt.xlabel("File A")
            plt.ylabel("File B")
            plt.tight_layout()
            plt.savefig(out_png, dpi=args.dpi)
            plt.close()
    print(f"Stations requested: {len(STATIONS)}")
    print(f"Stations found in both files: {len(results)}")
    print(f"Stations missing (column not found): {len(missing)}")

    # Save summary
    summary_df = pd.DataFrame(results)
    if summary_df.empty:
        summary_df = pd.DataFrame(columns=[
            "station","N","r","bias_mean","mae","rmse",
            "max_pos_err","max_neg_err","mean_ratio_pointwise","ratio_of_means"
        ])
    else:
        summary_df = summary_df.sort_values("station")


    summary_csv = out_dir / "summary_stats.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Save missing list
    if missing:
        (out_dir / "missing_stations.txt").write_text("\n".join(missing) + "\n")

    print(f"Done.\n- Summary: {summary_csv}\n- Plots: {plots_dir}")
    if missing:
        print(f"- Missing stations (column not found in one of the files): {len(missing)} -> {out_dir/'missing_stations.txt'}")
