#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd


data_dir = Path("/mnt/12TB/Sujan/Spatial_correlation/Raw_rain_measurement/OneDrive_1_1-26-2026/stormwatch_compiled_rawstyle_mm")
output_dir = Path("/mnt/12TB/Sujan/Spatial_correlation/Raw_rain_measurement/OneDrive_1_1-26-2026/Hari/stormwatch_instantaneous/")
output_dir.mkdir(parents=True, exist_ok=True)

AGG_FUNC = "sum"   # usually sum for rainfall amounts


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
    cols_norm = {c.strip().lower(): c for c in df.columns}
    for want in candidates:
        if want in cols_norm:
            return cols_norm[want]
    raise KeyError(f"None of these columns found: {candidates}. Have: {list(df.columns)}")


def aggregate_one_csv(in_csv: Path, out_csv: Path) -> dict:
    df = pd.read_csv(in_csv)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Find Reading/Value even if spacing/case differs
    reading_col = _pick_column(df, ["reading"])
    value_col   = _pick_column(df, ["value"])

    # Clean and parse
    reading_raw = df[reading_col].astype(str).str.strip()
    value_raw = df[value_col].astype(str).str.strip()

    # Parse Reading as UTC
    t = pd.to_datetime(reading_raw, utc=True, errors="coerce")

    # Parse Value
    v = pd.to_numeric(
        value_raw
        .str.replace(",", "", regex=False)
        .str.replace("mm", "", regex=False)
        .str.replace("in", "", regex=False)
        .str.strip(),
        errors="coerce"
    )

    n_total = len(df)
    n_time_ok = int(t.notna().sum())
    n_val_ok = int(v.notna().sum())

    out = pd.DataFrame({"Reading": t, "Value": v}).dropna(subset=["Reading", "Value"])
    n_kept = len(out)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if n_kept == 0:
        pd.DataFrame(columns=["bin_start_utc", "bin_end_utc", "mm_per_hr"]).to_csv(out_csv, index=False)
        return {
            "file": str(in_csv),
            "rows_total": n_total,
            "time_parsed_ok": n_time_ok,
            "value_parsed_ok": n_val_ok,
            "rows_kept_after_dropna": n_kept,
            "bins_written": 0,
        }

    # 5-min bin start; only bins that exist
    out["bin_start_utc"] = out["Reading"].dt.floor("5min")
    out["bin_end_utc"] = out["bin_start_utc"] + pd.Timedelta(minutes=5)

    # Aggregate to 5-min depth (sum is typical)
    grp = out.groupby("bin_start_utc")["Value"]
    if AGG_FUNC == "sum":
        val_5min = grp.sum()
    elif AGG_FUNC == "mean":
        val_5min = grp.mean()
    elif AGG_FUNC == "max":
        val_5min = grp.max()
    else:
        raise ValueError(f"Unsupported AGG_FUNC={AGG_FUNC}")

    agg = val_5min.reset_index().rename(columns={"Value": "depth_5min_mm"}).sort_values("bin_start_utc")

    # Convert 5-min depth to mm/hr
    agg["mm_per_hr"] = agg["depth_5min_mm"] * 12.0
    agg["bin_end_utc"] = agg["bin_start_utc"] + pd.Timedelta(minutes=5)

    # Final columns (keep depth too, useful for QA)
    agg = agg[["bin_start_utc", "bin_end_utc", "depth_5min_mm", "mm_per_hr"]]

    # WRITE OUTPUT
    agg.to_csv(out_csv, index=False)

    return {
        "file": str(in_csv),
        "rows_total": n_total,
        "time_parsed_ok": n_time_ok,
        "value_parsed_ok": n_val_ok,
        "rows_kept_after_dropna": n_kept,
        "bins_written": int(len(agg)),
    }


def main() -> None:
    csvs = sorted(data_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files under: {data_dir}")

    stats = []
    for in_csv in csvs:
        rel = in_csv.relative_to(data_dir)
        out_csv = (output_dir / rel).with_name(in_csv.stem + "_5min.csv")
        try:
            s = aggregate_one_csv(in_csv, out_csv)
        except Exception as e:
            s = {"file": str(in_csv), "error": repr(e)}
        stats.append(s)

    stats_df = pd.DataFrame(stats)
    summary_path = output_dir / "_aggregation_summary.csv"
    stats_df.to_csv(summary_path, index=False)

    # Print quick diagnostics
    if "error" in stats_df.columns:
        n_err = int(stats_df["error"].notna().sum())
    else:
        n_err = 0
    n_blank = int((stats_df.get("bins_written", 0) == 0).sum()) if "bins_written" in stats_df.columns else 0

    print(f"Processed files: {len(csvs)}")
    print(f"Files with errors: {n_err}")
    print(f"Files with 0 bins written: {n_blank}")
    print(f"Summary written: {summary_path}")


if __name__ == "__main__":
    main()
