#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_manifest(manifest_csv: Path) -> list[Path]:
    df = pd.read_csv(manifest_csv)

    if "rain_file" not in df.columns:
        raise ValueError(f"{manifest_csv} must contain a 'rain_file' column")

    files = [Path(p) for p in df["rain_file"].dropna().astype(str).tolist()]

    if not files:
        raise ValueError(f"No rainfall files found in {manifest_csv}")

    missing = [p for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing bin rainfall files. First few: {missing[:5]}")

    return files


def get_header_columns(first_file: Path) -> list[str]:
    return pd.read_csv(first_file, nrows=0).columns.tolist()


def stream_combine_bins(
    manifest_csv: Path,
    out_csv: Path,
    chunksize: int,
    overwrite: bool,
) -> None:
    if out_csv.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {out_csv}. Use --overwrite to replace it.")

    files = load_manifest(manifest_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    expected_cols = get_header_columns(files[0])

    first_write = True
    total_rows = 0

    print(f"Combining {len(files)} bin rainfall files")
    print(f"Output: {out_csv}")

    for i, fp in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {fp.name}")

        file_cols = get_header_columns(fp)
        if file_cols != expected_cols:
            raise ValueError(
                f"Column mismatch in {fp}\n"
                f"Expected first columns: {expected_cols[:5]}\n"
                f"Got first columns     : {file_cols[:5]}"
            )

        for chunk in pd.read_csv(fp, chunksize=chunksize):
            if "time_local" not in chunk.columns:
                raise ValueError(f"{fp} missing time_local column")

            # Keep same column order
            chunk = chunk[expected_cols]

            # Append safely
            chunk.to_csv(
                out_csv,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
            )

            total_rows += len(chunk)
            first_write = False

    print("=" * 80)
    print(f"Saved combined rainfall CSV: {out_csv}")
    print(f"Total rows written: {total_rows}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Memory-safe streaming combiner for per-bin continuous IDW rainfall CSVs."
    )

    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path(
            "/mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Continuous_IDW/"
            "03_Interpolated_Rain/bin_rainfall_manifest.csv"
        ),
    )

    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path(
            "/mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Continuous_IDW/"
            "03_Interpolated_Rain/continuous_idw_rainfall.csv"
        ),
    )

    parser.add_argument("--chunksize", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    stream_combine_bins(
        manifest_csv=args.manifest_csv,
        out_csv=args.out_csv,
        chunksize=args.chunksize,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()