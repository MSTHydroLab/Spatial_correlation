#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import rasterio
'''
python batch_raster_difference.py \
  --pipeline-root /mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Pipeline_Output \
  --mode full_grid \
  --input1-name input1 \
  --input2-name input2 \
  --output-prefix input1_minus_input2
  '''
def parse_args():
    p = argparse.ArgumentParser(
        description="Compute raster differences for all event folders."
    )

    p.add_argument(
        "--pipeline-root",
        type=Path,
        required=True,
        help="Root folder containing event1, event2, ... outputs.",
    )

    p.add_argument(
        "--mode",
        choices=["full_grid", "catchments_only"],
        default="full_grid",
        help="Which raster mode to compare.",
    )

    p.add_argument(
        "--input1-name",
        default="input1",
        help="Prefix used for input1 raster filenames.",
    )

    p.add_argument(
        "--input2-name",
        default="input2",
        help="Prefix used for input2 raster filenames.",
    )

    p.add_argument(
        "--out-subdir",
        default="raster_difference",
        help="Subfolder inside each event folder where difference rasters are saved.",
    )

    p.add_argument(
        "--output-prefix",
        default="input1_minus_input2",
        help="Output raster filename prefix.",
    )

    return p.parse_args()


def find_raster(event_dir: Path, input_folder: str, prefix: str, mode: str) -> Path:
    raster_path = (
        event_dir
        / input_folder
        / mode
        / f"{prefix}_total_event_rain_{mode}.asc"
    )

    if not raster_path.exists():
        raise FileNotFoundError(f"Missing raster: {raster_path}")

    return raster_path


def read_raster(path: Path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        profile = src.profile.copy()
        nodata = src.nodata

    if nodata is not None:
        arr[np.isclose(arr, nodata)] = np.nan

    return arr, profile, nodata


def write_difference_raster(
    out_path: Path,
    diff: np.ndarray,
    profile: dict,
    nodata_value: float = -9999.0,
):
    out = diff.copy()
    out[~np.isfinite(out)] = nodata_value

    profile.update(
        driver="GTiff",
        dtype="float32",
        nodata=nodata_value,
        compress="lzw",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out.astype("float32"), 1)


def process_event(event_dir: Path, args) -> None:
    input1_raster = find_raster(
        event_dir=event_dir,
        input_folder=f"input1_{args.input1_name}",
        prefix=args.input1_name,
        mode=args.mode,
    )

    input2_raster = find_raster(
        event_dir=event_dir,
        input_folder=f"input2_{args.input2_name}",
        prefix=args.input2_name,
        mode=args.mode,
    )

    arr1, profile1, nodata1 = read_raster(input1_raster)
    arr2, profile2, nodata2 = read_raster(input2_raster)

    if arr1.shape != arr2.shape:
        raise ValueError(
            f"Shape mismatch in {event_dir.name}: "
            f"{arr1.shape} vs {arr2.shape}"
        )

    diff = arr1 - arr2

    out_path = (
        event_dir
        / args.out_subdir
        / f"{args.output_prefix}_{args.mode}.tif"
    )

    write_difference_raster(out_path, diff, profile1)

    valid = diff[np.isfinite(diff)]
    print(f"\n{event_dir.name}")
    print(f"  input1: {input1_raster}")
    print(f"  input2: {input2_raster}")
    print(f"  saved : {out_path}")

    if valid.size:
        print(f"  min   : {np.nanmin(valid):.3f} mm")
        print(f"  max   : {np.nanmax(valid):.3f} mm")
        print(f"  mean  : {np.nanmean(valid):.3f} mm")
        print(f"  sum   : {np.nansum(valid):.3f} mm")
    else:
        print("  warning: no valid cells")


def main():
    args = parse_args()

    event_dirs = sorted(
        [p for p in args.pipeline_root.glob("event*") if p.is_dir()],
        key=lambda p: int(p.name.replace("event", "")) if p.name.replace("event", "").isdigit() else 9999,
    )

    if not event_dirs:
        raise FileNotFoundError(f"No event folders found in {args.pipeline_root}")

    for event_dir in event_dirs:
        try:
            process_event(event_dir, args)
        except Exception as e:
            print(f"\n{event_dir.name}: skipped")
            print(f"  reason: {e}")


if __name__ == "__main__":
    main()