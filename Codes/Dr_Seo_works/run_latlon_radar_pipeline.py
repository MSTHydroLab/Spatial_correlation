#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

'''
python run_latlon_radar_pipeline.py --event-start "2013-05-30 11:00:00" --event-end "2013-05-31 13:00:00" --compare-start "2013-05-30 12:00:00" --compare-end "2013-05-31 12:00:00" --out-dir /mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Pipeline_Output/event1 --input2-radar-dir /mnt/12TB/Sujan/Radar_products/RA/ --input1-radar-dir /mnt/12TB/Sujan/radar_package/Hydro-NEXRAD/output/event1/KEAX_20130530_20130531/KEAX/LATLON/

python run_latlon_radar_pipeline.py --event-start "2014-08-06 11:00:00" --event-end "2014-08-07 13:00:00" --compare-start "2014-08-06 12:00:00" --compare-end "2014-08-07 12:00:00" --out-dir /mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Pipeline_Output/event2 --input2-radar-dir /mnt/12TB/Sujan/Radar_products/RA/ --input1-radar-dir /mnt/12TB/Sujan/radar_package/Hydro-NEXRAD/output/event2/KEAX_20140806_20140807/KEAX/LATLON/

python run_latlon_radar_pipeline.py --event-start "2015-09-10 10:00:00" --event-end "2015-09-11 13:00:00" --compare-start "2015-09-10 11:00:00" --compare-end "2015-09-11 12:00:00" --out-dir /mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Pipeline_Output/event3 --input2-radar-dir /mnt/12TB/Sujan/Radar_products/RA/ --input1-radar-dir /mnt/12TB/Sujan/radar_package/Hydro-NEXRAD/output/event3/KEAX_20150910_20150911/KEAX/LATLON/

python run_latlon_radar_pipeline.py --event-start "2016-04-26 06:00:00" --event-end "2016-04-27 13:00:00" --compare-start "2016-04-26 07:00:00" --compare-end "2016-04-27 12:00:00" --out-dir /mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Pipeline_Output/event4 --input2-radar-dir /mnt/12TB/Sujan/Radar_products/RA/ --input1-radar-dir /mnt/12TB/Sujan/radar_package/Hydro-NEXRAD/output/event4/KEAX_20160426_20160427/KEAX/LATLON/

python run_latlon_radar_pipeline.py --event-start "2016-08-26 05:00:00" --event-end "2016-08-27 13:00:00" --compare-start "2016-08-26 06:00:00" --compare-end "2016-08-27 12:00:00" --out-dir /mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Pipeline_Output/event5 --input2-radar-dir /mnt/12TB/Sujan/Radar_products/RA/ --input1-radar-dir /mnt/12TB/Sujan/radar_package/Hydro-NEXRAD/output/event5/KEAX_20160826_20160827/KEAX/LATLON/

python run_latlon_radar_pipeline.py --event-start "2017-07-22 23:00:00" --event-end "2017-07-23 13:00:00" --compare-start "2017-07-23 00:00:00" --compare-end "2017-07-23 12:00:00" --out-dir /mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Pipeline_Output/event6 --input2-radar-dir /mnt/12TB/Sujan/Radar_products/RA/ --input1-radar-dir /mnt/12TB/Sujan/radar_package/Hydro-NEXRAD/output/event6/KEAX_20170723_20170723/KEAX/LATLON/

python run_latlon_radar_pipeline.py --event-start "2017-07-26 16:00:00" --event-end "2017-07-27 13:00:00" --compare-start "2017-07-26 17:00:00" --compare-end "2017-07-27 12:00:00" --out-dir /mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Pipeline_Output/event7 --input2-radar-dir /mnt/12TB/Sujan/Radar_products/RA/ --input1-radar-dir /mnt/12TB/Sujan/radar_package/Hydro-NEXRAD/output/event7/KEAX_20170726_20170727/KEAX/LATLON/

python run_latlon_radar_pipeline.py --event-start "2017-08-21 11:00:00" --event-end "2017-08-22 13:00:00" --compare-start "2017-08-21 12:00:00" --compare-end "2017-08-22 12:00:00" --out-dir /mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Pipeline_Output/event8 --input2-radar-dir /mnt/12TB/Sujan/Radar_products/RA/ --input1-radar-dir /mnt/12TB/Sujan/radar_package/Hydro-NEXRAD/output/event8/KEAX_20170821_20170822/KEAX/LATLON/

python run_latlon_radar_pipeline.py --event-start "2018-07-17 20:00:00" --event-end "2018-07-18 17:00:00" --compare-start "2018-07-17 21:00:00" --compare-end "2018-07-18 16:00:00" --out-dir /mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Pipeline_Output/event9 --input2-radar-dir /mnt/12TB/Sujan/Radar_products/RA/ --input1-radar-dir /mnt/12TB/Sujan/radar_package/Hydro-NEXRAD/output/event9/KEAX_20180717_20180718/KEAX/LATLON/

python run_latlon_radar_pipeline.py --event-start "2019-06-23 00:00:00" --event-end "2019-06-23 21:00:00" --compare-start "2019-06-23 01:00:00" --compare-end "2019-06-23 20:00:00" --out-dir /mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Pipeline_Output/event10 --input2-radar-dir /mnt/12TB/Sujan/Radar_products/RA/ --input1-radar-dir /mnt/12TB/Sujan/radar_package/Hydro-NEXRAD/output/event10/KEAX_20190623_20190623/KEAX/LATLON/

python run_latlon_radar_pipeline.py --event-start "2019-08-25 10:00:00" --event-end "2019-08-26 18:00:00" --compare-start "2019-08-25 11:00:00" --compare-end "2019-08-26 17:00:00" --out-dir /mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Pipeline_Output/event11 --input2-radar-dir /mnt/12TB/Sujan/Radar_products/RA/ --input1-radar-dir /mnt/12TB/Sujan/radar_package/Hydro-NEXRAD/output/event11/KEAX_20190825_20190826/KEAX/LATLON/

python run_latlon_radar_pipeline.py --event-start "2020-05-28 00:00:00" --event-end "2020-05-29 01:00:00" --compare-start "2020-05-28 01:00:00" --compare-end "2020-05-29 00:00:00" --out-dir /mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Pipeline_Output/event12 --input2-radar-dir /mnt/12TB/Sujan/Radar_products/RA/ --input1-radar-dir /mnt/12TB/Sujan/radar_package/Hydro-NEXRAD/output/event12/KEAX_20200528_20200529/KEAX/LATLON/

python run_latlon_radar_pipeline.py --event-start "2020-07-03 18:00:00" --event-end "2020-07-04 04:00:00" --compare-start "2020-07-03 19:00:00" --compare-end "2020-07-04 03:00:00" --out-dir /mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Pipeline_Output/event13 --input2-radar-dir /mnt/12TB/Sujan/Radar_products/RA/ --input1-radar-dir /mnt/12TB/Sujan/radar_package/Hydro-NEXRAD/output/event13/KEAX_20200703_20200704/KEAX/LATLON/

python run_latlon_radar_pipeline.py --event-start "2021-08-12 20:00:00" --event-end "2021-08-13 16:00:00" --compare-start "2021-08-12 21:00:00" --compare-end "2021-08-13 15:00:00" --out-dir /mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Pipeline_Output/event14 --input2-radar-dir /mnt/12TB/Sujan/Radar_products/RA/ --input1-radar-dir /mnt/12TB/Sujan/radar_package/Hydro-NEXRAD/output/event14/KEAX_20210812_20210813/KEAX/LATLON/

python run_latlon_radar_pipeline.py --event-start "2022-03-30 00:00:00" --event-end "2022-03-30 12:00:00" --compare-start "2022-03-30 01:00:00" --compare-end "2022-03-30 11:00:00" --out-dir /mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works/Pipeline_Output/event15 --input2-radar-dir /mnt/12TB/Sujan/Radar_products/RA/ --input1-radar-dir /mnt/12TB/Sujan/radar_package/Hydro-NEXRAD/output/event15/KEAX_20220330_20220330/KEAX/LATLON/
'''

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Process and compare two radar product folders using the same WGS grid. "
            "Outputs both catchment-clipped and full-grid products."
        )
    )

    p.add_argument("--event-start", required=True)
    p.add_argument("--event-end", required=True)
    p.add_argument(
        "--compare-start",
        default=None,
        help=(
            "Start time used only for Compare_timeseries_stats.py. "
            "If not provided, event-start is used."
        ),
    )

    p.add_argument(
        "--compare-end",
        default=None,
        help=(
            "End time used only for Compare_timeseries_stats.py. "
            "If not provided, event-end is used."
        ),
    )
    p.add_argument("--input1-radar-dir", type=Path, required=True)
    p.add_argument("--input2-radar-dir", type=Path, required=True)

    p.add_argument("--input1-name", default="input1")
    p.add_argument("--input2-name", default="input2")

    p.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/Dr_Seo_works"),
        help="Folder containing extract_latlon_radar_timeseries.py and Comparison/Compare_timeseries_stats.py",
    )

    p.add_argument(
        "--grid-csv",
        type=Path,
        default=Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK/dependent_files/grid_centers_wgs84.csv"),
    )

    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--catchments",
        nargs="*",
        type=Path,
        default=[
            Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6893390/6893390.shp"),
            Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/06893080/6893080.shp"),
            Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/dependent_files/6892513/6892513.shp"),
        ],
    )

    p.add_argument(
        "--append-time",
        default="2100-01-01 00:00:00",
        help="Synthetic timestamp used by comparison script for event-sum row.",
    )
    
    return p.parse_args()


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n" + "=" * 100)
    print(" ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def require_dir(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required folder not found: {path}")


def safe_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def run_total_raster(
    total_raster_script: Path,
    radar_dir: Path,
    out_dir: Path,
    output_prefix: str,
    event_start: str,
    event_end: str,
    catchments: list[Path],
    full_grid: bool,
) -> Path:
    """
    Build event-total precipitation raster and summary statistics.

    Outputs from build_total_precipitation_raster.py:
      - ASCII raster: *_total_event_rain_*.asc
      - metadata CSV: *_metadata.csv
      - summary CSV: *_summary.csv
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_tag = "full_grid" if full_grid else "catchments_only"

    cmd = [
        "python", str(total_raster_script),
        "--event-start", event_start,
        "--event-end", event_end,
        "--radar-dir", str(radar_dir),
        "--out-dir", str(out_dir),
        "--output-prefix", output_prefix,
        "--catchments", *[str(p) for p in catchments],
    ]

    if full_grid:
        cmd.append("--full-grid")

    run(cmd)

    asc_path = out_dir / f"{output_prefix}_total_event_rain_{mode_tag}.asc"
    summary_path = out_dir / f"{output_prefix}_total_event_rain_{mode_tag}_summary.csv"

    require_file(asc_path)
    require_file(summary_path)

    return asc_path

def run_extract(
    extract_script: Path,
    radar_dir: Path,
    grid_csv: Path,
    out_dir: Path,
    output_prefix: str,
    event_start: str,
    event_end: str,
    catchments: list[Path],
    full_grid: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_tag = "full_grid" if full_grid else "catchments_only"

    cmd = [
        "python", str(extract_script),
        "--event-start", event_start,
        "--event-end", event_end,
        "--radar-dir", str(radar_dir),
        "--grid-csv", str(grid_csv),
        "--out-dir", str(out_dir),
        "--output-prefix", output_prefix,
        "--catchments", *[str(p) for p in catchments],
        "--write-grid-centers-csv",
    ]

    if full_grid:
        cmd.append("--no-catchment-clip")

    run(cmd)

    out_csv = out_dir / f"{output_prefix}_grid_rain_timeseries_{mode_tag}.csv"
    require_file(out_csv)
    return out_csv


def run_compare(
    compare_script: Path,
    input1_csv: Path,
    input2_csv: Path,
    out_dir: Path,
    event_start: str,
    event_end: str,
    append_time: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    run([
        "python", str(compare_script),
        "--input1", str(input1_csv),
        "--input2", str(input2_csv),
        "--start", event_start,
        "--end", event_end,
        "--out-dir", str(out_dir),
        "--append-time", append_time,
        "--append-label-column",
    ])


def main():
    args = parse_args()
    compare_start = args.compare_start if args.compare_start is not None else args.event_start
    compare_end = args.compare_end if args.compare_end is not None else args.event_end
    input1_name = safe_name(args.input1_name)
    input2_name = safe_name(args.input2_name)

    require_dir(args.input1_radar_dir)
    require_dir(args.input2_radar_dir)
    require_file(args.grid_csv)

    extract_script = args.base_dir / "extract_latlon_radar_timeseries.py"
    compare_script = args.base_dir / "Comparison" / "Compare_timeseries_stats.py"
    total_raster_script = args.base_dir / "build_total_precipitation_raster.py"

    require_file(extract_script)
    require_file(compare_script)
    require_file(total_raster_script)

    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    input1_root = out_root / f"input1_{input1_name}"
    input2_root = out_root / f"input2_{input2_name}"
    comparison_root = out_root / "comparison"

    # ------------------------------------------------------------------
    # 1. Process input 1, catchments only
    # ------------------------------------------------------------------
    input1_catch_csv = run_extract(
        extract_script=extract_script,
        radar_dir=args.input1_radar_dir,
        grid_csv=args.grid_csv,
        out_dir=input1_root / "catchments_only",
        output_prefix=input1_name,
        event_start=args.event_start,
        event_end=args.event_end,
        catchments=args.catchments,
        full_grid=False,
    )
    input1_catch_asc = run_total_raster(
        total_raster_script=total_raster_script,
        radar_dir=args.input1_radar_dir,
        out_dir=input1_root / "catchments_only",
        output_prefix=input1_name,
        event_start=args.event_start,
        event_end=args.event_end,
        catchments=args.catchments,
        full_grid=False,
    )
    # ------------------------------------------------------------------
    # 2. Process input 1, full grid
    # ------------------------------------------------------------------
    input1_full_csv = run_extract(
        extract_script=extract_script,
        radar_dir=args.input1_radar_dir,
        grid_csv=args.grid_csv,
        out_dir=input1_root / "full_grid",
        output_prefix=input1_name,
        event_start=args.event_start,
        event_end=args.event_end,
        catchments=args.catchments,
        full_grid=True,
    )
    input1_full_asc = run_total_raster(
        total_raster_script=total_raster_script,
        radar_dir=args.input1_radar_dir,
        out_dir=input1_root / "full_grid",
        output_prefix=input1_name,
        event_start=args.event_start,
        event_end=args.event_end,
        catchments=args.catchments,
        full_grid=True,
    )
    # ------------------------------------------------------------------
    # 3. Process input 2, catchments only
    # ------------------------------------------------------------------
    input2_catch_csv = run_extract(
        extract_script=extract_script,
        radar_dir=args.input2_radar_dir,
        grid_csv=args.grid_csv,
        out_dir=input2_root / "catchments_only",
        output_prefix=input2_name,
        event_start=args.event_start,
        event_end=args.event_end,
        catchments=args.catchments,
        full_grid=False,
    )
    input2_catch_asc = run_total_raster(
        total_raster_script=total_raster_script,
        radar_dir=args.input2_radar_dir,
        out_dir=input2_root / "catchments_only",
        output_prefix=input2_name,
        event_start=args.event_start,
        event_end=args.event_end,
        catchments=args.catchments,
        full_grid=False,
    )
    # ------------------------------------------------------------------
    # 4. Process input 2, full grid
    # ------------------------------------------------------------------
    input2_full_csv = run_extract(
        extract_script=extract_script,
        radar_dir=args.input2_radar_dir,
        grid_csv=args.grid_csv,
        out_dir=input2_root / "full_grid",
        output_prefix=input2_name,
        event_start=args.event_start,
        event_end=args.event_end,
        catchments=args.catchments,
        full_grid=True,
    )
    input2_full_asc = run_total_raster(
        total_raster_script=total_raster_script,
        radar_dir=args.input2_radar_dir,
        out_dir=input2_root / "full_grid",
        output_prefix=input2_name,
        event_start=args.event_start,
        event_end=args.event_end,
        catchments=args.catchments,
        full_grid=True,
    )
    # ------------------------------------------------------------------
    # 5. Compare input1 vs input2, catchment-clipped
    # ------------------------------------------------------------------
    run_compare(
        compare_script=compare_script,
        input1_csv=input1_catch_csv,
        input2_csv=input2_catch_csv,
        out_dir=comparison_root / "catchments_only_input1_vs_input2",
        event_start=compare_start,
        event_end=compare_end,
        append_time=args.append_time,
    )

    # ------------------------------------------------------------------
    # 6. Compare input1 vs input2, full-grid
    # ------------------------------------------------------------------
    run_compare(
        compare_script=compare_script,
        input1_csv=input1_full_csv,
        input2_csv=input2_full_csv,
        out_dir=comparison_root / "full_grid_input1_vs_input2",
        event_start=compare_start,
        event_end=compare_end,
        append_time=args.append_time,
    )

    print("\nPipeline finished.")

    print(f"Input 1 catchments-only CSV : {input1_catch_csv}")
    print(f"Input 1 catchments-only ASC : {input1_catch_asc}")
    print(f"Input 1 full-grid CSV       : {input1_full_csv}")
    print(f"Input 1 full-grid ASC       : {input1_full_asc}")

    print(f"Input 2 catchments-only CSV : {input2_catch_csv}")
    print(f"Input 2 catchments-only ASC : {input2_catch_asc}")
    print(f"Input 2 full-grid CSV       : {input2_full_csv}")
    print(f"Input 2 full-grid ASC       : {input2_full_asc}")

    print(f"Comparison folder           : {comparison_root}")


if __name__ == "__main__":
    main()