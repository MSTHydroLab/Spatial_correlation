#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK")


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n" + "=" * 80)
    print("RUNNING:")
    print(" ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def script_path(name: str, base_dir: Path) -> Path:
    p = base_dir / name
    if not p.exists():
        raise FileNotFoundError(f"Required script not found: {p}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run the full WGS kriging pipeline: grid -> event correlation -> "
            "primary weights -> fallback weight options -> primary rainfall -> fallback rainfall screening"
        )
    )

    ap.add_argument("--event", type=int, nargs="+", required=True, help="One or more event numbers, e.g. 1 or 1 2 3")
    ap.add_argument("--base-dir", default=str(BASE_DIR), help="Folder containing the WGS_OK scripts")

    # Grid stage
    ap.add_argument("--build-grid", action="store_true", help="Run 00_build_wgs_grid_and_neighbors.py before the event stages")
    ap.add_argument("--stations-csv", default="", help="Optional stations CSV path for grid build")
    ap.add_argument("--grid-csv", default="", help="Optional prebuilt grid CSV; if empty, grid is generated from bounds")
    ap.add_argument("--grid-out-csv", default="", help="Optional output path for generated grid CSV")
    ap.add_argument("--neighbors-out-csv", default="", help="Optional output path for neighbor CSV")
    ap.add_argument("--start-lat", type=float, default=None)
    ap.add_argument("--end-lat", type=float, default=None)
    ap.add_argument("--start-lon", type=float, default=None)
    ap.add_argument("--end-lon", type=float, default=None)
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--lon-major", action="store_true")
    ap.add_argument("--start-km", type=int, default=None)
    ap.add_argument("--end-km", type=int, default=None)
    ap.add_argument("--want-n", type=int, default=None)
    ap.add_argument("--min-ang-sep-deg", type=float, default=None)

    # Shared weights options
    ap.add_argument("--nugget", type=float, default=0.0, help="Nugget used in the OK weights stages")

    # 02b options
    ap.add_argument("--inner-radius-km", type=float, default=7.0, help="Inner radius used by 02b fallback option builder")
    ap.add_argument("--min-within-inner-radius", type=int, default=2, help="Minimum gauges within inner radius for 02b")
    ap.add_argument("--max-options", type=int, default=5, help="Maximum fallback weight options per centroid for 02b/03b")
    ap.add_argument("--negative-tol", type=float, default=0.1, help="Negative-weight tolerance for 02b")
    ap.add_argument("--no-idw", action="store_true", help="Disable IDW fallback groups in 02b")

    # 03b options (matches the current 03b script in the project)
    ap.add_argument("--min-valid-in-window", type=int, default=8, help="Minimum valid cells in local screening window for 03b")
    ap.add_argument("--abs-threshold-mm", type=float, default=4.0, help="Absolute screening threshold for 03b")
    ap.add_argument("--no-require-improvement", action="store_true", help="Allow 03b replacement even if not strictly improved")

    # Stage skipping
    ap.add_argument("--skip-correlation", action="store_true")
    ap.add_argument("--skip-weights", action="store_true", help="Skip primary 02 weights")
    ap.add_argument("--skip-weight-options", action="store_true", help="Skip fallback 02b weight options")
    ap.add_argument("--skip-rain", action="store_true", help="Skip primary 03 rainfall")
    ap.add_argument("--skip-fallback-rain", action="store_true", help="Skip fallback 03b rainfall screening")

    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    s00 = script_path("00_build_wgs_grid_and_neighbors.py", base_dir)
    s01 = script_path("01_event_correlation_analysis.py", base_dir)
    s02 = script_path("02_WGS_Weights_Calculator.py", base_dir)
    s02b = script_path("02b_WGS_Weight_Options_Builder.py", base_dir)
    s03 = script_path("03_WGS_rainfall_estimator.py", base_dir)
    s03b = script_path("03b_WGS_rainfall_estimator_with_fallback.py", base_dir)

    py = sys.executable

    # Step 0: Build grid and neighbors (optional)
    if args.build_grid:
        cmd = [py, str(s00)]

        if args.stations_csv:
            cmd += ["--stations-csv", args.stations_csv]
        if args.grid_csv:
            cmd += ["--grid-csv", args.grid_csv]
        if args.grid_out_csv:
            cmd += ["--grid-out-csv", args.grid_out_csv]
        if args.neighbors_out_csv:
            cmd += ["--neighbors-out-csv", args.neighbors_out_csv]

        if not args.grid_csv:
            required_bounds = {
                "start_lat": args.start_lat,
                "end_lat": args.end_lat,
                "start_lon": args.start_lon,
                "end_lon": args.end_lon,
                "delta": args.delta,
            }
            missing = [k for k, v in required_bounds.items() if v is None]
            if missing:
                raise ValueError(
                    "When --build-grid is used without --grid-csv, the following are required: "
                    + ", ".join(missing)
                )

            cmd += ["--start-lat", str(args.start_lat)]
            cmd += ["--end-lat", str(args.end_lat)]
            cmd += ["--start-lon", str(args.start_lon)]
            cmd += ["--end-lon", str(args.end_lon)]
            cmd += ["--delta", str(args.delta)]

        if args.lon_major:
            cmd += ["--lon-major"]
        if args.start_km is not None:
            cmd += ["--start-km", str(args.start_km)]
        if args.end_km is not None:
            cmd += ["--end-km", str(args.end_km)]
        if args.want_n is not None:
            cmd += ["--want-n", str(args.want_n)]
        if args.min_ang_sep_deg is not None:
            cmd += ["--min-ang-sep-deg", str(args.min_ang_sep_deg)]

        run_cmd(cmd, cwd=base_dir)

    for ev in args.event:
        print("\n" + "#" * 80)
        print(f"STARTING EVENT {ev}")
        print("#" * 80)

        # Step 1: Event correlation analysis
        if not args.skip_correlation:
            cmd = [py, str(s01), "--event", str(ev)]
            run_cmd(cmd, cwd=base_dir)

        # Step 2: Primary weights
        if not args.skip_weights:
            cmd = [py, str(s02), "--event", str(ev), "--nugget", str(args.nugget)]
            run_cmd(cmd, cwd=base_dir)

        # Step 2b: Fallback weight options
        if not args.skip_weight_options:
            cmd = [
                py, str(s02b),
                "--event", str(ev),
                "--nugget", str(args.nugget),
                "--inner-radius-km", str(args.inner_radius_km),
                "--min-within-inner-radius", str(args.min_within_inner_radius),
                "--max-options", str(args.max_options),
                "--negative-tol", str(args.negative_tolerance if hasattr(args, 'negative_tolerance') else args.negative_tol),
            ]
            if args.no_idw:
                cmd += ["--no-idw"]
            run_cmd(cmd, cwd=base_dir)

        # Step 3: Primary interpolated rainfall
        if not args.skip_rain:
            cmd = [py, str(s03), "--event", str(ev)]
            run_cmd(cmd, cwd=base_dir)

        # Step 3b: Fallback rainfall screening
        if not args.skip_fallback_rain:
            cmd = [
                py, str(s03b),
                "--event", str(ev),
                "--max-options", str(args.max_options),
                "--min-valid-in-window", str(args.min_valid_in_window),
                "--abs-threshold-mm", str(args.abs_threshold_mm),
            ]

            if args.no_require_improvement:
                cmd += ["--no-require-improvement"]
            run_cmd(cmd, cwd=base_dir)

        print(f"\nFinished event {ev}")

    print("\nDone.")
    print(f"Finished events: {args.event}")


if __name__ == "__main__":
    main()
