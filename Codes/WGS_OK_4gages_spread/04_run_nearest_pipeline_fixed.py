#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys

BASE_DIR = Path("/mnt/12TB/Sujan/Spatial_correlation/Codes/WGS_OK_4gages_spread")


def run_cmd(cmd, cwd=None):
    print("\n" + "=" * 80)
    print("RUNNING:")
    print(" ".join(map(str, cmd)))
    print("=" * 80)
    subprocess.run([str(x) for x in cmd], check=True, cwd=str(cwd) if cwd else None)


def script_path(name: str, base_dir: Path) -> Path:
    p = base_dir / name
    if not p.exists():
        raise FileNotFoundError(f"Required script not found: {p}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run nearest-gauge interpolation pipeline for one or more events in sequence."
    )
    ap.add_argument("--event", type=int, nargs="+", required=True,
                    help="One or more event numbers, e.g. --event 1 2 3")
    ap.add_argument("--n-gauges", type=int, choices=[3, 4], required=True)
    ap.add_argument("--base-dir", default=str(BASE_DIR))

    ap.add_argument("--build-nearest-table", action="store_true")
    ap.add_argument("--search-radius-km", type=float, default=7.0)
    ap.add_argument("--keep-n", type=int, default=10)
    ap.add_argument("--nugget", type=float, default=0.0)

    ap.add_argument("--skip-correlation", action="store_true")
    ap.add_argument("--skip-weights", action="store_true")
    ap.add_argument("--skip-rain", action="store_true")
    ap.add_argument("--skip-stats", action="store_true")

    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    py = sys.executable

    s00 = script_path("00_build_nearest_gauge_candidates.py", base_dir)
    s01 = script_path("01_event_correlation_analysis.py", base_dir)
    s02 = script_path("02_nearest_weights_calculator.py", base_dir)
    s03 = script_path("03_nearest_rainfall_estimator.py", base_dir)
    s04 = script_path("04_negative_weight_catchment_stats.py", base_dir)

    if args.build_nearest_table:
        run_cmd(
            [
                py, s00,
                "--base-dir", base_dir,
                "--search-radius-km", args.search_radius_km,
                "--keep-n", args.keep_n,
            ],
            cwd=base_dir
        )

    for ev in args.event:
        print("\n" + "#" * 80)
        print(f"STARTING EVENT {ev}")
        print("#" * 80)

        if not args.skip_correlation:
            run_cmd(
                [py, s01, "--event", ev],
                cwd=base_dir
            )

        if not args.skip_weights:
            run_cmd(
                [
                    py, s02,
                    "--event", ev,
                    "--n-gauges", args.n_gauges,
                    "--base-dir", base_dir,
                    "--nugget", args.nugget,
                ],
                cwd=base_dir
            )

        if not args.skip_rain:
            run_cmd(
                [
                    py, s03,
                    "--event", ev,
                    "--n-gauges", args.n_gauges,
                    "--event-meta-dir", base_dir / "01_Event_TimeSeries",
                    "--weights-dir", base_dir / "02_OK_Weights",
                    "--out-dir", base_dir / "03_Interpolated_Rain",
                ],
                cwd=base_dir
            )

        if not args.skip_stats:
            run_cmd(
                [
                    py, s04,
                    "--base-dir", base_dir,
                    "--event", ev,
                    "--n-gauges", args.n_gauges,
                ],
                cwd=base_dir
            )

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
