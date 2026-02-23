#!/usr/bin/env python3
"""
json_features_to_csv.py

Extracts all features + properties from a GeoJSON-like "FeatureCollection" JSON file
and writes a CSV in the SAME folder.

What it does
- Reads the JSON
- Copies top-level metadata (view, cache, generated, data_date, etc.) into every row
- Flattens:
    geometry: type, coordinates (lon/lat and optional extra dims)
    properties: every key becomes a column
- Handles 100s/1000s of features safely (stream-ish in memory; 192 is easy)
- Writes: <input_stem>.csv next to the JSON

Usage
  python json_features_to_csv.py /path/to/file.json
  python json_features_to_csv.py /path/to/folder/with/jsons --glob "*.json"

Notes
- If keys are missing in some features, CSV will still be produced with blanks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_dict(d: Dict[str, Any], parent: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flattens nested dicts:
      {"a":{"b":1}} -> {"a.b":1}
    Leaves lists as-is (except geometry coords handled separately).
    """
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        key = f"{parent}{sep}{k}" if parent else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, parent=key, sep=sep))
        else:
            out[key] = v
    return out


def geometry_to_cols(geom: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(geom, dict):
        return out

    out["geom_type"] = geom.get("type")
    coords = geom.get("coordinates")

    # Most Stormwatch geojson points: [lon, lat]
    if isinstance(coords, (list, tuple)):
        out["coord_0"] = coords[0] if len(coords) > 0 else None
        out["coord_1"] = coords[1] if len(coords) > 1 else None

        # If there are extra dims (e.g., z), keep them too
        for i in range(2, len(coords)):
            out[f"coord_{i}"] = coords[i]
    else:
        out["coord_0"] = None
        out["coord_1"] = None

    return out


def feature_rows(obj: Dict[str, Any], src_file: Path) -> List[Dict[str, Any]]:
    if not isinstance(obj, dict):
        raise ValueError("JSON root is not an object/dict")

    # Keep any top-level metadata except the big features list
    top_meta = {k: v for k, v in obj.items() if k != "features"}
    top_meta_flat = flatten_dict(top_meta)

    feats = obj.get("features", [])
    if not isinstance(feats, list):
        raise ValueError("JSON does not contain a 'features' list")

    rows: List[Dict[str, Any]] = []
    for idx, f in enumerate(feats):
        if not isinstance(f, dict):
            continue

        row: Dict[str, Any] = {}
        row.update(top_meta_flat)

        # Track source
        row["src_file"] = str(src_file)
        row["feature_index"] = idx

        # Feature-level fields
        row["feature_type"] = f.get("type")

        # Geometry
        row.update(geometry_to_cols(f.get("geometry") or {}))

        # Properties (flatten nested if any)
        props = f.get("properties") or {}
        if isinstance(props, dict):
            props_flat = flatten_dict(props, parent="properties")
            row.update(props_flat)
        else:
            row["properties"] = props

        rows.append(row)

    return rows


def write_csv_for_json(json_path: Path) -> Path:
    obj = load_json(json_path)
    rows = feature_rows(obj, json_path)

    out_csv = json_path.with_suffix(".csv")
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return out_csv


def iter_json_files(path: Path, glob_pat: str) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for fp in sorted(path.glob(glob_pat)):
        if fp.is_file():
            yield fp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to a .json file OR a folder containing json files")
    ap.add_argument("--glob", default="*.json", help="Glob when path is a folder (default: *.json)")
    args = ap.parse_args()

    p = Path(args.path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"Not found: {p}")

    written: List[Tuple[Path, Path]] = []
    for json_fp in iter_json_files(p, args.glob):
        try:
            out_csv = write_csv_for_json(json_fp)
            written.append((json_fp, out_csv))
            print(f"Wrote {out_csv}  (from {json_fp.name})", flush=True)
        except Exception as e:
            print(f"[WARN] Failed on {json_fp}: {e}", flush=True)

    if not written:
        raise SystemExit("No CSVs written (no matching JSONs or all failed).")


if __name__ == "__main__":
    main()
