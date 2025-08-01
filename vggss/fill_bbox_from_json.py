#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fill_bbox_from_json.py
======================

Populate the `bbox` column in `metadata/windows_jepa.csv` using the
annotations from `vggss.json`. Flipped samples (UID ending with `_fh`)
receive horizontally mirrored bounding boxes.

Paths are hard‑coded below; adjust them if your directory structure differs.

Output
------
`metadata/windows_jepa_filled.csv` (same columns, bbox filled in JSON format)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


CSV_PATH = Path("vggss_jepa_flip/metadata/windows_jepa.csv")
JSON_PATH = Path("vggss.json")
OUT_PATH = Path("vggss_jepa_flip/metadata/windows_jepa_filled.csv")

# --------------------------------------------------------------------- #
# Helper functions                                                      #
# --------------------------------------------------------------------- #
def load_vggss_json(path: Path) -> Dict[str, List[List[float]]]:
    """Return a dict {file_uid: bbox_list} from the VGG‑SS JSON."""
    with open(path, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    return {item["file"]: item["bbox"] for item in annotations}


def flip_bbox_horiz(boxes: List[List[float]]) -> List[List[float]]:
    """Mirror bounding boxes horizontally in normalized 0‑1 coordinates."""
    return [[1 - x_max, y_min, 1 - x_min, y_max] for x_min, y_min, x_max, y_max in boxes]


# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #
def main() -> None:
    # Safety checks
    for p in (CSV_PATH, JSON_PATH):
        if not p.exists():
            raise FileNotFoundError(p)

    print(f"[INFO] Loading JSON annotations from {JSON_PATH}")
    bbox_map = load_vggss_json(JSON_PATH)

    print(f"[INFO] Loading CSV metadata from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    updated, missing = 0, 0
    new_bbox_column: List[str] = []

    for uid in df["uid"]:
        base_uid = uid[:-3] if uid.endswith("_fh") else uid

        if base_uid in bbox_map:
            boxes = bbox_map[base_uid]
            if uid.endswith("_fh"):
                boxes = flip_bbox_horiz(boxes)
            new_bbox_column.append(json.dumps(boxes))
            updated += 1
        else:
            new_bbox_column.append("[]")
            missing += 1

    df["bbox"] = new_bbox_column
    df.to_csv(OUT_PATH, index=False)

    print(f"[OK] Saved filled metadata → {OUT_PATH}")
    print(f"    Bounding boxes updated for {updated} rows")
    print(f"    Missing boxes for {missing} rows (left as empty lists)")

if __name__ == "__main__":
    main()
