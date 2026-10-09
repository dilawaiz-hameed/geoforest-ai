#!/usr/bin/env python3
"""Regenerate the cross-tile de-duplication table in the README.

Plants a known number of trees in a synthetic scene and runs the pipeline at
several overlaps. Raw detections and merged duplicates should both rise with
overlap while the count stays equal to the number of trees planted.

    PYTHONPATH=src python scripts/dedup_table.py
"""

from __future__ import annotations

import numpy as np

from geoforest.pipeline import run_pipeline
from geoforest.synthetic import PlantedTreeDetector

WIDTH, HEIGHT = 1000, 800
TILE_SIZE = 400
POSITIONS = [(50, 50), (320, 100), (620, 300), (700, 400), (900, 700)]
OVERLAPS = (0.0, 0.1, 0.25, 0.5)


def main() -> int:
    scene = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    print(
        f"{len(POSITIONS)} trees planted in a {WIDTH}x{HEIGHT} scene, "
        f"tile size {TILE_SIZE}\n"
    )
    print("| Overlap | Tiles | `raw_count` | Removed | `count` |")
    print("| --- | --- | --- | --- | --- |")

    wrong = 0
    for overlap in OVERLAPS:
        detector = PlantedTreeDetector(
            POSITIONS, WIDTH, HEIGHT, tile_size=TILE_SIZE, overlap=overlap
        )
        result = run_pipeline(
            scene, detector=detector, tile_size=TILE_SIZE, overlap=overlap
        )
        print(
            f"| {overlap:.2f} | {result.tile_count} | {result.raw_count} "
            f"| {result.duplicates_removed} | {result.count} |"
        )
        if result.count != len(POSITIONS):
            wrong += 1

    if wrong:
        print(f"\nFAIL: the count was wrong at {wrong} of {len(OVERLAPS)} overlaps.")
        return 1

    print("\nThe count held at", len(POSITIONS), "across every overlap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
