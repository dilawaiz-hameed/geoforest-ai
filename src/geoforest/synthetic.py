"""Scripted detectors with known answers.

A real detector's output cannot be predicted, so it cannot be used to prove
that the count is right — only that it is plausible. These detectors report
trees at positions chosen in advance, which turns the cross-tile merge into
something testable: the right answer is known before the pipeline runs.

Used by the test suite and by ``scripts/dedup_table.py``, so the figures in
the README come from the same code the tests assert on.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from geoforest.tiling import tile_grid

__all__ = ["FixedCountDetector", "PlantedTreeDetector", "SilentDetector"]


class SilentDetector:
    """Finds nothing. Establishes the empty-result path."""

    name = "silent"
    is_real_model = False

    def predict(self, tile: np.ndarray) -> np.ndarray:
        return np.empty((0, 5), dtype=float)


class FixedCountDetector:
    """Returns exactly ``n`` well-separated boxes per tile."""

    name = "fixed"
    is_real_model = False

    def __init__(self, n: int = 3, size: int = 20) -> None:
        self.n = n
        self.size = size

    def predict(self, tile: np.ndarray) -> np.ndarray:
        boxes = []
        for i in range(self.n):
            x = 10 + i * (self.size * 3)
            if x + self.size >= tile.shape[1] or self.size >= tile.shape[0]:
                break
            boxes.append([x, 10, x + self.size, 10 + self.size, 0.9])
        return np.asarray(boxes, dtype=float).reshape(-1, 5)


class PlantedTreeDetector:
    """Reports trees planted at fixed *scene* positions.

    Walks the same tile grid as the pipeline, in the same order, so it knows
    which tile it is being handed. Any planted tree fully inside that tile is
    reported in tile-local coordinates — exactly what a real detector does to
    a tree in an overlap region, where several tiles each report it. The
    pipeline must merge those reports back into one.
    """

    name = "planted"
    is_real_model = False

    def __init__(
        self,
        positions: Iterable[Sequence[int]],
        width: int,
        height: int,
        tile_size: int,
        overlap: float,
        size: int = 30,
    ) -> None:
        self.positions = [tuple(p) for p in positions]
        self.size = size
        self.tiles = tile_grid(width, height, tile_size=tile_size, overlap=overlap)
        self.calls = 0

    def predict(self, tile_array: np.ndarray) -> np.ndarray:
        tile = self.tiles[self.calls]
        self.calls += 1

        boxes = []
        for scene_x, scene_y in self.positions:
            local_x = scene_x - tile.x0
            local_y = scene_y - tile.y0
            if (
                local_x >= 0
                and local_y >= 0
                and local_x + self.size <= tile.width
                and local_y + self.size <= tile.height
            ):
                boxes.append(
                    [local_x, local_y, local_x + self.size, local_y + self.size, 0.9]
                )
        return np.asarray(boxes, dtype=float).reshape(-1, 5)
