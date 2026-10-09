"""Cut a large scene into overlapping tiles.

A crown sitting on a tile boundary is cut in half and missed, so tiles
overlap. The cost is that trees in the overlap are detected twice, which is
what :mod:`geoforest.geometry` de-duplicates later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["Tile", "tile_grid", "tile_origins"]


@dataclass(frozen=True)
class Tile:
    """One tile, in scene pixel coordinates. ``x1``/``y1`` are exclusive."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def offset(self) -> tuple[int, int]:
        return self.x0, self.y0


def tile_origins(length: int, tile_size: int, overlap: float) -> list[int]:
    """Origins along one axis.

    The final origin is clamped to ``length - tile_size`` so every tile is
    full size where the axis allows it; the last two tiles may therefore
    overlap by more than ``overlap``. Detectors behave badly on short tiles,
    so a uniform size matters more than a uniform stride.
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")

    if length <= tile_size:
        return [0]

    stride = max(1, round(tile_size * (1.0 - overlap)))
    n_tiles = math.ceil((length - tile_size) / stride) + 1
    return [min(i * stride, length - tile_size) for i in range(n_tiles)]


def tile_grid(
    width: int, height: int, tile_size: int = 400, overlap: float = 0.25
) -> list[Tile]:
    """Full tile grid for a scene, in row-major order."""
    xs = tile_origins(width, tile_size, overlap)
    ys = tile_origins(height, tile_size, overlap)
    return [
        Tile(x, y, min(x + tile_size, width), min(y + tile_size, height))
        for y in ys
        for x in xs
    ]
