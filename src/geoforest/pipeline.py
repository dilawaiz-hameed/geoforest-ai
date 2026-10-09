"""Core module: tile a scene, detect in each tile, merge across boundaries.

This is the fine-scale arm of GeoForest AI end to end. The de-duplication
step is the part that turns per-tile detections into a defensible count:
without it, every tree in an overlap region is counted twice.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from geoforest.detector import Detector, get_detector
from geoforest.geometry import nms, shift_boxes
from geoforest.tiling import tile_grid

__all__ = ["PipelineResult", "run_pipeline"]


@dataclass
class PipelineResult:
    """Outcome of one pipeline run over one scene."""

    detections: np.ndarray = field(repr=False)
    """``(N, 5)`` array of ``[x0, y0, x1, y1, score]`` in scene coordinates."""

    count: int
    raw_count: int
    duplicates_removed: int
    tile_count: int
    elapsed_s: float
    detector_name: str
    is_real_model: bool
    tile_size: int
    overlap: float
    iou_threshold: float
    scene_width: int
    scene_height: int

    def to_summary(self) -> dict:
        """JSON-safe summary, without the box array."""
        payload = asdict(self)
        payload.pop("detections")
        return payload

    def to_records(self) -> list[dict]:
        """One record per detection, for CSV export or an API response."""
        return [
            {
                "id": i,
                "x0": round(float(b[0]), 2),
                "y0": round(float(b[1]), 2),
                "x1": round(float(b[2]), 2),
                "y1": round(float(b[3]), 2),
                "score": round(float(b[4]), 4),
            }
            for i, b in enumerate(self.detections)
        ]


def run_pipeline(
    scene: np.ndarray,
    detector: Detector | None = None,
    tile_size: int = 400,
    overlap: float = 0.25,
    iou_threshold: float = 0.3,
) -> PipelineResult:
    """Run tiling, detection and cross-tile de-duplication over one scene.

    Args:
        scene: ``(H, W, 3)`` uint8 RGB array.
        detector: Any :class:`~geoforest.detector.Detector`. Defaults to
            whatever ``get_detector()`` resolves from the environment.
        tile_size: Tile edge in pixels.
        overlap: Fraction of a tile shared with its neighbour, in ``[0, 1)``.
        iou_threshold: Boxes overlapping more than this are treated as the
            same tree and merged.

    Raises:
        ValueError: If ``scene`` is not an RGB array.
    """
    scene = np.asarray(scene)
    if scene.ndim != 3 or scene.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB scene, got shape {scene.shape}")

    detector = detector or get_detector()
    started = time.perf_counter()

    height, width = scene.shape[:2]
    tiles = tile_grid(width, height, tile_size=tile_size, overlap=overlap)

    collected: list[np.ndarray] = []
    for tile in tiles:
        patch = scene[tile.y0 : tile.y1, tile.x0 : tile.x1]
        local = detector.predict(patch)
        if local.size == 0:
            continue
        boxes = shift_boxes(local[:, :4], tile.x0, tile.y0)
        collected.append(np.column_stack([boxes, local[:, 4]]))

    if collected:
        stacked = np.vstack(collected)
        keep = nms(stacked[:, :4], stacked[:, 4], iou_threshold=iou_threshold)
        merged = stacked[keep]
    else:
        stacked = np.empty((0, 5), dtype=float)
        merged = stacked

    return PipelineResult(
        detections=merged,
        count=int(merged.shape[0]),
        raw_count=int(stacked.shape[0]),
        duplicates_removed=int(stacked.shape[0] - merged.shape[0]),
        tile_count=len(tiles),
        elapsed_s=round(time.perf_counter() - started, 3),
        detector_name=detector.name,
        is_real_model=detector.is_real_model,
        tile_size=tile_size,
        overlap=overlap,
        iou_threshold=iou_threshold,
        scene_width=width,
        scene_height=height,
    )
