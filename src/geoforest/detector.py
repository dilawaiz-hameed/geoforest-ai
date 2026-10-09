"""Detector interface and implementations.

Two implementations share one interface so the pipeline, API and frontend can
be built and tested before the model is wired in:

``StubDetector``
    Deterministic, no ML dependencies. Used by CI so unit tests stay fast,
    and as a fallback if the real model fails to load. Its output is NOT a
    tree detection and is labelled as such everywhere it surfaces.

``DeepForestDetector``
    The real thing. Imports DeepForest lazily so the rest of the package
    stays importable without torch installed.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["DeepForestDetector", "Detector", "StubDetector", "get_detector"]


@runtime_checkable
class Detector(Protocol):
    """Anything that turns an RGB tile into scored boxes."""

    name: str
    is_real_model: bool

    def predict(self, tile: np.ndarray) -> np.ndarray:
        """Detect in one RGB tile.

        Args:
            tile: ``(H, W, 3)`` uint8 array.

        Returns:
            ``(N, 5)`` float array of ``[x0, y0, x1, y1, score]`` in
            tile-local pixel coordinates. May be empty.
        """
        ...


class StubDetector:
    """Placeholder detector producing deterministic boxes.

    Seeded from the tile's own bytes, so the same tile always yields the same
    boxes and tests are reproducible. It detects nothing — it exists so the
    surrounding system can be built and tested independently of the model.
    """

    name = "stub"
    is_real_model = False

    def __init__(self, boxes_per_tile: int = 40, box_size: int = 24) -> None:
        self.boxes_per_tile = boxes_per_tile
        self.box_size = box_size

    def predict(self, tile: np.ndarray) -> np.ndarray:
        tile = np.asarray(tile)
        if tile.ndim != 3 or tile.shape[2] != 3:
            raise ValueError(f"expected (H, W, 3) RGB tile, got shape {tile.shape}")

        height, width = tile.shape[:2]
        if min(height, width) <= self.box_size:
            return np.empty((0, 5), dtype=float)

        digest = hashlib.sha256(np.ascontiguousarray(tile).tobytes()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))

        n = self.boxes_per_tile
        x0 = rng.integers(0, width - self.box_size, size=n).astype(float)
        y0 = rng.integers(0, height - self.box_size, size=n).astype(float)
        jitter = rng.integers(-4, 5, size=n).astype(float)
        size = np.clip(self.box_size + jitter, 8, None)
        scores = rng.uniform(0.2, 0.9, size=n)

        return np.column_stack([x0, y0, x0 + size, y0 + size, scores])


class DeepForestDetector:
    """Pretrained DeepForest crown detector.

    Loads the released weights once and reuses them. ``score_threshold`` is
    deliberately permissive by default: the proposal's POC ran at 0.2 to
    characterise what the model finds rather than to flatter the numbers.
    """

    name = "deepforest"
    is_real_model = True

    def __init__(self, score_threshold: float = 0.2) -> None:
        self.score_threshold = score_threshold
        self._model = None

    def _load(self):  # pragma: no cover - needs torch and released weights
        if self._model is None:
            from deepforest import main as deepforest_main

            logger.info("loading DeepForest release weights")
            model = deepforest_main.deepforest()
            model.use_release()
            model.model.score_thresh = self.score_threshold
            self._model = model
        return self._model

    def predict(self, tile: np.ndarray) -> np.ndarray:
        tile = np.asarray(tile)
        if tile.ndim != 3 or tile.shape[2] != 3:
            raise ValueError(f"expected (H, W, 3) RGB tile, got shape {tile.shape}")

        return self._infer(self._load(), tile)

    def _infer(self, model, tile: np.ndarray) -> np.ndarray:
        """Translate DeepForest's dataframe into the protocol's array.

        Split out from :meth:`predict` so the column mapping and the score
        filter can be tested against a fake model, without torch or weights.
        """
        # DeepForest expects BGR when handed a raw array.
        frame = model.predict_image(image=tile[:, :, ::-1].astype("float32"))
        if frame is None or len(frame) == 0:
            return np.empty((0, 5), dtype=float)

        boxes = frame[["xmin", "ymin", "xmax", "ymax", "score"]].to_numpy(dtype=float)
        return boxes[boxes[:, 4] >= self.score_threshold]


def get_detector(name: str | None = None) -> Detector:
    """Pick a detector by name, falling back to the stub if loading fails.

    Resolution order: explicit ``name`` argument, then the
    ``GEOFOREST_DETECTOR`` environment variable, then ``"stub"``.
    """
    name = (name or os.environ.get("GEOFOREST_DETECTOR", "stub")).lower()

    if name == "stub":
        return StubDetector()

    if name == "deepforest":
        detector = DeepForestDetector()
        try:
            detector._load()
        except Exception:
            logger.exception("DeepForest unavailable, falling back to stub detector")
            return StubDetector()
        return detector

    raise ValueError(f"unknown detector {name!r}; expected 'stub' or 'deepforest'")
