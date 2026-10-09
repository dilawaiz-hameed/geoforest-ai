"""Box geometry: IoU and non-maximum suppression.

Pure NumPy, no ML dependencies, so these run in CI in milliseconds.
Boxes are ``(N, 4)`` arrays of ``[x0, y0, x1, y1]`` in pixel coordinates.
"""

from __future__ import annotations

import numpy as np

__all__ = ["iou_matrix", "nms", "shift_boxes"]


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise intersection-over-union.

    Returns an ``(N, M)`` array where entry ``(i, j)`` is the IoU of
    ``boxes_a[i]`` and ``boxes_b[j]``. Degenerate boxes (zero area) give 0.
    """
    a = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=float).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)

    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)

    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.3) -> np.ndarray:
    """Greedy non-maximum suppression.

    Keeps the highest-scoring box, discards everything whose IoU with it is
    *strictly greater* than ``iou_threshold``, and repeats. The strict
    comparison matches torchvision, so swapping in a GPU implementation later
    will not change the counts.

    Returns kept indices, highest score first. Ties break on original order,
    so the result is deterministic.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")

    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if boxes.shape[0] != scores.shape[0]:
        raise ValueError(
            f"boxes and scores length mismatch: {boxes.shape[0]} vs {scores.shape[0]}"
        )
    if boxes.shape[0] == 0:
        return np.empty(0, dtype=int)

    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        ious = iou_matrix(boxes[current : current + 1], boxes[rest])[0]
        order = rest[ious <= iou_threshold]

    return np.asarray(keep, dtype=int)


def shift_boxes(boxes: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate tile-local boxes into scene coordinates."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if boxes.size == 0:
        return boxes.copy()
    return boxes + np.array([dx, dy, dx, dy], dtype=float)
