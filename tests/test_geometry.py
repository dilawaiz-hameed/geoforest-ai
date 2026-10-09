"""Tests for IoU and non-maximum suppression.

NMS is what turns per-tile detections into a count, so these cases are the
difference between a defensible number and a double-counted one.
"""

import numpy as np
import pytest

from geoforest.geometry import iou_matrix, nms, shift_boxes


class TestIoU:
    def test_identical_boxes_give_one(self):
        box = np.array([[0, 0, 10, 10]])
        assert iou_matrix(box, box)[0, 0] == pytest.approx(1.0)

    def test_disjoint_boxes_give_zero(self):
        a = np.array([[0, 0, 10, 10]])
        b = np.array([[20, 20, 30, 30]])
        assert iou_matrix(a, b)[0, 0] == pytest.approx(0.0)

    def test_touching_edges_give_zero(self):
        a = np.array([[0, 0, 10, 10]])
        b = np.array([[10, 0, 20, 10]])
        assert iou_matrix(a, b)[0, 0] == pytest.approx(0.0)

    def test_half_overlap(self):
        # Intersection 50, union 150.
        a = np.array([[0, 0, 10, 10]])
        b = np.array([[5, 0, 15, 10]])
        assert iou_matrix(a, b)[0, 0] == pytest.approx(50 / 150)

    def test_contained_box(self):
        # Intersection 25, union 100.
        a = np.array([[0, 0, 10, 10]])
        b = np.array([[2, 2, 7, 7]])
        assert iou_matrix(a, b)[0, 0] == pytest.approx(25 / 100)

    def test_shape_is_n_by_m(self):
        a = np.zeros((3, 4))
        b = np.zeros((5, 4))
        assert iou_matrix(a, b).shape == (3, 5)

    def test_empty_input_is_safe(self):
        assert iou_matrix(np.empty((0, 4)), np.array([[0, 0, 1, 1]])).shape == (0, 1)

    def test_degenerate_box_gives_zero(self):
        a = np.array([[5, 5, 5, 5]])
        b = np.array([[0, 0, 10, 10]])
        assert iou_matrix(a, b)[0, 0] == pytest.approx(0.0)


class TestNMS:
    def test_merges_near_duplicate(self):
        """Two boxes for the same tree collapse to one."""
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]])
        scores = np.array([0.9, 0.5])
        assert nms(boxes, scores, iou_threshold=0.3).tolist() == [0]

    def test_keeps_distinct_trees(self):
        boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]])
        scores = np.array([0.9, 0.5])
        assert sorted(nms(boxes, scores, iou_threshold=0.3).tolist()) == [0, 1]

    def test_keeps_the_higher_score(self):
        """Survivor is chosen by score, not by input order."""
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]])
        scores = np.array([0.2, 0.95])
        assert nms(boxes, scores, iou_threshold=0.3).tolist() == [1]

    def test_suppression_is_strictly_above_the_threshold(self):
        """Matches the torchvision convention: suppress only when IoU > threshold."""
        boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10]])
        scores = np.array([0.9, 0.8])
        iou = iou_matrix(boxes[:1], boxes[1:])[0, 0]  # 50 / 150

        # Threshold above the pair's IoU: both survive.
        assert len(nms(boxes, scores, iou_threshold=iou + 1e-6)) == 2
        # Threshold below it: the lower-scoring box is suppressed.
        assert nms(boxes, scores, iou_threshold=iou - 1e-6).tolist() == [0]
        # Exactly at it: kept, because suppression needs a strict excess.
        assert len(nms(boxes, scores, iou_threshold=iou)) == 2

    def test_empty_input_returns_empty(self):
        assert nms(np.empty((0, 4)), np.empty(0)).size == 0

    def test_output_is_ordered_by_score(self):
        boxes = np.array([[0, 0, 5, 5], [100, 100, 105, 105], [200, 200, 205, 205]])
        scores = np.array([0.3, 0.9, 0.6])
        assert nms(boxes, scores).tolist() == [1, 2, 0]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            nms(np.zeros((3, 4)), np.zeros(2))

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="iou_threshold"):
            nms(np.zeros((1, 4)), np.zeros(1), iou_threshold=1.5)

    def test_is_deterministic(self):
        rng = np.random.default_rng(0)
        boxes = rng.integers(0, 100, size=(50, 2))
        boxes = np.column_stack([boxes, boxes + 10]).astype(float)
        scores = rng.uniform(size=50)
        first = nms(boxes, scores).tolist()
        assert all(nms(boxes, scores).tolist() == first for _ in range(3))


class TestShiftBoxes:
    def test_applies_tile_offset(self):
        boxes = np.array([[0, 0, 10, 10]])
        assert shift_boxes(boxes, 100, 200).tolist() == [[100, 200, 110, 210]]

    def test_zero_offset_is_identity(self):
        boxes = np.array([[1.0, 2.0, 3.0, 4.0]])
        np.testing.assert_array_equal(shift_boxes(boxes, 0, 0), boxes)

    def test_empty_input_is_safe(self):
        assert shift_boxes(np.empty((0, 4)), 5, 5).shape == (0, 4)
