"""Tests for the core module: tile, detect, de-duplicate, count.

The cross-tile cases are the point. A tree in an overlap region is reported
by every tile that sees it, and the count is only defensible if those
reports collapse back into one.
"""

import numpy as np
import pytest

from geoforest.detector import StubDetector
from geoforest.pipeline import run_pipeline
from geoforest.synthetic import FixedCountDetector, PlantedTreeDetector


class TestContract:
    def test_rejects_grayscale_scene(self):
        with pytest.raises(ValueError, match="RGB"):
            run_pipeline(np.zeros((100, 100), dtype=np.uint8), detector=StubDetector())

    def test_rejects_four_channel_scene(self):
        with pytest.raises(ValueError, match="RGB"):
            run_pipeline(np.zeros((100, 100, 4), dtype=np.uint8), detector=StubDetector())

    def test_silent_detector_gives_zero_count(self, scene, silent_detector):
        result = run_pipeline(scene, detector=silent_detector)
        assert result.count == 0
        assert result.raw_count == 0
        assert result.duplicates_removed == 0
        assert result.detections.shape == (0, 5)

    def test_result_records_the_configuration(self, scene, fixed_detector):
        result = run_pipeline(
            scene, detector=fixed_detector, tile_size=256, overlap=0.1, iou_threshold=0.4
        )
        assert result.tile_size == 256
        assert result.overlap == 0.1
        assert result.iou_threshold == 0.4
        assert result.scene_width == 1000
        assert result.scene_height == 800
        assert result.detector_name == "fixed"
        assert result.is_real_model is False

    def test_detections_are_in_scene_coordinates(self, scene, fixed_detector):
        result = run_pipeline(scene, detector=fixed_detector, tile_size=400)
        boxes = result.detections[:, :4]
        assert boxes[:, 0].min() >= 0
        assert boxes[:, 1].min() >= 0
        assert boxes[:, 2].max() <= result.scene_width
        assert boxes[:, 3].max() <= result.scene_height
        # Something must lie beyond the first tile, or nothing was shifted.
        assert boxes[:, 0].max() > 400


class TestCrossTileDeduplication:
    """The core claim: overlap-region trees are counted once."""

    def test_tree_in_overlap_is_counted_once(self, scene):
        # Tiles at x=0 and x=300 both contain x=320 when tile_size=400.
        planted = PlantedTreeDetector(
            positions=[(320, 100)],
            width=1000,
            height=800,
            tile_size=400,
            overlap=0.25,
        )
        result = run_pipeline(
            scene, detector=planted, tile_size=400, overlap=0.25, iou_threshold=0.3
        )
        assert result.raw_count > 1, "test is vacuous unless the tree is seen twice"
        assert result.count == 1
        assert result.duplicates_removed == result.raw_count - 1

    def test_merged_box_sits_at_the_planted_position(self, scene):
        planted = PlantedTreeDetector(
            positions=[(320, 100)], width=1000, height=800, tile_size=400, overlap=0.25
        )
        result = run_pipeline(scene, detector=planted, tile_size=400, overlap=0.25)
        np.testing.assert_allclose(result.detections[0, :4], [320, 100, 350, 130])

    def test_distinct_trees_survive_deduplication(self, scene):
        positions = [(50, 50), (320, 100), (700, 400), (900, 700)]
        planted = PlantedTreeDetector(
            positions=positions, width=1000, height=800, tile_size=400, overlap=0.25
        )
        result = run_pipeline(scene, detector=planted, tile_size=400, overlap=0.25)
        assert result.count == len(positions)

    def test_count_is_invariant_to_overlap(self, scene):
        """More overlap means more duplicates but the same number of trees."""
        positions = [(50, 50), (320, 100), (700, 400)]
        counts = []
        for overlap in (0.0, 0.25, 0.5):
            planted = PlantedTreeDetector(
                positions=positions,
                width=1000,
                height=800,
                tile_size=400,
                overlap=overlap,
            )
            counts.append(
                run_pipeline(
                    scene, detector=planted, tile_size=400, overlap=overlap
                ).count
            )
        assert counts == [len(positions)] * 3

    def test_duplicates_rise_with_overlap(self, scene):
        """Sanity check that the de-duplication is doing more work, not less."""
        positions = [(320, 100), (620, 300)]
        removed = []
        for overlap in (0.1, 0.5):
            planted = PlantedTreeDetector(
                positions=positions,
                width=1000,
                height=800,
                tile_size=400,
                overlap=overlap,
            )
            removed.append(
                run_pipeline(
                    scene, detector=planted, tile_size=400, overlap=overlap
                ).duplicates_removed
            )
        assert removed[1] > removed[0]


class TestAccounting:
    def test_counts_reconcile(self, scene):
        result = run_pipeline(scene, detector=FixedCountDetector(n=4), tile_size=400)
        assert result.count + result.duplicates_removed == result.raw_count
        assert result.count == len(result.detections)

    def test_tile_count_matches_the_grid(self, scene, fixed_detector):
        result = run_pipeline(scene, detector=fixed_detector, tile_size=400, overlap=0.25)
        # 1000px -> 3 origins, 800px -> 3 origins.
        assert result.tile_count == 9

    def test_small_scene_runs_as_one_tile(self, small_scene, fixed_detector):
        result = run_pipeline(small_scene, detector=fixed_detector, tile_size=400)
        assert result.tile_count == 1

    def test_summary_is_json_safe(self, scene, fixed_detector):
        summary = run_pipeline(scene, detector=fixed_detector).to_summary()
        assert "detections" not in summary
        assert isinstance(summary["count"], int)
        assert isinstance(summary["elapsed_s"], float)

    def test_records_match_detections(self, scene, fixed_detector):
        result = run_pipeline(scene, detector=fixed_detector)
        records = result.to_records()
        assert len(records) == result.count
        assert [r["id"] for r in records] == list(range(result.count))
        assert all(r["x1"] > r["x0"] and r["y1"] > r["y0"] for r in records)


class TestStubDetector:
    def test_is_deterministic_for_the_same_scene(self, scene):
        a = run_pipeline(scene, detector=StubDetector(), tile_size=400)
        b = run_pipeline(scene, detector=StubDetector(), tile_size=400)
        assert a.count == b.count
        np.testing.assert_array_equal(a.detections, b.detections)

    def test_is_flagged_as_not_a_real_model(self, scene):
        assert run_pipeline(scene, detector=StubDetector()).is_real_model is False

    def test_rejects_non_rgb_tile(self):
        with pytest.raises(ValueError, match="RGB"):
            StubDetector().predict(np.zeros((50, 50), dtype=np.uint8))
