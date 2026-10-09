"""Tests for the detector layer.

The resolution logic matters more than it looks: it decides what the
deployed service is actually running, and a silent fall back to the stub
must still be a *labelled* stub rather than an unexplained set of boxes.
"""

import numpy as np
import pytest

from geoforest.detector import (
    DeepForestDetector,
    Detector,
    StubDetector,
    get_detector,
)
from geoforest.synthetic import FixedCountDetector, PlantedTreeDetector, SilentDetector


class TestProtocol:
    @pytest.mark.parametrize(
        "detector",
        [
            StubDetector(),
            DeepForestDetector(),
            SilentDetector(),
            FixedCountDetector(),
            PlantedTreeDetector([(0, 0)], 400, 400, 400, 0.0),
        ],
    )
    def test_implementations_satisfy_the_protocol(self, detector):
        assert isinstance(detector, Detector)
        assert isinstance(detector.name, str)
        assert isinstance(detector.is_real_model, bool)


class TestStubDetector:
    def test_returns_the_requested_number_of_boxes(self):
        tile = np.zeros((200, 200, 3), dtype=np.uint8)
        assert StubDetector(boxes_per_tile=7).predict(tile).shape == (7, 5)

    def test_boxes_stay_inside_the_tile(self):
        tile = np.zeros((200, 300, 3), dtype=np.uint8)
        boxes = StubDetector().predict(tile)
        assert boxes[:, 0].min() >= 0
        assert boxes[:, 1].min() >= 0
        assert boxes[:, 2].max() <= 300
        assert boxes[:, 3].max() <= 200

    def test_scores_are_probabilities(self):
        boxes = StubDetector().predict(np.zeros((200, 200, 3), dtype=np.uint8))
        assert ((boxes[:, 4] >= 0.0) & (boxes[:, 4] <= 1.0)).all()

    def test_same_tile_gives_the_same_boxes(self, rng):
        tile = rng.integers(0, 255, size=(200, 200, 3), dtype=np.uint8)
        np.testing.assert_array_equal(
            StubDetector().predict(tile), StubDetector().predict(tile)
        )

    def test_different_tiles_give_different_boxes(self, rng):
        a = rng.integers(0, 255, size=(200, 200, 3), dtype=np.uint8)
        b = rng.integers(0, 255, size=(200, 200, 3), dtype=np.uint8)
        assert not np.array_equal(StubDetector().predict(a), StubDetector().predict(b))

    def test_tile_smaller_than_a_box_returns_nothing(self):
        tile = np.zeros((10, 10, 3), dtype=np.uint8)
        assert StubDetector(box_size=24).predict(tile).shape == (0, 5)

    def test_is_not_labelled_as_a_real_model(self):
        assert StubDetector().is_real_model is False


class TestDeepForestDetector:
    def test_rejects_a_non_rgb_tile_before_loading_weights(self):
        """Validation happens first, so this needs no model installed."""
        with pytest.raises(ValueError, match="RGB"):
            DeepForestDetector().predict(np.zeros((50, 50), dtype=np.uint8))

    def test_is_labelled_as_a_real_model(self):
        assert DeepForestDetector().is_real_model is True

    def test_score_threshold_is_configurable(self):
        assert DeepForestDetector(score_threshold=0.45).score_threshold == 0.45


class FakeDeepForest:
    """Stands in for the loaded model, returning DeepForest's own columns."""

    def __init__(self, frame):
        self.frame = frame
        self.seen = None

    def predict_image(self, image):
        self.seen = image
        return self.frame


class TestDeepForestAdapter:
    """The dataframe-to-array translation, tested without torch."""

    def frame(self, rows):
        import pandas as pd

        return pd.DataFrame(
            rows, columns=["xmin", "ymin", "xmax", "ymax", "score", "label"]
        )

    def test_maps_columns_in_protocol_order(self):
        model = FakeDeepForest(self.frame([[10, 20, 40, 55, 0.8, "Tree"]]))
        boxes = DeepForestDetector()._infer(
            model, np.zeros((100, 100, 3), dtype=np.uint8)
        )
        np.testing.assert_allclose(boxes, [[10, 20, 40, 55, 0.8]])

    def test_drops_boxes_below_the_score_threshold(self):
        model = FakeDeepForest(
            self.frame(
                [
                    [0, 0, 10, 10, 0.9, "Tree"],
                    [20, 20, 30, 30, 0.1, "Tree"],
                    [40, 40, 50, 50, 0.3, "Tree"],
                ]
            )
        )
        boxes = DeepForestDetector(score_threshold=0.3)._infer(
            model, np.zeros((100, 100, 3), dtype=np.uint8)
        )
        assert boxes.shape == (2, 5)
        assert (boxes[:, 4] >= 0.3).all()

    def test_empty_frame_gives_an_empty_array(self):
        model = FakeDeepForest(self.frame([]))
        boxes = DeepForestDetector()._infer(
            model, np.zeros((100, 100, 3), dtype=np.uint8)
        )
        assert boxes.shape == (0, 5)

    def test_none_result_gives_an_empty_array(self):
        model = FakeDeepForest(None)
        boxes = DeepForestDetector()._infer(
            model, np.zeros((100, 100, 3), dtype=np.uint8)
        )
        assert boxes.shape == (0, 5)

    def test_channels_are_reversed_for_deepforest(self):
        """DeepForest wants BGR; the protocol hands us RGB."""
        tile = np.zeros((4, 4, 3), dtype=np.uint8)
        tile[:, :, 0] = 255  # pure red in RGB
        model = FakeDeepForest(self.frame([]))
        DeepForestDetector()._infer(model, tile)
        assert model.seen[0, 0, 2] == 255  # arrives in the blue slot
        assert model.seen[0, 0, 0] == 0


class TestGetDetector:
    def test_defaults_to_the_stub(self, monkeypatch):
        monkeypatch.delenv("GEOFOREST_DETECTOR", raising=False)
        assert get_detector().name == "stub"

    def test_reads_the_environment(self, monkeypatch):
        monkeypatch.setenv("GEOFOREST_DETECTOR", "stub")
        assert get_detector().name == "stub"

    def test_argument_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv("GEOFOREST_DETECTOR", "deepforest")
        assert get_detector("stub").name == "stub"

    def test_name_is_case_insensitive(self, monkeypatch):
        monkeypatch.delenv("GEOFOREST_DETECTOR", raising=False)
        assert get_detector("STUB").name == "stub"

    def test_unknown_name_raises(self, monkeypatch):
        monkeypatch.delenv("GEOFOREST_DETECTOR", raising=False)
        with pytest.raises(ValueError, match="unknown detector"):
            get_detector("yolov8")

    def test_unavailable_model_falls_back_to_a_labelled_stub(self, monkeypatch, caplog):
        """A missing model must degrade visibly, never pretend to be real."""

        def explode(self):
            raise ImportError("deepforest is not installed")

        monkeypatch.setattr(DeepForestDetector, "_load", explode)
        with caplog.at_level("ERROR"):
            detector = get_detector("deepforest")

        assert detector.name == "stub"
        assert detector.is_real_model is False
        assert "falling back to stub" in caplog.text

    def test_returns_the_model_when_it_loads(self, monkeypatch):
        monkeypatch.setattr(DeepForestDetector, "_load", lambda self: object())
        detector = get_detector("deepforest")
        assert detector.name == "deepforest"
        assert detector.is_real_model is True
