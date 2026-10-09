"""API contract tests."""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def png_bytes(width=600, height=500, seed=7):
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


class TestHealth:
    def test_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_reports_which_detector_is_loaded(self, client):
        body = client.get("/health").json()
        assert "detector" in body
        assert isinstance(body["is_real_model"], bool)


class TestDetect:
    def test_returns_a_count_and_detections(self, client):
        response = client.post(
            "/detect", files={"file": ("tile.png", png_bytes(), "image/png")}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == len(body["detections"])
        assert body["count"] + body["duplicates_removed"] == body["raw_count"]

    def test_echoes_scene_dimensions(self, client):
        response = client.post(
            "/detect", files={"file": ("tile.png", png_bytes(640, 480), "image/png")}
        )
        body = response.json()
        assert (body["scene_width"], body["scene_height"]) == (640, 480)

    def test_honours_tile_parameters(self, client):
        response = client.post(
            "/detect",
            params={"tile_size": 256, "overlap": 0.1, "iou_threshold": 0.45},
            files={"file": ("tile.png", png_bytes(), "image/png")},
        )
        body = response.json()
        assert body["tile_size"] == 256
        assert body["overlap"] == 0.1
        assert body["iou_threshold"] == 0.45

    def test_detections_have_the_expected_shape(self, client):
        body = client.post(
            "/detect", files={"file": ("tile.png", png_bytes(), "image/png")}
        ).json()
        if body["detections"]:
            record = body["detections"][0]
            assert set(record) == {"id", "x0", "y0", "x1", "y1", "score"}
            assert 0.0 <= record["score"] <= 1.0

    def test_rejects_a_non_image(self, client):
        response = client.post(
            "/detect", files={"file": ("notes.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 415

    def test_rejects_an_oversized_image(self, client):
        response = client.post(
            "/detect", files={"file": ("huge.png", png_bytes(5000, 100), "image/png")}
        )
        assert response.status_code == 413

    def test_requires_a_file(self, client):
        assert client.post("/detect").status_code == 422

    def test_greyscale_upload_is_converted_not_rejected(self, client):
        buffer = io.BytesIO()
        Image.fromarray(np.zeros((300, 300), dtype=np.uint8), mode="L").save(
            buffer, format="PNG"
        )
        response = client.post(
            "/detect", files={"file": ("grey.png", buffer.getvalue(), "image/png")}
        )
        assert response.status_code == 200
