"""Tests for the ingestion and validation path of module M0.

The service accepts imagery from anyone, so what it refuses matters as much as
what it accepts. These cases cover the boundaries: the exact pixel limit rather
than a number comfortably inside it, a file whose extension disagrees with its
contents, and input degenerate enough to break arithmetic downstream.

Every rejection test is paired with an accepted equivalent, so a test cannot
pass because the service has started refusing everything.
"""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import main as api_main
from api.main import MAX_EDGE_PX, app


@pytest.fixture
def client():
    return TestClient(app)


def png_bytes(width=600, height=400, seed=3, mode="RGB"):
    """A valid PNG of the requested size, generated rather than stored."""
    rng = np.random.default_rng(seed)
    if mode == "L":
        array = rng.integers(0, 255, size=(height, width), dtype=np.uint8)
    else:
        array = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buffer, format="PNG")
    return buffer.getvalue()


def post(client, payload, name="scene.png", content_type="image/png", **params):
    return client.post(
        "/detect",
        params=params or None,
        files={"file": (name, payload, content_type)},
    )


class TestWhatIsAccepted:
    """The paired half of the rejection tests below."""

    def test_an_ordinary_png_is_accepted(self, client):
        assert post(client, png_bytes()).status_code == 200

    def test_a_non_square_scene_echoes_its_dimensions(self, client):
        body = post(client, png_bytes(width=4000, height=100)).json()
        assert (body["scene_width"], body["scene_height"]) == (4000, 100)

    def test_a_jpeg_is_accepted(self, client):
        buffer = io.BytesIO()
        rng = np.random.default_rng(5)
        Image.fromarray(rng.integers(0, 255, size=(300, 300, 3), dtype=np.uint8)).save(
            buffer, format="JPEG"
        )
        response = post(
            client, buffer.getvalue(), name="scene.jpg", content_type="image/jpeg"
        )
        assert response.status_code == 200


class TestContentDecidesNotExtension:
    """A filename is a hint from the client, not a fact about the bytes."""

    def test_a_png_named_tif_is_accepted(self, client):
        response = post(client, png_bytes(), name="scene.tif", content_type="image/tiff")
        assert response.status_code == 200

    def test_a_text_file_named_png_is_rejected(self, client):
        response = post(client, b"this is not an image", name="scene.png")
        assert response.status_code == 415


class TestDegenerateInput:
    def test_an_empty_upload_is_rejected_not_crashed(self, client):
        response = post(client, b"")
        assert response.status_code == 415

    def test_a_truncated_png_is_rejected(self, client):
        """Half a PNG is not a PNG; the decoder must not be trusted with it."""
        payload = png_bytes()
        response = post(client, payload[: len(payload) // 2])
        assert response.status_code == 415

    def test_a_single_pixel_scene_counts_zero(self, client):
        """Smaller than one tile and smaller than one box: no crash, no trees."""
        response = post(client, png_bytes(width=1, height=1))
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 0
        assert body["detections"] == []

    def test_a_greyscale_scene_is_converted_not_rejected(self, client):
        response = post(client, png_bytes(mode="L"))
        assert response.status_code == 200


class TestLimits:
    """The limits are tested at their exact values, not near them."""

    def test_an_image_at_the_edge_limit_is_accepted(self, client):
        payload = png_bytes(width=MAX_EDGE_PX, height=64)
        assert post(client, payload).status_code == 200

    def test_an_image_one_pixel_over_the_edge_limit_is_rejected(self, client):
        payload = png_bytes(width=MAX_EDGE_PX + 1, height=64)
        assert post(client, payload).status_code == 413

    def test_an_oversized_upload_is_rejected(self, client, monkeypatch):
        """The byte limit is lowered rather than a 32 MB file generated.

        Substituting the limit keeps the suite fast and tests the same branch;
        generating tens of megabytes of incompressible noise would not.
        """
        monkeypatch.setattr(api_main, "MAX_UPLOAD_BYTES", 500)
        payload = png_bytes(width=600, height=400)
        assert len(payload) > 500, "payload must exceed the lowered limit"
        assert post(client, payload).status_code == 413

    def test_the_same_upload_passes_under_the_normal_limit(self, client):
        """Pairs with the test above: the payload itself is not the problem."""
        assert post(client, png_bytes(width=600, height=400)).status_code == 200


class TestParameterValidation:
    def test_an_overlap_of_one_is_rejected(self, client):
        response = post(client, png_bytes(), overlap=1.0)
        assert response.status_code == 422

    def test_a_negative_overlap_is_rejected(self, client):
        response = post(client, png_bytes(), overlap=-0.1)
        assert response.status_code == 422

    def test_a_zero_tile_size_is_rejected(self, client):
        response = post(client, png_bytes(), tile_size=0)
        assert response.status_code == 422

    def test_valid_parameters_are_echoed(self, client):
        body = post(
            client, png_bytes(), tile_size=256, overlap=0.1, iou_threshold=0.45
        ).json()
        assert (body["tile_size"], body["overlap"], body["iou_threshold"]) == (
            256,
            0.1,
            0.45,
        )


class TestRequestsAreIndependent:
    def test_two_uploads_do_not_share_state(self, client):
        """A scene must not influence the next one through the detector."""
        first = post(client, png_bytes(width=600, height=400, seed=1)).json()
        second = post(client, png_bytes(width=800, height=600, seed=2)).json()

        assert (first["scene_width"], first["scene_height"]) == (600, 400)
        assert (second["scene_width"], second["scene_height"]) == (800, 600)

    def test_the_same_scene_twice_gives_the_same_count(self, client):
        payload = png_bytes(seed=9)
        first = post(client, payload).json()
        second = post(client, payload).json()
        assert first["count"] == second["count"]
