"""Tests for the tiling grid."""

from itertools import pairwise

import pytest

from geoforest.tiling import tile_grid, tile_origins


class TestTileOrigins:
    def test_single_tile_when_axis_shorter_than_tile(self):
        assert tile_origins(300, tile_size=400, overlap=0.25) == [0]

    def test_single_tile_on_exact_fit(self):
        assert tile_origins(400, tile_size=400, overlap=0.25) == [0]

    def test_no_overlap_gives_contiguous_tiles(self):
        assert tile_origins(1200, tile_size=400, overlap=0.0) == [0, 400, 800]

    def test_quarter_overlap_gives_expected_stride(self):
        # stride = 400 * 0.75 = 300
        assert tile_origins(1000, tile_size=400, overlap=0.25) == [0, 300, 600]

    def test_final_origin_is_clamped_to_fit(self):
        """A ragged edge pulls the last tile back instead of shrinking it."""
        origins = tile_origins(1024, tile_size=400, overlap=0.25)
        assert origins == [0, 300, 600, 624]
        assert origins[-1] + 400 == 1024

    def test_origins_cover_the_whole_axis(self):
        length, tile = 1500, 400
        origins = tile_origins(length, tile_size=tile, overlap=0.25)
        assert origins[0] == 0
        assert origins[-1] + tile >= length

    def test_origins_are_strictly_increasing(self):
        origins = tile_origins(2000, tile_size=256, overlap=0.3)
        assert all(b > a for a, b in pairwise(origins))

    @pytest.mark.parametrize("overlap", [-0.1, 1.0, 1.5])
    def test_invalid_overlap_raises(self, overlap):
        with pytest.raises(ValueError, match="overlap"):
            tile_origins(1000, tile_size=400, overlap=overlap)

    @pytest.mark.parametrize("tile_size", [0, -10])
    def test_invalid_tile_size_raises(self, tile_size):
        with pytest.raises(ValueError, match="tile_size"):
            tile_origins(1000, tile_size=tile_size, overlap=0.25)


class TestTileGrid:
    def test_grid_is_product_of_axes(self):
        tiles = tile_grid(1000, 1000, tile_size=400, overlap=0.25)
        assert len(tiles) == 3 * 3

    def test_small_scene_yields_one_tile_of_scene_size(self):
        tiles = tile_grid(200, 150, tile_size=400, overlap=0.25)
        assert len(tiles) == 1
        assert (tiles[0].width, tiles[0].height) == (200, 150)

    def test_tiles_stay_inside_the_scene(self):
        width, height = 1024, 768
        for tile in tile_grid(width, height, tile_size=400, overlap=0.25):
            assert 0 <= tile.x0 < tile.x1 <= width
            assert 0 <= tile.y0 < tile.y1 <= height

    def test_tiles_are_full_size_when_scene_allows(self):
        for tile in tile_grid(1024, 1024, tile_size=400, overlap=0.25):
            assert (tile.width, tile.height) == (400, 400)

    def test_every_pixel_is_covered(self):
        """No gaps: each column and row falls inside at least one tile."""
        width, height, size = 900, 700, 256
        tiles = tile_grid(width, height, tile_size=size, overlap=0.2)
        covered_x = {x for t in tiles for x in range(t.x0, t.x1)}
        covered_y = {y for t in tiles for y in range(t.y0, t.y1)}
        assert covered_x == set(range(width))
        assert covered_y == set(range(height))

    def test_overlap_produces_shared_pixels(self):
        """The overlap that causes double-counting is really there."""
        tiles = tile_grid(1000, 400, tile_size=400, overlap=0.25)
        first, second = tiles[0], tiles[1]
        assert second.x0 < first.x1

    def test_offset_matches_origin(self):
        tile = tile_grid(1000, 1000, tile_size=400, overlap=0.25)[4]
        assert tile.offset == (tile.x0, tile.y0)
