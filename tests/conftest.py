"""Shared fixtures.

The scripted detectors live in :mod:`geoforest.synthetic` so that the test
suite and ``scripts/dedup_table.py`` assert and report on the same code.
"""

import numpy as np
import pytest

from geoforest.synthetic import FixedCountDetector, SilentDetector


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def scene(rng):
    """A 1000x800 RGB scene of plausible canopy-ish noise."""
    return rng.integers(0, 255, size=(800, 1000, 3), dtype=np.uint8)


@pytest.fixture
def small_scene(rng):
    """Smaller than one tile, so the grid is a single tile."""
    return rng.integers(0, 255, size=(200, 200, 3), dtype=np.uint8)


@pytest.fixture
def silent_detector():
    return SilentDetector()


@pytest.fixture
def fixed_detector():
    return FixedCountDetector()
