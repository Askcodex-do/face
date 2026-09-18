"""Pytest fixtures for the LightSwapConverter test suite.

Path setup lives here so every test module can import the package by name
without an installed distribution. Synthetic media builders live in
``tests.helpers``; this module only wires them into fixtures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
for candidate in (PROJECT_ROOT, PACKAGE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tests.helpers import draw_synthetic_face, write_test_image, write_test_video


@pytest.fixture(autouse=True)
def _quiet_logging():
    """Send log output to a memory buffer only, never to the filesystem."""
    from LightSwapConverter.utils.logger import configure_for_tests

    configure_for_tests("DEBUG")
    yield


@pytest.fixture
def synthetic_face() -> np.ndarray:
    """An in-memory synthetic face image."""
    return draw_synthetic_face()


@pytest.fixture
def face_image(tmp_path: Path) -> Path:
    """A synthetic face image written to a temporary file."""
    return write_test_image(tmp_path / "face.png")


@pytest.fixture
def target_video(tmp_path: Path) -> Path:
    """A short synthetic video with a face in every frame."""
    return write_test_video(tmp_path / "target.mp4", frames=10)
