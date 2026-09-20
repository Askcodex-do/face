"""Face detection.

Uses OpenCV's bundled Haar cascade classifier: a few hundred kilobytes, no
model download, no neural network runtime. That fits the offline / 2 GB RAM /
CPU-only constraints. The class is written against a small interface so a
different detector can be swapped in later without touching the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from utils.logger import get_logger

try:  # pragma: no cover - exercised only on machines without OpenCV
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

logger = get_logger("core.face_detector")

DEFAULT_CASCADE = "haarcascade_frontalface_default.xml"


@dataclass(frozen=True)
class FaceRegion:
    """A detected face as a pixel-space bounding box."""

    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def is_valid(self) -> bool:
        return self.width > 0 and self.height > 0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    def scaled(self, factor: float) -> "FaceRegion":
        """Return the box scaled by ``factor`` about its center."""
        cx, cy = self.center
        width = max(int(self.width * factor), 1)
        height = max(int(self.height * factor), 1)
        return FaceRegion(cx - width // 2, cy - height // 2, width, height, self.confidence)

    def clamp(self, frame_width: int, frame_height: int) -> "FaceRegion":
        """Clip the box to the frame boundaries."""
        x1 = max(min(self.x, frame_width - 1), 0)
        y1 = max(min(self.y, frame_height - 1), 0)
        x2 = max(min(self.x2, frame_width), x1 + 1)
        y2 = max(min(self.y2, frame_height), y1 + 1)
        return FaceRegion(x1, y1, x2 - x1, y2 - y1, self.confidence)


class FaceDetector:
    """Detect faces in BGR frames using a Haar cascade."""

    def __init__(
        self,
        cascade_path: Optional[str | Path] = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: int = 48,
        max_faces: int = 2,
    ) -> None:
        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)
        self.min_size = int(min_size)
        self.max_faces = int(max_faces)
        self._cascade_path = Path(cascade_path) if cascade_path else self._default_cascade_path()
        self._cascade = None
        self._loaded = False

    # -------------------------------------------------------------- lifecycle
    def load(self) -> bool:
        """Load the cascade lazily. Returns ``True`` when detection is ready."""
        if self._loaded:
            return self._cascade is not None
        self._loaded = True

        if cv2 is None:
            logger.warning("OpenCV is not installed; face detection is disabled.")
            return False
        if not hasattr(cv2, "CascadeClassifier"):
            # OpenCV 5 removed the legacy objdetect cascades.
            logger.warning("This OpenCV build has no CascadeClassifier; detection is disabled.")
            return False
        if self._cascade_path is None or not Path(self._cascade_path).is_file():
            logger.warning("Cascade file not found at %s; detection is disabled.", self._cascade_path)
            return False

        cascade = cv2.CascadeClassifier(str(self._cascade_path))
        if cascade.empty():
            logger.warning("Cascade file at %s could not be parsed.", self._cascade_path)
            return False

        self._cascade = cascade
        logger.info("Loaded cascade %s", self._cascade_path.name)
        return True

    @property
    def is_ready(self) -> bool:
        return self._cascade is not None

    # -------------------------------------------------------------- detection
    def detect(self, frame: np.ndarray, max_faces: Optional[int] = None) -> List[FaceRegion]:
        """Return detected faces sorted by area, largest first."""
        if frame is None or frame.size == 0:
            return []
        if not self.load():
            return []

        gray = self._to_gray(frame)
        height, width = gray.shape[:2]
        min_side = max(min(self.min_size, height, width), 8)

        detect_kwargs = {
            "scaleFactor": self.scale_factor,
            "minNeighbors": self.min_neighbors,
            "minSize": (min_side, min_side),
        }
        if hasattr(cv2, "CASCADE_SCALE_IMAGE"):
            detect_kwargs["flags"] = cv2.CASCADE_SCALE_IMAGE

        boxes = self._cascade.detectMultiScale(gray, **detect_kwargs)

        limit = self.max_faces if max_faces is None else int(max_faces)
        regions = [
            FaceRegion(int(x), int(y), int(w), int(h)).clamp(width, height)
            for x, y, w, h in boxes
        ]
        regions = [r for r in regions if r.is_valid]
        regions.sort(key=lambda r: r.area, reverse=True)
        if limit > 0:
            regions = regions[:limit]
        return regions

    def detect_primary(self, frame: np.ndarray) -> Optional[FaceRegion]:
        """Convenience wrapper returning only the largest face, if any."""
        faces = self.detect(frame, max_faces=1)
        return faces[0] if faces else None

    def detect_batch(self, frames: Sequence[np.ndarray]) -> List[List[FaceRegion]]:
        """Detect faces on several frames, preserving order."""
        return [self.detect(frame) for frame in frames]

    # ------------------------------------------------------------------ helper
    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _default_cascade_path() -> Optional[Path]:
        """Locate the cascade shipped inside the OpenCV package."""
        if cv2 is None:
            return None
        data_dir = Path(cv2.data.haarcascades) if hasattr(cv2, "data") else None
        if data_dir is None:
            return None
        return data_dir / DEFAULT_CASCADE
