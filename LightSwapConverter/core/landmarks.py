"""Facial landmark estimation.

Two strategies are provided, both dependency light and fully offline:

``template``
    A fixed 5-point layout placed geometrically inside the detected face box.
    Rough, but instant and stable, which is enough to drive a similarity
    alignment.
``cascade``
    Per-feature Haar cascades (eyes, nose, mouth) refine the template when the
    cascades are available locally. Still no neural network involved.

``FaceLandmarks`` exposes the result as a numpy array plus named accessors, so
downstream code never depends on how the points were produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from core.face_detector import FaceDetector, FaceRegion
from utils.logger import get_logger

try:  # pragma: no cover - exercised only on machines without OpenCV
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

logger = get_logger("core.landmarks")

# Canonical 5-point layout in normalised face-box coordinates.
# Order: left eye, right eye, nose tip, left mouth corner, right mouth corner.
TEMPLATE_LAYOUT: Dict[str, tuple[float, float]] = {
    "left_eye": (0.32, 0.38),
    "right_eye": (0.68, 0.38),
    "nose": (0.50, 0.58),
    "mouth_left": (0.38, 0.76),
    "mouth_right": (0.62, 0.76),
}

LANDMARK_NAMES = tuple(TEMPLATE_LAYOUT.keys())

CASCADE_FILES = {
    "left_eye": "haarcascade_lefteye_2splits.xml",
    "right_eye": "haarcascade_righteye_2splits.xml",
    "mouth": "haarcascade_smile.xml",
}


@dataclass
class FaceLandmarks:
    """Landmark points for one face.

    ``points`` is an ``(N, 2)`` float32 array in pixel coordinates.
    """

    points: np.ndarray
    names: tuple[str, ...] = LANDMARK_NAMES
    source: str = "template"
    face: Optional[FaceRegion] = None
    metadata: Dict[str, float] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return int(self.points.shape[0]) if self.points.size else 0

    @property
    def is_valid(self) -> bool:
        return self.points.ndim == 2 and self.points.shape[0] >= 3 and self.points.shape[1] == 2

    @property
    def center(self) -> np.ndarray:
        return self.points.mean(axis=0) if self.count else np.zeros(2, dtype=np.float32)

    @property
    def eye_center(self) -> np.ndarray:
        return (self["left_eye"] + self["right_eye"]) / 2.0

    @property
    def inter_ocular_distance(self) -> float:
        return float(np.linalg.norm(self["right_eye"] - self["left_eye"]))

    @property
    def roll_angle(self) -> float:
        """In-plane head rotation in degrees."""
        delta = self["right_eye"] - self["left_eye"]
        return float(np.degrees(np.arctan2(delta[1], delta[0])))

    def __getitem__(self, name: str) -> np.ndarray:
        try:
            index = self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"Unknown landmark {name!r}; known: {self.names}") from exc
        return self.points[index]

    def as_dict(self) -> Dict[str, tuple[float, float]]:
        return {
            name: (float(self.points[i][0]), float(self.points[i][1]))
            for i, name in enumerate(self.names)
            if i < self.count
        }

    def translated(self, dx: float, dy: float) -> "FaceLandmarks":
        offset = np.array([dx, dy], dtype=np.float32)
        return FaceLandmarks(self.points + offset, self.names, self.source, self.face, dict(self.metadata))


class LandmarkExtractor:
    """Produce landmarks for a face box."""

    def __init__(
        self,
        mode: str = "template",
        detector: Optional[FaceDetector] = None,
        cascades_dir: Optional[str | Path] = None,
    ) -> None:
        self.mode = mode
        self.detector = detector
        self._cascades_dir = Path(cascades_dir) if cascades_dir else self._default_cascades_dir()
        self._feature_cascades: Dict[str, object] = {}
        self._cascades_loaded = False

    # -------------------------------------------------------------- extraction
    def extract(self, frame: np.ndarray, face: FaceRegion) -> FaceLandmarks:
        """Estimate landmarks for ``face`` inside ``frame``."""
        if face is None or not face.is_valid:
            raise ValueError("A valid FaceRegion is required to extract landmarks.")

        if self.mode == "cascade" and frame is not None and frame.size:
            refined = self._extract_with_cascades(frame, face)
            if refined is not None:
                return refined

        return self._template_landmarks(face)

    def extract_many(self, frame: np.ndarray, faces: List[FaceRegion]) -> List[FaceLandmarks]:
        return [self.extract(frame, face) for face in faces]

    # ---------------------------------------------------------------- template
    @staticmethod
    def _template_landmarks(face: FaceRegion) -> FaceLandmarks:
        points = np.array(
            [
                (face.x + face.width * fx, face.y + face.height * fy)
                for fx, fy in TEMPLATE_LAYOUT.values()
            ],
            dtype=np.float32,
        )
        return FaceLandmarks(points=points, source="template", face=face)

    # ----------------------------------------------------------------- cascade
    def _extract_with_cascades(self, frame: np.ndarray, face: FaceRegion) -> Optional[FaceLandmarks]:
        if not self._load_cascades():
            return None

        points = self._template_landmarks(face).points.copy()
        names = list(LANDMARK_NAMES)
        gray = FaceDetector._to_gray(frame)
        region = gray[face.y : face.y2, face.x : face.x2]
        if region.size == 0:
            return None

        hits = 0
        for feature in ("left_eye", "right_eye", "mouth"):
            cascade = self._feature_cascades.get(feature)
            if cascade is None:
                continue
            # Search only the anatomically plausible band of the face box.
            y_from, y_to = self._search_band(feature, face.height)
            strip = region[y_from:y_to, :]
            if strip.size == 0:
                continue
            found = cascade.detectMultiScale(strip, scaleFactor=1.1, minNeighbors=4, minSize=(8, 8))
            if len(found) == 0:
                continue
            x, y, w, h = max(found, key=lambda b: b[2] * b[3])
            cx = face.x + x + w / 2.0
            cy = face.y + y_from + y + h / 2.0

            if feature == "mouth":
                half = face.width * 0.12
                points[names.index("mouth_left")] = (cx - half, cy)
                points[names.index("mouth_right")] = (cx + half, cy)
            else:
                points[names.index(feature)] = (cx, cy)
            hits += 1

        if hits == 0:
            return None

        landmarks = FaceLandmarks(points=points, source="cascade", face=face)
        landmarks.metadata["refined_features"] = float(hits)
        return landmarks

    def _load_cascades(self) -> bool:
        if self._cascades_loaded:
            return bool(self._feature_cascades)
        self._cascades_loaded = True

        if cv2 is None or self._cascades_dir is None:
            return False
        if not hasattr(cv2, "CascadeClassifier"):
            return False

        for feature, filename in CASCADE_FILES.items():
            path = self._cascades_dir / filename
            if not path.is_file():
                continue
            cascade = cv2.CascadeClassifier(str(path))
            if not cascade.empty():
                self._feature_cascades[feature] = cascade

        if not self._feature_cascades:
            logger.info("No feature cascades available; using the geometric template.")
        return bool(self._feature_cascades)

    @staticmethod
    def _search_band(feature: str, face_height: int) -> tuple[int, int]:
        if feature == "mouth":
            return int(face_height * 0.55), int(face_height * 0.95)
        return int(face_height * 0.15), int(face_height * 0.60)

    @staticmethod
    def _default_cascades_dir() -> Optional[Path]:
        if cv2 is None or not hasattr(cv2, "data"):
            return None
        return Path(cv2.data.haarcascades)
