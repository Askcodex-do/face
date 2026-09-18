"""Face detection using OpenCV's bundled Haar cascades.

Haar cascades ship with OpenCV, need no model download, run on the CPU and use a
few megabytes of memory, which is exactly what the target machine allows. They
are less accurate than a modern detector, so this module adds two cheap
mitigations: a region of interest around the previous detection, and a list of
fallback cascades.

The public surface is intentionally narrow so a different detector can be
dropped in later without touching the rest of the pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..utils.config import DetectionConfig
from ..utils.logger import get_logger

log = get_logger("core.face_detector")


class DetectorError(RuntimeError):
    """Raised when no cascade file can be loaded."""


@dataclass(frozen=True)
class FaceBox:
    """An axis aligned face rectangle in pixel coordinates."""

    x: int
    y: int
    width: int
    height: int
    #: Detector confidence proxy. Haar has no score, so this is the number of
    #: neighbours that agreed, normalised by the configured minimum.
    score: float = 1.0

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> Tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    def clamp(self, frame_width: int, frame_height: int) -> "FaceBox":
        """Return a copy clipped to the frame bounds."""
        x = max(0, min(self.x, frame_width - 1))
        y = max(0, min(self.y, frame_height - 1))
        x2 = max(x + 1, min(self.x2, frame_width))
        y2 = max(y + 1, min(self.y2, frame_height))
        return FaceBox(x, y, x2 - x, y2 - y, self.score)

    def pad(self, ratio: float, frame_width: int, frame_height: int) -> "FaceBox":
        """Return a copy grown by ``ratio`` on every side, then clamped."""
        if ratio <= 0:
            return self.clamp(frame_width, frame_height)
        dx = int(round(self.width * ratio))
        dy = int(round(self.height * ratio))
        grown = FaceBox(
            self.x - dx, self.y - dy, self.width + 2 * dx, self.height + 2 * dy,
            self.score,
        )
        return grown.clamp(frame_width, frame_height)

    def scale(self, factor: float) -> "FaceBox":
        """Return a copy scaled about its centre."""
        cx, cy = self.center
        width = max(1, int(round(self.width * factor)))
        height = max(1, int(round(self.height * factor)))
        return FaceBox(
            cx - width // 2, cy - height // 2, width, height, self.score
        )

    def expand(self, factor: float) -> "FaceBox":
        """Return a square box grown by ``factor``, used to track motion."""
        cx, cy = self.center
        side = max(self.width, self.height) * factor
        half = side / 2.0
        return FaceBox(
            int(round(cx - half)),
            int(round(cy - half)),
            int(round(side)),
            int(round(side)),
            self.score,
        )


def _candidate_cascade_paths(file_name: str) -> List[str]:
    """Return plausible absolute paths for an OpenCV cascade file."""
    candidates: List[str] = []
    data_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if data_dir:
        candidates.append(os.path.join(data_dir, file_name))
    # Some builds only expose the cascades relative to the cv2 package folder.
    package_dir = os.path.dirname(getattr(cv2, "__file__", "") or "")
    if package_dir:
        candidates.append(
            os.path.join(package_dir, "data", "haarcascades", file_name)
        )
    candidates.append(file_name)
    return candidates


def load_cascade(file_name: str) -> cv2.CascadeClassifier:
    """Load a cascade by file name, searching the OpenCV installation."""
    for candidate in _candidate_cascade_paths(file_name):
        if os.path.isfile(candidate):
            cascade = cv2.CascadeClassifier(candidate)
            if not cascade.empty():
                log.debug("loaded cascade %s", candidate)
                return cascade
    raise DetectorError(
        f"cannot locate cascade {file_name!r}; expected it inside the OpenCV "
        f"package data directory"
    )


class FaceDetector:
    """Detect the most prominent face in each frame.

    The detector keeps a little state: the region of interest from the previous
    frame is searched first, which both speeds up detection and keeps the
    identity of the tracked face stable across frames.
    """

    def __init__(self, config: Optional[DetectionConfig] = None) -> None:
        self.config = config or DetectionConfig()
        self._primary = load_cascade(self.config.cascade_file)
        self._fallbacks: List[cv2.CascadeClassifier] = []
        for name in self.config.fallback_cascades:
            try:
                self._fallbacks.append(load_cascade(name))
            except DetectorError as exc:
                log.debug("skipping fallback cascade: %s", exc)
        self._last_box: Optional[FaceBox] = None

    # -- public API ---------------------------------------------------------

    def detect(self, frame: np.ndarray) -> Optional[FaceBox]:
        """Return the best face box in ``frame``, or ``None`` when there is none."""
        if frame is None or frame.size == 0:
            return None

        height, width = frame.shape[:2]
        min_side = max(20, int(round(min(height, width) * self.config.min_face_ratio)))
        gray = self._to_gray(frame)

        box = self._detect_in_roi(gray, width, height, min_side)
        if box is None:
            box = self._detect_full(gray, width, height, min_side)
        if box is None:
            self._last_box = None
            return None

        padded = box.pad(self.config.padding, width, height)
        self._last_box = padded
        return padded

    def detect_all(self, frame: np.ndarray) -> List[FaceBox]:
        """Return every face found in ``frame``, best first.

        The converter only replaces one face, but exposing the full list keeps
        the detector useful for previews and future multi-face support.
        """
        if frame is None or frame.size == 0:
            return []
        height, width = frame.shape[:2]
        min_side = max(20, int(round(min(height, width) * self.config.min_face_ratio)))
        gray = self._to_gray(frame)
        detections = self._primary.detectMultiScale(
            gray,
            scaleFactor=self.config.scale_factor,
            minNeighbors=self.config.min_neighbors,
            minSize=(min_side, min_side),
        )
        boxes = [self._to_box(item, min_side, width, height) for item in detections]
        boxes.sort(key=lambda item: item.area, reverse=True)
        return boxes

    def reset(self) -> None:
        """Forget the tracked region, e.g. when starting a new video."""
        self._last_box = None

    # -- internals ----------------------------------------------------------

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        """Convert to grayscale for the cascade.

        No histogram equalisation is applied. A global contrast stretch
        amplifies compression noise in video frames, which measurably reduces
        the detection rate on compressed input, so the plain luminance channel
        is the better choice here.
        """
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _to_box(
        self, raw: Sequence[int], min_side: int, width: int, height: int
    ) -> FaceBox:
        x, y, w, h = (int(value) for value in raw[:4])
        score = 1.0
        if min_side > 0:
            score = max(0.1, min(1.0, w / float(min_side * 4)))
        return FaceBox(x, y, w, h, score).clamp(width, height)

    def _detect_in_roi(
        self, gray: np.ndarray, width: int, height: int, min_side: int
    ) -> Optional[FaceBox]:
        if not self.config.use_tracking or self._last_box is None:
            return None

        roi_box = self._last_box.expand(1.6).clamp(width, height)
        if roi_box.width < min_side or roi_box.height < min_side:
            return None

        roi = gray[roi_box.y : roi_box.y2, roi_box.x : roi_box.x2]
        if roi.size == 0:
            return None

        detections = self._primary.detectMultiScale(
            roi,
            scaleFactor=self.config.scale_factor,
            minNeighbors=max(2, self.config.min_neighbors - 2),
            minSize=(min_side, min_side),
        )
        if len(detections) == 0:
            return None

        # Translate ROI coordinates back into frame coordinates and keep the
        # detection closest to the previous centre.
        prev_center = np.array(self._last_box.center, dtype=np.float32)
        best: Optional[FaceBox] = None
        best_distance = float("inf")
        for raw in detections:
            x, y, w, h = (int(value) for value in raw[:4])
            candidate = FaceBox(x + roi_box.x, y + roi_box.y, w, h)
            distance = float(np.linalg.norm(np.array(candidate.center) - prev_center))
            if distance < best_distance:
                best_distance = distance
                best = candidate
        if best is None:
            return None
        return best.clamp(width, height)

    def _detect_full(
        self, gray: np.ndarray, width: int, height: int, min_side: int
    ) -> Optional[FaceBox]:
        """Search the whole frame with the primary cascade, then the fallbacks."""
        detections = self._primary.detectMultiScale(
            gray,
            scaleFactor=self.config.scale_factor,
            minNeighbors=self.config.min_neighbors,
            minSize=(min_side, min_side),
        )
        if len(detections) > 0:
            candidates = [
                self._to_box(raw, min_side, width, height) for raw in detections
            ]
            candidates.sort(key=lambda item: item.area, reverse=True)
            return candidates[0]

        for cascade in self._fallbacks:
            detections = cascade.detectMultiScale(
                gray,
                scaleFactor=self.config.scale_factor,
                minNeighbors=self.config.min_neighbors,
                minSize=(min_side, min_side),
            )
            if len(detections) > 0:
                candidates = [
                    self._to_box(raw, min_side, width, height) for raw in detections
                ]
                candidates.sort(key=lambda item: item.area, reverse=True)
                log.debug("primary cascade missed, fallback found a face")
                return candidates[0]
        return None


def detect_largest_face(
    frame: np.ndarray, config: Optional[DetectionConfig] = None
) -> Optional[FaceBox]:
    """One shot helper that detects a face without keeping detector state."""
    return FaceDetector(config).detect(frame)
