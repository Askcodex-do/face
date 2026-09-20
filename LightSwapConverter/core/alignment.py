"""Geometric face alignment.

Alignment is a pure similarity transform (rotation + uniform scale +
translation) estimated from the eye centres. No projective or dense warp is
used, so the cost per frame stays low and the output cannot shear.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from core.landmarks import FaceLandmarks
from utils.logger import get_logger

try:  # pragma: no cover - exercised only on machines without OpenCV
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

logger = get_logger("core.alignment")

# Reference eye layout for a canonical, upright face of size 256x256.
CANONICAL_SIZE = (256, 256)
CANONICAL_LEFT_EYE = np.array([0.32 * CANONICAL_SIZE[0], 0.38 * CANONICAL_SIZE[1]], dtype=np.float32)
CANONICAL_RIGHT_EYE = np.array([0.68 * CANONICAL_SIZE[0], 0.38 * CANONICAL_SIZE[1]], dtype=np.float32)


@dataclass
class AlignmentResult:
    """Result of aligning one face."""

    matrix: np.ndarray
    size: tuple[int, int]
    scale: float = 1.0
    angle: float = 0.0
    valid: bool = True

    @property
    def inverse(self) -> np.ndarray:
        return cv2.invertAffineTransform(self.matrix)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Map an ``(N, 2)`` point array into aligned space."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.transform(pts, self.matrix).reshape(-1, 2)

    def transform_point(self, x: float, y: float) -> tuple[float, float]:
        out = self.transform_points(np.array([[x, y]], dtype=np.float32))[0]
        return float(out[0]), float(out[1])


class FaceAligner:
    """Build and apply similarity transforms for face crops."""

    def __init__(self, output_size: tuple[int, int] = CANONICAL_SIZE) -> None:
        self.output_size = (int(output_size[0]), int(output_size[1]))

    # ------------------------------------------------------------- estimation
    def compute_matrix(self, landmarks: FaceLandmarks) -> AlignmentResult:
        """Estimate the transform that maps a face onto the canonical layout."""
        if landmarks is None or not landmarks.is_valid:
            return self.identity_result(valid=False)

        left_eye = np.asarray(landmarks["left_eye"], dtype=np.float32)
        right_eye = np.asarray(landmarks["right_eye"], dtype=np.float32)

        delta = right_eye - left_eye
        source_distance = float(np.hypot(delta[0], delta[1]))
        if source_distance < 1e-3:
            # Degenerate landmarks (both eyes on the same pixel): fall back to
            # a translation-only transform instead of dividing by zero.
            return self.identity_result(valid=False)

        target_left, target_right = self._canonical_eyes()
        target_distance = float(np.hypot(*(target_right - target_left)))
        scale = target_distance / source_distance
        angle = float(np.degrees(np.arctan2(delta[1], delta[0])))

        source_center = (left_eye + right_eye) / 2.0
        target_center = (target_left + target_right) / 2.0

        rotation = cv2.getRotationMatrix2D(
            (float(source_center[0]), float(source_center[1])), angle, scale
        )
        rotation[0, 2] += target_center[0] - source_center[0]
        rotation[1, 2] += target_center[1] - source_center[1]

        return AlignmentResult(
            matrix=rotation.astype(np.float32),
            size=self.output_size,
            scale=scale,
            angle=angle,
            valid=True,
        )

    def align(self, frame: np.ndarray, landmarks: FaceLandmarks) -> tuple[np.ndarray, AlignmentResult]:
        """Warp the face region of ``frame`` into canonical space."""
        result = self.compute_matrix(landmarks)
        if frame is None or frame.size == 0:
            return np.zeros((self.output_size[1], self.output_size[0], 3), dtype=np.uint8), result
        warped = cv2.warpAffine(
            frame,
            result.matrix,
            self.output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return warped, result

    def align_face_crop(
        self, frame: np.ndarray, landmarks: FaceLandmarks
    ) -> tuple[np.ndarray, AlignmentResult]:
        """Align only the bounding box region, which is cheaper than the full frame."""
        face = landmarks.face
        if face is None:
            return self.align(frame, landmarks)

        patch = frame[face.y : face.y2, face.x : face.x2]
        if patch.size == 0:
            return self.align(frame, landmarks)

        shifted = landmarks.translated(-face.x, -face.y)
        return self.align(patch, shifted)

    # ------------------------------------------------------------------ helper
    def identity_result(self, valid: bool = True) -> AlignmentResult:
        matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        return AlignmentResult(matrix=matrix, size=self.output_size, scale=1.0, angle=0.0, valid=valid)

    def unalign(self, image: np.ndarray, result: AlignmentResult, frame_size: tuple[int, int]) -> np.ndarray:
        """Map an aligned image back into frame space."""
        width, height = frame_size
        return cv2.warpAffine(
            image,
            result.inverse,
            (int(width), int(height)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def _canonical_eyes(self) -> tuple[np.ndarray, np.ndarray]:
        scale_x = self.output_size[0] / CANONICAL_SIZE[0]
        scale_y = self.output_size[1] / CANONICAL_SIZE[1]
        left = CANONICAL_LEFT_EYE * np.array([scale_x, scale_y], dtype=np.float32)
        right = CANONICAL_RIGHT_EYE * np.array([scale_x, scale_y], dtype=np.float32)
        return left, right

    @staticmethod
    def average_landmarks(sequence: Sequence[FaceLandmarks]) -> Optional[FaceLandmarks]:
        """Temporally average landmarks to damp jitter between frames."""
        valid = [item for item in sequence if item is not None and item.is_valid]
        if not valid:
            return None
        stacked = np.stack([item.points for item in valid], axis=0)
        averaged = stacked.mean(axis=0).astype(np.float32)
        return FaceLandmarks(points=averaged, names=valid[0].names, source="averaged", face=valid[-1].face)
