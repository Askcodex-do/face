"""Geometric alignment between a source face and a target face.

The job of this module is to answer one question: given the source face image
and the target face in a frame, what transform maps the source onto the target?
It returns plain matrices and sizes, so it can be unit tested without any image
processing involved.

Similarity transforms (rotation, uniform scale, translation) are used by default
because they preserve the shape of the face and cannot introduce the shearing
that makes a swap look rubbery. Affine and homography fits are available for
callers that want to match a strong head pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from ..utils.logger import get_logger
from .face_detector import FaceBox
from .landmarks import ALIGNMENT_INDICES, Landmarks

log = get_logger("core.alignment")

#: The five points used for a similarity fit, by index into the 68 point layout.
DEFAULT_FIT_INDICES: Tuple[int, ...] = ALIGNMENT_INDICES


class AlignmentError(RuntimeError):
    """Raised when an alignment transform cannot be estimated."""


@dataclass(frozen=True)
class AlignmentResult:
    """The outcome of aligning a source face to a target face."""

    #: 2x3 transform mapping source image coordinates to frame coordinates.
    matrix: np.ndarray
    #: Size of the aligned source patch, i.e. the face box in the frame.
    size: Tuple[int, int]
    #: Mean reprojection error of the fitted points, in pixels.
    error: float
    #: ``"similarity"``, ``"affine"`` or ``"homography"``.
    method: str

    @property
    def is_reliable(self) -> bool:
        """True when the fit error is small relative to the patch size."""
        width, height = self.size
        limit = max(4.0, min(width, height) * 0.10)
        return self.error <= limit

    def __str__(self) -> str:  # pragma: no cover - display helper
        return (
            f"{self.method} {self.size[0]}x{self.size[1]} "
            f"error={self.error:.2f}px reliable={self.is_reliable}"
        )


def _eye_and_nose_points(landmarks: Landmarks) -> np.ndarray:
    """Return the alignment points of a landmark set, shape (5, 2)."""
    points = landmarks.points
    indices = [index for index in DEFAULT_FIT_INDICES if index < len(points)]
    if len(indices) < 3:
        raise AlignmentError("need at least three alignment points")
    return points[indices]


def fit_similarity(
    source_points: np.ndarray, target_points: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Fit a similarity transform from source to target points.

    Uses :func:`cv2.estimateAffinePartial2D`, which is the least squares
    similarity fit with RANSAC available for outliers.
    """
    source = np.asarray(source_points, dtype=np.float32)
    target = np.asarray(target_points, dtype=np.float32)
    if source.shape != target.shape or source.shape[0] < 2:
        raise AlignmentError("source and target point sets must match and hold >= 2 points")

    matrix, _ = cv2.estimateAffinePartial2D(
        source, target, method=cv2.LMEDS
    )
    if matrix is None:
        # LMEDS can fail on perfectly collinear input; fall back to the plain
        # least squares fit before giving up.
        matrix, _ = cv2.estimateAffinePartial2D(source, target, method=0)
    if matrix is None:
        raise AlignmentError("similarity fit failed")
    return matrix.astype(np.float32), _reprojection_error(matrix, source, target)


def fit_affine(
    source_points: np.ndarray, target_points: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Fit a full affine transform, allowing shear and non uniform scale."""
    source = np.asarray(source_points, dtype=np.float32)
    target = np.asarray(target_points, dtype=np.float32)
    if source.shape != target.shape or source.shape[0] < 3:
        raise AlignmentError("affine fit needs matching sets of >= 3 points")

    matrix = cv2.getAffineTransform(source[:3], target[:3])
    if matrix is None or not np.all(np.isfinite(matrix)):
        raise AlignmentError("affine fit failed")
    return matrix.astype(np.float32), _reprojection_error(matrix, source, target)


def fit_homography(
    source_points: np.ndarray, target_points: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Fit a perspective transform. Only useful for a strong head turn."""
    source = np.asarray(source_points, dtype=np.float32)
    target = np.asarray(target_points, dtype=np.float32)
    if source.shape != target.shape or source.shape[0] < 4:
        raise AlignmentError("homography fit needs matching sets of >= 4 points")

    matrix, _ = cv2.findHomography(source, target, method=0)
    if matrix is None or not np.all(np.isfinite(matrix)):
        raise AlignmentError("homography fit failed")
    # The full 3x3 matrix is passed on: a projective transform cannot be
    # evaluated from its first two rows alone, because the perspective divide
    # by the third row is part of the mapping.
    return matrix.astype(np.float32), _reprojection_error(matrix, source, target)


def _reprojection_error(
    matrix: np.ndarray, source: np.ndarray, target: np.ndarray
) -> float:
    """Mean distance between the mapped source points and the target points.

    Accepts a 2x3 affine matrix or a full 3x3 projective one.
    """
    if matrix.shape == (2, 3):
        projected = cv2.transform(source.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    else:
        projected = cv2.perspectiveTransform(
            source.reshape(-1, 1, 2), matrix
        ).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(projected - target, axis=1)))


def similarity_from_eyes(
    source_landmarks: Landmarks, target_landmarks: Landmarks
) -> Tuple[np.ndarray, float]:
    """Build a similarity transform from the two eye centres and the nose tip.

    This is the cheapest reliable alignment and the one the processor uses as a
    fallback when the full fit is rejected.
    """
    source = _eye_and_nose_points(source_landmarks)
    target = _eye_and_nose_points(target_landmarks)
    return fit_similarity(source, target)


def scale_matrix(factor: float, offset_x: float = 0.0, offset_y: float = 0.0) -> np.ndarray:
    """Return a 2x3 matrix that scales then translates."""
    return np.array(
        [[factor, 0.0, offset_x], [0.0, factor, offset_y]], dtype=np.float32
    )


def compose(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """Return the transform that applies ``inner`` first, then ``outer``."""
    outer_3x3 = np.vstack([outer, [0.0, 0.0, 1.0]]) if outer.shape == (2, 3) else outer
    inner_3x3 = np.vstack([inner, [0.0, 0.0, 1.0]]) if inner.shape == (2, 3) else inner
    result = outer_3x3 @ inner_3x3
    if outer.shape == (2, 3) and inner.shape == (2, 3):
        return result[:2].astype(np.float32)
    return result.astype(np.float32)


def invert(matrix: np.ndarray) -> np.ndarray:
    """Return the inverse of a 2x3 or 3x3 transform."""
    if matrix.shape == (2, 3):
        square = np.vstack([matrix, [0.0, 0.0, 1.0]])
        return np.linalg.inv(square)[:2].astype(np.float32)
    return np.linalg.inv(matrix).astype(np.float32)


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 2x3 transform to an ``(N, 2)`` array of points."""
    array = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if matrix.shape == (2, 3):
        return cv2.transform(array, matrix).reshape(-1, 2)
    return cv2.perspectiveTransform(array, matrix).reshape(-1, 2)


def align_to_face(
    source_landmarks: Landmarks,
    target_landmarks: Landmarks,
    target_box: FaceBox,
    *,
    method: str = "similarity",
) -> AlignmentResult:
    """Align a source face onto a target face inside a frame.

    ``method`` selects the transform family: ``"similarity"``, ``"affine"`` or
    ``"homography"``. The returned matrix maps source image coordinates into the
    frame, and ``size`` is the patch size the caller should warp into.
    """
    size = (int(target_box.width), int(target_box.height))
    if size[0] < 2 or size[1] < 2:
        raise AlignmentError(f"target box is too small to align to: {size}")

    try:
        if method == "affine":
            matrix, error = fit_affine(
                source_landmarks.points, target_landmarks.points
            )
        elif method == "homography":
            matrix, error = fit_homography(
                source_landmarks.points, target_landmarks.points
            )
        else:
            matrix, error = fit_similarity(
                _eye_and_nose_points(source_landmarks),
                _eye_and_nose_points(target_landmarks),
            )
    except AlignmentError as exc:
        log.warning("alignment with method %s failed: %s", method, exc)
        raise

    result = AlignmentResult(matrix, size, error, method)
    if not result.is_reliable:
        log.debug("alignment error is high: %s", result)
    return result


def resize_source_to_box(
    source_landmarks: Landmarks,
    target_box: FaceBox,
) -> AlignmentResult:
    """Align by scaling only, ignoring rotation. Used as a last resort."""
    source_points = _eye_and_nose_points(source_landmarks)
    target_points = _eye_and_nose_points(target_landmarks_from_box(target_box))
    matrix, error = fit_similarity(source_points, target_points)
    return AlignmentResult(
        matrix,
        (int(target_box.width), int(target_box.height)),
        error,
        "similarity",
    )


def target_landmarks_from_box(box: FaceBox) -> Landmarks:
    """Build a landmark set from a face box using the geometric prior only."""
    from .landmarks import reference_layout

    return Landmarks(reference_layout(box), box, prior_only=True)
