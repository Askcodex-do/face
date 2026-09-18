"""Landmark estimation with classical computer vision only.

There is no neural network here on purpose: the rules forbid ONNX, TensorFlow,
PyTorch and large models, and a 2 GB CPU only machine could not run them anyway.

What this module does instead is estimate a 68 point layout from geometry and
image evidence. The detection box gives a statistical prior for where the parts
of a face sit, then local search refines the points that can be measured:
pupils via a bright spot search, nostrils via a dark region search, mouth via a
dark region search. Points that cannot be measured stay on the prior.

The result is a stable, plausible layout that is good enough to drive an affine
alignment. It is explicitly not an anatomically accurate 3D face model, and the
docstrings say so, so nobody mistakes it for one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..utils.config import LandmarkConfig
from ..utils.logger import get_logger
from .face_detector import FaceBox

log = get_logger("core.landmarks")

# ---------------------------------------------------------------------------
# The canonical 68 point layout
# ---------------------------------------------------------------------------

#: Region name to index range, following the common 68 point convention.
REGIONS: dict[str, Tuple[int, int]] = {
    "jaw": (0, 17),
    "right_eyebrow": (17, 22),
    "left_eyebrow": (22, 27),
    "nose": (27, 36),
    "right_eye": (36, 42),
    "left_eye": (42, 48),
    "mouth": (48, 68),
}

#: Landmark indices used to fit the alignment transform. The eye centres and the
#: nose tip give a stable similarity transform without needing the jaw contour,
#: which is the least reliable part of the estimate.
ALIGNMENT_INDICES: Tuple[int, ...] = (36, 39, 42, 45, 30)

#: Reference layout in a normalised face space, origin at the top left of the
#: face box with both coordinates in ``[0, 1]``. Values come from averaged
#: measurements of frontal faces and act as the prior for every point.
_BASE_LAYOUT: Tuple[Tuple[float, float], ...] = (
    # jaw contour, 17 points, image order from the left side round to the right
    (0.00, 0.18), (0.01, 0.30), (0.04, 0.42), (0.08, 0.54), (0.14, 0.65),
    (0.22, 0.75), (0.31, 0.83), (0.41, 0.88), (0.50, 0.90), (0.59, 0.88),
    (0.69, 0.83), (0.78, 0.75), (0.86, 0.65), (0.92, 0.54), (0.96, 0.42),
    (0.99, 0.30), (1.00, 0.18),
    # right eyebrow, 5 points
    (0.16, 0.22), (0.24, 0.17), (0.33, 0.15), (0.41, 0.17), (0.47, 0.22),
    # left eyebrow, 5 points
    (0.53, 0.22), (0.59, 0.17), (0.67, 0.15), (0.76, 0.17), (0.84, 0.22),
    # nose bridge and tip, 9 points; index 30 is the tip
    (0.50, 0.26), (0.50, 0.34), (0.50, 0.42), (0.50, 0.50), (0.41, 0.55),
    (0.46, 0.58), (0.54, 0.58), (0.59, 0.55), (0.50, 0.56),
    # right eye, 6 points; indices 36 and 39 are the corners
    (0.30, 0.35), (0.34, 0.31), (0.40, 0.31), (0.44, 0.35), (0.40, 0.38),
    (0.34, 0.38),
    # left eye, 6 points; indices 42 and 45 are the corners
    (0.56, 0.35), (0.60, 0.31), (0.66, 0.31), (0.70, 0.35), (0.66, 0.38),
    (0.60, 0.38),
    # outer mouth, 12 points
    (0.33, 0.73), (0.38, 0.70), (0.44, 0.68), (0.50, 0.69), (0.56, 0.68),
    (0.62, 0.70), (0.67, 0.73), (0.62, 0.77), (0.56, 0.80), (0.50, 0.81),
    (0.44, 0.80), (0.38, 0.77),
    # inner mouth, 8 points
    (0.39, 0.73), (0.44, 0.71), (0.50, 0.72), (0.56, 0.71), (0.61, 0.73),
    (0.56, 0.76), (0.50, 0.77), (0.44, 0.76),
)

NUM_POINTS = len(_BASE_LAYOUT)
assert NUM_POINTS == 68, "the reference layout must contain 68 points"


@dataclass
class Landmarks:
    """A set of landmark points in pixel coordinates of a specific frame."""

    points: np.ndarray  # shape (68, 2), float32
    box: Optional[FaceBox] = None
    #: True when every point is on the geometric prior rather than measured.
    prior_only: bool = False

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float32)
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")

    # -- accessors ----------------------------------------------------------

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def __getitem__(self, index) -> np.ndarray:
        return self.points[index]

    def region(self, name: str) -> np.ndarray:
        """Return the points of a named region, e.g. ``"left_eye"``."""
        if name not in REGIONS:
            raise KeyError(f"unknown region {name!r}, expected one of {list(REGIONS)}")
        start, stop = REGIONS[name]
        return self.points[start:stop]

    @property
    def right_eye_center(self) -> np.ndarray:
        """Centre of the eye that appears on the left of the image."""
        return self.region("right_eye").mean(axis=0)

    @property
    def left_eye_center(self) -> np.ndarray:
        """Centre of the eye that appears on the right of the image."""
        return self.region("left_eye").mean(axis=0)

    @property
    def eye_distance(self) -> float:
        """Interocular distance in pixels, the natural scale of the face."""
        return float(np.linalg.norm(self.left_eye_center - self.right_eye_center))

    @property
    def mouth_center(self) -> np.ndarray:
        return self.region("mouth").mean(axis=0)

    @property
    def nose_tip(self) -> np.ndarray:
        return self.points[30]

    @property
    def jaw(self) -> np.ndarray:
        return self.region("jaw")

    @property
    def face_center(self) -> np.ndarray:
        return self.points.mean(axis=0)

    @property
    def bounding_rect(self) -> Tuple[int, int, int, int]:
        """Axis aligned rect covering every point, as ``(x, y, w, h)``."""
        minimum = self.points.min(axis=0)
        maximum = self.points.max(axis=0)
        x, y = int(np.floor(minimum[0])), int(np.floor(minimum[1]))
        w = int(np.ceil(maximum[0])) - x
        h = int(np.ceil(maximum[1])) - y
        return x, y, max(w, 1), max(h, 1)

    # -- transformations ----------------------------------------------------

    def copy(self) -> "Landmarks":
        return Landmarks(self.points.copy(), self.box, self.prior_only)

    def translate(self, dx: float, dy: float) -> "Landmarks":
        moved = self.points + np.array([dx, dy], dtype=np.float32)
        return Landmarks(moved, self.box, self.prior_only)

    def scale(self, factor: float, origin: Optional[np.ndarray] = None) -> "Landmarks":
        pivot = np.asarray(
            origin if origin is not None else self.face_center, dtype=np.float32
        )
        scaled = (self.points - pivot) * factor + pivot
        return Landmarks(scaled, self.box, self.prior_only)

    def to_normalized(self) -> np.ndarray:
        """Map the points into the ``[0, 1]`` box space."""
        if self.box is None:
            raise ValueError("cannot normalise without a face box")
        scale = np.array(
            [max(self.box.width, 1), max(self.box.height, 1)], dtype=np.float32
        )
        offset = np.array([self.box.x, self.box.y], dtype=np.float32)
        return (self.points - offset) / scale

    def as_list(self) -> List[List[float]]:
        """Return the points as plain Python floats, e.g. for JSON output."""
        return self.points.astype(float).tolist()


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def reference_layout(box: FaceBox) -> np.ndarray:
    """Map the normalised reference layout onto ``box`` in pixel coordinates."""
    base = np.asarray(_BASE_LAYOUT, dtype=np.float32)
    scale = np.array([box.width, box.height], dtype=np.float32)
    offset = np.array([box.x, box.y], dtype=np.float32)
    return base * scale + offset


def _search_darkest_in_rect(
    gray: np.ndarray, rect: Tuple[int, int, int, int], window: int
) -> Optional[np.ndarray]:
    """Find the darkest ``window`` sized patch inside ``rect``.

    ``rect`` is ``(x, y, w, h)``. Returns the centre of the darkest patch, or
    ``None`` when the rectangle is too small or lies outside the frame.
    """
    height, width = gray.shape[:2]
    x, y, w, h = rect
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 - x0 < window or y1 - y0 < window:
        return None

    patch = gray[y0:y1, x0:x1]
    means = cv2.boxFilter(
        patch.astype(np.float32),
        ddepth=cv2.CV_32F,
        ksize=(window, window),
        normalize=True,
        borderType=cv2.BORDER_REPLICATE,
    )
    flat = int(np.argmin(means))
    iy, ix = divmod(flat, means.shape[1])
    return np.array([x0 + ix, y0 + iy], dtype=np.float32)


def _clamp_offset(offset: np.ndarray, limit: float) -> np.ndarray:
    """Limit the length of an offset vector to ``limit`` pixels."""
    norm = float(np.linalg.norm(offset))
    if norm <= limit or norm == 0.0:
        return offset
    return offset / norm * limit


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class LandmarkEstimator:
    """Estimate the 68 point layout for a detected face box.

    The estimator is stateful only for temporal smoothing: when enabled, each
    frame's estimate is blended with the previous one so the alignment does not
    jitter frame to frame.
    """

    def __init__(self, config: Optional[LandmarkConfig] = None) -> None:
        self.config = config or LandmarkConfig()
        self._previous: Optional[np.ndarray] = None

    # -- public API ---------------------------------------------------------

    def estimate(
        self, frame: np.ndarray, box: FaceBox, *, refine: bool = True
    ) -> Landmarks:
        """Return the landmark layout for ``box`` inside ``frame``."""
        points = reference_layout(box)
        measured = False

        if refine and self.config.num_points == 68:
            gray = self._to_gray(frame)
            points, measured = self._refine(gray, points, box)

        points = self._apply_smoothing(points, box)
        self._previous = points.copy()

        return Landmarks(points, box, prior_only=not measured)

    def reset(self) -> None:
        """Drop the smoothing history, e.g. at the start of a new video."""
        self._previous = None

    # -- internals ----------------------------------------------------------

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)
        if frame.ndim == 2:
            gray = frame
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.config.blur_radius > 0:
            size = self.config.blur_radius * 2 + 1
            gray = cv2.GaussianBlur(gray, (size, size), 0)
        return gray

    def _refine(
        self, gray: np.ndarray, points: np.ndarray, box: FaceBox
    ) -> Tuple[np.ndarray, bool]:
        """Nudge the measurable points towards image evidence.

        Two rules keep this honest and stable:

        * Every refinement is clamped to a small fraction of the face box, so a
          false measurement can never drag the layout far off the prior.
        * The whole layout is never re-scaled from a measurement. Alignment only
          ever uses the *relative* geometry between the source and target
          landmarks, and both are measured with this same estimator, so any
          systematic bias cancels out. A locally wrong measurement does not.

        Eyes are searched inside a rectangle derived from the eye points
        themselves, so an eyebrow or a nostril cannot win the search. The nose
        and mouth are searched in a small window around their prior.
        """
        refined = points.copy()
        measured = False
        window = max(2, int(round(min(box.width, box.height) * 0.045)))

        for region in ("right_eye", "left_eye"):
            start, stop = REGIONS[region]
            eye_points = refined[start:stop]
            centre = eye_points.mean(axis=0)
            minimum = eye_points.min(axis=0)
            maximum = eye_points.max(axis=0)
            rect = (
                int(round(minimum[0])),
                int(round(minimum[1])),
                int(round(maximum[0] - minimum[0])),
                int(round(maximum[1] - minimum[1])),
            )
            found = _search_darkest_in_rect(gray, rect, window)
            if found is None:
                continue
            offset = _clamp_offset(found - centre, max(box.width * 0.04, 2.0))
            refined[start:stop] = eye_points + offset
            measured = True

        # Nose: the nostrils sit just below the tip prior.
        nose_start, nose_stop = REGIONS["nose"]
        nose_points = refined[nose_start:nose_stop]
        nose_center = nose_points.mean(axis=0)
        found = _search_darkest_in_rect(
            gray,
            (
                int(round(nose_center[0] - box.width * 0.10)),
                int(round(nose_center[1] - box.height * 0.03)),
                int(round(box.width * 0.20)),
                int(round(box.height * 0.09)),
            ),
            window,
        )
        if found is not None:
            offset = _clamp_offset(found - nose_center, box.height * 0.05)
            refined[nose_start:nose_stop] = nose_points + offset
            measured = True

        # Mouth: the line between the lips is darker than the surrounding skin.
        mouth_start, mouth_stop = REGIONS["mouth"]
        mouth_points = refined[mouth_start:mouth_stop]
        mouth_center = mouth_points.mean(axis=0)
        found = _search_darkest_in_rect(
            gray,
            (
                int(round(mouth_center[0] - box.width * 0.18)),
                int(round(mouth_center[1] - box.height * 0.05)),
                int(round(box.width * 0.36)),
                int(round(box.height * 0.10)),
            ),
            window,
        )
        if found is not None:
            offset = _clamp_offset(found - mouth_center, box.height * 0.05)
            refined[mouth_start:mouth_stop] = mouth_points + offset
            measured = True

        return refined, measured

    def _apply_smoothing(
        self, points: np.ndarray, box: FaceBox
    ) -> np.ndarray:
        """Blend with the previous estimate when the box has not jumped."""
        previous = self._previous
        weight = float(self.config.smoothing)
        if previous is None or weight <= 0.0 or previous.shape != points.shape:
            return points

        # A large jump means the tracked face changed or the box is wrong, so
        # the history is stale and smoothing would drag the layout off the face.
        previous_center = previous.mean(axis=0)
        current_center = points.mean(axis=0)
        jump = float(np.linalg.norm(current_center - previous_center))
        if jump > max(box.width, box.height):
            log.debug("landmark history reset, centre moved %.1f px", jump)
            return points

        return (previous * weight + points * (1.0 - weight)).astype(np.float32)


def estimate_landmarks(
    frame: np.ndarray,
    box: FaceBox,
    config: Optional[LandmarkConfig] = None,
) -> Landmarks:
    """One shot helper that estimates landmarks without keeping state."""
    return LandmarkEstimator(config).estimate(frame, box)
