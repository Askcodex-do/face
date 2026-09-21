"""Landmark estimation with classical computer vision only.

There is no neural network here on purpose: the rules forbid ONNX, TensorFlow,
PyTorch and large models, and a 2 GB CPU only machine could not run them anyway.

What this module does instead is estimate a 68 point layout from geometry and
image evidence. The detection box gives a statistical prior for where the parts
of a face sit, then a tightly bounded local search nudges the eye centres and
the mouth towards image evidence. Points that cannot be measured stay on the
prior.

The prior is derived from real ground truth rather than invented: it is the
bilateral average of the two 68 point annotations published with OpenCV's test
data, mapped into the detector's own box frame. See ``_BASE_LAYOUT`` for the
provenance and ``tests/test_landmark_quality.py`` for the verification.

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

#: Landmark indices used to fit the alignment transform. The four eye corners and
#: the nose tip give a stable similarity transform without needing the jaw contour,
#: which is the least reliable part of the estimate.
ALIGNMENT_INDICES: Tuple[int, ...] = (36, 39, 42, 45, 30)

#: Interocular distance of the prior layout, expressed as a fraction of the face
#: box width. Measured from ground truth against the detector's own output: the
#: padded box from :meth:`FaceDetector.detect` is 0.2557 wide per interocular
#: distance on an average frontal face.
_PRIOR_INTEROCULAR_FRACTION = 0.2557

#: Centre of the prior eye pair inside the face box, as a fraction of box width
#: and height. Measured the same way; the vertical value is the more important
#: of the two because it sets where the alignment puts the eye line.
_PRIOR_EYE_CENTRE = (0.5064, 0.3981)

#: Reference layout in a normalised face space, origin at the top left of the
#: face box with both coordinates in ``[0, 1]``. Values come from averaged
#: measurements of frontal faces and act as the prior for every point.
#:
#: Provenance: the bilateral average of ``david1.pts`` and ``david2.pts``, the two
#: 68 point ground truth annotations that ship with OpenCV's test data
#: (``opencv_extra``, Apache-2.0). The shapes were normalised by interocular
#: distance and eye-line rotation, averaged, mirrored onto each other so the
#: layout is exactly symmetric, then placed into the detector's box frame using
#: ``_PRIOR_INTEROCULAR_FRACTION`` and ``_PRIOR_EYE_CENTRE``. Earlier revisions
#: used a hand-drawn layout which placed the jaw 70% too wide and the mouth 46%
#: too wide relative to the same ground truth.
_BASE_LAYOUT: Tuple[Tuple[float, float], ...] = (
    # jaw contour, 17 points, image order from the left side round to the right
    (0.2251, 0.4118), (0.2320, 0.4955), (0.2451, 0.5737), (0.2604, 0.6507),
    (0.2886, 0.7196),
    (0.3311, 0.7811), (0.3822, 0.8321), (0.4377, 0.8728), (0.5064, 0.8847),
    (0.5751, 0.8728),
    (0.6306, 0.8321), (0.6817, 0.7811), (0.7242, 0.7196), (0.7524, 0.6507),
    (0.7677, 0.5737),
    (0.7808, 0.4955), (0.7877, 0.4118),
    # right eyebrow, 5 points
    (0.2666, 0.3593), (0.3027, 0.3233), (0.3568, 0.3106), (0.4125, 0.3142),
    (0.4656, 0.3324),
    # left eyebrow, 5 points
    (0.5472, 0.3324), (0.6003, 0.3142), (0.6560, 0.3106), (0.7101, 0.3233),
    (0.7462, 0.3593),
    # nose bridge and tip, 9 points; index 30 is the tip
    (0.5064, 0.3956), (0.5064, 0.4542), (0.5064, 0.5121), (0.5064, 0.5716),
    (0.4413, 0.6032),
    (0.4727, 0.6158), (0.5064, 0.6271), (0.5401, 0.6158), (0.5715, 0.6032),
    # right eye, 6 points; indices 36 and 39 are the corners
    (0.3289, 0.3979), (0.3597, 0.3801), (0.3971, 0.3805), (0.4313, 0.4075),
    (0.3958, 0.4115), (0.3585, 0.4112),
    # left eye, 6 points; indices 42 and 45 are the corners
    (0.5815, 0.4075), (0.6157, 0.3805), (0.6531, 0.3801), (0.6839, 0.3979),
    (0.6543, 0.4112), (0.6170, 0.4115),
    # outer mouth, 12 points
    (0.3919, 0.6970), (0.4336, 0.6890), (0.4735, 0.6831), (0.5064, 0.6921),
    (0.5393, 0.6831), (0.5792, 0.6890), (0.6209, 0.6970), (0.5801, 0.7290),
    (0.5408, 0.7440), (0.5064, 0.7477), (0.4720, 0.7440), (0.4327, 0.7290),
    # inner mouth, 8 points
    (0.4087, 0.6994), (0.4731, 0.7057), (0.5064, 0.7101), (0.5397, 0.7057),
    (0.6041, 0.6994),
    (0.5397, 0.7087), (0.5064, 0.7124), (0.4731, 0.7087),
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


def _refine_dark_centre(
    gray: np.ndarray,
    predicted: np.ndarray,
    search_rect: Tuple[int, int, int, int],
    *,
    window: int,
    max_shift: float,
    min_darkness: float,
    confidence_soft: float,
) -> Optional[Tuple[np.ndarray, float]]:
    """Find a genuinely dark feature near ``predicted`` and score the evidence.

    Returns ``(point, confidence)`` or ``None`` when there is nothing to
    measure. Confidence is ``0`` when the darkest patch is no darker than the
    surrounding skin and ramps to ``1`` once it is ``confidence_soft`` times
    further below the local mean, which is the difference between a shadowed
    answer and a coincidence.

    Three guard rails keep the caller honest:

    * the candidate must be at least ``min_darkness`` below the local mean, so a
      flat cheek cannot win the search;
    * the shift is clamped to ``max_shift`` pixels, so one bad search cannot
      drag a point across the face;
    * the confidence is returned rather than assumed, so the caller can refuse a
      weak measurement instead of applying it.
    """
    found = _search_darkest_in_rect(gray, search_rect, window)
    if found is None:
        return None

    height, width = gray.shape[:2]
    skin_rect = _scale_rect(search_rect, 1.5, width, height)
    local = _mean_in_rect(gray, skin_rect)
    feature = _mean_in_rect(
        gray,
        (int(round(found[0] - window / 2)), int(round(found[1] - window / 2)),
         window, window),
    )
    if local is None or feature is None:
        return None

    depth = float(local - feature)
    if depth < min_darkness:
        return None

    confidence = min(1.0, depth / max(confidence_soft, 1e-6))
    shift = _clamp_offset(found - predicted, max_shift)
    return (predicted + shift).astype(np.float32), float(confidence)


def _scale_rect(
    rect: Tuple[int, int, int, int], factor: float, width: int, height: int
) -> Tuple[int, int, int, int]:
    """Grow ``rect`` about its centre by ``factor``, then clip to the frame."""
    x, y, w, h = rect
    cx, cy = x + w / 2.0, y + h / 2.0
    nw, nh = max(1, int(round(w * factor))), max(1, int(round(h * factor)))
    nx, ny = int(round(cx - nw / 2.0)), int(round(cy - nh / 2.0))
    nx, ny = max(0, nx), max(0, ny)
    nw, nh = min(nw, max(1, width - nx)), min(nh, max(1, height - ny))
    return nx, ny, nw, nh


def _grow_rect_to_at_least(
    rect: Tuple[int, int, int, int], minimum: int, width: int, height: int
) -> Tuple[int, int, int, int]:
    """Expand ``rect`` about its centre until both sides reach ``minimum`` px.

    The landmark prior puts the eye region about 3% of the face box high, which
    can be smaller than the search window used to look for a pupil. Without this
    the pupil search would be handed a rectangle too small to hold its window,
    quietly return ``None``, and leave the eyes unrefined on exactly the images
    where the prior is tightest.
    """
    x, y, w, h = rect
    cx, cy = x + w / 2.0, y + h / 2.0
    nw, nh = max(w, minimum), max(h, minimum)
    nx, ny = int(round(cx - nw / 2.0)), int(round(cy - nh / 2.0))
    nx, ny = max(0, nx), max(0, ny)
    nw, nh = min(nw, max(1, width - nx)), min(nh, max(1, height - ny))
    return nx, ny, nw, nh


def _mean_in_rect(
    gray: np.ndarray, rect: Tuple[int, int, int, int]
) -> Optional[float]:
    """Mean intensity inside ``rect``, or ``None`` when it is degenerate."""
    height, width = gray.shape[:2]
    x, y, w, h = rect
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return float(gray[y0:y1, x0:x1].mean())


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
        """Nudge a few well-defined points towards image evidence.

        Only the things that can actually be measured from intensity are touched:

        * the two **eye centres**, found as the dark pupil/iris patch inside the
          eye region, and
        * the **mouth centre**, found as the dark line between the lips.

        Everything else, and in particular every point the alignment transform is
        fitted from, stays exactly on the prior. That is deliberate. A previous
        revision also moved the eye *corners* and the whole *nose* region, and
        measurement against ground truth showed it made the alignment worse: the
        five point fit residual rose from 2.15 px to 3.55 px. Moving a point that
        the fit depends on, on evidence this weak, is simply a regression.

        The whole layout is never re-scaled from a measurement. Alignment only
        ever uses the *relative* geometry between the source and target
        landmarks, and both are measured with this same estimator, so any
        systematic bias cancels out. A locally wrong measurement does not, which
        is why each one has to clear the evidence thresholds in
        :func:`_refine_dark_centre` before it is applied at all, and why a weak
        measurement is blended in proportion to its confidence instead of being
        taken at face value.
        """
        refined = points.copy()
        measured = False

        scale = min(box.width, box.height)
        window = max(2, int(round(scale * 0.045)))
        # The pupil is much darker than the eye white around it, so the contrast
        # thresholds can be fairly strict without losing real eyes.
        min_depth = 18.0
        soft_depth = 45.0

        for region in ("right_eye", "left_eye"):
            start, stop = REGIONS[region]
            eye_points = refined[start:stop]
            centre = eye_points.mean(axis=0)
            minimum = eye_points.min(axis=0)
            maximum = eye_points.max(axis=0)
            # Search inside the eye itself, so a brow or a nostril cannot win.
            rect = (
                int(round(minimum[0])),
                int(round(minimum[1])),
                int(round(maximum[0] - minimum[0])),
                int(round(maximum[1] - minimum[1])),
            )
            # The eye region of the prior is only about 3% of the box high,
            # which can be smaller than the pupil search window. Widen it to
            # fit before searching, or the pupil would never be found on a
            # small face.
            rect = _grow_rect_to_at_least(
                rect, window + 1, gray.shape[1], gray.shape[0]
            )
            result = _refine_dark_centre(
                gray,
                centre,
                rect,
                window=window,
                max_shift=max(scale * 0.04, 2.0),
                min_darkness=min_depth,
                confidence_soft=soft_depth,
            )
            if result is None:
                continue
            found, confidence = result
            # Move the whole eye as one rigid unit and scale the move by how
            # strong the evidence was.
            offset = _clamp_offset(found - centre, max(scale * 0.04, 2.0))
            refined[start:stop] = eye_points + offset * confidence
            measured = True

        # The nose is deliberately left on the prior: the nose tip cannot be
        # found reliably by darkness alone, and it is one of the five points the
        # alignment is fitted from.

        # Mouth: the line between the lips is darker than the surrounding skin.
        mouth_start, mouth_stop = REGIONS["mouth"]
        mouth_points = refined[mouth_start:mouth_stop]
        mouth_centre = mouth_points.mean(axis=0)
        result = _refine_dark_centre(
            gray,
            mouth_centre,
            (
                int(round(mouth_centre[0] - box.width * 0.18)),
                int(round(mouth_centre[1] - box.height * 0.05)),
                int(round(box.width * 0.36)),
                int(round(box.height * 0.10)),
            ),
            window=window,
            max_shift=max(box.height * 0.05, 2.0),
            min_darkness=12.0,
            confidence_soft=35.0,
        )
        if result is not None:
            found, confidence = result
            offset = _clamp_offset(found - mouth_centre, max(box.height * 0.05, 2.0))
            refined[mouth_start:mouth_stop] = mouth_points + offset * confidence
            measured = True

        return self._clamp_to_prior(refined, box), measured

    def _clamp_to_prior(self, points: np.ndarray, box: FaceBox) -> np.ndarray:
        """Cap how far any single landmark may stray from its prior position.

        Refinement is a small correction to a good prior, so no point should ever
        travel far. Capping the per-point displacement does two useful things:

        * a confidently wrong measurement can only do bounded damage, and
        * because every point is capped independently, the *shape* of the layout
          is preserved. An earlier revision instead rescaled the head of the
          layout towards its own centroid, which quietly shrank the face and
          cost a quarter of the paste area before it was caught.

        The budget is a fraction of the face box, so it scales with the face.
        """
        budget = float(self.config.max_refine_shift) * min(box.width, box.height)
        prior = reference_layout(box)
        if budget <= 0.0:
            # A zero budget means every landmark stays exactly on the prior.
            return prior
        delta = points - prior
        norms = np.linalg.norm(delta, axis=1, keepdims=True)
        over = (norms > budget).ravel()
        if over.any():
            delta[over] = delta[over] / norms[over] * budget
            log.debug("%d landmark(s) clamped to the %.1f px prior budget",
                      int(over.sum()), budget)
        return (prior + delta).astype(np.float32)

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
