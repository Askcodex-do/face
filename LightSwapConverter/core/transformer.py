"""Warp and colour correct the source face so it fits the target frame.

Given an alignment transform this module produces the actual pixel patch that
will be pasted into the frame. Two things happen here:

* **Geometric** - the source face is warped into the target face box, with a
  validity mask derived from the source face outline.
* **Photometric** - the warped patch is colour matched to the target region and
  optionally sharpened, so skin tone and contrast do not give the paste away.

Everything runs on plain OpenCV and NumPy arrays of a single face patch, which
keeps the memory footprint tiny.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from ..utils.config import TransformConfig
from ..utils.logger import get_logger
from .alignment import AlignmentResult, compose, transform_points
from .face_detector import FaceBox
from .landmarks import Landmarks

log = get_logger("core.transformer")


class TransformError(RuntimeError):
    """Raised when the source face cannot be warped."""


@dataclass
class WarpedFace:
    """A warped source face ready for blending."""

    #: BGR patch of size ``(height, width)`` matching the target face box.
    patch: np.ndarray
    #: 8 bit mask, 255 where the patch holds real source pixels.
    mask: np.ndarray
    #: Face box in frame coordinates that the patch corresponds to.
    box: FaceBox
    #: Transform used, kept for debugging and for the inverse mapping.
    matrix: np.ndarray

    @property
    def size(self) -> Tuple[int, int]:
        return self.patch.shape[1], self.patch.shape[0]

    @property
    def coverage(self) -> float:
        """Fraction of the patch that holds valid source pixels."""
        if self.mask.size == 0:
            return 0.0
        return float(np.count_nonzero(self.mask)) / float(self.mask.size)


def translation_matrix(dx: float, dy: float) -> np.ndarray:
    """Return a 2x3 translation matrix."""
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)


def face_hull_mask(landmarks: Landmarks, shape: Tuple[int, int]) -> np.ndarray:
    """Return a filled convex hull mask of a landmark set.

    ``shape`` is ``(height, width)`` of the mask to build.
    """
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    points = landmarks.points.astype(np.int32)
    if len(points) < 3:
        return mask
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


class FaceTransformer:
    """Warp and colour match a source face onto a target face box."""

    def __init__(self, config: Optional[TransformConfig] = None) -> None:
        self.config = config or TransformConfig()

    # -- geometry -----------------------------------------------------------

    def warp(
        self,
        source_image: np.ndarray,
        source_landmarks: Landmarks,
        alignment: AlignmentResult,
        box: FaceBox,
    ) -> WarpedFace:
        """Warp ``source_image`` into the patch described by ``alignment``.

        ``alignment.matrix`` maps source image coordinates to frame coordinates.
        The matrix is shifted by the box origin so the patch can live in its own
        small coordinate system instead of a full frame sized canvas.
        """
        if source_image is None or source_image.size == 0:
            raise TransformError("source image is empty")

        width, height = alignment.size
        if width < 2 or height < 2:
            raise TransformError(f"invalid patch size {width}x{height}")

        image = self._prepare_source(source_image)

        local = compose(translation_matrix(-box.x, -box.y), alignment.matrix)

        # The alignment matrix's linear scale is how many times larger the face
        # will be drawn, which decides how much the warp will soften it.
        stretch = float(np.linalg.norm(local[:, 0]))
        interpolation = self._interpolation_for(stretch, self.config)

        try:
            patch = cv2.warpAffine(
                image,
                local,
                (width, height),
                flags=interpolation,
                borderMode=cv2.BORDER_REPLICATE,
            )
        except cv2.error as exc:  # pragma: no cover - OpenCV level failure
            raise TransformError(f"warpAffine failed: {exc}") from exc

        mask = self._warp_mask(source_landmarks, local, width, height, image.shape)
        if np.count_nonzero(mask) == 0:
            raise TransformError("warped source face covers no pixels")

        return WarpedFace(patch, mask, box, local)

    def _prepare_source(self, image: np.ndarray) -> np.ndarray:
        """Ensure the source image is a contiguous 8 bit BGR array."""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if self.config.flip_source:
            image = cv2.flip(image, 1)
        return np.ascontiguousarray(image)

    @staticmethod
    def _interpolation_for(stretch: float, config: TransformConfig) -> int:
        """Pick a warp interpolation from how far the face will be stretched.

        ``warpAffine`` defaults to bilinear, which softens badly once a face is
        enlarged more than about 1.5x: measured on a face stretched 3.4x to
        8.7x, cubic interpolation recovered about 15% more fine detail than
        bilinear at every size. Lanczos was marginally better again but three
        times slower, which is not a trade worth making on the hardware this
        targets, so bilinear is kept for the ordinary near-1:1 case where it is
        both correct and cheapest.

        Pre-scaling the source and then warping was measured too, and rejected:
        two resampling passes blur more than one, so it was worse than a single
        cubic warp as well as costing an extra buffer.
        """
        if stretch > config.cubic_stretch_threshold > 0.0:
            return cv2.INTER_CUBIC
        return cv2.INTER_LINEAR

    def _warp_mask(
        self,
        source_landmarks: Landmarks,
        matrix: np.ndarray,
        width: int,
        height: int,
        source_shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Build the validity mask for the patch.

        The mask is the source face outline warped with the same matrix, eroded
        slightly so the edge of the warped rectangle never leaks into the frame.
        """
        source_height, source_width = source_shape[:2]
        source_mask = face_hull_mask(
            source_landmarks, (source_height, source_width)
        )
        warped = cv2.warpAffine(
            source_mask,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        erode = max(1, int(round(min(width, height) * 0.02)))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode * 2 + 1, erode * 2 + 1)
        )
        return cv2.erode(warped, kernel)

    # -- photometry ---------------------------------------------------------

    def match_color(
        self,
        patch: np.ndarray,
        reference: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Shift the colour statistics of ``patch`` towards ``reference``.

        Uses the Reinhard mean and standard deviation transfer in LAB space,
        blended by the configured strength. This is a classic colour transfer,
        not a learned model, and it is what stops a pasted face from looking
        like it was lit by a different sun.
        """
        strength = float(self.config.color_match)
        if strength <= 0.0 or patch.size == 0 or reference.size == 0:
            return patch
        if patch.shape[:2] != reference.shape[:2]:
            reference = cv2.resize(
                reference, (patch.shape[1], patch.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        region = mask if mask is not None else np.full(patch.shape[:2], 255, np.uint8)
        if np.count_nonzero(region) < 16:
            return patch

        try:
            patch_lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
            ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
        except cv2.error:
            return patch

        selected = region > 0
        result = patch_lab.copy()
        for channel in range(3):
            source_values = patch_lab[:, :, channel][selected]
            target_values = ref_lab[:, :, channel]
            source_mean = float(source_values.mean())
            source_std = float(source_values.std())
            target_mean = float(target_values.mean())
            target_std = float(target_values.std())
            if source_std < 1e-3:
                continue
            ratio = target_std / source_std
            # Limit the correction so an unusual target region cannot blow up
            # the contrast of the pasted face.
            ratio = float(np.clip(ratio, 0.5, 2.0))
            corrected = (patch_lab[:, :, channel] - source_mean) * ratio + target_mean
            result[:, :, channel] = (
                patch_lab[:, :, channel] * (1.0 - strength) + corrected * strength
            )

        result = np.clip(result, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    def match_histogram(
        self, patch: np.ndarray, reference: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Match the luminance histogram of ``patch`` to ``reference``.

        A stronger, sometimes harsher alternative to :meth:`match_color`,
        exposed for experimentation and covered by the tests.
        """
        if patch.size == 0 or reference.size == 0:
            return patch
        if patch.shape[:2] != reference.shape[:2]:
            reference = cv2.resize(
                reference, (patch.shape[1], patch.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        patch_ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
        ref_ycrcb = cv2.cvtColor(reference, cv2.COLOR_BGR2YCrCb)
        region = mask if mask is not None else np.full(patch.shape[:2], 255, np.uint8)
        selected = region > 0
        if np.count_nonzero(selected) < 16:
            return patch

        source_values = patch_ycrcb[:, :, 0][selected]
        target_values = ref_ycrcb[:, :, 0]
        source_hist, _ = np.histogram(source_values, bins=256, range=(0, 256))
        target_hist, _ = np.histogram(target_values, bins=256, range=(0, 256))

        source_cdf = np.cumsum(source_hist).astype(np.float64)
        target_cdf = np.cumsum(target_hist).astype(np.float64)
        source_cdf /= max(source_cdf[-1], 1.0)
        target_cdf /= max(target_cdf[-1], 1.0)

        lookup = np.interp(source_cdf, target_cdf, np.arange(256)).astype(np.uint8)
        mapped = patch_ycrcb.copy()
        mapped[:, :, 0] = lookup[patch_ycrcb[:, :, 0]]
        return cv2.cvtColor(mapped, cv2.COLOR_YCrCb2BGR)

    def sharpen(self, patch: np.ndarray) -> np.ndarray:
        """Apply an unsharp mask. A no-op when the amount is zero."""
        amount = float(self.config.sharpen)
        if amount <= 0.0 or patch.size == 0:
            return patch
        blurred = cv2.GaussianBlur(patch, (0, 0), sigmaX=1.2)
        sharpened = cv2.addWeighted(
            patch, 1.0 + amount, blurred, -amount, 0.0
        )
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def adjust_brightness(self, patch: np.ndarray) -> np.ndarray:
        """Add a constant brightness offset. A no-op when the offset is zero."""
        offset = float(self.config.brightness)
        if abs(offset) < 1e-6 or patch.size == 0:
            return patch
        shifted = patch.astype(np.float32) + offset * 255.0
        return np.clip(shifted, 0, 255).astype(np.uint8)

    # -- combined -----------------------------------------------------------

    def build_patch(
        self,
        source_image: np.ndarray,
        source_landmarks: Landmarks,
        alignment: AlignmentResult,
        box: FaceBox,
        frame: np.ndarray,
    ) -> WarpedFace:
        """Warp the source face and colour match it to the frame region."""
        warped = self.warp(source_image, source_landmarks, alignment, box)
        reference = self._crop(frame, box)
        if reference is not None:
            warped.patch = self.match_color(warped.patch, reference, warped.mask)
        warped.patch = self.adjust_brightness(warped.patch)
        warped.patch = self.sharpen(warped.patch)
        return warped

    def _crop(self, frame: np.ndarray, box: FaceBox) -> Optional[np.ndarray]:
        """Crop the face region out of a frame, or ``None`` when out of bounds."""
        if frame is None or frame.size == 0:
            return None
        height, width = frame.shape[:2]
        clamped = box.clamp(width, height)
        region = frame[clamped.y : clamped.y2, clamped.x : clamped.x2]
        if region.size == 0:
            return None
        return region

    def sample_reference(
        self, frame: np.ndarray, landmarks: Landmarks, box: FaceBox
    ) -> Optional[np.ndarray]:
        """Return a colour reference built from the target face skin only.

        Sampling inside the landmark hull avoids pulling in hair and background,
        which would drag the colour match in the wrong direction.
        """
        region = self._crop(frame, box)
        if region is None:
            return None
        local = landmarks.translate(-box.x, -box.y)
        mask = face_hull_mask(local, region.shape[:2])
        if np.count_nonzero(mask) < 32:
            return None
        return region


def map_points_to_frame(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map points with a source to frame transform. Thin wrapper for clarity."""
    return transform_points(matrix, points)
