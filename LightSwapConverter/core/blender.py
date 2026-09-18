"""Composite the warped face back into the frame.

A hard paste leaves a visible rectangle and a seam along the jaw. This module
builds a soft, feathered mask around the warped face and alpha blends it into a
copy of the frame, optionally using OpenCV's seamless cloning to match gradients
across the seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from ..utils.config import BlendConfig
from ..utils.logger import get_logger
from .face_detector import FaceBox
from .landmarks import Landmarks
from .transformer import WarpedFace, face_hull_mask

log = get_logger("core.blender")


class BlendError(RuntimeError):
    """Raised when the warped face cannot be composited into the frame."""


@dataclass
class BlendResult:
    """Statistics describing one blend operation."""

    box: FaceBox
    #: Fraction of the patch that was actually written into the frame.
    coverage: float
    #: ``"alpha"`` or ``"seamless"``.
    method: str


def feather_mask(mask: np.ndarray, feather: float) -> np.ndarray:
    """Blur the edges of a binary mask to produce a soft alpha channel.

    ``feather`` is a fraction of the mask's smaller side, so the same value
    looks consistent on small and large faces. Returns a float32 mask in
    ``[0, 1]``.

    Blurring alone would also attenuate the interior of a small or thin mask,
    which makes the pasted face look ghostly. The eroded core of the mask is
    therefore re-imposed with a maximum, so the interior stays fully opaque and
    only the rim is soft.
    """
    if mask is None or mask.size == 0:
        return np.zeros((1, 1), dtype=np.float32)

    alpha = mask.astype(np.float32) / 255.0
    if feather <= 0.0:
        return alpha

    height, width = mask.shape[:2]
    sigma = max(1.0, min(width, height) * float(feather) * 0.25)
    # Blur in a padded canvas so the soft edge is not clipped at the border.
    pad = int(round(sigma * 3))
    padded = cv2.copyMakeBorder(
        alpha, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0.0
    )
    blurred = cv2.GaussianBlur(padded, (0, 0), sigmaX=sigma)
    blurred = blurred[pad : pad + height, pad : pad + width]

    core_radius = max(1, int(round(sigma)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (core_radius * 2 + 1, core_radius * 2 + 1)
    )
    core = cv2.erode((alpha > 0.5).astype(np.float32), kernel)

    return np.clip(np.maximum(blurred, core), 0.0, 1.0).astype(np.float32)


def build_alpha(
    warped: WarpedFace,
    landmarks: Landmarks,
    box: FaceBox,
    config: BlendConfig,
) -> np.ndarray:
    """Build the alpha channel used to blend a warped face.

    The hull of the target landmarks constrains the paste to the face itself, so
    the corners of the warped rectangle never show up. The result is then
    feathered and scaled by the configured opacity.
    """
    if config.use_hull_mask:
        local = landmarks.translate(-box.x, -box.y)
        hull = face_hull_mask(local, warped.patch.shape[:2])
        combined = cv2.bitwise_and(hull, warped.mask)
    else:
        combined = warped.mask

    if np.count_nonzero(combined) == 0:
        # Fall back to the warped validity mask rather than pasting nothing.
        combined = warped.mask

    alpha = feather_mask(combined, config.feather)
    return np.clip(alpha * float(config.opacity), 0.0, 1.0).astype(np.float32)


class FaceBlender:
    """Composite warped faces into frames."""

    def __init__(self, config: Optional[BlendConfig] = None) -> None:
        self.config = config or BlendConfig()

    # -- public API ---------------------------------------------------------

    def blend(
        self,
        frame: np.ndarray,
        warped: WarpedFace,
        landmarks: Landmarks,
    ) -> Tuple[np.ndarray, BlendResult]:
        """Return a copy of ``frame`` with the warped face composited in."""
        if frame is None or frame.size == 0:
            raise BlendError("cannot blend into an empty frame")

        box = warped.box.clamp(frame.shape[1], frame.shape[0])
        if box.width < 2 or box.height < 2:
            raise BlendError(f"face box is too small to blend: {box.as_tuple()}")

        alpha = build_alpha(warped, landmarks, box, self.config)
        patch = warped.patch
        if patch.shape[:2] != (box.height, box.width):
            patch = cv2.resize(
                patch, (box.width, box.height), interpolation=cv2.INTER_LINEAR
            )
            alpha = cv2.resize(
                alpha, (box.width, box.height), interpolation=cv2.INTER_LINEAR
            )

        if float(self.config.opacity) <= 0.0:
            return frame.copy(), BlendResult(box, 0.0, "alpha")

        method = "alpha"
        if self.config.seamless:
            result = self._blend_seamless(frame, patch, alpha, box)
            if result is None:
                log.debug("seamless cloning unavailable, using alpha blend")
            else:
                coverage = float(np.count_nonzero(alpha > 0.01)) / float(alpha.size)
                return result, BlendResult(box, coverage, "seamless")

        result = self._blend_alpha(frame, patch, alpha, box)
        coverage = float(np.count_nonzero(alpha > 0.01)) / float(alpha.size)
        return result, BlendResult(box, coverage, method)

    def blend_in_place(
        self,
        frame: np.ndarray,
        warped: WarpedFace,
        landmarks: Landmarks,
    ) -> BlendResult:
        """Composite directly into ``frame``, avoiding one frame sized copy.

        Useful in the processing loop on memory constrained machines.
        """
        result, stats = self.blend(frame, warped, landmarks)
        np.copyto(frame, result)
        return stats

    # -- internals ----------------------------------------------------------

    def _blend_alpha(
        self,
        frame: np.ndarray,
        patch: np.ndarray,
        alpha: np.ndarray,
        box: FaceBox,
    ) -> np.ndarray:
        """Standard premultiplied alpha blend of a patch into a frame copy."""
        output = frame.copy()
        region = output[box.y : box.y2, box.x : box.x2]
        if region.shape[:2] != patch.shape[:2]:
            raise BlendError(
                f"patch {patch.shape[:2]} does not fit region {region.shape[:2]}"
            )

        weights = alpha[:, :, None]
        blended = region.astype(np.float32) * (1.0 - weights) + patch.astype(
            np.float32
        ) * weights
        region[:] = np.clip(blended, 0, 255).astype(np.uint8)
        return output

    def _blend_seamless(
        self,
        frame: np.ndarray,
        patch: np.ndarray,
        alpha: np.ndarray,
        box: FaceBox,
    ) -> Optional[np.ndarray]:
        """Gradient aware cloning, falling back to ``None`` when it cannot run."""
        centre = (box.x + box.width // 2, box.y + box.height // 2)
        hard_mask = (alpha > 0.5).astype(np.uint8) * 255
        if np.count_nonzero(hard_mask) == 0:
            return None
        try:
            output = frame.copy()
            cloned = cv2.seamlessClone(
                patch, output, hard_mask, centre, cv2.NORMAL_CLONE
            )
        except cv2.error as exc:
            log.debug("seamlessClone failed: %s", exc)
            return None

        # seamlessClone ignores soft edges, so re-apply the feathered alpha
        # outside the hard mask to keep the transition smooth.
        soft = np.clip(alpha - (hard_mask > 0).astype(np.float32), 0.0, 1.0)
        if np.count_nonzero(soft) == 0:
            return cloned
        return self._blend_alpha(cloned, patch, soft, box)


def composite(
    frame: np.ndarray,
    warped: WarpedFace,
    landmarks: Landmarks,
    config: Optional[BlendConfig] = None,
) -> Tuple[np.ndarray, BlendResult]:
    """One shot helper that blends a warped face into a frame."""
    return FaceBlender(config).blend(frame, warped, landmarks)


def mask_preview(
    frame: np.ndarray,
    warped: WarpedFace,
    landmarks: Landmarks,
    config: Optional[BlendConfig] = None,
) -> np.ndarray:
    """Return a visualisation of the alpha mask over the face box.

    Used by the GUI to show why a swap looks the way it does.
    """
    alpha = build_alpha(warped, landmarks, warped.box, config or BlendConfig())
    box = warped.box.clamp(frame.shape[1], frame.shape[0])
    preview = frame.copy()
    region = preview[box.y : box.y2, box.x : box.x2]
    if region.shape[:2] != alpha.shape[:2]:
        alpha = cv2.resize(
            alpha, (region.shape[1], region.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    overlay = np.zeros_like(region)
    overlay[:, :, 1] = (alpha * 255).astype(np.uint8)
    return cv2.addWeighted(region, 0.6, overlay, 0.4, 0.0)
