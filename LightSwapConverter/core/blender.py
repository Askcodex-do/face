"""Seamless face blending.

The aligned replacement patch is warped back into frame space and composited
through a feathered mask. ``cv2.seamlessClone`` is used when available because
it hides the seam better, with a plain alpha composite as the always-available
fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.alignment import AlignmentResult
from utils.logger import get_logger

try:  # pragma: no cover - exercised only on machines without OpenCV
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

logger = get_logger("core.blender")


@dataclass
class BlendResult:
    """Outcome of one blend operation."""

    frame: np.ndarray
    method: str = "alpha"
    strength: float = 1.0
    applied: bool = True


class FaceBlender:
    """Composite a transformed face patch back onto the target frame."""

    def __init__(
        self,
        strength: float = 0.85,
        feather: int = 21,
        use_seamless: bool = True,
        mask_scale: float = 1.0,
    ) -> None:
        self.strength = float(np.clip(strength, 0.0, 1.0))
        self.feather = max(int(feather) | 1, 3)
        self.use_seamless = bool(use_seamless)
        self.mask_scale = float(mask_scale)

    # ------------------------------------------------------------------ blend
    def blend(
        self,
        frame: np.ndarray,
        patch: np.ndarray,
        mask: np.ndarray,
        alignment: AlignmentResult,
    ) -> BlendResult:
        """Place ``patch`` onto ``frame`` using the inverse alignment transform."""
        if frame is None or frame.size == 0:
            return BlendResult(frame=frame, method="none", strength=0.0, applied=False)
        if patch is None or patch.size == 0 or not alignment.valid:
            return BlendResult(frame=frame, method="none", strength=0.0, applied=False)

        height, width = frame.shape[:2]
        warped_patch, warped_mask = self._warp_to_frame(patch, mask, alignment, width, height)
        if warped_mask.max() == 0:
            return BlendResult(frame=frame, method="none", strength=0.0, applied=False)

        region = self._bounding_rect(warped_mask)
        if region is None:
            return BlendResult(frame=frame, method="none", strength=0.0, applied=False)

        x, y, w, h = region
        target_crop = frame[y : y + h, x : x + w]
        patch_crop = warped_patch[y : y + h, x : x + w]
        mask_crop = warped_mask[y : y + h, x : x + w]

        if self.use_seamless and w > 8 and h > 8:
            composited = self._seamless(target_crop, patch_crop, mask_crop)
            if composited is not None:
                frame[y : y + h, x : x + w] = self._apply_strength(target_crop, composited, mask_crop)
                return BlendResult(frame=frame, method="seamless", strength=self.strength)

        composited = self._alpha(target_crop, patch_crop, mask_crop)
        frame[y : y + h, x : x + w] = self._apply_strength(target_crop, composited, mask_crop)
        return BlendResult(frame=frame, method="alpha", strength=self.strength)

    def blend_patch_only(self, patch: np.ndarray, mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Resize a patch for preview widgets without touching a full frame."""
        return cv2.resize(patch, size, interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------ modes
    def _seamless(
        self, target_crop: np.ndarray, patch_crop: np.ndarray, mask_crop: np.ndarray
    ) -> Optional[np.ndarray]:
        """Try ``cv2.seamlessClone``; return ``None`` when it is unusable."""
        try:
            binary = (mask_crop > 32).astype(np.uint8) * 255
            if cv2.countNonZero(binary) == 0:
                return None
            height, width = binary.shape[:2]
            center = (width // 2, height // 2)
            return cv2.seamlessClone(
                patch_crop.astype(np.uint8),
                target_crop.astype(np.uint8),
                binary,
                center,
                cv2.NORMAL_CLONE,
            )
        except Exception as exc:  # OpenCV raises cv2.error on degenerate inputs
            logger.debug("seamlessClone unavailable (%s); falling back to alpha.", exc)
            return None

    def _alpha(self, target_crop: np.ndarray, patch_crop: np.ndarray, mask_crop: np.ndarray) -> np.ndarray:
        alpha = (mask_crop.astype(np.float32) / 255.0)[..., None]
        blended = patch_crop.astype(np.float32) * alpha + target_crop.astype(np.float32) * (1.0 - alpha)
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _apply_strength(self, target: np.ndarray, composited: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Dial the swap back towards the original frame."""
        if self.strength >= 0.999:
            return composited
        alpha = (mask.astype(np.float32) / 255.0)[..., None] * self.strength
        result = composited.astype(np.float32) * alpha + target.astype(np.float32) * (1.0 - alpha)
        return np.clip(result, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ helper
    def _warp_to_frame(
        self,
        patch: np.ndarray,
        mask: np.ndarray,
        alignment: AlignmentResult,
        width: int,
        height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        inverse = alignment.inverse
        warped_patch = cv2.warpAffine(
            patch,
            inverse,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        if mask is None:
            mask = np.full(patch.shape[:2], 255, dtype=np.uint8)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if self.mask_scale != 1.0:
            mask = cv2.resize(mask, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_NEAREST)

        warped_mask = cv2.warpAffine(
            mask,
            inverse,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if self.feather > 3:
            warped_mask = cv2.GaussianBlur(warped_mask, (self.feather, self.feather), 0)
        return warped_patch, warped_mask

    @staticmethod
    def _bounding_rect(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        ys, xs = np.nonzero(mask > 8)
        if xs.size == 0 or ys.size == 0:
            return None
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        return x1, y1, x2 - x1, y2 - y1
