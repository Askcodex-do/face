"""Source face preparation.

This module prepares the replacement face: it aligns the source image onto the
canonical layout and applies a cheap photometric correction so it matches the
target frame's colour and brightness. No generative model is involved - the
"transformation" here is deterministic image processing, which is what keeps
the application offline and CPU friendly.

A future learned face-swap model can be plugged in by implementing the same
``prepare`` / ``transform`` interface and registering it in ``TRANSFORMERS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Type

import numpy as np

from core.alignment import FaceAligner
from core.landmarks import FaceLandmarks
from utils.logger import get_logger

try:  # pragma: no cover - exercised only on machines without OpenCV
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

logger = get_logger("core.transformer")


class TransformerError(RuntimeError):
    """Raised when a source face cannot be prepared."""


@dataclass
class TransformResult:
    """Prepared source face in canonical space."""

    image: np.ndarray
    mask: np.ndarray
    size: tuple[int, int]
    method: str = "copy"
    color_shift: Optional[tuple[float, float, float]] = None

    @property
    def is_valid(self) -> bool:
        return self.image is not None and self.image.size > 0 and self.mask is not None


class FaceTransformer:
    """Turn a source face into a patch that can be blended onto a target.

    Parameters
    ----------
    method:
        ``"copy"``    - aligned copy with optional colour matching (default).
        ``"blend"``   - 50/50 mix of source and target, softer identity swap.
        ``"monochrome`` - luminance transfer, useful for stylised output.
    color_match:
        When ``True`` the source is shifted towards the target's mean colour.
    """

    METHODS = ("copy", "blend", "monochrome")

    def __init__(
        self,
        method: str = "copy",
        color_match: bool = True,
        aligner: Optional[FaceAligner] = None,
        blur_radius: int = 0,
    ) -> None:
        if method not in self.METHODS:
            raise ValueError(f"Unknown transform method {method!r}; expected one of {self.METHODS}")
        self.method = method
        self.color_match = bool(color_match)
        self.aligner = aligner or FaceAligner()
        self.blur_radius = int(blur_radius)

    # ---------------------------------------------------------------- source
    def load_source(self, path: str | Path) -> np.ndarray:
        """Load a source face image from disk as BGR."""
        image_path = Path(path)
        if not image_path.is_file():
            raise TransformerError(f"Source image not found: {image_path}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise TransformerError(f"Could not decode source image: {image_path}")
        return image

    def prepare(
        self,
        source_image: np.ndarray,
        source_landmarks: FaceLandmarks,
        target_landmarks: Optional[FaceLandmarks] = None,
    ) -> TransformResult:
        """Align the source face and optionally match it to the target."""
        if source_image is None or source_image.size == 0:
            raise TransformerError("Source image is empty.")

        if source_landmarks is None or not source_landmarks.is_valid:
            # No landmarks: use a plain resize so the pipeline can still run.
            patch = cv2.resize(source_image, self.aligner.output_size, interpolation=cv2.INTER_AREA)
            mask = self._soft_mask(patch.shape[:2])
            return TransformResult(patch, mask, self.aligner.output_size, self.method)

        aligned, _ = self.aligner.align(source_image, source_landmarks)
        mask = self._soft_mask(aligned.shape[:2])
        return TransformResult(aligned, mask, self.aligner.output_size, self.method)

    # ------------------------------------------------------------- transform
    def transform(
        self,
        source: TransformResult,
        target_patch: np.ndarray,
        target_landmarks: Optional[FaceLandmarks] = None,
    ) -> TransformResult:
        """Produce the final patch to blend onto ``target_patch``."""
        if source is None or not source.is_valid:
            raise TransformerError("Prepared source is missing or invalid.")

        image = source.image
        if image.shape[:2] != target_patch.shape[:2]:
            image = cv2.resize(image, (target_patch.shape[1], target_patch.shape[0]), interpolation=cv2.INTER_AREA)

        if self.blur_radius > 0:
            radius = self.blur_radius | 1
            image = cv2.GaussianBlur(image, (radius, radius), 0)

        if self.method == "monochrome":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        shift = None
        if self.color_match:
            image, shift = self.match_color(image, target_patch)

        if self.method == "blend":
            image = cv2.addWeighted(image, 0.5, target_patch, 0.5, 0.0)

        return TransformResult(
            image=image,
            mask=source.mask,
            size=source.size,
            method=self.method,
            color_shift=shift,
        )

    # ---------------------------------------------------------------- colour
    @staticmethod
    def match_color(
        source: np.ndarray, target: np.ndarray, strength: float = 1.0
    ) -> tuple[np.ndarray, tuple[float, float, float]]:
        """Shift source mean colour towards the target mean colour.

        A mean/std transfer rather than a full histogram match: constant time
        and stable enough for video, where lighting changes gradually.
        """
        source_float = source.astype(np.float32)
        target_float = target.astype(np.float32)

        source_mean = source_float.reshape(-1, 3).mean(axis=0)
        target_mean = target_float.reshape(-1, 3).mean(axis=0)
        source_std = source_float.reshape(-1, 3).std(axis=0) + 1e-6
        target_std = target_float.reshape(-1, 3).std(axis=0) + 1e-6

        ratio = np.clip(target_std / source_std, 0.5, 2.0)
        adjusted = (source_float - source_mean) * ratio + target_mean

        blended = source_float * (1.0 - strength) + adjusted * strength
        shift = tuple(float(v) for v in (target_mean - source_mean))
        return np.clip(blended, 0, 255).astype(np.uint8), shift

    # ------------------------------------------------------------------ mask
    @staticmethod
    def _soft_mask(shape: tuple[int, int], feather: int = 21) -> np.ndarray:
        """Build an elliptical, feathered mask for the canonical face patch."""
        height, width = shape
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, int(height * 0.52))
        axes = (int(width * 0.40), int(height * 0.48))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        feather = max(int(feather) | 1, 3)
        mask = cv2.GaussianBlur(mask, (feather, feather), 0)
        return mask

    # ------------------------------------------------------------ registry
    @classmethod
    def register_method(cls, name: str, factory: Type["FaceTransformer"]) -> None:
        """Register a custom transformer subclass by name.

        This is the extension point for a future model-based swapper.
        """
        cls.METHODS = tuple(dict.fromkeys((*cls.METHODS, name)))
        _CUSTOM_METHODS[name] = factory
        logger.info("Registered transformer method %r", name)


_CUSTOM_METHODS: Dict[str, Type[FaceTransformer]] = {}


def create_transformer(name: str = "copy", **kwargs) -> FaceTransformer:
    """Factory that resolves built-in and custom transformer methods."""
    factory = _CUSTOM_METHODS.get(name)
    if factory is not None:
        return factory(**kwargs)
    return FaceTransformer(method=name, **kwargs)
