"""Face pipeline orchestration: source face plus target video to output.

This module knows the order of the face stages. Every stage is injected as an
object with a small interface, so the orchestrator contains no image processing
code itself.

The frame loop, progress reporting, cancellation and resource handling are not
implemented here. They belong to :mod:`~LightSwapConverter.core.processor`, the
Phase 1 video engine, and this module drives it through a ``transform``
callback. That keeps one implementation of the loop and one place that can leak
a decoder.

Kept separate from ``processor`` so the video engine stays free of any face or
model code, as Phase 1 requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..utils.config import AppConfig
from ..utils.logger import get_logger
from .alignment import AlignmentError, align_to_face
from .blender import BlendError, FaceBlender
from .face_detector import FaceBox, FaceDetector
from .landmarks import LandmarkEstimator, Landmarks
from .processor import (
    ErrorCallback,
    FrameTransform,
    PreviewCallback,
    ProgressCallback,
    ProcessorError,
    VideoProcessor,
    make_side_by_side,
    read_image,
    write_image,
)
from .transformer import FaceTransformer, TransformError

log = get_logger("core.swapper")

__all__ = [
    "SourceFace",
    "SwapStats",
    "VideoFaceProcessor",
    "read_image",
    "write_image",
    "make_side_by_side",
]


@dataclass
class SourceFace:
    """A prepared source face: the image plus its landmarks."""

    image: np.ndarray
    landmarks: Landmarks
    box: FaceBox

    @property
    def size(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]


@dataclass
class SwapStats:
    """Face oriented counters for a conversion.

    The timing and frame counts come from the video engine; the swap counters
    are accumulated by the transform this module installs.
    """

    frames_read: int = 0
    frames_written: int = 0
    frames_swapped: int = 0
    frames_skipped: int = 0
    errors: int = 0
    cancelled: bool = False
    elapsed_seconds: float = 0.0
    output_path: str = ""
    messages: List[str] = field(default_factory=list)

    @property
    def swap_rate(self) -> float:
        if self.frames_read == 0:
            return 0.0
        return self.frames_swapped / float(self.frames_read)

    @property
    def fps(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.frames_written / self.elapsed_seconds

    def summary(self) -> str:
        return (
            f"{self.frames_written} frames written in "
            f"{self.elapsed_seconds:.1f}s ({self.fps:.1f} fps), "
            f"{self.frames_swapped} swapped, {self.frames_skipped} without a face, "
            f"{self.errors} errors"
        )


#: Historical name for :class:`SwapStats`, kept so existing callers and tests
#: that import ``ProcessingStats`` from this module keep working.
ProcessingStats = SwapStats


class VideoFaceProcessor:
    """Convert a video by replacing the face in every frame with a source face.

    Parameters
    ----------
    config:
        Application configuration. Defaults are used when omitted.
    detector, landmark_estimator, transformer, blender:
        Stage implementations. They may be replaced with compatible objects,
        which is how the tests exercise the loop without touching real images.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        *,
        detector: Optional[FaceDetector] = None,
        landmark_estimator: Optional[LandmarkEstimator] = None,
        transformer: Optional[FaceTransformer] = None,
        blender: Optional[FaceBlender] = None,
    ) -> None:
        self.config = config or AppConfig()
        self.detector = detector or FaceDetector(self.config.detection)
        self.landmark_estimator = landmark_estimator or LandmarkEstimator(
            self.config.landmarks
        )
        self.transformer = transformer or FaceTransformer(self.config.transform)
        self.blender = blender or FaceBlender(self.config.blend)

        self._source: Optional[SourceFace] = None
        #: The engine running the current conversion, if any. Cancellation has
        #: to reach the engine because that is where the loop lives.
        self._engine: Optional[VideoProcessor] = None
        #: Set when cancel() is called before a run starts, so the request is
        #: not lost between the call and the loop.
        self._cancel_requested = False

    # -- cancellation -------------------------------------------------------

    def cancel(self) -> None:
        """Ask the running conversion to stop after the current frame.

        Safe to call from another thread. A cancel issued just as a conversion
        is starting is not lost, because :meth:`process_video` re-checks the
        flag after the engine exists.
        """
        self._cancel_requested = True
        engine = self._engine
        if engine is not None:
            engine.cancel()

    def reset(self) -> None:
        """Clear the cancel flag and per-run detector state."""
        self._cancel_requested = False
        self._engine = None
        self.detector.reset()
        self.landmark_estimator.reset()

    # -- source preparation -------------------------------------------------

    def prepare_source_face(
        self, image: np.ndarray, *, refine: bool = True
    ) -> SourceFace:
        """Detect and measure the face in a source image.

        Raises :class:`ProcessorError` when no face is found, because a source
        without a face makes the whole conversion pointless.
        """
        if image is None or image.size == 0:
            raise ProcessorError("source face image is empty")

        self.detector.reset()
        box = self.detector.detect(image)
        if box is None:
            raise ProcessorError(
                "no face detected in the source image; use a clear frontal photo"
            )

        landmarks = self.landmark_estimator.estimate(image, box, refine=refine)
        self._source = SourceFace(image, landmarks, box)
        log.info(
            "source face prepared: box=%s interocular=%.1fpx",
            box.as_tuple(),
            landmarks.eye_distance,
        )
        return self._source

    def load_source_face(self, path: str | Path, *, refine: bool = True) -> SourceFace:
        """Read an image from disk and prepare it as the source face."""
        image = read_image(path)
        return self.prepare_source_face(image, refine=refine)

    # -- frame level --------------------------------------------------------

    def process_frame(
        self, frame: np.ndarray, source: SourceFace
    ) -> tuple[np.ndarray, bool]:
        """Replace the face in one frame.

        Returns ``(frame, swapped)``. The original frame object is returned
        unchanged when no face is found, so the caller can write it directly.
        """
        box = self.detector.detect(frame)
        if box is None:
            return frame, False

        target_landmarks = self.landmark_estimator.estimate(frame, box)

        try:
            alignment = align_to_face(
                source.landmarks, target_landmarks, box, method="similarity"
            )
        except AlignmentError as exc:
            log.debug("alignment failed, skipping frame: %s", exc)
            return frame, False

        if not alignment.is_reliable:
            # A bad fit means the head pose is too far from the source. Pasting
            # anyway produces a smeared face, so the frame is left alone.
            log.debug("alignment rejected: %s", alignment)
            return frame, False

        try:
            warped = self.transformer.build_patch(
                source.image, source.landmarks, alignment, box, frame
            )
        except TransformError as exc:
            log.debug("warp failed, skipping frame: %s", exc)
            return frame, False

        try:
            blended, _stats = self.blender.blend(frame, warped, target_landmarks)
        except BlendError as exc:
            log.debug("blend failed, skipping frame: %s", exc)
            return frame, False

        return blended, True

    # -- video level --------------------------------------------------------

    def _swap_transform(
        self, source: SourceFace, stats: ProcessingStats
    ) -> FrameTransform:
        """Build the per-frame callable handed to the video engine.

        The closure keeps only the prepared source face and the statistics, so
        the engine stays unaware of faces entirely. Returning ``None`` tells the
        engine to pass the frame through unchanged.
        """

        def transform(frame: np.ndarray, index: int) -> Optional[np.ndarray]:
            result, swapped = self.process_frame(frame, source)
            if swapped:
                stats.frames_swapped += 1
                return result
            stats.frames_skipped += 1
            return None

        return transform

    def process_video(
        self,
        source_face_path: str | Path,
        target_video_path: str | Path,
        output_path: str | Path,
        *,
        progress: Optional[ProgressCallback] = None,
        preview: Optional[PreviewCallback] = None,
        on_error: Optional[ErrorCallback] = None,
        start_frame: int = 0,
    ) -> ProcessingStats:
        """Run the full conversion and return the collected statistics."""
        self.reset()
        source = self.load_source_face(source_face_path)
        stats = ProcessingStats()

        engine = VideoProcessor(self.config)
        self._engine = engine
        if self._cancel_requested:
            # A cancel that arrived while the source face was being prepared
            # must still be honoured.
            engine.cancel()

        engine_stats = engine.process_video(
            target_video_path,
            output_path,
            self._swap_transform(source, stats),
            progress=progress,
            preview=preview,
            on_error=on_error,
            start_frame=start_frame,
        )

        # Mirror the engine counters into the face-oriented statistics so the
        # existing GUI and callers keep working unchanged.
        stats.frames_read = engine_stats.frames_read
        stats.frames_written = engine_stats.frames_written
        stats.errors = engine_stats.errors
        stats.cancelled = engine_stats.cancelled
        stats.elapsed_seconds = engine_stats.elapsed_seconds
        stats.output_path = engine_stats.output_path
        stats.messages = engine_stats.messages
        return stats

    def preview_swap(
        self,
        source_face_path: str | Path,
        frame: np.ndarray,
        *,
        side_by_side: bool = True,
    ) -> np.ndarray:
        """Swap a single frame and return a preview image for the GUI."""
        source = self.load_source_face(source_face_path)
        result, swapped = self.process_frame(frame, source)
        if not swapped:
            log.info("preview frame contained no detectable face")
        if side_by_side:
            return make_side_by_side(frame, result)
        return result
