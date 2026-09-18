"""Pipeline orchestration: turn a source face plus a target video into output.

This module is the only place that knows the order of the stages. Every stage is
injected as an object with a small interface, so the processor itself has no
image processing code and is straightforward to test with stubs.

The processing loop is deliberately conservative about memory: one frame is
decoded, converted and written before the next is read. Nothing accumulates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

from ..utils.config import AppConfig
from ..utils.logger import get_logger
from .alignment import AlignmentError, align_to_face
from .blender import BlendError, FaceBlender
from .face_detector import FaceBox, FaceDetector
from .landmarks import LandmarkEstimator, Landmarks
from .transformer import FaceTransformer, TransformError
from .video_reader import VideoReader
from .video_writer import VideoWriter

log = get_logger("core.processor")

#: ``(completed, total, message)``. ``total`` is 0 when it is unknown.
ProgressCallback = Callable[[int, int, str], None]
#: ``(frame_index, frame)`` used for live previews. The frame is a copy.
PreviewCallback = Callable[[int, np.ndarray], None]
#: Called for each frame that could not be processed, for reporting.
ErrorCallback = Callable[[int, str], None]


class ProcessorError(RuntimeError):
    """Raised when a conversion cannot start or must be abandoned."""


class CancelledError(RuntimeError):
    """Raised inside the loop when the caller cancels the conversion."""


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
class ProcessingStats:
    """Counters collected while a conversion runs."""

    frames_read: int = 0
    frames_written: int = 0
    frames_swapped: int = 0
    frames_skipped: int = 0
    errors: int = 0
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

        self._cancelled = False
        self._source: Optional[SourceFace] = None

    # -- cancellation -------------------------------------------------------

    def cancel(self) -> None:
        """Ask the running conversion to stop after the current frame."""
        self._cancelled = True

    def reset(self) -> None:
        """Clear the cancel flag and per-run detector state."""
        self._cancelled = False
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
        stats = ProcessingStats(output_path=str(output_path))
        started = time.time()

        source = self.load_source_face(source_face_path)
        reader = VideoReader(target_video_path, self.config.video)
        info = reader.open().info

        if info.megapixels > self.config.performance.max_frame_megapixels:
            log.warning(
                "frame size %.2f MP exceeds the configured limit of %.2f MP; "
                "the conversion may be slow on this machine",
                info.megapixels,
                self.config.performance.max_frame_megapixels,
            )

        width, height = self.config.video.output_size(info.width, info.height)
        writer = VideoWriter(
            output_path,
            width,
            height,
            info.fps,
            self.config.video,
            audio_from=target_video_path if self.config.video.copy_audio else None,
        )
        writer.open()

        total = info.frame_count or 0
        limit = self.config.video.max_frames or 0
        if limit > 0:
            total = min(total, limit) if total else limit
        if start_frame > 0:
            reader.seek(start_frame)
            stats.frames_read = start_frame

        try:
            for index, frame in enumerate(
                reader.frames(limit=limit, start=start_frame), start=start_frame
            ):
                if self._cancelled:
                    stats.messages.append(f"cancelled at frame {index}")
                    log.info("conversion cancelled at frame %d", index)
                    break

                stats.frames_read += 1
                try:
                    result, swapped = self.process_frame(frame, source)
                except (cv2.error, ValueError) as exc:
                    stats.errors += 1
                    message = f"frame {index}: {exc}"
                    stats.messages.append(message)
                    log.warning("frame %d failed: %s", index, exc)
                    if on_error is not None:
                        on_error(index, str(exc))
                    result, swapped = frame, False

                if swapped:
                    stats.frames_swapped += 1
                else:
                    stats.frames_skipped += 1

                writer.write(result)
                stats.frames_written += 1

                interval = self.config.performance.preview_interval
                if preview is not None and index % interval == 0:
                    preview(index, result)

                report_every = self.config.performance.progress_interval
                if progress is not None and index % report_every == 0:
                    progress(index, total, f"frame {index}")
        finally:
            reader.release()
            stats.output_path = writer.close() or str(output_path)

        stats.elapsed_seconds = time.time() - started
        if progress is not None:
            progress(stats.frames_written, total or stats.frames_written, "done")
        log.info("conversion finished: %s", stats.summary())
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
            height = min(frame.shape[0], result.shape[0])
            original = cv2.resize(
                frame, (int(frame.shape[1] * height / frame.shape[0]), height)
            )
            updated = cv2.resize(
                result, (int(result.shape[1] * height / result.shape[0]), height)
            )
            return np.hstack([original, updated])
        return result


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def read_image(path: str | Path) -> np.ndarray:
    """Read an image, raising :class:`ProcessorError` when it cannot be decoded.

    ``cv2.imread`` handles non ASCII paths poorly on Windows, so the file is
    read as bytes first and decoded from memory.
    """
    target = Path(path)
    if not target.is_file():
        raise ProcessorError(f"image file not found: {target}")
    try:
        buffer = np.fromfile(str(target), dtype=np.uint8)
    except OSError as exc:
        raise ProcessorError(f"cannot read {target}: {exc}") from exc
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ProcessorError(f"unsupported or corrupt image: {target}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> Path:
    """Write an image, supporting non ASCII paths on Windows."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    extension = target.suffix or ".png"
    ok, buffer = cv2.imencode(extension, image)
    if not ok:
        raise ProcessorError(f"cannot encode image as {extension}")
    buffer.tofile(str(target))
    return target


def make_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Stack two images horizontally, scaling them to a common height."""
    height = min(left.shape[0], right.shape[0])
    left_scaled = cv2.resize(
        left, (int(left.shape[1] * height / left.shape[0]), height)
    )
    right_scaled = cv2.resize(
        right, (int(right.shape[1] * height / right.shape[0]), height)
    )
    return np.hstack([left_scaled, right_scaled])
