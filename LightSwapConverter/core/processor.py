"""Phase 1: the lightweight video processing engine.

This module turns an input video into an output video one frame at a time. It
owns the processing loop, progress reporting, cancellation and resource
cleanup, and nothing else. The per-frame work is supplied by the caller as a
``transform`` callable, so the engine has no opinion about what is done to the
pixels.

Design notes for the 2 GB target machine:

* Exactly one decoded frame and one output frame are alive at a time. No frame
  list, ring buffer or queue is ever built.
* Each frame is decoded, transformed and written before the next decode, so
  peak memory is a small multiple of one frame rather than of the whole video.
* ``VideoCapture`` and ``VideoWriter`` are released on every exit path,
  including a raising transform, a cancel, and a write failure.

The engine deliberately contains no face detection, no face swapping and no
model inference. Those plug in through ``transform`` in a later phase.
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
from .video_reader import VideoInfo, VideoReadError, VideoReader
from .video_writer import VideoWriteError, VideoWriter

log = get_logger("core.processor")

#: ``(completed, total, message)``. ``total`` is 0 when it is unknown.
ProgressCallback = Callable[[int, int, str], None]
#: ``(frame_index, frame)`` for live previews. The buffer can be reused after
#: the callback returns, so copy it if you intend to keep it.
PreviewCallback = Callable[[int, np.ndarray], None]
#: ``(frame_index, message)`` called for each frame that could not be handled.
ErrorCallback = Callable[[int, str], None]
#: ``(frame, index) -> frame | None``. Returning ``None`` passes the frame
#: through untouched, which is how a transform declines a frame it cannot use.
FrameTransform = Callable[[np.ndarray, int], Optional[np.ndarray]]

#: Cap on retained per-frame error messages. A pathological file could fail on
#: every frame, and an unbounded list would grow with the video length.
MAX_RETAINED_MESSAGES = 200


class ProcessorError(RuntimeError):
    """Raised when a conversion cannot start or must be abandoned."""


class CancelledError(RuntimeError):
    """Raised inside the loop when the caller cancels the conversion."""


@dataclass
class ProcessingStats:
    """Counters collected while a conversion runs.

    ``frames_read`` counts decoded frames and ``frames_written`` counts encoded
    frames; they differ only when a frame cannot be written. ``frames_processed``
    splits into the frames the transform changed and the frames it declined.
    """

    frames_read: int = 0
    frames_written: int = 0
    frames_transformed: int = 0
    frames_passed_through: int = 0
    errors: int = 0
    cancelled: bool = False
    elapsed_seconds: float = 0.0
    source_path: str = ""
    output_path: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration_seconds: float = 0.0
    messages: List[str] = field(default_factory=list)

    def add_message(self, message: str) -> None:
        """Record a diagnostic, keeping the list bounded on long failures."""
        if len(self.messages) < MAX_RETAINED_MESSAGES:
            self.messages.append(message)

    @property
    def frames_processed(self) -> int:
        """Frames the loop handled, whether changed or passed through."""
        return self.frames_transformed + self.frames_passed_through

    @property
    def completed(self) -> bool:
        """True when the run reached the end of the input without cancelling."""
        return not self.cancelled

    @property
    def transform_rate(self) -> float:
        """Fraction of processed frames the transform actually changed."""
        processed = self.frames_processed
        if processed == 0:
            return 0.0
        return self.frames_transformed / float(processed)

    @property
    def processing_fps(self) -> float:
        """Frames written per second of wall clock time."""
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.frames_written / self.elapsed_seconds

    def summary(self) -> str:
        state = "cancelled" if self.cancelled else "finished"
        return (
            f"{state}: {self.frames_written} frames written in "
            f"{self.elapsed_seconds:.1f}s ({self.processing_fps:.1f} fps), "
            f"{self.frames_transformed} transformed, "
            f"{self.frames_passed_through} unchanged, {self.errors} errors"
        )


def identity_transform(frame: np.ndarray, index: int) -> np.ndarray:
    """A transform that changes nothing, useful as a default and in tests."""
    return frame


class VideoProcessor:
    """Frame-by-frame video conversion engine.

    One instance can run several conversions. ``cancel`` applies to whichever
    run is active::

        def darker(frame, index):
            return (frame * 0.8).astype(frame.dtype)

        stats = VideoProcessor().process_video("in.mp4", "out.mp4", darker)
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()
        self._cancelled = False

    # -- control ------------------------------------------------------------

    def cancel(self) -> None:
        """Ask the active conversion to stop after the current frame.

        Cancellation is cooperative. The frame in flight is finished, the writer
        is closed and the partial output is kept, so the caller never ends up
        with an unfinalised container.
        """
        self._cancelled = True
        log.info("cancellation requested")

    def reset(self) -> None:
        """Clear a stale cancel flag so the next run starts cleanly."""
        self._cancelled = False

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    # -- metadata -----------------------------------------------------------

    def inspect(self, path: str | Path) -> VideoInfo:
        """Return the metadata of a video without decoding its frames."""
        from .video_reader import probe

        return probe(path)

    # -- conversion ---------------------------------------------------------

    def process_video(
        self,
        input_path: str | Path,
        output_path: str | Path,
        transform: Optional[FrameTransform] = None,
        *,
        progress: Optional[ProgressCallback] = None,
        preview: Optional[PreviewCallback] = None,
        on_error: Optional[ErrorCallback] = None,
        start_frame: int = 0,
        max_frames: Optional[int] = None,
    ) -> ProcessingStats:
        """Convert ``input_path`` into ``output_path``, one frame at a time.

        Parameters
        ----------
        transform:
            Called as ``transform(frame, index)`` for each decoded frame. It may
            return a new frame, the same frame, or ``None`` to leave the frame
            untouched. An exception is caught per frame and counted, and the
            original frame is written instead, so a single bad frame does not
            abandon a long conversion. Raise :class:`CancelledError` to stop.
        progress:
            Called as ``progress(completed, total, message)``. ``total`` is 0
            when the container does not report a frame count.
        preview:
            Called as ``preview(index, frame)`` every
            ``performance.preview_interval`` frames, for a live view.
        on_error:
            Called as ``on_error(index, message)`` for each failed frame.
        start_frame:
            Skip this many frames before processing.
        max_frames:
            Stop after this many frames. Defaults to ``video.max_frames`` from
            the configuration, where ``0`` means "to the end of the file".

        Returns the collected :class:`ProcessingStats`.

        Raises
        ------
        VideoReadError
            The input is missing, corrupt, or not a readable video.
        VideoWriteError
            The output file could not be created, for example an unsupported
            container extension or an unavailable codec.
        ProcessorError
            Any other failure while opening either end of the pipeline.
        """
        self.reset()
        stats = ProcessingStats(
            source_path=str(input_path), output_path=str(output_path)
        )
        started = time.time()

        reader = VideoReader(input_path, self.config.video)
        writer: Optional[VideoWriter] = None
        total = 0
        try:
            info = self._open_reader(reader)
            stats.width, stats.height = info.width, info.height
            stats.fps = info.fps
            stats.duration_seconds = info.duration_seconds
            self._warn_on_large_frames(info)

            limit = self._resolve_limit(max_frames)
            total = self._resolve_total(info, limit, start_frame)

            writer = self._open_writer(output_path, info)
            self._run_loop(
                reader,
                writer,
                transform,
                stats,
                total=total,
                limit=limit,
                start_frame=start_frame,
                progress=progress,
                preview=preview,
                on_error=on_error,
            )
        finally:
            # Release both handles on every path, so no decoder or file lock
            # outlives the call even when an unexpected error escapes.
            reader.release()
            if writer is not None:
                stats.output_path = writer.close() or str(output_path)

        stats.elapsed_seconds = time.time() - started
        if progress is not None:
            progress(
                stats.frames_written,
                total or stats.frames_written,
                "cancelled" if stats.cancelled else "done",
            )
        log.info("%s -> %s: %s", input_path, output_path, stats.summary())
        return stats

    def convert(
        self,
        input_path: str | Path,
        output_path: str | Path,
        transform: Optional[FrameTransform] = None,
        **kwargs,
    ) -> ProcessingStats:
        """Alias for :meth:`process_video`."""
        return self.process_video(input_path, output_path, transform, **kwargs)

    # -- internals ----------------------------------------------------------

    def _open_reader(self, reader: VideoReader) -> VideoInfo:
        try:
            return reader.open().info
        except VideoReadError:
            raise
        except (cv2.error, OSError, ValueError) as exc:
            raise ProcessorError(f"cannot read {reader.path}: {exc}") from exc

    def _open_writer(self, output_path: str | Path, info: VideoInfo) -> VideoWriter:
        width, height = self.config.video.output_size(info.width, info.height)
        try:
            return VideoWriter(
                output_path,
                width,
                height,
                info.fps,
                self.config.video,
                audio_from=info.path if self.config.video.copy_audio else None,
            ).open()
        except VideoWriteError:
            raise
        except (cv2.error, OSError, ValueError) as exc:
            raise ProcessorError(f"cannot write {output_path}: {exc}") from exc

    def _warn_on_large_frames(self, info: VideoInfo) -> None:
        limit = self.config.performance.max_frame_megapixels
        if limit > 0 and info.megapixels > limit:
            log.warning(
                "frame size %.2f MP exceeds the configured %.2f MP; expect slow "
                "going on a 2 GB machine",
                info.megapixels,
                limit,
            )

    def _resolve_limit(self, max_frames: Optional[int]) -> int:
        if max_frames is not None:
            return max(0, int(max_frames))
        return max(0, int(self.config.video.max_frames or 0))

    @staticmethod
    def _resolve_total(info: VideoInfo, limit: int, start_frame: int) -> int:
        """Best available frame total for progress reporting.

        Containers with no frame index report 0, so progress is reported against
        an unknown total rather than a misleading one.
        """
        total = max(0, info.frame_count - max(0, start_frame))
        if limit > 0:
            total = min(total, limit) if total else limit
        return total

    def _run_loop(
        self,
        reader: VideoReader,
        writer: VideoWriter,
        transform: Optional[FrameTransform],
        stats: ProcessingStats,
        *,
        total: int,
        limit: int,
        start_frame: int,
        progress: Optional[ProgressCallback],
        preview: Optional[PreviewCallback],
        on_error: Optional[ErrorCallback],
    ) -> None:
        report_every = max(1, int(self.config.performance.progress_interval))
        preview_every = max(1, int(self.config.performance.preview_interval))

        for index, frame in enumerate(
            reader.frames(limit=limit, start=start_frame), start=start_frame
        ):
            if self._cancelled:
                stats.cancelled = True
                # Appended directly: a cancel is a single terminal event and
                # must survive even if the message cap has been reached.
                stats.messages.append(f"cancelled at frame {index}")
                log.info("conversion cancelled at frame %d", index)
                break

            stats.frames_read += 1
            result = self._apply_transform(transform, frame, index, stats, on_error)

            if result is None:
                stats.frames_passed_through += 1
                result = frame
            else:
                stats.frames_transformed += 1

            if not self._write_frame(writer, result, index, stats, on_error):
                continue
            stats.frames_written += 1

            if preview is not None and index % preview_every == 0:
                preview(index, result)
            if progress is not None and index % report_every == 0:
                progress(index, total, f"frame {index}")

    def _write_frame(
        self,
        writer: VideoWriter,
        frame: np.ndarray,
        index: int,
        stats: ProcessingStats,
        on_error: Optional[ErrorCallback],
    ) -> bool:
        """Write one frame, reporting rather than raising on failure."""
        try:
            writer.write(frame)
        except (VideoWriteError, cv2.error, ValueError) as exc:
            stats.errors += 1
            message = f"frame {index}: cannot write ({exc})"
            stats.add_message(message)
            log.warning("%s", message)
            if on_error is not None:
                on_error(index, str(exc))
            return False
        return True

    @staticmethod
    def _apply_transform(
        transform: Optional[FrameTransform],
        frame: np.ndarray,
        index: int,
        stats: ProcessingStats,
        on_error: Optional[ErrorCallback],
    ) -> Optional[np.ndarray]:
        """Run the transform, turning any failure into a passed-through frame."""
        if transform is None:
            return None
        try:
            result = transform(frame, index)
        except Exception as exc:  # noqa: BLE001 - one bad frame must not end the run
            stats.errors += 1
            message = f"frame {index}: {type(exc).__name__}: {exc}"
            stats.add_message(message)
            log.warning("%s", message)
            if on_error is not None:
                on_error(index, str(exc))
            return None
        if result is None:
            return None
        if not isinstance(result, np.ndarray):
            stats.errors += 1
            message = f"frame {index}: transform returned {type(result).__name__}"
            stats.add_message(message)
            log.warning("%s", message)
            if on_error is not None:
                on_error(index, message)
            return None
        return result


# ---------------------------------------------------------------------------
# Image helpers
#
# Format utilities rather than video helpers, kept here so the GUI and the later
# phases share a single import site for file IO.
# ---------------------------------------------------------------------------


def read_image(path: str | Path) -> np.ndarray:
    """Read an image, raising :class:`ProcessorError` when it cannot be decoded.

    ``cv2.imread`` handles non ASCII paths poorly on Windows, so the file is
    read as bytes and decoded from memory. That keeps paths containing spaces or
    accented characters working on the target machine.
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
