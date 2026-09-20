"""Sequential video decoding.

OpenCV's ``VideoCapture`` is used because it is the smallest reliable decoder
available without shipping extra binaries. Frames are read one at a time and
optionally downscaled, which is what keeps memory flat on 2 GB machines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from utils.logger import get_logger

try:  # pragma: no cover - exercised only on machines without OpenCV
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

logger = get_logger("core.video_reader")


class VideoReaderError(RuntimeError):
    """Raised when a video file cannot be opened or decoded."""


@dataclass(frozen=True)
class VideoInfo:
    """Metadata describing the opened video."""

    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    codec: str = ""

    @property
    def duration_seconds(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps

    @property
    def is_valid(self) -> bool:
        return self.width > 0 and self.height > 0 and self.frame_count > 0


@dataclass(frozen=True)
class FrameInfo:
    """Positional information for a decoded frame."""

    index: int
    timestamp: float
    width: int
    height: int
    scale: float = 1.0


class VideoReader:
    """Read frames from a video file sequentially.

    Example::

        with VideoReader("input.mp4") as reader:
            for frame, info in reader.frames():
                ...
    """

    def __init__(
        self,
        path: str | Path,
        max_width: int = 0,
        max_height: int = 0,
        fps_fallback: float = 25.0,
    ) -> None:
        self.path = Path(path)
        self.max_width = int(max_width)
        self.max_height = int(max_height)
        self.fps_fallback = float(fps_fallback)
        self._capture = None
        self._info: Optional[VideoInfo] = None

    # ------------------------------------------------------------- lifecycle
    def open(self) -> VideoInfo:
        """Open the file and return its metadata."""
        if cv2 is None:
            raise VideoReaderError("OpenCV is not installed; cannot decode video.")
        if not self.path.is_file():
            raise VideoReaderError(f"Video file not found: {self.path}")

        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise VideoReaderError(f"Unsupported or corrupt video: {self.path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = self.fps_fallback
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        self._capture = capture
        self._info = VideoInfo(
            path=str(self.path),
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            codec=_fourcc_to_string(capture.get(cv2.CAP_PROP_FOURCC)),
        )
        logger.info(
            "Opened %s (%dx%d, %.2f fps, %d frames)",
            self.path.name,
            width,
            height,
            fps,
            frame_count,
        )
        return self._info

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "VideoReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -------------------------------------------------------------- accessors
    @property
    def info(self) -> VideoInfo:
        if self._info is None:
            raise VideoReaderError("Reader is not open; call open() first.")
        return self._info

    @property
    def is_open(self) -> bool:
        return self._capture is not None

    # ------------------------------------------------------------------ frames
    def read(self) -> tuple[Optional[np.ndarray], Optional[FrameInfo]]:
        """Read the next frame.

        Returns ``(frame, info)`` or ``(None, None)`` at end of stream.
        """
        if self._capture is None or self._info is None:
            raise VideoReaderError("Reader is not open; call open() first.")

        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None, None

        index = int(self._capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        frame, scale = self._resize(frame)
        height, width = frame.shape[:2]
        info = FrameInfo(
            index=max(index, 0),
            timestamp=max(index, 0) / self._info.fps,
            width=width,
            height=height,
            scale=scale,
        )
        return frame, info

    def frames(self) -> Iterator[tuple[np.ndarray, FrameInfo]]:
        """Yield ``(frame, info)`` pairs until the stream ends."""
        while True:
            frame, info = self.read()
            if frame is None or info is None:
                return
            yield frame, info

    def release(self) -> None:
        """Alias for :meth:`close`, kept for readability in callers."""
        self.close()

    # ------------------------------------------------------------------ helper
    def _resize(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """Downscale the frame when it exceeds the configured limits."""
        height, width = frame.shape[:2]
        limit_w = self.max_width or width
        limit_h = self.max_height or height
        scale = min(limit_w / width, limit_h / height, 1.0)
        if scale >= 1.0:
            return frame, 1.0

        target = (max(int(width * scale), 1), max(int(height * scale), 1))
        resized = cv2.resize(frame, target, interpolation=cv2.INTER_AREA)
        return resized, scale


def _fourcc_to_string(value: float) -> str:
    """Convert OpenCV's numeric FOURCC back into a readable tag."""
    if not value:
        return ""
    code = int(value)
    chars = [chr((code >> (8 * i)) & 0xFF) for i in range(4)]
    return "".join(c for c in chars if c.isprintable()).strip()
