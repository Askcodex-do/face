"""Frame accurate video reading built on OpenCV's ``VideoCapture``.

The reader wraps ``cv2.VideoCapture`` with the conveniences the rest of the
application needs: lazy opening, metadata inspection, iteration, seeking and
guaranteed release. It decodes one frame at a time and never keeps a frame
buffer, which matters on the 2 GB target machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from ..utils.config import VideoConfig
from ..utils.logger import get_logger

log = get_logger("core.video_reader")


class VideoReadError(RuntimeError):
    """Raised when a video cannot be opened or a frame cannot be decoded."""


@dataclass(frozen=True)
class VideoInfo:
    """Metadata describing an opened video stream."""

    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    fourcc: str

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000.0

    @property
    def duration_seconds(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps if self.frame_count else 0.0

    def __str__(self) -> str:  # pragma: no cover - display helper
        return (
            f"{Path(self.path).name}: {self.width}x{self.height} @ "
            f"{self.fps:.2f} fps, {self.frame_count or 'unknown'} frames"
        )


class VideoReader:
    """Sequential reader for a single video file.

    Use it as a context manager to guarantee the capture is released::

        with VideoReader("input.mp4") as reader:
            for index, frame in reader.frames():
                ...
    """

    def __init__(self, path: str | os.PathLike[str], config: Optional[VideoConfig] = None):
        self.path = str(path)
        self.config = config or VideoConfig()
        self._capture: Optional[cv2.VideoCapture] = None
        self._info: Optional[VideoInfo] = None
        self._position = 0

    # -- lifecycle ----------------------------------------------------------

    def open(self) -> "VideoReader":
        """Open the file and read its metadata."""
        if self._capture is not None:
            return self
        if not os.path.isfile(self.path):
            raise VideoReadError(f"video file not found: {self.path}")

        capture = cv2.VideoCapture(self.path)
        if not capture.isOpened():
            capture.release()
            raise VideoReadError(f"cannot open video: {self.path}")

        self._capture = capture
        self._info = self._read_info()
        self._position = 0
        log.info("opened %s", self._info)
        return self

    def _read_info(self) -> VideoInfo:
        assert self._capture is not None
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_value = int(self._capture.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_value >> (8 * i)) & 0xFF) for i in range(4))
        if fps <= 0 or fps != fps:  # NaN guard for broken containers
            fps = self.config.fallback_fps
        return VideoInfo(
            path=self.path,
            width=width,
            height=height,
            fps=fps,
            frame_count=max(count, 0),
            fourcc=fourcc.strip("\x00"),
        )

    def release(self) -> None:
        """Release the underlying capture. Safe to call repeatedly."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._info = None
        self._position = 0

    def close(self) -> None:
        self.release()

    def __enter__(self) -> "VideoReader":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def __iter__(self) -> Iterator[np.ndarray]:
        return self.frames()

    def __len__(self) -> int:
        return self.info.frame_count

    # -- metadata -----------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    @property
    def info(self) -> VideoInfo:
        if self._info is None:
            raise VideoReadError("video is not open, call open() first")
        return self._info

    @property
    def position(self) -> int:
        """Index of the next frame that will be returned."""
        return self._position

    # -- reading ------------------------------------------------------------

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame. Returns ``(ok, frame)`` like OpenCV."""
        self._ensure_open()
        assert self._capture is not None
        ok, frame = self._capture.read()
        if ok and frame is not None:
            self._position += 1
            frame = self._prepare(frame)
            return True, frame
        return False, None

    def read_frame(self) -> np.ndarray:
        """Read a frame or raise :class:`VideoReadError` when exhausted."""
        ok, frame = self.read()
        if not ok or frame is None:
            raise VideoReadError(f"no frame at index {self._position} in {self.path}")
        return frame

    def frames(self, limit: int = 0, start: int = 0) -> Iterator[np.ndarray]:
        """Yield frames from ``start``, at most ``limit`` of them.

        ``limit`` of ``0`` means "until the end of the file".
        """
        self._ensure_open()
        if start > 0:
            self.seek(start)
        emitted = 0
        while limit <= 0 or emitted < limit:
            ok, frame = self.read()
            if not ok or frame is None:
                break
            emitted += 1
            yield frame

    def seek(self, index: int) -> bool:
        """Move to frame ``index``.

        Returns ``True`` when the container accepted the seek. Files without a
        valid frame index are decoded forward instead, which is slower but
        always correct.
        """
        self._ensure_open()
        assert self._capture is not None
        if index < 0:
            raise ValueError("frame index must be >= 0")
        if index == 0:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._position = 0
            return True

        self._capture.set(cv2.CAP_PROP_POS_FRAMES, float(index))
        landed = int(self._capture.get(cv2.CAP_PROP_POS_FRAMES))
        self._position = landed
        if landed == index:
            return True

        # Fall back to a linear scan from wherever the container landed.
        while self._position < index:
            ok, _ = self.read()
            if not ok:
                return False
        return self._position == index

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        """Apply the configured resize and keep the frame contiguous."""
        target_w, target_h = self.config.output_size(
            frame.shape[1], frame.shape[0]
        )
        if (target_w, target_h) != (frame.shape[1], frame.shape[0]):
            frame = cv2.resize(
                frame, (target_w, target_h), interpolation=cv2.INTER_AREA
            )
        return np.ascontiguousarray(frame)

    def _ensure_open(self) -> None:
        if not self.is_open:
            raise VideoReadError("video is not open, call open() first")


def probe(path: str | os.PathLike[str]) -> VideoInfo:
    """Open ``path`` briefly and return its metadata."""
    reader = VideoReader(path)
    try:
        return reader.open().info
    finally:
        reader.release()


def extract_frame(
    path: str | os.PathLike[str], index: int = 0
) -> Optional[np.ndarray]:
    """Return a single decoded frame, or ``None`` when it is unavailable."""
    reader = VideoReader(path)
    try:
        reader.open()
        if not reader.seek(index):
            return None
        ok, frame = reader.read()
        return frame if ok else None
    except VideoReadError as exc:
        log.warning("extract_frame failed: %s", exc)
        return None
    finally:
        reader.release()
