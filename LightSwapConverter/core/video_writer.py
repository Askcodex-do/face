"""Sequential video encoding.

Frames are written straight to disk as they are produced so the pipeline never
needs to buffer more than a handful of frames in RAM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.logger import get_logger

try:  # pragma: no cover - exercised only on machines without OpenCV
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

logger = get_logger("core.video_writer")


class VideoWriterError(RuntimeError):
    """Raised when the output video cannot be created or written."""


class VideoWriter:
    """Write frames to an output video file.

    Example::

        with VideoWriter("out.mp4", fps=25, size=(640, 360)) as writer:
            writer.write(frame)
    """

    def __init__(
        self,
        path: str | Path,
        fps: float,
        size: tuple[int, int],
        codec: str = "mp4v",
    ) -> None:
        self.path = Path(path)
        self.fps = float(fps) if fps and fps > 0 else 25.0
        self.size = (int(size[0]), int(size[1]))
        self.codec = codec
        self._writer = None
        self._frames_written = 0

    # ------------------------------------------------------------- lifecycle
    def open(self) -> None:
        if cv2 is None:
            raise VideoWriterError("OpenCV is not installed; cannot encode video.")
        if self.size[0] <= 0 or self.size[1] <= 0:
            raise VideoWriterError(f"Invalid frame size: {self.size}")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, self.size)
        if not writer.isOpened():
            writer.release()
            raise VideoWriterError(
                f"Could not open output video {self.path} with codec {self.codec!r}"
            )
        self._writer = writer
        logger.info(
            "Writing %s (%dx%d, %.2f fps, codec %s)",
            self.path.name,
            self.size[0],
            self.size[1],
            self.fps,
            self.codec,
        )

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("Closed %s after %d frames", self.path.name, self._frames_written)

    def __enter__(self) -> "VideoWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ----------------------------------------------------------------- writing
    def write(self, frame: np.ndarray) -> bool:
        """Append a frame, resizing it if it does not match the output size."""
        if self._writer is None:
            raise VideoWriterError("Writer is not open; call open() first.")
        if frame is None or frame.size == 0:
            return False

        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)

        self._writer.write(frame)
        self._frames_written += 1
        return True

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def is_open(self) -> bool:
        return self._writer is not None
