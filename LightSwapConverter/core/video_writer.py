"""Video writing built on OpenCV's ``VideoWriter``.

OpenCV writes video only, so audio is copied separately by muxing the original
track into the finished file with ffmpeg when that executable happens to be on
the machine. The application stays fully offline either way: no download is ever
attempted and a missing ffmpeg simply means the output has no audio.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..utils.config import VideoConfig
from ..utils.logger import get_logger

log = get_logger("core.video_writer")


class VideoWriteError(RuntimeError):
    """Raised when the output file cannot be created or written."""


def ffmpeg_available() -> bool:
    """Return ``True`` when an ``ffmpeg`` executable can be found on PATH."""
    return shutil.which("ffmpeg") is not None


class VideoWriter:
    """Write frames to a video file, optionally with the source audio track.

    Typical use::

        with VideoWriter("out.mp4", width, height, fps) as writer:
            writer.write(frame)

    When ``audio_from`` is set and ffmpeg is available, the audio track of that
    file is muxed into the result when the writer is closed.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        width: int,
        height: int,
        fps: float,
        config: Optional[VideoConfig] = None,
        *,
        audio_from: Optional[str | os.PathLike[str]] = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise VideoWriteError(f"invalid frame size {width}x{height}")
        if fps <= 0:
            raise VideoWriteError(f"invalid frame rate {fps}")

        self.config = config or VideoConfig()
        self.path = str(path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.audio_from = str(audio_from) if audio_from else None

        self._writer: Optional[cv2.VideoWriter] = None
        self._frames_written = 0
        #: Frames are staged here when audio muxing is requested, because the
        #: mux step needs a complete video-only file to read from.
        self._temp_path: Optional[str] = None
        self._final_path = self.path

    # -- lifecycle ----------------------------------------------------------

    def open(self) -> "VideoWriter":
        """Create the output file and return ``self``."""
        if self._writer is not None:
            return self

        target = Path(self._final_path)
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)

        write_path = self._final_path
        if self._should_mux_audio():
            handle, temp_path = tempfile.mkstemp(
                suffix=Path(self._final_path).suffix or ".mp4"
            )
            os.close(handle)
            self._temp_path = temp_path
            write_path = temp_path

        fourcc = cv2.VideoWriter_fourcc(*self.config.codec)
        writer = cv2.VideoWriter(
            write_path, fourcc, self.fps, (self.width, self.height)
        )
        if not writer.isOpened():
            writer.release()
            self._cleanup_temp()
            raise VideoWriteError(
                f"cannot create output video {self._final_path} "
                f"with codec {self.config.codec!r}"
            )

        self._writer = writer
        log.info(
            "writing %dx%d @ %.2f fps with codec %s to %s",
            self.width,
            self.height,
            self.fps,
            self.config.codec,
            self._final_path,
        )
        return self

    def _should_mux_audio(self) -> bool:
        return bool(
            self.audio_from
            and self.config.copy_audio
            and ffmpeg_available()
            and os.path.isfile(self.audio_from)
        )

    def write(self, frame: np.ndarray) -> None:
        """Append a frame, resizing it when it does not match the output size."""
        if self._writer is None:
            raise VideoWriteError("writer is not open, call open() first")
        if frame is None:
            raise VideoWriteError("cannot write an empty frame")

        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(
                frame, (self.width, self.height), interpolation=cv2.INTER_AREA
            )
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        self._writer.write(np.ascontiguousarray(frame))
        self._frames_written += 1

    def close(self) -> Optional[str]:
        """Finalise the file and mux audio when requested.

        Returns the path of the finished file, or ``None`` when the writer had
        not been opened.
        """
        if self._writer is None:
            return None
        self._writer.release()
        self._writer = None

        if self._temp_path is not None:
            self._mux_audio()
            self._cleanup_temp()
        log.info("wrote %d frames to %s", self._frames_written, self._final_path)
        return self._final_path

    def release(self) -> None:
        self.close()

    def __enter__(self) -> "VideoWriter":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- audio muxing -------------------------------------------------------

    def _mux_audio(self) -> None:
        """Copy the audio track of ``audio_from`` into the finished video."""
        if not self._temp_path or not self.audio_from:
            return
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            self._temp_path,
            "-i",
            self.audio_from,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            self._final_path,
        ]
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=900,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("audio mux failed (%s), keeping silent output", exc)
            self._promote_temp()
            return

        if result.returncode != 0:
            log.warning(
                "ffmpeg returned %d, keeping silent output: %s",
                result.returncode,
                result.stderr.decode("utf-8", "replace").strip()[:300],
            )
            self._promote_temp()

    def _promote_temp(self) -> None:
        """Move the video-only file into place when muxing did not work."""
        if not self._temp_path:
            return
        try:
            os.replace(self._temp_path, self._final_path)
            self._temp_path = None
        except OSError as exc:
            log.error("cannot move %s into place: %s", self._temp_path, exc)

    def _cleanup_temp(self) -> None:
        if self._temp_path and os.path.isfile(self._temp_path):
            try:
                os.remove(self._temp_path)
            except OSError:
                pass
        self._temp_path = None

    # -- metadata -----------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._writer is not None

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def output_path(self) -> str:
        return self._final_path
