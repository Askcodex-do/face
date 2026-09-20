"""End-to-end conversion test plus cancellation and progress reporting."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from core.processor import ConversionJob, Processor  # noqa: E402
from core.video_writer import VideoWriter  # noqa: E402
from utils.config import AppConfig  # noqa: E402


def make_config(tmp_path) -> AppConfig:
    return AppConfig(
        models_dir=str(tmp_path / "models"),
        assets_dir=str(tmp_path / "assets"),
        output_dir=str(tmp_path / "output"),
        log_dir=str(tmp_path / "logs"),
    )


def write_clip(path, frames: int = 6, size=(160, 120)) -> None:
    with VideoWriter(path, fps=10.0, size=size, codec="MJPG") as writer:
        for i in range(frames):
            frame = np.full((size[1], size[0], 3), 30 + i * 10, dtype=np.uint8)
            cv2.rectangle(frame, (40, 30), (120, 100), (120, 150, 200), -1)
            writer.write(frame)


def write_face_image(path) -> None:
    image = np.full((120, 120, 3), 200, dtype=np.uint8)
    cv2.circle(image, (60, 60), 45, (150, 160, 190), -1)
    cv2.imwrite(str(path), image)


def test_processor_converts_video_end_to_end(tmp_path):
    source = tmp_path / "in.avi"
    face = tmp_path / "face.png"
    output = tmp_path / "out.avi"
    write_clip(source)
    write_face_image(face)

    processor = Processor(make_config(tmp_path))
    job = ConversionJob(
        source_video=str(source),
        target_face_image=str(face),
        output_video=str(output),
        codec="MJPG",
        max_width=160,
        max_height=120,
    )

    seen = []
    result = processor.process(job, progress=lambda done, total, msg: seen.append((done, total)))

    assert not result.errors, result.errors
    assert result.frames_read == 6
    assert result.frames_written == 6
    assert output.is_file()
    assert seen and seen[-1][0] == seen[-1][1]


def test_processor_frame_skip_reduces_output(tmp_path):
    source = tmp_path / "in.avi"
    face = tmp_path / "face.png"
    output = tmp_path / "out.avi"
    write_clip(source, frames=6)
    write_face_image(face)

    processor = Processor(make_config(tmp_path))
    job = ConversionJob(
        source_video=str(source),
        target_face_image=str(face),
        output_video=str(output),
        codec="MJPG",
        frame_skip=3,
        max_width=160,
        max_height=120,
    )
    result = processor.process(job)
    assert not result.errors
    assert result.frames_read == 6
    assert result.frames_written == 2


def test_processor_can_be_cancelled(tmp_path):
    source = tmp_path / "in.avi"
    face = tmp_path / "face.png"
    output = tmp_path / "out.avi"
    write_clip(source, frames=10)
    write_face_image(face)

    processor = Processor(make_config(tmp_path))
    job = ConversionJob(
        source_video=str(source),
        target_face_image=str(face),
        output_video=str(output),
        codec="MJPG",
        max_width=160,
        max_height=120,
    )
    result = processor.process(job, is_cancelled=lambda: True)
    assert result.cancelled
    assert result.succeeded is False


def test_processor_reports_missing_video_as_error(tmp_path):
    face = tmp_path / "face.png"
    write_face_image(face)

    processor = Processor(make_config(tmp_path))
    job = ConversionJob(
        source_video=str(tmp_path / "missing.avi"),
        target_face_image=str(face),
        output_video=str(tmp_path / "out.avi"),
    )
    result = processor.process(job)
    assert result.errors
    assert result.succeeded is False


def test_default_output_path_uses_output_dir(tmp_path):
    processor = Processor(make_config(tmp_path))
    output = processor.default_output_path(str(tmp_path / "holiday.mp4"))
    assert output.name == "holiday_swapped.mp4"
    assert output.parent == processor.config.output_path()
