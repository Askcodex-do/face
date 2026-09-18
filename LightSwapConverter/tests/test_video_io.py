"""Tests for video reading and writing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from LightSwapConverter.core.video_reader import (
    VideoInfo,
    VideoReadError,
    VideoReader,
    extract_frame,
    probe,
)
from LightSwapConverter.core.video_writer import (
    VideoWriteError,
    VideoWriter,
    ffmpeg_available,
)
from LightSwapConverter.utils.config import VideoConfig

from tests.helpers import draw_synthetic_face, write_test_video


class TestVideoReader:
    def test_open_reads_metadata(self, target_video: Path):
        with VideoReader(target_video) as reader:
            info = reader.info
            assert info.width == 320
            assert info.height == 240
            assert info.frame_count == 10
            assert info.fps == pytest.approx(12.0, abs=0.5)
            assert info.path == str(target_video)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(VideoReadError, match="not found"):
            VideoReader(tmp_path / "absent.mp4").open()

    def test_non_video_file_raises(self, tmp_path: Path):
        fake = tmp_path / "fake.mp4"
        fake.write_text("this is not a video", encoding="utf-8")
        with pytest.raises(VideoReadError):
            VideoReader(fake).open()

    def test_iterating_yields_every_frame(self, target_video: Path):
        with VideoReader(target_video) as reader:
            frames = list(reader.frames())
        assert len(frames) == 10
        assert all(frame.shape == (240, 320, 3) for frame in frames)

    def test_frames_honours_the_limit(self, target_video: Path):
        with VideoReader(target_video) as reader:
            frames = list(reader.frames(limit=3))
        assert len(frames) == 3

    def test_frames_can_start_from_an_offset(self, target_video: Path):
        with VideoReader(target_video) as reader:
            all_frames = list(reader.frames())
        with VideoReader(target_video) as reader:
            tail = list(reader.frames(start=6))
        assert len(tail) == 4
        assert np.array_equal(tail[0], all_frames[6])

    def test_reader_is_its_own_iterator(self, target_video: Path):
        with VideoReader(target_video) as reader:
            assert len([frame for frame in reader]) == 10

    def test_len_reports_the_frame_count(self, target_video: Path):
        with VideoReader(target_video) as reader:
            assert len(reader) == 10

    def test_read_frame_raises_when_exhausted(self, target_video: Path):
        with VideoReader(target_video) as reader:
            for _ in range(10):
                reader.read_frame()
            with pytest.raises(VideoReadError, match="no frame"):
                reader.read_frame()

    def test_read_returns_false_at_the_end(self, target_video: Path):
        with VideoReader(target_video) as reader:
            for _ in range(10):
                assert reader.read()[0]
            ok, frame = reader.read()
            assert ok is False
            assert frame is None

    def test_position_advances(self, target_video: Path):
        with VideoReader(target_video) as reader:
            assert reader.position == 0
            reader.read()
            assert reader.position == 1

    def test_seek_to_zero_rewinds(self, target_video: Path):
        with VideoReader(target_video) as reader:
            first = reader.read_frame()
            reader.seek(0)
            assert np.array_equal(reader.read_frame(), first)

    def test_seek_negative_raises(self, target_video: Path):
        with VideoReader(target_video) as reader:
            with pytest.raises(ValueError):
                reader.seek(-1)

    def test_reading_without_open_raises(self, target_video: Path):
        reader = VideoReader(target_video)
        with pytest.raises(VideoReadError, match="not open"):
            reader.read()

    def test_info_before_open_raises(self, target_video: Path):
        with pytest.raises(VideoReadError, match="not open"):
            _ = VideoReader(target_video).info

    def test_release_is_idempotent(self, target_video: Path):
        reader = VideoReader(target_video).open()
        reader.release()
        reader.release()
        assert reader.is_open is False

    def test_open_is_idempotent(self, target_video: Path):
        reader = VideoReader(target_video).open().open()
        try:
            assert reader.is_open
        finally:
            reader.release()

    def test_resize_config_scales_frames(self, target_video: Path):
        config = VideoConfig(resize_width=160, resize_height=120)
        with VideoReader(target_video, config) as reader:
            frame = reader.read_frame()
        assert frame.shape == (120, 160, 3)


class TestVideoInfo:
    def test_megapixels_and_duration(self):
        info = VideoInfo("x.mp4", 1920, 1080, 30.0, 300, "mp4v")
        assert info.megapixels == pytest.approx(2.0736, abs=1e-4)
        assert info.duration_seconds == pytest.approx(10.0)

    def test_duration_is_zero_without_fps(self):
        assert VideoInfo("x.mp4", 10, 10, 0.0, 5, "mp4v").duration_seconds == 0.0

    def test_duration_is_zero_without_frame_count(self):
        assert VideoInfo("x.mp4", 10, 10, 25.0, 0, "mp4v").duration_seconds == 0.0


class TestModuleHelpers:
    def test_probe_returns_info(self, target_video: Path):
        assert probe(target_video).frame_count == 10

    def test_probe_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(VideoReadError):
            probe(tmp_path / "absent.mp4")

    def test_extract_frame_defaults_to_the_first(self, target_video: Path):
        frame = extract_frame(target_video)
        assert frame is not None and frame.shape == (240, 320, 3)

    def test_extract_frame_out_of_range_returns_none(self, target_video: Path):
        assert extract_frame(target_video, 9999) is None

    def test_extract_frame_on_bad_file_returns_none(self, tmp_path: Path):
        bad = tmp_path / "bad.mp4"
        bad.write_text("nope", encoding="utf-8")
        assert extract_frame(bad) is None


class TestVideoWriter:
    def test_writes_frames_and_reports_count(self, tmp_path: Path):
        output = tmp_path / "out.mp4"
        with VideoWriter(output, 64, 48, 10.0) as writer:
            for _ in range(5):
                writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            assert writer.frames_written == 5
        assert output.is_file()

    def test_output_is_readable_by_the_reader(self, tmp_path: Path):
        output = tmp_path / "out.mp4"
        frame = draw_synthetic_face(64, 48)
        with VideoWriter(output, 64, 48, 10.0) as writer:
            for _ in range(4):
                writer.write(frame)
        info = probe(output)
        assert (info.width, info.height) == (64, 48)
        assert info.frame_count == 4

    def test_frames_are_resized_to_the_output_size(self, tmp_path: Path):
        output = tmp_path / "out.mp4"
        with VideoWriter(output, 32, 24, 10.0) as writer:
            writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
        assert probe(output).width == 32

    def test_non_uint8_frames_are_clipped(self, tmp_path: Path):
        output = tmp_path / "out.mp4"
        with VideoWriter(output, 16, 16, 10.0) as writer:
            writer.write(np.full((16, 16, 3), 900.0, dtype=np.float32))
        assert output.is_file()

    def test_invalid_dimensions_raise(self, tmp_path: Path):
        with pytest.raises(VideoWriteError, match="invalid frame size"):
            VideoWriter(tmp_path / "out.mp4", 0, 48, 10.0)

    def test_invalid_fps_raises(self, tmp_path: Path):
        with pytest.raises(VideoWriteError, match="invalid frame rate"):
            VideoWriter(tmp_path / "out.mp4", 64, 48, 0.0)

    def test_empty_frame_raises(self, tmp_path: Path):
        writer = VideoWriter(tmp_path / "out.mp4", 16, 16, 10.0).open()
        try:
            with pytest.raises(VideoWriteError, match="empty frame"):
                writer.write(None)  # type: ignore[arg-type]
        finally:
            writer.close()

    def test_write_without_open_raises(self, tmp_path: Path):
        with pytest.raises(VideoWriteError, match="not open"):
            VideoWriter(tmp_path / "out.mp4", 16, 16, 10.0).write(
                np.zeros((16, 16, 3), dtype=np.uint8)
            )

    def test_close_without_open_returns_none(self, tmp_path: Path):
        assert VideoWriter(tmp_path / "out.mp4", 16, 16, 10.0).close() is None

    def test_close_is_idempotent(self, tmp_path: Path):
        writer = VideoWriter(tmp_path / "out.mp4", 16, 16, 10.0).open()
        assert writer.close() is not None
        assert writer.close() is None

    def test_parent_directories_are_created(self, tmp_path: Path):
        output = tmp_path / "deep" / "nested" / "out.mp4"
        with VideoWriter(output, 16, 16, 10.0) as writer:
            writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
        assert output.is_file()

    def test_invalid_codec_raises(self, tmp_path: Path):
        config = VideoConfig(codec="zzzz")
        with pytest.raises(VideoWriteError, match="cannot create"):
            VideoWriter(tmp_path / "out.mp4", 16, 16, 10.0, config).open()

    def test_no_temp_files_are_left_behind(self, tmp_path: Path):
        output = tmp_path / "out.mp4"
        with VideoWriter(output, 16, 16, 10.0) as writer:
            writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
        assert [item.name for item in tmp_path.iterdir()] == ["out.mp4"]


class TestAudioMuxing:
    def test_ffmpeg_availability_is_a_boolean(self):
        assert isinstance(ffmpeg_available(), bool)

    def test_audio_from_missing_file_does_not_mux(self, tmp_path: Path):
        output = tmp_path / "out.mp4"
        writer = VideoWriter(
            output, 16, 16, 10.0, audio_from=tmp_path / "no_audio.mp4"
        )
        with writer:
            writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
        # The video must still be produced when there is no audio to copy.
        assert output.is_file()

    def test_audio_copy_disabled_skips_muxing(self, tmp_path: Path):
        source = write_test_video(tmp_path / "source.mp4", frames=3)
        output = tmp_path / "out.mp4"
        config = VideoConfig(copy_audio=False)
        with VideoWriter(output, 320, 240, 12.0, config, audio_from=source) as writer:
            writer.write(draw_synthetic_face(320, 240))
        assert output.is_file()
        assert probe(output).frame_count == 1
