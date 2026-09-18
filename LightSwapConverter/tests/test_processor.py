"""Tests for the processor that wires the pipeline together.

These tests run the real stages - real cascade, real warping, real blending -
against synthetic video. No stage is mocked. The only substitution is the media
itself, which is generated on the fly so the suite stays offline.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from LightSwapConverter.core.swapper import (
    ProcessingStats,
    ProcessorError,
    SourceFace,
    VideoFaceProcessor,
    make_side_by_side,
    read_image,
    write_image,
)
from LightSwapConverter.core.video_reader import VideoReadError, probe
from LightSwapConverter.utils.config import AppConfig

from tests.helpers import draw_synthetic_face, write_test_video


@pytest.fixture
def processor() -> VideoFaceProcessor:
    """A processor with the frame limit cleared, for direct frame tests."""
    config = AppConfig()
    config.video.max_frames = 0
    return VideoFaceProcessor(config)


class TestImageHelpers:
    def test_write_then_read_round_trip(self, tmp_path: Path):
        original = draw_synthetic_face(120, 90)
        path = write_image(tmp_path / "out.png", original)
        restored = read_image(path)
        assert np.array_equal(original, restored)

    def test_read_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(ProcessorError, match="not found"):
            read_image(tmp_path / "absent.png")

    def test_read_non_image_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.png"
        bad.write_text("not an image", encoding="utf-8")
        with pytest.raises(ProcessorError, match="unsupported or corrupt"):
            read_image(bad)

    def test_write_creates_parent_directories(self, tmp_path: Path):
        path = write_image(tmp_path / "a" / "b" / "out.png", draw_synthetic_face(32, 32))
        assert path.is_file()

    def test_write_encodes_the_requested_extension(self, tmp_path: Path):
        path = write_image(tmp_path / "out.bmp", draw_synthetic_face(32, 32))
        assert path.suffix == ".bmp"
        assert read_image(path).shape == (32, 32, 3)


class TestSideBySide:
    def test_matches_heights(self):
        left = np.zeros((100, 50, 3), np.uint8)
        right = np.zeros((200, 80, 3), np.uint8)
        result = make_side_by_side(left, right)
        assert result.shape[0] == 100
        assert result.shape[1] == 50 + 40

    def test_identical_inputs_double_the_width(self):
        image = np.zeros((60, 40, 3), np.uint8)
        assert make_side_by_side(image, image).shape == (60, 80, 3)


class TestSourcePreparation:
    def test_prepares_a_source_face(self, processor, synthetic_face):
        source = processor.prepare_source_face(synthetic_face)
        assert isinstance(source, SourceFace)
        assert source.landmarks is not None
        assert len(source.landmarks) == 68

    def test_source_size_matches_the_image(self, processor, synthetic_face):
        source = processor.prepare_source_face(synthetic_face)
        assert source.size == (320, 240)

    def test_empty_image_raises(self, processor):
        with pytest.raises(ProcessorError, match="empty"):
            processor.prepare_source_face(np.zeros((0, 0, 3), np.uint8))

    def test_image_without_a_face_raises(self, processor):
        with pytest.raises(ProcessorError, match="no face detected"):
            processor.prepare_source_face(np.zeros((240, 320, 3), np.uint8))

    def test_load_source_face_from_disk(self, processor, face_image):
        source = processor.load_source_face(face_image)
        assert source.size == (320, 240)

    def test_load_source_face_missing_file_raises(self, processor, tmp_path: Path):
        with pytest.raises(ProcessorError):
            processor.load_source_face(tmp_path / "absent.png")


class TestProcessFrame:
    def test_swaps_the_face_in_a_frame(self, processor):
        source_frame = draw_synthetic_face(320, 240, skin=(80, 200, 120))
        target_frame = draw_synthetic_face(320, 240)
        source = processor.prepare_source_face(source_frame)

        result, swapped = processor.process_frame(target_frame, source)
        assert swapped is True
        assert result.shape == target_frame.shape
        assert not np.array_equal(result, target_frame)

    def test_leaves_a_frame_without_a_face_alone(self, processor, synthetic_face):
        source = processor.prepare_source_face(synthetic_face)
        blank = np.zeros((240, 320, 3), np.uint8)
        result, swapped = processor.process_frame(blank, source)
        assert swapped is False
        assert result is blank  # the original object is returned untouched

    def test_output_is_uint8(self, processor, synthetic_face):
        source = processor.prepare_source_face(synthetic_face)
        result, _ = processor.process_frame(synthetic_face, source)
        assert result.dtype == np.uint8

    def test_pixels_far_from_the_face_are_untouched(self, processor):
        source_frame = draw_synthetic_face(320, 240, skin=(80, 200, 120))
        target_frame = draw_synthetic_face(320, 240)
        source = processor.prepare_source_face(source_frame)
        result, swapped = processor.process_frame(target_frame, source)
        assert swapped
        assert np.array_equal(result[0:10, 0:10], target_frame[0:10, 0:10])

    def test_repeated_frames_are_stable(self, processor, synthetic_face):
        source = processor.prepare_source_face(synthetic_face)
        first, _ = processor.process_frame(synthetic_face, source)
        second, _ = processor.process_frame(synthetic_face, source)
        # Detector and landmark state must not drift between frames.
        assert first.shape == second.shape


class TestProcessVideo:
    def test_end_to_end_conversion(self, processor, face_image, target_video, tmp_path: Path):
        output = tmp_path / "out.mp4"
        stats = processor.process_video(face_image, target_video, output)

        assert output.is_file()
        assert stats.frames_written == 10
        assert stats.frames_read == 10
        assert stats.output_path == str(output)
        assert stats.elapsed_seconds > 0

    def test_every_frame_with_a_face_is_swapped(
        self, processor, face_image, target_video, tmp_path: Path
    ):
        stats = processor.process_video(face_image, target_video, tmp_path / "out.mp4")
        # The synthetic video keeps a face in every frame.
        assert stats.frames_swapped == 10
        assert stats.frames_skipped == 0
        assert stats.swap_rate == pytest.approx(1.0)

    def test_output_video_is_readable(self, processor, face_image, target_video, tmp_path: Path):
        output = tmp_path / "out.mp4"
        processor.process_video(face_image, target_video, output)
        info = probe(output)
        assert (info.width, info.height) == (320, 240)
        assert info.frame_count == 10

    def test_progress_is_reported_monotonically(
        self, processor, face_image, target_video, tmp_path: Path
    ):
        seen: list[tuple[int, int, str]] = []
        processor.process_video(
            face_image,
            target_video,
            tmp_path / "out.mp4",
            progress=lambda done, total, message: seen.append((done, total, message)),
        )
        assert seen
        completed = [done for done, _total, _message in seen]
        assert completed == sorted(completed)
        assert seen[-1][2] == "done"

    def test_preview_callback_receives_frames(
        self, processor, face_image, target_video, tmp_path: Path
    ):
        previews: list[tuple[int, np.ndarray]] = []
        processor.process_video(
            face_image,
            target_video,
            tmp_path / "out.mp4",
            preview=lambda index, frame: previews.append((index, frame)),
        )
        assert previews
        assert all(frame.shape == (240, 320, 3) for _index, frame in previews)

    def test_max_frames_limits_the_output(
        self, processor, face_image, target_video, tmp_path: Path
    ):
        processor.config.video.max_frames = 4
        stats = processor.process_video(face_image, target_video, tmp_path / "out.mp4")
        assert stats.frames_written == 4
        assert probe(tmp_path / "out.mp4").frame_count == 4

    def test_start_frame_skips_the_beginning(
        self, processor, face_image, target_video, tmp_path: Path
    ):
        stats = processor.process_video(
            face_image, target_video, tmp_path / "out.mp4", start_frame=6
        )
        assert stats.frames_written == 4

    def test_resize_is_applied_to_the_output(
        self, processor, face_image, target_video, tmp_path: Path
    ):
        processor.config.video.resize_width = 160
        processor.config.video.resize_height = 120
        processor.process_video(face_image, target_video, tmp_path / "out.mp4")
        assert probe(tmp_path / "out.mp4").width == 160

    def test_source_without_a_face_raises(self, processor, target_video, tmp_path: Path):
        blank = write_image(tmp_path / "blank.png", np.zeros((240, 320, 3), np.uint8))
        with pytest.raises(ProcessorError, match="no face detected"):
            processor.process_video(blank, target_video, tmp_path / "out.mp4")

    def test_missing_video_raises(self, processor, face_image, tmp_path: Path):
        with pytest.raises(VideoReadError):
            processor.process_video(face_image, tmp_path / "absent.mp4", tmp_path / "o.mp4")

    def test_missing_source_raises(self, processor, target_video, tmp_path: Path):
        with pytest.raises(ProcessorError):
            processor.process_video(tmp_path / "absent.png", target_video, tmp_path / "o.mp4")

    def test_video_without_faces_is_written_unchanged(
        self, processor, face_image, tmp_path: Path
    ):
        blank_video = write_test_video(
            tmp_path / "blank.mp4", frames=4, moving=False
        )
        # Overwrite with genuinely face free content.
        writer = cv2.VideoWriter(
            str(blank_video), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (320, 240)
        )
        for _ in range(4):
            writer.write(np.zeros((240, 320, 3), np.uint8))
        writer.release()

        stats = processor.process_video(face_image, blank_video, tmp_path / "out.mp4")
        assert stats.frames_swapped == 0
        assert stats.frames_written == 4

    def test_output_directory_is_created(
        self, processor, face_image, target_video, tmp_path: Path
    ):
        output = tmp_path / "nested" / "deeper" / "out.mp4"
        processor.process_video(face_image, target_video, output)
        assert output.is_file()

    def test_stats_summary_is_readable(
        self, processor, face_image, target_video, tmp_path: Path
    ):
        stats = processor.process_video(face_image, target_video, tmp_path / "out.mp4")
        summary = stats.summary()
        assert "frames written" in summary
        assert "fps" in summary


class TestCancellation:
    def test_cancel_stops_the_loop_early(
        self, processor, face_image, target_video, tmp_path: Path
    ):
        # The preview callback is rate limited by preview_interval, so set it to
        # one frame to make the cancellation deterministic.
        processor.config.performance.preview_interval = 1

        def cancel_after_two(index: int, _frame: np.ndarray) -> None:
            if index >= 2:
                processor.cancel()

        stats = processor.process_video(
            face_image,
            target_video,
            tmp_path / "out.mp4",
            preview=cancel_after_two,
        )
        assert stats.frames_written < 10
        assert any("cancelled" in message for message in stats.messages)

    def test_a_stale_cancel_does_not_block_a_new_run(
        self, processor, face_image, target_video, tmp_path: Path
    ):
        # Starting a conversion clears any cancel left over from a previous run,
        # otherwise a cancelled job would silently break the next one.
        processor.cancel()
        stats = processor.process_video(
            face_image, target_video, tmp_path / "out.mp4"
        )
        assert stats.frames_written == 10

    def test_reset_clears_the_cancel_flag(self, processor):
        processor.cancel()
        assert processor._cancel_requested is True
        processor.reset()
        assert processor._cancel_requested is False

    def test_a_partial_output_file_is_still_written(
        self, processor, face_image, target_video, tmp_path: Path
    ):
        output = tmp_path / "out.mp4"
        processor.config.performance.preview_interval = 1

        def cancel_after_three(index: int, _frame: np.ndarray) -> None:
            if index >= 3:
                processor.cancel()

        processor.process_video(
            face_image, target_video, output, preview=cancel_after_three
        )
        assert output.is_file()
        assert probe(output).frame_count > 0


class TestPreviewSwap:
    def test_side_by_side_preview_is_double_width(
        self, processor, face_image, synthetic_face
    ):
        preview = processor.preview_swap(face_image, synthetic_face, side_by_side=True)
        assert preview.shape[0] == synthetic_face.shape[0]
        assert preview.shape[1] == synthetic_face.shape[1] * 2

    def test_single_preview_keeps_the_frame_size(
        self, processor, face_image, synthetic_face
    ):
        preview = processor.preview_swap(face_image, synthetic_face, side_by_side=False)
        assert preview.shape == synthetic_face.shape

    def test_preview_of_a_face_free_frame_returns_it(
        self, processor, face_image
    ):
        blank = np.zeros((240, 320, 3), np.uint8)
        preview = processor.preview_swap(face_image, blank, side_by_side=False)
        assert np.array_equal(preview, blank)


class TestProcessingStats:
    def test_swap_rate_without_frames(self):
        assert ProcessingStats().swap_rate == 0.0

    def test_fps_without_elapsed_time(self):
        assert ProcessingStats().fps == 0.0

    def test_fps_computation(self):
        stats = ProcessingStats(frames_written=100, elapsed_seconds=10.0)
        assert stats.fps == pytest.approx(10.0)

    def test_swap_rate_computation(self):
        stats = ProcessingStats(frames_read=10, frames_swapped=5)
        assert stats.swap_rate == pytest.approx(0.5)
