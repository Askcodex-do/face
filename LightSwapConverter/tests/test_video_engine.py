"""Phase 1 tests: the lightweight video processing engine.

Every test builds its own media with OpenCV, so the suite runs offline and needs
no sample files. No stage is mocked: real ``VideoCapture`` and ``VideoWriter``
objects are exercised, including the failure and cancellation paths.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from LightSwapConverter.core.processor import (
    CancelledError,
    ProcessingStats,
    ProcessorError,
    VideoProcessor,
    identity_transform,
    make_side_by_side,
    read_image,
    write_image,
)
from LightSwapConverter.core.video_reader import VideoReadError, probe
from LightSwapConverter.core.video_writer import VideoWriteError
from LightSwapConverter.utils.config import AppConfig, VideoConfig

from tests.helpers import draw_synthetic_face, write_test_video


def grey(frame: np.ndarray, index: int) -> np.ndarray:
    """A real transform: convert to grey and back, so the frame stays 3-channel."""
    return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


@pytest.fixture
def engine() -> VideoProcessor:
    return VideoProcessor(AppConfig())


@pytest.fixture
def input_video(tmp_path: Path) -> Path:
    return write_test_video(tmp_path / "in.mp4", frames=10)


class TestMetadata:
    """Requirement 8: width, height, fps, frame count and duration."""

    def test_all_metadata_is_reported(self, engine, input_video, tmp_path):
        stats = engine.process_video(input_video, tmp_path / "out.mp4")
        assert stats.width == 320
        assert stats.height == 240
        assert stats.fps == pytest.approx(12.0, abs=0.5)
        assert stats.duration_seconds == pytest.approx(10 / 12.0, abs=0.1)
        assert stats.frames_written == 10

    def test_inspect_returns_info_without_writing(self, engine, input_video):
        info = engine.inspect(input_video)
        assert (info.width, info.height) == (320, 240)
        assert info.frame_count == 10
        assert info.fps == pytest.approx(12.0, abs=0.5)
        assert info.duration_seconds == pytest.approx(10 / 12.0, abs=0.1)

    def test_duration_matches_frame_count_over_fps(self, engine, input_video):
        info = engine.inspect(input_video)
        assert info.duration_seconds == pytest.approx(
            info.frame_count / info.fps, abs=1e-6
        )


class TestFrameByFrameProcessing:
    """Requirements 6, 7 and 11: one frame at a time, with progress."""

    def test_transform_runs_once_per_frame(self, engine, input_video, tmp_path):
        seen: list[int] = []

        def record(frame: np.ndarray, index: int) -> np.ndarray:
            seen.append(index)
            return frame

        stats = engine.process_video(input_video, tmp_path / "out.mp4", record)
        assert seen == list(range(10))
        assert stats.frames_processed == 10

    def test_the_transform_actually_changes_the_output(self, engine, input_video, tmp_path):
        output = tmp_path / "out.mp4"
        engine.process_video(input_video, output, grey)

        capture = cv2.VideoCapture(str(output))
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        assert ok
        # Grey-and-back leaves the three channels equal. Lossy encoding allows a
        # few levels of drift, so compare with a small tolerance.
        assert np.abs(frame[:, :, 0].astype(int) - frame[:, :, 1]).max() <= 4
        assert np.abs(frame[:, :, 1].astype(int) - frame[:, :, 2]).max() <= 4

    def test_progress_is_monotonic_and_ends_with_done(self, engine, input_video, tmp_path):
        seen: list[tuple[int, int, str]] = []
        engine.process_video(
            input_video,
            tmp_path / "out.mp4",
            progress=lambda done, total, msg: seen.append((done, total, msg)),
        )
        assert seen
        assert [done for done, _, _ in seen] == sorted(done for done, _, _ in seen)
        assert seen[-1][2] == "done"
        assert seen[-1][0] == 10

    def test_progress_reports_the_total(self, engine, input_video, tmp_path):
        totals: list[int] = []
        engine.process_video(
            input_video,
            tmp_path / "out.mp4",
            progress=lambda done, total, msg: totals.append(total),
        )
        assert totals[0] == 10

    def test_preview_receives_frames(self, engine, input_video, tmp_path):
        previews: list[int] = []
        engine.process_video(
            input_video,
            tmp_path / "out.mp4",
            preview=lambda index, frame: previews.append(index),
        )
        assert previews
        assert all(isinstance(index, int) for index in previews)

    def test_max_frames_limits_the_run(self, engine, input_video, tmp_path):
        output = tmp_path / "out.mp4"
        stats = engine.process_video(input_video, output, max_frames=4)
        assert stats.frames_written == 4
        assert probe(output).frame_count == 4

    def test_start_frame_skips_the_beginning(self, engine, input_video, tmp_path):
        stats = engine.process_video(
            input_video, tmp_path / "out.mp4", start_frame=7
        )
        assert stats.frames_written == 3

    def test_config_max_frames_is_honoured(self, input_video, tmp_path):
        config = AppConfig()
        config.video.max_frames = 5
        stats = VideoProcessor(config).process_video(input_video, tmp_path / "out.mp4")
        assert stats.frames_written == 5

    def test_resize_config_scales_the_output(self, input_video, tmp_path):
        config = AppConfig()
        config.video.resize_width = 160
        config.video.resize_height = 120
        VideoProcessor(config).process_video(input_video, tmp_path / "out.mp4")
        assert probe(tmp_path / "out.mp4").width == 160


class TestNoMemoryAccumulation:
    """Requirement 6: never load the whole video into memory."""

    def test_engine_keeps_no_frame_buffer(self, engine, input_video, tmp_path):
        engine.process_video(input_video, tmp_path / "out.mp4")
        # Only the config and the cancel flag may survive a run. No list of
        # frames, no cache, no queue.
        attributes = {name: type(value).__name__ for name, value in vars(engine).items()}
        assert attributes == {"config": "AppConfig", "_cancelled": "bool"}

    def test_a_long_video_does_not_grow_the_engine(self, engine, tmp_path):
        long_video = write_test_video(tmp_path / "long.mp4", frames=40)
        before = len(gc.get_objects())
        engine.process_video(long_video, tmp_path / "out.mp4")
        gc.collect()
        after = len(gc.get_objects())
        # A frame buffer per frame would show up as tens of thousands of extra
        # objects. Allow generous slack for interpreter noise.
        assert after - before < 20000

    def test_message_list_stays_bounded_when_every_frame_fails(
        self, engine, input_video, tmp_path
    ):
        def always_fails(frame: np.ndarray, index: int) -> np.ndarray:
            raise ValueError("nope")

        stats = engine.process_video(input_video, tmp_path / "out.mp4", always_fails)
        assert stats.errors == 10
        assert len(stats.messages) <= 200


class TestErrorHandling:
    """Requirements 10 and 13: clean errors, and no leaked handles."""

    def test_missing_input_raises(self, engine, tmp_path):
        with pytest.raises(VideoReadError, match="not found"):
            engine.process_video(tmp_path / "absent.mp4", tmp_path / "out.mp4")

    def test_corrupt_input_raises(self, engine, tmp_path):
        bad = tmp_path / "bad.mp4"
        bad.write_text("definitely not a video", encoding="utf-8")
        with pytest.raises(VideoReadError):
            engine.process_video(bad, tmp_path / "out.mp4")

    def test_a_failing_transform_does_not_abort_the_run(self, engine, input_video, tmp_path):
        def fails_on_one_frame(frame: np.ndarray, index: int) -> np.ndarray:
            if index == 5:
                raise RuntimeError("boom")
            return frame

        output = tmp_path / "out.mp4"
        stats = engine.process_video(input_video, output, fails_on_one_frame)
        assert stats.errors == 1
        assert stats.frames_written == 10
        assert probe(output).frame_count == 10

    def test_a_failing_transform_is_reported(self, engine, input_video, tmp_path):
        reported: list[tuple[int, str]] = []

        def always_fails(frame: np.ndarray, index: int) -> np.ndarray:
            raise ValueError("boom")

        engine.process_video(
            input_video,
            tmp_path / "out.mp4",
            always_fails,
            on_error=lambda index, message: reported.append((index, message)),
        )
        assert reported
        assert "boom" in reported[0][1]

    def test_a_transform_returning_a_non_array_is_handled(self, engine, input_video, tmp_path):
        stats = engine.process_video(
            input_video, tmp_path / "out.mp4", lambda frame, index: "nonsense"
        )
        assert stats.errors == 10
        # The original frames are still written, so the output is complete.
        assert stats.frames_written == 10

    def test_returning_none_passes_the_frame_through(self, engine, input_video, tmp_path):
        stats = engine.process_video(
            input_video, tmp_path / "out.mp4", lambda frame, index: None
        )
        assert stats.frames_passed_through == 10
        assert stats.frames_transformed == 0
        assert stats.frames_written == 10

    def test_unwritable_output_raises(self, engine, input_video, tmp_path):
        config = AppConfig()
        config.video = VideoConfig(codec="zzzz")
        # Start-up failures surface as the precise writer error, which names the
        # codec that was refused.
        with pytest.raises(VideoWriteError, match="cannot create"):
            VideoProcessor(config).process_video(input_video, tmp_path / "out.mp4")

    def test_reader_is_released_after_a_failed_run(self, engine, tmp_path):
        # On Linux an unreleased capture keeps the file open; count the handles.
        video = write_test_video(tmp_path / "in.mp4", frames=3)
        before = _open_fd_count()
        for _ in range(5):
            with pytest.raises(VideoReadError):
                engine.process_video(tmp_path / "absent.mp4", tmp_path / "out.mp4")
        assert _open_fd_count() <= before + 1
        assert video.exists()

    def test_repeated_runs_do_not_leak_file_handles(self, engine, input_video, tmp_path):
        before = _open_fd_count()
        for run in range(5):
            engine.process_video(input_video, tmp_path / f"out{run}.mp4")
        assert _open_fd_count() <= before + 2


class TestCancellation:
    """Requirement 12: processing can be stopped safely."""

    def test_cancel_stops_the_loop(self, engine, input_video, tmp_path):
        output = tmp_path / "out.mp4"
        engine.config.performance.preview_interval = 1

        def cancel_early(index: int, frame: np.ndarray) -> None:
            if index >= 3:
                engine.cancel()

        stats = engine.process_video(
            input_video, output, preview=cancel_early
        )
        assert stats.cancelled is True
        assert stats.frames_written < 10

    def test_cancelled_output_is_still_playable(self, engine, input_video, tmp_path):
        output = tmp_path / "out.mp4"
        engine.config.performance.preview_interval = 1

        def cancel_early(index: int, frame: np.ndarray) -> None:
            if index >= 4:
                engine.cancel()

        engine.process_video(input_video, output, preview=cancel_early)
        info = probe(output)
        assert 0 < info.frame_count < 10

    def test_cancel_is_reported_in_the_messages(self, engine, input_video, tmp_path):
        engine.config.performance.preview_interval = 1

        def cancel_early(index: int, frame: np.ndarray) -> None:
            if index >= 2:
                engine.cancel()

        stats = engine.process_video(
            input_video, tmp_path / "out.mp4", preview=cancel_early
        )
        assert any("cancelled" in message for message in stats.messages)

    def test_progress_final_message_says_cancelled(self, engine, input_video, tmp_path):
        engine.config.performance.preview_interval = 1
        messages: list[str] = []

        def cancel_early(index: int, frame: np.ndarray) -> None:
            if index >= 2:
                engine.cancel()

        engine.process_video(
            input_video,
            tmp_path / "out.mp4",
            preview=cancel_early,
            progress=lambda done, total, msg: messages.append(msg),
        )
        assert messages[-1] == "cancelled"

    def test_reset_clears_a_stale_cancel(self, engine, input_video, tmp_path):
        engine.cancel()
        assert engine.is_cancelled is True
        stats = engine.process_video(input_video, tmp_path / "out.mp4")
        # process_video resets first, so a stale cancel cannot break a new run.
        assert stats.frames_written == 10
        assert stats.cancelled is False

    def test_cancel_from_the_transform_stops_immediately(
        self, engine, input_video, tmp_path
    ):
        # Cancelling from inside the transform is the tightest possible timing:
        # the flag is set before the engine re-checks it.
        calls: list[int] = []

        def cancel_on_first(frame: np.ndarray, index: int) -> np.ndarray:
            calls.append(index)
            engine.cancel()
            return frame

        stats = engine.process_video(input_video, tmp_path / "out.mp4", cancel_on_first)
        assert calls == [0]
        assert stats.cancelled is True
        # The frame in flight is still written, so the output is not truncated
        # mid-frame.
        assert stats.frames_written == 1

    def test_cancelled_error_is_available_for_transforms(self):
        # Documented in the transform contract, so it must exist and be an
        # Exception subclass.
        assert issubclass(CancelledError, Exception)


class TestFormats:
    """Requirement 9: MP4, AVI and MKV."""

    @pytest.mark.parametrize("extension", [".mp4", ".avi", ".mkv"])
    def test_round_trip_in_each_container(self, engine, tmp_path, extension):
        source = write_test_video(tmp_path / f"in{extension}", frames=6)
        output = tmp_path / f"out{extension}"
        stats = engine.process_video(source, output, grey)
        assert stats.frames_written == 6
        assert output.is_file() and output.stat().st_size > 0

    @pytest.mark.parametrize("extension", [".mp4", ".avi", ".mkv"])
    def test_output_is_readable_in_each_container(self, engine, tmp_path, extension):
        source = write_test_video(tmp_path / f"in{extension}", frames=6)
        output = tmp_path / f"out{extension}"
        engine.process_video(source, output)
        info = probe(output)
        assert info.frame_count == 6
        assert (info.width, info.height) == (320, 240)

    def test_unknown_extension_is_rejected_clearly(self, engine, input_video, tmp_path):
        # OpenCV cannot infer a container from an unknown extension, so it
        # refuses the writer. The error must name the file.
        with pytest.raises(VideoWriteError, match="cannot create"):
            engine.process_video(input_video, tmp_path / "out.zzz")


class TestResourceCleanup:
    """Requirement 13: VideoCapture and VideoWriter are released."""

    def test_reader_and_writer_are_closed_after_success(self, engine, input_video, tmp_path):
        output = tmp_path / "out.mp4"
        engine.process_video(input_video, output)
        # If the writer were still open, the file could not be removed on
        # Windows; on Linux, assert it is finalised and readable instead.
        assert probe(output).frame_count == 10

    def test_output_file_is_not_locked_after_a_failed_transform(
        self, engine, input_video, tmp_path
    ):
        output = tmp_path / "out.mp4"

        def always_fails(frame: np.ndarray, index: int) -> np.ndarray:
            raise RuntimeError("boom")

        engine.process_video(input_video, output, always_fails)
        assert output.is_file()

    def test_context_free_api_leaves_no_open_capture(self, engine, input_video, tmp_path):
        before = _open_fd_count()
        engine.process_video(input_video, tmp_path / "out.mp4")
        assert _open_fd_count() <= before + 1

    def test_identity_transform_is_a_no_op(self, engine, input_video, tmp_path):
        output = tmp_path / "out.mp4"
        stats = engine.process_video(input_video, output, identity_transform)
        assert stats.frames_written == 10
        assert stats.frames_transformed == 10


class TestImageHelpers:
    def test_image_round_trip(self, tmp_path):
        original = draw_synthetic_face(120, 90)
        path = write_image(tmp_path / "img.png", original)
        assert np.array_equal(read_image(path), original)

    def test_read_missing_image_raises(self, tmp_path):
        with pytest.raises(ProcessorError, match="not found"):
            read_image(tmp_path / "absent.png")

    def test_read_corrupt_image_raises(self, tmp_path):
        bad = tmp_path / "bad.png"
        bad.write_text("not an image", encoding="utf-8")
        with pytest.raises(ProcessorError, match="corrupt"):
            read_image(bad)

    def test_side_by_side_doubles_the_width(self):
        image = np.zeros((60, 40, 3), np.uint8)
        assert make_side_by_side(image, image).shape == (60, 80, 3)

    def test_side_by_side_matches_heights(self):
        left = np.zeros((100, 50, 3), np.uint8)
        right = np.zeros((200, 80, 3), np.uint8)
        assert make_side_by_side(left, right).shape[0] == 100


class TestStats:
    def test_empty_stats_are_safe(self):
        stats = ProcessingStats()
        assert stats.transform_rate == 0.0
        assert stats.processing_fps == 0.0
        assert stats.frames_processed == 0
        assert stats.completed is True

    def test_transform_rate(self):
        stats = ProcessingStats(frames_transformed=3, frames_passed_through=1)
        assert stats.transform_rate == pytest.approx(0.75)

    def test_processing_fps(self):
        stats = ProcessingStats(frames_written=100, elapsed_seconds=10.0)
        assert stats.processing_fps == pytest.approx(10.0)

    def test_summary_is_readable(self):
        stats = ProcessingStats(frames_written=5, elapsed_seconds=1.0)
        summary = stats.summary()
        assert "5 frames written" in summary
        assert "fps" in summary

    def test_messages_are_capped(self):
        stats = ProcessingStats()
        for index in range(500):
            stats.add_message(f"message {index}")
        assert len(stats.messages) == 200


def _open_fd_count() -> int:
    """Number of file descriptors this process has open, or a large number.

    Linux only; the suite is developed there. Returns a sentinel on platforms
    without ``/proc`` so the assertion stays permissive rather than flaky.
    """
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return 0
