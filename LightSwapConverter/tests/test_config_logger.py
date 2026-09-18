"""Tests for the configuration and logging utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from LightSwapConverter.utils.config import (
    AppConfig,
    BlendConfig,
    DetectionConfig,
    LoggingConfig,
    PerformanceConfig,
    TransformConfig,
    VideoConfig,
)
from LightSwapConverter.utils.logger import (
    MemoryHandler,
    configure_for_tests,
    get_logger,
    get_memory_handler,
    setup_logging,
)


class TestDefaults:
    def test_defaults_are_valid(self):
        assert AppConfig().validate() == []

    def test_every_section_validates_cleanly(self):
        sections = (
            VideoConfig(),
            DetectionConfig(),
            TransformConfig(),
            BlendConfig(),
            PerformanceConfig(),
            LoggingConfig(),
        )
        for section in sections:
            assert section.validate() == [], type(section).__name__

    def test_defaults_target_a_low_end_machine(self):
        config = AppConfig()
        assert config.performance.num_threads <= 4
        assert config.performance.max_frame_megapixels <= 3.0
        assert config.blend.seamless is False  # seamless cloning costs CPU
        assert config.video.codec == "mp4v"


class TestValidation:
    @pytest.mark.parametrize(
        "section,attribute,value,expected",
        [
            (VideoConfig(), "resize_width", -1, "resize_width"),
            (VideoConfig(), "resize_width", 640, "resize_width"),
            (VideoConfig(), "fallback_fps", 0.0, "fallback_fps"),
            (VideoConfig(), "codec", "mp4", "codec"),
            (DetectionConfig(), "scale_factor", 1.0, "scale_factor"),
            (DetectionConfig(), "min_neighbors", 0, "min_neighbors"),
            (DetectionConfig(), "min_face_ratio", 0.0, "min_face_ratio"),
            (DetectionConfig(), "padding", 1.5, "padding"),
            (TransformConfig(), "color_match", 1.5, "color_match"),
            (TransformConfig(), "brightness", 2.0, "brightness"),
            (BlendConfig(), "feather", 0.0, "feather"),
            (BlendConfig(), "opacity", -0.1, "opacity"),
            (PerformanceConfig(), "num_threads", 0, "num_threads"),
            (PerformanceConfig(), "progress_interval", 0, "progress_interval"),
            (LoggingConfig(), "level", "LOUD", "level"),
        ],
    )
    def test_invalid_values_are_reported(self, section, attribute, value, expected):
        setattr(section, attribute, value)
        errors = section.validate()
        assert any(expected in message for message in errors), errors

    def test_app_config_reports_the_failing_section(self):
        config = AppConfig()
        config.blend.opacity = 5.0
        errors = config.validate()
        assert len(errors) == 1
        assert errors[0].startswith("blend:")

    def test_resize_requires_both_dimensions(self):
        config = VideoConfig(resize_width=640, resize_height=0)
        assert any("both" in message for message in config.validate())


class TestSerialization:
    def test_round_trip_preserves_values(self):
        config = AppConfig()
        config.video.resize_width = 640
        config.video.resize_height = 360
        config.blend.seamless = True
        config.transform.color_match = 0.25
        config.last_source_face = "C:/faces/me.png"

        restored = AppConfig.from_dict(config.to_dict())
        assert restored == config

    def test_unknown_keys_are_ignored(self):
        data = AppConfig().to_dict()
        data["unknown_section"] = {"anything": 1}
        data["blend"]["unknown_option"] = 42
        restored = AppConfig.from_dict(data)
        assert restored.validate() == []

    def test_malformed_section_is_skipped(self):
        data = AppConfig().to_dict()
        data["video"] = "not a mapping"
        restored = AppConfig.from_dict(data)
        assert restored.video.codec == VideoConfig().codec

    def test_wrong_value_type_is_skipped(self):
        data = AppConfig().to_dict()
        data["detection"]["min_neighbors"] = "many"
        restored = AppConfig.from_dict(data)
        assert restored.detection.min_neighbors == DetectionConfig().min_neighbors

    def test_from_dict_tolerates_none_and_scalars(self):
        assert AppConfig.from_dict(None).validate() == []
        assert AppConfig.from_dict("nonsense").validate() == []  # type: ignore[arg-type]

    def test_numeric_strings_are_coerced(self):
        data = AppConfig().to_dict()
        data["video"]["max_frames"] = "25"
        assert AppConfig.from_dict(data).video.max_frames == 25

    def test_list_values_are_coerced(self):
        data = AppConfig().to_dict()
        data["detection"]["fallback_cascades"] = ["a.xml", "b.xml"]
        assert AppConfig.from_dict(data).detection.fallback_cascades == ["a.xml", "b.xml"]

    def test_save_and_load_round_trip(self, tmp_path: Path):
        path = tmp_path / "config.json"
        config = AppConfig()
        config.video.max_frames = 7
        config.save(path)
        assert json.loads(path.read_text(encoding="utf-8"))["video"]["max_frames"] == 7
        assert AppConfig.load(path).video.max_frames == 7

    def test_save_creates_parent_directories(self, tmp_path: Path):
        path = tmp_path / "nested" / "deeper" / "config.json"
        AppConfig().save(path)
        assert path.is_file()

    def test_load_missing_file_returns_defaults(self, tmp_path: Path):
        assert AppConfig.load(tmp_path / "absent.json").validate() == []

    def test_load_corrupt_file_returns_defaults(self, tmp_path: Path):
        path = tmp_path / "config.json"
        path.write_text("{not json", encoding="utf-8")
        assert AppConfig.load(path).validate() == []

    def test_save_leaves_no_temporary_file(self, tmp_path: Path):
        path = tmp_path / "config.json"
        AppConfig().save(path)
        leftovers = [item.name for item in tmp_path.iterdir() if item.suffix == ".tmp"]
        assert leftovers == []


class TestOutputSize:
    def test_output_size_without_resize(self):
        assert VideoConfig().output_size(1280, 720) == (1280, 720)

    def test_output_size_with_resize(self):
        config = VideoConfig(resize_width=640, resize_height=360)
        assert config.output_size(1920, 1080) == (640, 360)


class TestLogging:
    def test_setup_installs_a_memory_handler(self):
        setup_logging(LoggingConfig(file_name=""), stream=False)
        assert isinstance(get_memory_handler(), MemoryHandler)

    def test_records_capture_messages(self):
        logger = configure_for_tests()
        logger.info("hello from the test suite")
        lines = get_memory_handler().records()  # type: ignore[union-attr]
        assert any("hello from the test suite" in line for line in lines)

    def test_child_logger_is_namespaced(self):
        configure_for_tests()
        child = get_logger("core.video_reader")
        assert child.name == "lightswapconverter.core.video_reader"

    def test_repeated_setup_does_not_duplicate_handlers(self):
        for _ in range(3):
            setup_logging(LoggingConfig(file_name=""), stream=False)
        logger = logging.getLogger("lightswapconverter")
        assert len(logger.handlers) == 1

    def test_level_is_applied(self):
        setup_logging(LoggingConfig(level="WARNING", file_name=""), stream=False)
        assert logging.getLogger("lightswapconverter").level == logging.WARNING

    def test_memory_handler_is_bounded(self):
        handler = MemoryHandler(capacity=5)
        record = logging.LogRecord(
            "t", logging.INFO, __file__, 1, "line %d", (0,), None
        )
        for index in range(20):
            record = logging.LogRecord(
                "t", logging.INFO, __file__, 1, "line %d", (index,), None
            )
            handler.emit(record)
        assert len(handler.records()) == 5
        assert "line 19" in handler.records()[-1]

    def test_drain_empties_the_buffer(self):
        handler = MemoryHandler()
        handler.emit(logging.LogRecord("t", logging.INFO, __file__, 1, "x", (), None))
        assert handler.drain()
        assert handler.records() == []

    def test_file_handler_writes_when_a_path_is_given(self, tmp_path: Path):
        log_file = tmp_path / "app.log"
        logger = setup_logging(
            LoggingConfig(level="INFO", file_name="app.log"),
            log_file=log_file,
            stream=False,
        )
        logger.info("written to disk")
        for handler in logger.handlers:
            handler.flush()
        assert log_file.is_file()
        assert "written to disk" in log_file.read_text(encoding="utf-8")

    def test_unwritable_log_directory_does_not_raise(self, tmp_path: Path):
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        logger = setup_logging(
            LoggingConfig(file_name="app.log"),
            log_file=blocker / "nested" / "app.log",
            stream=False,
        )
        assert logger is not None
