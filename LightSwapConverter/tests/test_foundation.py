"""Foundation tests: config, logging, geometry helpers and video round trip."""

from __future__ import annotations

from utils.config import AppConfig, DetectionConfig
from utils.logger import get_logger, setup_logging


# --------------------------------------------------------------------- config
def test_config_roundtrip(tmp_path):
    path = tmp_path / "config.json"
    config = AppConfig()
    config.processing.frame_skip = 3
    config.save(path)

    loaded = AppConfig.load(path)
    assert loaded.processing.frame_skip == 3
    assert loaded.video.codec == config.video.codec


def test_config_ignores_unknown_and_partial_keys(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        '{"version": "9.9", "unknown_section": {"a": 1}, "detection": {"min_size": 64}}',
        encoding="utf-8",
    )

    loaded = AppConfig.load(path)
    assert loaded.version == "9.9"
    assert loaded.detection.min_size == 64
    # Unspecified fields keep their defaults.
    assert loaded.detection.max_faces == DetectionConfig().max_faces


def test_config_missing_file_returns_defaults(tmp_path):
    loaded = AppConfig.load(tmp_path / "does_not_exist.json")
    assert loaded.app_name == AppConfig().app_name


def test_config_corrupt_file_returns_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")
    assert AppConfig.load(path).video.codec == AppConfig().video.codec


def test_ensure_directories(tmp_path):
    config = AppConfig(
        models_dir=str(tmp_path / "models"),
        assets_dir=str(tmp_path / "assets"),
        output_dir=str(tmp_path / "output"),
        log_dir=str(tmp_path / "logs"),
    )
    config.ensure_directories()
    assert config.output_path().is_dir()
    assert config.log_path().is_dir()


# --------------------------------------------------------------------- logger
def test_logging_is_idempotent(tmp_path):
    first = setup_logging(tmp_path, "DEBUG")
    second = setup_logging(tmp_path, "DEBUG")
    assert first is second
    assert get_logger("core.processor").name == "lightswapconverter.core.processor"

    get_logger("tests").info("hello from the test suite")
    assert (tmp_path / "lightswapconverter.log").is_file()
