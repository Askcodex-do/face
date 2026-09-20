"""Utility layer: configuration and logging."""

from .config import (
    APP_NAME,
    APP_VERSION,
    AppConfig,
    DetectionConfig,
    ProcessingConfig,
    VideoConfig,
)
from .logger import get_logger, setup_logging

__all__ = [
    "APP_NAME",
    "APP_VERSION",
    "AppConfig",
    "VideoConfig",
    "DetectionConfig",
    "ProcessingConfig",
    "setup_logging",
    "get_logger",
]
