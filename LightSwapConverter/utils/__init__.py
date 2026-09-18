"""Utility helpers shared across LightSwapConverter.

Exports are resolved lazily so importing a single helper never pulls in the
whole package, which keeps startup fast on low end machines.
"""

from __future__ import annotations

__all__ = [
    "AppConfig",
    "BlendConfig",
    "DetectionConfig",
    "LandmarkConfig",
    "LoggingConfig",
    "PerformanceConfig",
    "TransformConfig",
    "VideoConfig",
    "configure_for_tests",
    "default_config_path",
    "default_log_path",
    "get_logger",
    "get_memory_handler",
    "setup_logging",
]


def __getattr__(name: str):
    if name in {
        "AppConfig",
        "BlendConfig",
        "DetectionConfig",
        "LandmarkConfig",
        "LoggingConfig",
        "PerformanceConfig",
        "TransformConfig",
        "VideoConfig",
        "default_config_path",
    }:
        from . import config as _config

        return getattr(_config, name)
    if name in {
        "configure_for_tests",
        "default_log_path",
        "get_logger",
        "get_memory_handler",
        "setup_logging",
    }:
        from . import logger as _logger

        return getattr(_logger, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
