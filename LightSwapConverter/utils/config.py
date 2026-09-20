"""Application configuration for LightSwapConverter.

The whole configuration is a single JSON file so the application stays fully
offline and dependency free. Defaults are tuned for low-end machines:
Windows 8.1 x64, Python 3.10, 2 GB RAM, CPU only.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

APP_NAME = "LightSwapConverter"
APP_VERSION = "0.1.0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"

MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = PROJECT_ROOT / "logs"


@dataclass
class VideoConfig:
    """Video decoding/encoding parameters kept deliberately conservative."""

    codec: str = "mp4v"
    container: str = "mp4"
    fps_fallback: float = 25.0
    max_width: int = 960
    max_height: int = 540
    resize_if_larger: bool = True


@dataclass
class DetectionConfig:
    """Face detection tuning. Small images and no GPU are assumed."""

    backend: str = "haar"
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_size: int = 48
    max_faces: int = 2
    smooth_window: int = 5


@dataclass
class ProcessingConfig:
    """Frame pipeline behaviour."""

    frame_skip: int = 1
    keep_audio: bool = False
    queue_size: int = 4
    max_frames_in_memory: int = 8
    preview_enabled: bool = True
    mask_feather: int = 21
    blend_strength: float = 0.85
    color_match: bool = True


@dataclass
class AppConfig:
    """Top level configuration object."""

    app_name: str = APP_NAME
    version: str = APP_VERSION
    video: VideoConfig = field(default_factory=VideoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    models_dir: str = str(MODELS_DIR)
    assets_dir: str = str(ASSETS_DIR)
    output_dir: str = str(OUTPUT_DIR)
    log_dir: str = str(LOG_DIR)
    log_level: str = "INFO"

    # ------------------------------------------------------------------ paths
    def models_path(self) -> Path:
        return Path(self.models_dir)

    def assets_path(self) -> Path:
        return Path(self.assets_dir)

    def output_path(self) -> Path:
        return Path(self.output_dir)

    def log_path(self) -> Path:
        return Path(self.log_dir)

    def ensure_directories(self) -> None:
        """Create the runtime directories if they are missing."""
        for path in (self.models_path(), self.assets_path(), self.output_path(), self.log_path()):
            path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- (de)serial
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        """Build a config from a dict, ignoring unknown keys.

        Nested sections are merged over the defaults so a partial or outdated
        JSON file never breaks startup.
        """
        known = {f.name: f for f in fields(cls)}
        kwargs: Dict[str, Any] = {}

        for name, value in data.items():
            if name not in known:
                continue
            if name == "video" and isinstance(value, dict):
                kwargs[name] = _merge_section(VideoConfig, value)
            elif name == "detection" and isinstance(value, dict):
                kwargs[name] = _merge_section(DetectionConfig, value)
            elif name == "processing" and isinstance(value, dict):
                kwargs[name] = _merge_section(ProcessingConfig, value)
            else:
                kwargs[name] = value

        return cls(**kwargs)

    @classmethod
    def load(cls, path: Optional[os.PathLike | str] = None) -> "AppConfig":
        """Load configuration from disk, falling back to defaults."""
        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not config_path.is_file():
            return cls()
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return cls()
        if not isinstance(data, dict):
            return cls()
        return cls.from_dict(data)

    def save(self, path: Optional[os.PathLike | str] = None) -> Path:
        """Persist the configuration as pretty printed UTF-8 JSON."""
        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = config_path.with_suffix(config_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
        os.replace(tmp_path, config_path)
        return config_path


def _merge_section(section_cls: type, values: Dict[str, Any]) -> Any:
    """Instantiate a nested config dataclass from a possibly partial dict."""
    allowed = {f.name for f in fields(section_cls)}
    return section_cls(**{k: v for k, v in values.items() if k in allowed})
