"""Application configuration for LightSwapConverter.

Configuration is stored as a single JSON file so the application stays fully
offline and dependency free. The module exposes small dataclasses instead of a
loose dictionary so that every option has a documented type, a default and a
validation rule.

The target machine is a 2 GB RAM, CPU only Windows 8.1 box, therefore every
default here favours low memory usage and predictable run times over quality.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
ASSETS_DIR = PACKAGE_ROOT / "assets"
MODELS_DIR = PACKAGE_ROOT / "models"
DEFAULT_CONFIG_NAME = "config.json"


def default_config_path() -> Path:
    """Return the per-user location of the configuration file.

    On Windows this is ``%APPDATA%\\LightSwapConverter\\config.json``.  If
    ``APPDATA`` is missing (portable install, Linux development machine) we fall
    back to a hidden folder inside the user's home directory.
    """
    appdata = os.environ.get("APPDATA")
    base = Path(appdata) if appdata else Path.home() / ".lightswapconverter"
    return base / "LightSwapConverter" / DEFAULT_CONFIG_NAME


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclass
class VideoConfig:
    """Options describing how video is read and written."""

    #: Frames are decoded at native resolution; a value <= 0 disables resizing.
    resize_width: int = 0
    resize_height: int = 0
    #: Frames per second of the produced file when the source is unreadable.
    fallback_fps: float = 25.0
    #: OpenCV four character code for the output file.
    codec: str = "mp4v"
    #: Constant rate factor used by the optional ffmpeg audio mux step.
    quality: int = 23
    #: Copy the source audio track into the result when ffmpeg is available.
    copy_audio: bool = True
    #: Frames actually written are capped by this value (0 means no limit).
    max_frames: int = 0

    def output_size(self, width: int, height: int) -> tuple[int, int]:
        """Return the output ``(width, height)`` for a source frame size."""
        target_w = self.resize_width if self.resize_width > 0 else width
        target_h = self.resize_height if self.resize_height > 0 else height
        return target_w, target_h

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.resize_width < 0 or self.resize_height < 0:
            errors.append("resize_width/resize_height must be >= 0")
        if (self.resize_width > 0) != (self.resize_height > 0):
            errors.append(
                "resize_width and resize_height must both be set, or both be 0"
            )
        if not 0 < self.fallback_fps <= 240:
            errors.append("fallback_fps must be within (0, 240]")
        if len(self.codec) != 4:
            errors.append("codec must be a four character code, e.g. 'mp4v'")
        if self.max_frames < 0:
            errors.append("max_frames must be >= 0")
        return errors


@dataclass
class DetectionConfig:
    """Options for the face detector stage."""

    #: Cascade file resolved from the OpenCV installation when left empty.
    cascade_file: str = "haarcascade_frontalface_default.xml"
    #: Extra cascades used as a fallback when the primary one finds nothing.
    fallback_cascades: List[str] = field(
        default_factory=lambda: [
            "haarcascade_frontalface_alt2.xml",
            "haarcascade_profileface.xml",
        ]
    )
    scale_factor: float = 1.1
    min_neighbors: int = 5
    #: Faces smaller than this fraction of the frame height are ignored.
    min_face_ratio: float = 0.04
    #: Search a region of interest around the previous frame first. This keeps
    #: CPU cost low on video, where a face rarely moves far between frames.
    use_tracking: bool = True
    #: Relative padding added around the raw detection box.
    padding: float = 0.18

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.scale_factor <= 1.0:
            errors.append("scale_factor must be > 1.0")
        if self.min_neighbors < 1:
            errors.append("min_neighbors must be >= 1")
        if not 0.0 < self.min_face_ratio < 1.0:
            errors.append("min_face_ratio must be within (0, 1)")
        if not 0.0 <= self.padding <= 1.0:
            errors.append("padding must be within [0, 1]")
        return errors


@dataclass
class LandmarkConfig:
    """Options for the landmark estimation stage."""

    #: Number of landmarks produced. Only the 68 point layout is supported.
    num_points: int = 68
    #: Smooth landmark positions across frames to reduce jitter.
    smoothing: float = 0.6
    #: Pixels of blur applied before the eye/eyebrow heuristics.
    blur_radius: int = 3
    #: How far refinement may move a single landmark from its prior position, as a
    #: fraction of the face box. Refinement is a small correction to a good
    #: prior, so this bounds how much a confidently wrong measurement can
    #: distort the layout. Each point is capped on its own, which keeps the
    #: shape of the layout intact. Set to 0 to use the prior unchanged.
    max_refine_shift: float = 0.06

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.num_points not in (5, 68):
            errors.append("num_points must be 5 or 68")
        if not 0.0 <= self.smoothing < 1.0:
            errors.append("smoothing must be within [0, 1)")
        if self.blur_radius < 0:
            errors.append("blur_radius must be >= 0")
        if not 0.0 <= self.max_refine_shift <= 0.5:
            errors.append("max_refine_shift must be within [0, 0.5]")
        return errors


@dataclass
class TransformConfig:
    """Options controlling how the source face is warped onto the target."""

    #: Colour transfer strength, 0 disables histogram matching.
    color_match: float = 0.7
    #: Unsharp amount applied to the warped patch.
    sharpen: float = 0.25
    #: Random horizontal flips are pointless for a converter, keep deterministic.
    flip_source: bool = False
    #: Brightness correction applied after warping, in the range [-1, 1].
    brightness: float = 0.0
    #: How far the face may be stretched before the warp switches from bilinear
    #: to cubic interpolation. Bilinear is kept for near 1:1 swaps because it is
    #: the cheapest and loses nothing there; above this the extra sharpness of
    #: cubic is worth the cost. Set to 0 to always use bilinear.
    cubic_stretch_threshold: float = 1.5

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not 0.0 <= self.color_match <= 1.0:
            errors.append("color_match must be within [0, 1]")
        if not 0.0 <= self.sharpen <= 2.0:
            errors.append("sharpen must be within [0, 2]")
        if not -1.0 <= self.brightness <= 1.0:
            errors.append("brightness must be within [-1, 1]")
        if self.cubic_stretch_threshold < 0.0:
            errors.append("cubic_stretch_threshold must be >= 0")
        return errors


@dataclass
class BlendConfig:
    """Options for compositing the warped face back onto the frame."""

    #: Feather width of the mask, as a fraction of the face box size.
    feather: float = 0.35
    #: Overall opacity of the pasted face.
    opacity: float = 1.0
    #: Poisson blending removes visible seams but costs extra CPU.
    seamless: bool = False
    #: Restrict the mask to the convex hull of the face landmarks.
    use_hull_mask: bool = True

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not 0.0 < self.feather <= 1.0:
            errors.append("feather must be within (0, 1]")
        if not 0.0 <= self.opacity <= 1.0:
            errors.append("opacity must be within [0, 1]")
        return errors


@dataclass
class PerformanceConfig:
    """Resource guards for low memory machines."""

    #: Worker threads handed to OpenCV. The target is a CPU only machine.
    num_threads: int = 2
    #: Emit a progress callback every N frames.
    progress_interval: int = 10
    #: Abort when the estimated memory use of one frame exceeds this value.
    max_frame_megapixels: float = 2.1
    #: Keep the preview widget updated at most this often, in frames.
    preview_interval: int = 15

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.num_threads < 1:
            errors.append("num_threads must be >= 1")
        if self.progress_interval < 1:
            errors.append("progress_interval must be >= 1")
        if self.max_frame_megapixels <= 0:
            errors.append("max_frame_megapixels must be > 0")
        if self.preview_interval < 1:
            errors.append("preview_interval must be >= 1")
        return errors


@dataclass
class LoggingConfig:
    """Options for the application logger."""

    level: str = "INFO"
    #: Log file name inside the log directory, empty disables file logging.
    file_name: str = "lightswapconverter.log"
    #: Maximum size of a single log file before it is rotated.
    max_bytes: int = 512 * 1024
    backup_count: int = 2

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.level.upper() not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            errors.append("level must be a standard logging level name")
        if self.max_bytes <= 0:
            errors.append("max_bytes must be > 0")
        if self.backup_count < 0:
            errors.append("backup_count must be >= 0")
        return errors


# ---------------------------------------------------------------------------
# Root configuration object
# ---------------------------------------------------------------------------


@dataclass
class AppConfig:
    """Aggregate configuration for the whole application."""

    video: VideoConfig = field(default_factory=VideoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    landmarks: LandmarkConfig = field(default_factory=LandmarkConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    blend: BlendConfig = field(default_factory=BlendConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    #: Remembered locations of the last used media files.
    last_source_video: str = ""
    last_source_face: str = ""
    last_output_video: str = ""

    # -- validation ---------------------------------------------------------

    def validate(self) -> List[str]:
        """Return a list of human readable problems; empty means valid."""
        errors: List[str] = []
        for section in fields(self):
            value = getattr(self, section.name)
            validate = getattr(value, "validate", None)
            if validate is None:
                continue
            errors.extend(
                f"{section.name}: {message}" for message in validate()
            )
        return errors

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AppConfig":
        """Build a config from a mapping, ignoring unknown or malformed keys.

        Unknown keys are dropped so an older configuration file never prevents
        the application from starting after an upgrade.
        """
        config = cls()
        if not isinstance(data, dict):
            return config
        for section in fields(cls):
            raw = data.get(section.name)
            current = getattr(config, section.name)
            if raw is None:
                continue
            if isinstance(current, (VideoConfig, DetectionConfig, LandmarkConfig,
                                    TransformConfig, BlendConfig,
                                    PerformanceConfig, LoggingConfig)):
                if not isinstance(raw, dict):
                    continue
                known = {f.name: f for f in fields(current)}
                for key, value in raw.items():
                    spec = known.get(key)
                    if spec is None:
                        continue
                    try:
                        setattr(current, key, _coerce(value, spec.type))
                    except (TypeError, ValueError):
                        continue
            elif isinstance(current, str):
                setattr(config, section.name, str(raw))
        return config

    # -- persistence --------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        """Write the configuration to ``path`` and return the path used."""
        target = Path(path) if path is not None else default_config_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
        os.replace(tmp, target)
        return target

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AppConfig":
        """Load the configuration, falling back to defaults on any problem."""
        target = Path(path) if path is not None else default_config_path()
        if not target.is_file():
            return cls()
        try:
            with open(target, "r", encoding="utf-8") as handle:
                return cls.from_dict(json.load(handle))
        except (OSError, ValueError):
            return cls()


def _coerce(value: Any, type_hint: Any) -> Any:
    """Best effort conversion of a JSON value to the annotated field type."""
    hint = type_hint if isinstance(type_hint, str) else getattr(
        type_hint, "__name__", str(type_hint)
    )
    if hint.startswith("List"):
        if isinstance(value, list):
            return [str(item) for item in value]
        raise TypeError("expected a list")
    if hint == "bool":
        if isinstance(value, bool):
            return value
        raise TypeError("expected a boolean")
    if hint == "int":
        if isinstance(value, bool):
            raise TypeError("expected an integer")
        return int(value)
    if hint == "float":
        return float(value)
    return str(value)
