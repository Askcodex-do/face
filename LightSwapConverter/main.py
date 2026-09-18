"""LightSwapConverter entry point.

Run the graphical interface::

    python main.py

Run the headless command line converter, useful for batch jobs and for checking
that a machine can process video at all::

    python main.py --cli --face face.jpg --video input.mp4 --output out.mp4

The application is fully offline. Nothing here opens a network connection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ``main.py`` sits next to the ``core``, ``gui`` and ``utils`` packages, so the
# directory that holds the ``LightSwapConverter`` package is the project root.
# Adding it to sys.path lets the file be launched from anywhere, which matters
# for Windows shortcuts created by the installer.
PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = PROJECT_ROOT.parent
for candidate in (PACKAGE_PARENT, PROJECT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from LightSwapConverter.core.processor import (  # noqa: E402
    ProcessorError,
    VideoFaceProcessor,
)
from LightSwapConverter.core.video_reader import VideoReadError, probe  # noqa: E402
from LightSwapConverter.utils.config import AppConfig  # noqa: E402
from LightSwapConverter.utils.logger import get_logger, setup_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="LightSwapConverter",
        description=(
            "Lightweight offline video face replacement converter. "
            "Classical computer vision only, no neural network models."
        ),
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="run the headless converter instead of the graphical interface",
    )
    parser.add_argument("--face", help="source face image")
    parser.add_argument("--video", help="target video")
    parser.add_argument("--output", help="output video path")
    parser.add_argument(
        "--config", help="path to a configuration file to load and save"
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="override the configured log level",
    )
    parser.add_argument(
        "--resize",
        metavar="WIDTHxHEIGHT",
        help="resize frames before processing, e.g. 640x360",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="stop after this many frames"
    )
    parser.add_argument(
        "--no-audio", action="store_true", help="do not copy the source audio track"
    )
    parser.add_argument(
        "--seamless", action="store_true", help="use gradient aware blending"
    )
    parser.add_argument(
        "--probe", help="print information about a video and exit"
    )
    return parser


def load_config(path: str | None) -> AppConfig:
    """Load the configuration from an explicit path or the user profile."""
    if path:
        return AppConfig.load(Path(path))
    return AppConfig.load()


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Apply command line options on top of the loaded configuration."""
    if args.log_level:
        config.logging.level = args.log_level
    if args.resize:
        try:
            width_text, height_text = args.resize.lower().split("x")
            config.video.resize_width = int(width_text)
            config.video.resize_height = int(height_text)
        except ValueError as exc:
            raise SystemExit(
                f"invalid --resize value {args.resize!r}, expected WIDTHxHEIGHT"
            ) from exc
    if args.max_frames is not None:
        config.video.max_frames = max(0, args.max_frames)
    if args.no_audio:
        config.video.copy_audio = False
    if args.seamless:
        config.blend.seamless = True
    return config


def run_cli(args: argparse.Namespace, config: AppConfig) -> int:
    """Run the headless conversion and return a process exit code."""
    if not args.face or not args.video:
        print("error: --cli requires --face and --video", file=sys.stderr)
        return 2

    output = args.output or str(
        Path(args.video).with_name(f"{Path(args.video).stem}_swapped.mp4")
    )

    def report(done: int, total: int, message: str) -> None:
        if total > 0:
            percent = min(100.0, done * 100.0 / total)
            print(f"\r{percent:5.1f}%  {message}", end="", flush=True)
        else:
            print(f"\r{message}", end="", flush=True)

    processor = VideoFaceProcessor(config)
    try:
        stats = processor.process_video(
            args.face, args.video, output, progress=report
        )
    except (ProcessorError, VideoReadError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    print()
    print(stats.summary())
    print(f"output: {stats.output_path}")
    return 0 if stats.frames_written > 0 else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = apply_overrides(load_config(args.config), args)
    setup_logging(config.logging)

    errors = config.validate()
    if errors:
        for message in errors:
            print(f"configuration error: {message}", file=sys.stderr)
        return 2

    if args.probe:
        try:
            info = probe(args.probe)
        except VideoReadError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(info)
        return 0

    if args.cli:
        return run_cli(args, config)

    try:
        from LightSwapConverter.gui.window import run
    except ImportError as exc:  # pragma: no cover - depends on the install
        print(
            "error: the graphical interface is unavailable "
            f"({exc}). Install the Tkinter package for your Python build, "
            "or run with --cli.",
            file=sys.stderr,
        )
        return 1

    run(config)
    return 0


if __name__ == "__main__":
    get_logger("main").debug("starting LightSwapConverter")
    raise SystemExit(main())
