"""LightSwapConverter entry point.

Run with::

    python main.py

The application is fully offline: it never opens a network connection and
never downloads a model. Everything it needs ships with the Python
installation or the wheel files listed in ``requirements.txt``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``python main.py`` from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import APP_NAME, APP_VERSION, AppConfig  # noqa: E402
from utils.logger import setup_logging  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="LightSwapConverter",
        description="Lightweight offline video face replacement converter.",
    )
    parser.add_argument("--source", help="Input video file")
    parser.add_argument("--face", help="Replacement face image")
    parser.add_argument("--output", help="Output video file")
    parser.add_argument("--method", default="copy", choices=("copy", "blend", "monochrome"))
    parser.add_argument("--frame-skip", type=int, default=None, help="Process every Nth frame")
    parser.add_argument("--config", help="Path to a JSON configuration file")
    parser.add_argument("--headless", action="store_true", help="Run without opening the GUI")
    parser.add_argument("--check", action="store_true", help="Verify the environment and exit")
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {APP_VERSION}")
    return parser.parse_args(argv)


def check_environment() -> int:
    """Report which optional dependencies are available."""
    print(f"{APP_NAME} {APP_VERSION} - environment check")
    ok = True

    try:
        import numpy

        print(f"  numpy          : {numpy.__version__}")
    except ImportError:
        ok = False
        print("  numpy          : MISSING (required)")

    try:
        import cv2

        print(f"  opencv         : {cv2.__version__}")
        cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        print(f"  haar cascade   : {'found' if cascade.is_file() else 'MISSING'}")
    except ImportError:
        ok = False
        print("  opencv         : MISSING (required)")

    try:
        import tkinter

        print(f"  tkinter        : {tkinter.TkVersion}")
    except ImportError:
        print("  tkinter        : missing (GUI disabled, --headless still works)")

    print(f"  python         : {sys.version.split()[0]}")
    print("Result:", "OK" if ok else "INCOMPLETE")
    return 0 if ok else 1


def run_gui(config: AppConfig) -> int:
    from gui.window import MainWindow

    window = MainWindow(config)
    window.run()
    return 0


def run_cli(config: AppConfig, args: argparse.Namespace) -> int:
    from core.processor import ConversionJob, Processor

    if not args.source or not args.face:
        print("--source and --face are required in headless mode.", file=sys.stderr)
        return 2

    processor = Processor(config)
    job = ConversionJob(
        source_video=args.source,
        target_face_image=args.face,
        output_video=args.output,
        frame_skip=args.frame_skip or config.processing.frame_skip,
        method=args.method,
    )

    def progress(done: int, total: int, message: str) -> None:
        if total:
            print(f"\r{message}: {done}/{total}", end="", flush=True)

    result = processor.process(job, progress=progress)
    print()
    if result.errors:
        for error in result.errors:
            print(f"Error: {error}", file=sys.stderr)
        return 1
    if result.cancelled:
        print("Cancelled.")
        return 130
    print(f"Wrote {result.frames_written} frames to {result.output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = AppConfig.load(args.config)
    config.ensure_directories()

    logger = setup_logging(config.log_path(), config.log_level, console=args.headless)
    logger.info("%s %s starting", APP_NAME, APP_VERSION)

    if args.check:
        return check_environment()

    if args.headless or args.source:
        return run_cli(config, args)

    try:
        return run_gui(config)
    except ImportError as exc:
        logger.error("GUI unavailable (%s); falling back to headless mode.", exc)
        print("Tkinter is not available. Use --headless with --source and --face.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
