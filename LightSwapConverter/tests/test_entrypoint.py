"""Regression tests for the application entry point and the core exports.

These protect against two shipping blockers found in the ``origin/main`` audit:

* ``main.py`` imported ``VideoFaceProcessor`` from ``core.processor``, but the
  class lives in ``core.swapper``. The whole application failed to start with
  an ``ImportError`` while the rest of the suite stayed green, because no test
  imported the entry point.
* ``core/__init__.py`` mapped ``SourceFace`` and ``VideoFaceProcessor`` to the
  wrong module, so the lazy re-exports raised ``AttributeError``.

The suite deliberately imports ``main`` and runs it in a subprocess, so a broken
entry point can never pass unnoticed again. No existing test is modified.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent


class TestEntryPointImport:
    """``main.py`` must be importable; it is the application's front door."""

    def test_main_module_imports(self):
        """Importing the entry point must not raise."""
        module = importlib.import_module("LightSwapConverter.main")
        assert module is not None

    def test_main_exposes_the_names_run_cli_uses(self):
        """``run_cli`` and ``main`` reference these, so they must be present."""
        module = importlib.import_module("LightSwapConverter.main")
        for name in (
            "VideoFaceProcessor",
            "ProcessorError",
            "VideoReadError",
            "AppConfig",
            "probe",
            "build_parser",
            "run_cli",
            "main",
        ):
            assert hasattr(module, name), f"main.py is missing {name}"

    def test_video_face_processor_comes_from_swapper(self):
        """The exact bug: main.py must source it from ``core.swapper``.

        ``core.processor`` is the Phase 1 video engine and must stay free of
        face orchestration, so the class is genuinely not defined there.
        """
        import LightSwapConverter.core.processor as processor
        import LightSwapConverter.core.swapper as swapper

        main = importlib.import_module("LightSwapConverter.main")

        assert not hasattr(processor, "VideoFaceProcessor"), (
            "VideoFaceProcessor unexpectedly moved back into core.processor; "
            "update main.py and this test together"
        )
        assert hasattr(swapper, "VideoFaceProcessor")
        assert main.VideoFaceProcessor is swapper.VideoFaceProcessor

    def test_processor_error_still_comes_from_processor(self):
        """``ProcessorError`` legitimately belongs to the video engine."""
        import LightSwapConverter.core.processor as processor

        main = importlib.import_module("LightSwapConverter.main")

        assert main.ProcessorError is processor.ProcessorError


class TestCoreLazyExports:
    """Every advertised ``core`` name must resolve to the real object."""

    def test_all_advertised_names_resolve(self):
        """``__all__`` and ``_EXPORTS`` must agree and every name must resolve."""
        core = importlib.import_module("LightSwapConverter.core")

        assert set(core.__all__) == set(core._EXPORTS), (
            "core.__all__ and core._EXPORTS disagree; consumers would see "
            "AttributeError for the difference"
        )
        for name in core.__all__:
            assert getattr(core, name) is not None, f"core.{name} resolved to None"

    def test_source_face_export(self):
        """``SourceFace`` must come from ``core.swapper``."""
        core = importlib.import_module("LightSwapConverter.core")
        swapper = importlib.import_module("LightSwapConverter.core.swapper")

        assert core.SourceFace is swapper.SourceFace

    def test_video_face_processor_export(self):
        """``VideoFaceProcessor`` must come from ``core.swapper``."""
        core = importlib.import_module("LightSwapConverter.core")
        swapper = importlib.import_module("LightSwapConverter.core.swapper")

        assert core.VideoFaceProcessor is swapper.VideoFaceProcessor

    def test_app_config_export(self):
        """``AppConfig`` must come from ``utils.config``, not a phantom module."""
        core = importlib.import_module("LightSwapConverter.core")
        config = importlib.import_module("LightSwapConverter.utils.config")

        assert core.AppConfig is config.AppConfig

    def test_exports_that_stayed_in_processor(self):
        """The names that really do live in the video engine still resolve."""
        core = importlib.import_module("LightSwapConverter.core")
        processor = importlib.import_module("LightSwapConverter.core.processor")

        for name in ("ProcessorError", "ProcessingStats", "read_image", "write_image"):
            assert getattr(core, name) is getattr(processor, name)

    def test_unknown_name_raises_attribute_error(self):
        """The lazy loader must not invent attributes."""
        core = importlib.import_module("LightSwapConverter.core")
        with pytest.raises(AttributeError):
            core.definitely_not_a_real_name


class TestEntryPointSubprocess:
    """Run the entry point the way a user or a shortcut would."""

    @staticmethod
    def _run(args, tmp_path):
        env = dict(os.environ)
        # Keep any log or config file out of the real user profile.
        env["APPDATA"] = str(tmp_path)
        env["HOME"] = str(tmp_path)
        return subprocess.run(
            [sys.executable, str(PACKAGE_ROOT / "main.py"), *args],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180,
        )

    def test_help_succeeds(self, tmp_path):
        """``python main.py --help`` must exit 0 and print usage."""
        result = self._run(["--help"], tmp_path)

        assert result.returncode == 0, (
            f"--help failed with {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "usage:" in result.stdout.lower()
        assert "ImportError" not in result.stderr

    def test_cli_without_inputs_reports_usage_not_importerror(self, tmp_path):
        """``--cli`` with no files is a clean usage error, never an ImportError."""
        result = self._run(["--cli"], tmp_path)

        combined = result.stdout + result.stderr
        assert "ImportError" not in combined, combined
        assert "cannot import name" not in combined, combined
        assert result.returncode == 2, (
            f"expected the usage exit code 2, got {result.returncode}\n{combined}"
        )
        assert "--cli requires --face and --video" in combined