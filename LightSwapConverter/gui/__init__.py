"""Tkinter user interface for LightSwapConverter.

Importing this package does not import Tkinter, so the core pipeline stays
usable on a machine without a display server.
"""

from __future__ import annotations

__all__ = ["ConverterWindow", "run"]


def __getattr__(name: str):
    if name in {"ConverterWindow", "run"}:
        from . import window as _window

        return getattr(_window, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
