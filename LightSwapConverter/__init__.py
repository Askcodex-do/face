"""LightSwapConverter - a lightweight, fully offline video face replacement tool.

Target platform: Windows 8.1 64-bit, Python 3.10.11, CPU only, 2 GB RAM.

The application is split into three packages:

``core``
    The processing pipeline: video reading and writing, face detection,
    landmark estimation, alignment, warping and blending.
``gui``
    A Tkinter interface, imported lazily so the core works headless.
``utils``
    Configuration and logging.

Entry points::

    python main.py                          # graphical interface
    python main.py --cli --face f.jpg --video in.mp4 --output out.mp4

No component of this project opens a network connection or downloads a model.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
