"""Synthetic media builders shared by the test suite.

The suite never downloads sample media and never ships binary fixtures. Every
video and image a test needs is drawn here with OpenCV at test time, which keeps
the suite offline, fast and self contained.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


def draw_synthetic_face(
    width: int = 320,
    height: int = 240,
    *,
    cx: int = 160,
    cy: int = 120,
    face_width: int = 90,
    face_height: int = 110,
    skin: tuple[int, int, int] = (150, 180, 215),
    background: tuple[int, int, int] = (60, 60, 60),
) -> np.ndarray:
    """Draw a simple, clearly detectable frontal face.

    A Haar cascade will not fire on a flat ellipse, so the drawing includes the
    contrast the detector looks for: dark eyes and brows, a shaded nose and a
    mouth line on a lighter skin tone.
    """
    frame = np.full((height, width, 3), background, dtype=np.uint8)
    cv2.ellipse(
        frame,
        (cx, cy),
        (face_width // 2, face_height // 2),
        0,
        0,
        360,
        skin,
        thickness=-1,
    )

    eye_dx = int(face_width * 0.22)
    eye_y = cy - int(face_height * 0.16)
    eye_w = max(3, int(face_width * 0.13))
    eye_h = max(2, int(face_height * 0.07))
    for direction in (-1, 1):
        centre = (cx + direction * eye_dx, eye_y)
        cv2.ellipse(
            frame, centre, (eye_w, eye_h), 0, 0, 360, (35, 35, 35), thickness=-1
        )
        cv2.circle(frame, centre, max(1, eye_w // 3), (240, 240, 240), thickness=-1)
        cv2.ellipse(
            frame,
            (centre[0], centre[1] - eye_h * 2),
            (eye_w + 2, max(2, eye_h // 2)),
            0,
            0,
            360,
            (45, 45, 45),
            thickness=-1,
        )

    nose_y = cy + int(face_height * 0.05)
    cv2.ellipse(
        frame,
        (cx, nose_y),
        (max(3, int(face_width * 0.09)), max(4, int(face_height * 0.10))),
        0,
        0,
        360,
        (120, 145, 180),
        thickness=-1,
    )

    mouth_y = cy + int(face_height * 0.26)
    cv2.ellipse(
        frame,
        (cx, mouth_y),
        (int(face_width * 0.20), max(2, int(face_height * 0.04))),
        0,
        0,
        360,
        (70, 70, 110),
        thickness=-1,
    )

    return frame


def write_test_video(
    path: Path,
    frames: int = 12,
    width: int = 320,
    height: int = 240,
    fps: float = 12.0,
    codec: str = "mp4v",
    moving: bool = True,
) -> Path:
    """Write a short synthetic video containing a moving face."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
    )
    if not writer.isOpened():
        pytest.skip(f"codec {codec!r} is unavailable in this OpenCV build")
    try:
        for index in range(frames):
            offset = (index - frames // 2) * 2 if moving else 0
            frame = draw_synthetic_face(
                width, height, cx=width // 2 + offset, cy=height // 2
            )
            writer.write(frame)
    finally:
        writer.release()
    return path


def write_test_image(path: Path, **kwargs) -> Path:
    """Write a synthetic face image to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = draw_synthetic_face(**kwargs)
    ok, buffer = cv2.imencode(path.suffix or ".png", image)
    assert ok, "failed to encode the synthetic test image"
    buffer.tofile(str(path))
    return path
