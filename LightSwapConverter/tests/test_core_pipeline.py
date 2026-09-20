"""Pipeline tests using real OpenCV operations on synthetic frames."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from core.alignment import FaceAligner  # noqa: E402
from core.blender import FaceBlender  # noqa: E402
from core.face_detector import FaceDetector, FaceRegion  # noqa: E402
from core.landmarks import LandmarkExtractor  # noqa: E402
from core.transformer import FaceTransformer  # noqa: E402
from core.video_reader import VideoReader, VideoReaderError  # noqa: E402
from core.video_writer import VideoWriter  # noqa: E402


def make_frame(width: int = 320, height: int = 240) -> np.ndarray:
    frame = np.full((height, width, 3), 40, dtype=np.uint8)
    cv2.rectangle(frame, (60, 60), (200, 200), (90, 120, 160), -1)
    return frame


def synthetic_face_landmarks(region: FaceRegion):
    detector = FaceDetector()
    extractor = LandmarkExtractor(mode="template", detector=detector)
    return extractor.extract(make_frame(), region)


# ------------------------------------------------------------------ detector
def test_detector_handles_empty_frame():
    detector = FaceDetector()
    assert detector.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []


def test_detector_returns_no_faces_on_flat_image():
    detector = FaceDetector()
    flat = np.full((120, 120, 3), 127, dtype=np.uint8)
    assert detector.detect(flat) == []


def test_face_region_geometry():
    region = FaceRegion(10, 20, 100, 50)
    assert region.x2 == 110 and region.y2 == 70
    assert region.center == (60, 45)
    assert region.area == 5000

    scaled = region.scaled(0.5)
    assert scaled.width == 50 and scaled.height == 25

    clamped = FaceRegion(-10, -10, 40, 40).clamp(100, 100)
    assert clamped.x == 0 and clamped.y == 0
    assert clamped.x2 <= 100 and clamped.y2 <= 100


# ----------------------------------------------------------------- landmarks
def test_template_landmarks_are_ordered_and_valid():
    landmarks = synthetic_face_landmarks(FaceRegion(50, 50, 100, 100))
    assert landmarks.is_valid
    assert landmarks.count == 5
    assert landmarks["left_eye"][0] < landmarks["right_eye"][0]
    assert landmarks["left_eye"][1] < landmarks["mouth_left"][1]
    assert landmarks.inter_ocular_distance > 0
    assert abs(landmarks.roll_angle) < 1.0


def test_landmarks_reject_unknown_name():
    landmarks = synthetic_face_landmarks(FaceRegion(50, 50, 100, 100))
    with pytest.raises(KeyError):
        _ = landmarks["chin"]


def test_landmarks_translation_keeps_shape():
    landmarks = synthetic_face_landmarks(FaceRegion(50, 50, 100, 100))
    moved = landmarks.translated(10, -5)
    assert np.allclose(moved.points, landmarks.points + np.array([10, -5], dtype=np.float32))


# ----------------------------------------------------------------- alignment
def test_alignment_maps_eyes_onto_canonical_layout():
    landmarks = synthetic_face_landmarks(FaceRegion(50, 50, 120, 120))
    aligner = FaceAligner(output_size=(128, 128))
    result = aligner.compute_matrix(landmarks)
    assert result.valid

    projected = result.transform_points(landmarks.points)
    eye_delta = projected[1] - projected[0]
    assert abs(eye_delta[1]) < 1.0  # eyes become level
    assert projected[0][0] < projected[1][0]


def test_alignment_handles_degenerate_landmarks():
    landmarks = synthetic_face_landmarks(FaceRegion(10, 10, 40, 40))
    landmarks.points[0] = landmarks.points[1]  # both eyes on the same pixel
    result = FaceAligner().compute_matrix(landmarks)
    assert result.valid is False
    assert np.allclose(result.matrix, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))


def test_alignment_roundtrip_warp_is_reversible():
    frame = make_frame()
    landmarks = synthetic_face_landmarks(FaceRegion(60, 60, 140, 140))
    aligner = FaceAligner(output_size=(96, 96))
    warped, result = aligner.align(frame, landmarks)
    assert warped.shape == (96, 96, 3)

    restored = aligner.unalign(warped, result, (frame.shape[1], frame.shape[0]))
    assert restored.shape == frame.shape


def test_average_landmarks_smooths_jitter():
    a = synthetic_face_landmarks(FaceRegion(50, 50, 100, 100))
    b = synthetic_face_landmarks(FaceRegion(52, 50, 100, 100))
    averaged = FaceAligner.average_landmarks([a, b])
    assert averaged is not None
    assert np.allclose(averaged.points, (a.points + b.points) / 2.0, atol=1e-5)
    assert FaceAligner.average_landmarks([]) is None


# --------------------------------------------------------------- transformer
def test_color_match_moves_source_towards_target():
    source = np.full((32, 32, 3), 50, dtype=np.uint8)
    target = np.full((32, 32, 3), 200, dtype=np.uint8)
    adjusted, shift = FaceTransformer.match_color(source, target)
    assert adjusted.mean() > source.mean()
    assert all(value > 0 for value in shift)


def test_transformer_rejects_unknown_method():
    with pytest.raises(ValueError):
        FaceTransformer(method="neural")


def test_transformer_prepare_and_transform():
    frame = make_frame()
    landmarks = synthetic_face_landmarks(FaceRegion(60, 60, 140, 140))
    transformer = FaceTransformer(method="copy", color_match=True)

    prepared = transformer.prepare(frame, landmarks)
    assert prepared.is_valid
    assert prepared.mask.shape == prepared.image.shape[:2]

    patch, _ = FaceAligner().align(frame, landmarks)
    transformed = transformer.transform(prepared, patch, landmarks)
    assert transformed.image.shape == patch.shape
    assert transformed.color_shift is not None


def test_soft_mask_is_feathered_and_bounded():
    mask = FaceTransformer._soft_mask((64, 64))
    assert mask.dtype == np.uint8
    assert mask.max() > 0
    assert mask[32, 32] > mask[0, 0]  # centre opaque, corner transparent


# -------------------------------------------------------------------- blender
def test_blend_modifies_only_the_masked_area():
    frame = make_frame()
    landmarks = synthetic_face_landmarks(FaceRegion(60, 60, 140, 140))
    aligner = FaceAligner(output_size=(128, 128))
    alignment = aligner.compute_matrix(landmarks)

    patch = np.full((128, 128, 3), 255, dtype=np.uint8)
    mask = FaceTransformer._soft_mask((128, 128))
    corner_before = frame[0:5, 0:5].copy()

    blender = FaceBlender(strength=1.0, use_seamless=False)
    result = blender.blend(frame, patch, mask, alignment)

    assert result.applied
    assert result.method == "alpha"
    assert np.array_equal(frame[0:5, 0:5], corner_before)
    assert frame[120:140, 120:140].mean() > 200


def test_blend_is_a_noop_for_invalid_alignment():
    frame = make_frame()
    aligner = FaceAligner()
    result = FaceBlender().blend(
        frame, np.zeros((64, 64, 3), np.uint8), np.zeros((64, 64), np.uint8), aligner.identity_result(valid=False)
    )
    assert result.applied is False
    assert result.frame is frame


def test_blend_strength_zero_keeps_frame_unchanged():
    frame = make_frame()
    original = frame.copy()
    landmarks = synthetic_face_landmarks(FaceRegion(60, 60, 140, 140))
    aligner = FaceAligner(output_size=(128, 128))
    alignment = aligner.compute_matrix(landmarks)
    patch = np.full((128, 128, 3), 255, dtype=np.uint8)
    mask = FaceTransformer._soft_mask((128, 128))

    FaceBlender(strength=0.0, use_seamless=False).blend(frame, patch, mask, alignment)
    assert np.allclose(frame, original, atol=2)


# ------------------------------------------------------------ video read/write
def test_video_roundtrip(tmp_path):
    path = tmp_path / "clip.avi"
    size = (64, 48)

    with VideoWriter(path, fps=10.0, size=size, codec="MJPG") as writer:
        for i in range(5):
            writer.write(np.full((size[1], size[0], 3), i * 20, dtype=np.uint8))
    assert writer.frames_written == 5
    assert path.is_file()

    with VideoReader(path) as reader:
        info = reader.open() if not reader.is_open else reader.info
        assert info.width == size[0]
        assert info.height == size[1]
        assert info.fps == pytest.approx(10.0, abs=0.5)
        assert info.is_valid

        frames = list(reader.frames())
        assert len(frames) == 5
        assert frames[0][1].index == 0
        assert frames[0][1].width == size[0]


def test_reader_rejects_missing_file(tmp_path):
    with pytest.raises(VideoReaderError):
        VideoReader(tmp_path / "nope.mp4").open()


def test_reader_downscales_large_frames(tmp_path):
    path = tmp_path / "big.avi"
    with VideoWriter(path, fps=5.0, size=(400, 300), codec="MJPG") as writer:
        writer.write(np.zeros((300, 400, 3), dtype=np.uint8))

    with VideoReader(path, max_width=200, max_height=150) as reader:
        reader.open()
        frame, info = reader.read()
        assert frame.shape[:2] == (150, 200)
        assert info.scale == pytest.approx(0.5, abs=0.01)
