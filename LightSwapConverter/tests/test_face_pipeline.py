"""Tests for detection, landmarks, alignment, warping and blending.

The synthetic face from ``conftest`` is a drawing, not a photograph, so these
tests assert the invariants the pipeline actually depends on - point counts,
region ordering, bounded refinement, correct transform maths, mask shape - and
not pixel level accuracy against a cartoon. Where a real image property can be
checked, it is.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from LightSwapConverter.core.alignment import (
    AlignmentError,
    AlignmentResult,
    align_to_face,
    compose,
    fit_affine,
    fit_homography,
    fit_similarity,
    invert,
    scale_matrix,
    similarity_from_eyes,
    transform_points,
)
from LightSwapConverter.core.blender import (
    BlendError,
    FaceBlender,
    build_alpha,
    feather_mask,
    mask_preview,
)
from LightSwapConverter.core.face_detector import (
    DetectionConfig,
    DetectorError,
    FaceBox,
    FaceDetector,
    detect_largest_face,
    load_cascade,
)
from LightSwapConverter.core.landmarks import (
    ALIGNMENT_INDICES,
    NUM_POINTS,
    REGIONS,
    LandmarkEstimator,
    Landmarks,
    estimate_landmarks,
    reference_layout,
)
from LightSwapConverter.core.transformer import (
    FaceTransformer,
    TransformError,
    face_hull_mask,
    translation_matrix,
)
from LightSwapConverter.utils.config import BlendConfig
from LightSwapConverter.utils.config import LandmarkConfig, TransformConfig

from tests.helpers import draw_synthetic_face


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class TestFaceBox:
    def test_geometry_helpers(self):
        box = FaceBox(10, 20, 40, 60)
        assert box.x2 == 50 and box.y2 == 80
        assert box.center == (30, 50)
        assert box.area == 2400
        assert box.as_tuple() == (10, 20, 40, 60)

    def test_clamp_keeps_the_box_inside_the_frame(self):
        clamped = FaceBox(-20, -10, 400, 400).clamp(320, 240)
        assert clamped.x >= 0 and clamped.y >= 0
        assert clamped.x2 <= 320 and clamped.y2 <= 240

    def test_pad_grows_the_box(self):
        padded = FaceBox(50, 50, 100, 100).pad(0.2, 1000, 1000)
        assert padded.width == 140 and padded.height == 140

    def test_pad_with_zero_ratio_is_a_clamp(self):
        assert FaceBox(50, 50, 100, 100).pad(0.0, 1000, 1000).as_tuple() == (
            50, 50, 100, 100
        )

    def test_scale_keeps_the_centre(self):
        scaled = FaceBox(50, 50, 100, 100).scale(0.5)
        assert scaled.center == (100, 100)
        assert scaled.width == 50

    def test_expand_produces_a_square(self):
        expanded = FaceBox(0, 0, 40, 80).expand(2.0)
        assert expanded.width == expanded.height

    def test_boxes_are_frozen(self):
        box = FaceBox(0, 0, 10, 10)
        with pytest.raises(Exception):
            box.x = 5  # type: ignore[misc]


class TestCascadeLoading:
    def test_default_cascade_loads(self):
        assert not load_cascade("haarcascade_frontalface_default.xml").empty()

    def test_missing_cascade_raises(self):
        with pytest.raises(DetectorError, match="cannot locate"):
            load_cascade("definitely_not_a_cascade.xml")

    def test_detector_loads_its_cascades(self):
        assert FaceDetector() is not None

    def test_detector_skips_unloadable_fallback_cascades(self):
        config = DetectionConfig(fallback_cascades=["nope.xml"])
        detector = FaceDetector(config)
        assert detector is not None  # must not raise


class TestFaceDetection:
    def test_detects_the_synthetic_face(self, synthetic_face):
        box = FaceDetector().detect(synthetic_face)
        assert box is not None
        # The drawn face is centred at (160, 120).
        cx, cy = box.center
        assert abs(cx - 160) < 30
        assert abs(cy - 120) < 30

    def test_returns_none_for_a_blank_frame(self):
        assert FaceDetector().detect(np.zeros((240, 320, 3), np.uint8)) is None

    def test_returns_none_for_empty_input(self):
        assert FaceDetector().detect(np.zeros((0, 0, 3), np.uint8)) is None

    def test_detects_in_a_grayscale_frame(self):
        gray = cv2.cvtColor(draw_synthetic_face(320, 240), cv2.COLOR_BGR2GRAY)
        assert FaceDetector().detect(gray) is not None

    def test_detect_all_returns_a_list(self, synthetic_face):
        assert isinstance(FaceDetector().detect_all(synthetic_face), list)

    def test_tracking_reuses_the_previous_box(self, synthetic_face):
        detector = FaceDetector()
        first = detector.detect(synthetic_face)
        second = detector.detect(synthetic_face)
        assert first is not None and second is not None
        assert abs(first.center[0] - second.center[0]) <= 8

    def test_reset_clears_tracking_state(self, synthetic_face):
        detector = FaceDetector()
        detector.detect(synthetic_face)
        detector.reset()
        assert detector._last_box is None

    def test_tracking_can_be_disabled(self, synthetic_face):
        config = DetectionConfig(use_tracking=False)
        assert FaceDetector(config).detect(synthetic_face) is not None

    def test_detection_is_deterministic(self, synthetic_face):
        first = FaceDetector().detect(synthetic_face)
        second = FaceDetector().detect(synthetic_face)
        assert first.as_tuple() == second.as_tuple()

    def test_detected_box_stays_inside_the_frame(self, synthetic_face):
        box = FaceDetector().detect(synthetic_face)
        assert box.x >= 0 and box.y >= 0
        assert box.x2 <= 320 and box.y2 <= 240

    def test_one_shot_helper(self, synthetic_face):
        assert detect_largest_face(synthetic_face) is not None


# ---------------------------------------------------------------------------
# Landmarks
# ---------------------------------------------------------------------------


class TestReferenceLayout:
    def test_layout_has_68_points(self):
        assert NUM_POINTS == 68

    def test_regions_tile_the_layout_exactly(self):
        covered = sorted(
            index for start, stop in REGIONS.values() for index in range(start, stop)
        )
        assert covered == list(range(68))

    def test_region_sizes_match_the_convention(self):
        assert REGIONS["jaw"] == (0, 17)
        assert REGIONS["right_eye"] == (36, 42)
        assert REGIONS["left_eye"] == (42, 48)
        assert REGIONS["mouth"] == (48, 68)

    def test_alignment_indices_are_inside_the_layout(self):
        assert all(0 <= index < 68 for index in ALIGNMENT_INDICES)

    def test_layout_maps_onto_the_box(self):
        box = FaceBox(100, 50, 200, 200)
        points = reference_layout(box)
        assert points.shape == (68, 2)
        assert points[:, 0].min() >= 100
        assert points[:, 1].min() >= 50
        assert points[:, 0].max() <= 300
        assert points[:, 1].max() <= 250

    def test_eyes_are_above_the_mouth(self):
        points = reference_layout(FaceBox(0, 0, 100, 100))
        assert points[REGIONS["right_eye"][0] : REGIONS["right_eye"][1], 1].mean() < points[
            REGIONS["mouth"][0] : REGIONS["mouth"][1], 1
        ].mean()

    def test_left_and_right_eyes_are_on_the_expected_sides(self):
        points = reference_layout(FaceBox(0, 0, 100, 100))
        right = points[36:42, 0].mean()
        left = points[42:48, 0].mean()
        assert right < left  # "right_eye" appears on the left of the image

    def test_jaw_sits_inside_the_box_not_on_its_edge(self):
        # The Haar box is wider than the face: the padded box extends past the
        # jaw on both sides. A layout whose jaw reached the box edge would be
        # around 70% too wide, which is the defect this asserts against.
        points = reference_layout(FaceBox(0, 0, 100, 100))
        assert 10.0 < points[0, 0] < 35.0
        assert 65.0 < points[16, 0] < 90.0
        jaw_width = points[16, 0] - points[0, 0]
        assert 40.0 < jaw_width < 65.0

    def test_layout_is_symmetric_about_its_own_vertical_axis(self):
        points = reference_layout(FaceBox(0, 0, 100, 100))
        # The layout's axis of symmetry is where its centre features sit (the
        # nose bridge, which the annotations place in the middle of the face),
        # not the centre of the detection box.
        nose_x = points[27:31, 0].mean()
        assert points[0, 0] + points[16, 0] == pytest.approx(2 * nose_x, abs=1e-3)
        assert points[36, 0] + points[45, 0] == pytest.approx(2 * nose_x, abs=1e-3)
        assert points[42, 0] + points[39, 0] == pytest.approx(2 * nose_x, abs=1e-3)
        assert points[48, 0] + points[54, 0] == pytest.approx(2 * nose_x, abs=1e-3)


class TestLandmarks:
    def test_region_lookup(self):
        points = reference_layout(FaceBox(0, 0, 100, 100))
        landmarks = Landmarks(points)
        assert landmarks.region("jaw").shape == (17, 2)
        assert landmarks.region("mouth").shape == (20, 2)

    def test_unknown_region_raises(self):
        landmarks = Landmarks(reference_layout(FaceBox(0, 0, 10, 10)))
        with pytest.raises(KeyError, match="unknown region"):
            landmarks.region("chin")

    def test_rejects_badly_shaped_input(self):
        with pytest.raises(ValueError, match="shape"):
            Landmarks(np.zeros((68, 3)))

    def test_eye_distance_is_positive(self):
        landmarks = Landmarks(reference_layout(FaceBox(0, 0, 100, 100)))
        assert landmarks.eye_distance > 0

    def test_translate_moves_every_point(self):
        landmarks = Landmarks(reference_layout(FaceBox(0, 0, 100, 100)))
        moved = landmarks.translate(10, -5)
        assert np.allclose(moved.points - landmarks.points, [10, -5])

    def test_translate_does_not_mutate_the_original(self):
        landmarks = Landmarks(reference_layout(FaceBox(0, 0, 100, 100)))
        before = landmarks.points.copy()
        landmarks.translate(10, 10)
        assert np.array_equal(landmarks.points, before)

    def test_scale_about_the_face_centre(self):
        landmarks = Landmarks(reference_layout(FaceBox(0, 0, 100, 100)))
        centre = landmarks.face_center
        scaled = landmarks.scale(2.0)
        assert np.allclose(scaled.face_center, centre)

    def test_to_normalized_maps_into_the_unit_box(self):
        box = FaceBox(50, 50, 100, 100)
        normalized = Landmarks(reference_layout(box), box).to_normalized()
        assert normalized.min() >= -1e-5
        assert normalized.max() <= 1.0 + 1e-5

    def test_to_normalized_without_a_box_raises(self):
        with pytest.raises(ValueError, match="without a face box"):
            Landmarks(reference_layout(FaceBox(0, 0, 10, 10))).to_normalized()

    def test_bounding_rect_covers_every_point(self):
        landmarks = Landmarks(reference_layout(FaceBox(10, 10, 100, 100)))
        x, y, w, h = landmarks.bounding_rect
        assert x <= landmarks.points[:, 0].min()
        assert y <= landmarks.points[:, 1].min()
        assert x + w >= landmarks.points[:, 0].max()
        assert y + h >= landmarks.points[:, 1].max()

    def test_as_list_returns_plain_floats(self):
        landmarks = Landmarks(reference_layout(FaceBox(0, 0, 10, 10)))
        values = landmarks.as_list()
        assert len(values) == 68
        assert isinstance(values[0][0], float)

    def test_copy_is_independent(self):
        landmarks = Landmarks(reference_layout(FaceBox(0, 0, 10, 10)))
        duplicate = landmarks.copy()
        duplicate.points[0, 0] += 100
        assert duplicate.points[0, 0] != landmarks.points[0, 0]


class TestLandmarkEstimator:
    def test_estimates_68_points(self, synthetic_face):
        box = FaceDetector().detect(synthetic_face)
        landmarks = LandmarkEstimator().estimate(synthetic_face, box)
        assert len(landmarks) == 68

    def test_points_stay_near_the_face_box(self, synthetic_face):
        box = FaceDetector().detect(synthetic_face)
        landmarks = LandmarkEstimator().estimate(synthetic_face, box)
        # Refinement is clamped, so nothing may land far outside the box.
        assert landmarks.points[:, 0].min() > box.x - box.width
        assert landmarks.points[:, 0].max() < box.x2 + box.width

    def test_eyes_are_found_left_of_the_right_eye(self, synthetic_face):
        box = FaceDetector().detect(synthetic_face)
        landmarks = LandmarkEstimator().estimate(synthetic_face, box)
        assert landmarks.right_eye_center[0] < landmarks.left_eye_center[0]

    def test_eyes_are_above_the_mouth(self, synthetic_face):
        box = FaceDetector().detect(synthetic_face)
        landmarks = LandmarkEstimator().estimate(synthetic_face, box)
        assert landmarks.eye_distance > 0
        assert landmarks.right_eye_center[1] < landmarks.mouth_center[1]

    def test_refine_false_uses_the_prior_only(self, synthetic_face):
        box = FaceDetector().detect(synthetic_face)
        landmarks = LandmarkEstimator().estimate(synthetic_face, box, refine=False)
        assert landmarks.prior_only is True
        assert np.allclose(landmarks.points, reference_layout(box))

    def test_refinement_marks_the_result_as_measured(self, synthetic_face):
        box = FaceDetector().detect(synthetic_face)
        landmarks = LandmarkEstimator().estimate(synthetic_face, box, refine=True)
        assert landmarks.prior_only is False

    def test_estimation_is_deterministic_on_a_fresh_estimator(self, synthetic_face):
        box = FaceDetector().detect(synthetic_face)
        first = LandmarkEstimator().estimate(synthetic_face, box)
        second = LandmarkEstimator().estimate(synthetic_face, box)
        assert np.allclose(first.points, second.points)

    def test_smoothing_reduces_frame_to_frame_change(self, synthetic_face):
        box = FaceDetector().detect(synthetic_face)
        shifted = np.roll(synthetic_face, 6, axis=1)

        smooth = LandmarkEstimator(LandmarkConfig(smoothing=0.9))
        smooth.estimate(synthetic_face, box)
        smoothed = smooth.estimate(shifted, box)

        raw = LandmarkEstimator(LandmarkConfig(smoothing=0.0))
        raw.estimate(synthetic_face, box)
        unsmoothed = raw.estimate(shifted, box)

        smooth_delta = np.linalg.norm(smoothed.points - unsmoothed.points)
        assert smooth_delta > 0  # the two differ, so smoothing did something

    def test_history_is_dropped_on_a_large_jump(self, synthetic_face):
        estimator = LandmarkEstimator(LandmarkConfig(smoothing=0.9))
        box = FaceDetector().detect(synthetic_face)
        estimator.estimate(synthetic_face, box)
        far_box = FaceBox(900, 700, 60, 60)
        result = estimator.estimate(synthetic_face, far_box)
        # A stale history must not drag the layout towards the old position.
        assert np.allclose(result.points, reference_layout(far_box))

    def test_reset_clears_history(self, synthetic_face):
        estimator = LandmarkEstimator()
        box = FaceDetector().detect(synthetic_face)
        estimator.estimate(synthetic_face, box)
        estimator.reset()
        assert estimator._previous is None

    def test_handles_an_empty_frame(self):
        box = FaceBox(10, 10, 40, 40)
        assert len(LandmarkEstimator().estimate(np.zeros((0, 0, 3), np.uint8), box)) == 68

    def test_one_shot_helper(self, synthetic_face):
        box = FaceDetector().detect(synthetic_face)
        assert len(estimate_landmarks(synthetic_face, box)) == 68


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


class TestAlignmentMaths:
    def test_translation_is_recovered_exactly(self):
        source = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], np.float32)
        target = source + np.array([25, -15], np.float32)
        matrix, error = fit_similarity(source, target)
        assert error == pytest.approx(0.0, abs=1e-3)
        assert np.allclose(transform_points(matrix, source), target, atol=1e-3)

    def test_scaling_is_recovered(self):
        source = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], np.float32)
        target = source * 2.5
        matrix, error = fit_similarity(source, target)
        assert error == pytest.approx(0.0, abs=1e-3)

    def test_rotation_is_recovered(self):
        source = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], np.float32)
        angle = np.deg2rad(30)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            np.float32,
        )
        target = source @ rotation.T
        matrix, error = fit_similarity(source, target)
        assert error == pytest.approx(0.0, abs=1e-3)

    def test_similarity_preserves_angles(self):
        source = np.array([[0, 0], [10, 0], [0, 10]], np.float32)
        target = np.array([[5, 5], [15, 5], [5, 15]], np.float32)
        matrix, _ = fit_similarity(source, target)
        linear = matrix[:, :2]
        # Orthogonal columns of equal length is the definition of a similarity.
        assert np.isclose(linear[0] @ linear[1], 0.0, atol=1e-4)
        assert np.isclose(np.linalg.norm(linear[:, 0]), np.linalg.norm(linear[:, 1]), atol=1e-4)
        assert np.isclose(np.linalg.det(linear), 1.0, atol=1e-4)

    def test_mismatched_shapes_raise(self):
        with pytest.raises(AlignmentError, match="must match"):
            fit_similarity(np.zeros((4, 2), np.float32), np.zeros((3, 2), np.float32))

    def test_too_few_points_raise(self):
        with pytest.raises(AlignmentError):
            fit_similarity(np.zeros((1, 2), np.float32), np.zeros((1, 2), np.float32))

    def test_affine_needs_three_points(self):
        with pytest.raises(AlignmentError, match=">= 3"):
            fit_affine(np.zeros((2, 2), np.float32), np.zeros((2, 2), np.float32))

    def test_affine_recovers_a_shear(self):
        source = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], np.float32)
        shear = np.array([[1.0, 0.4, 0.0], [0.0, 1.0, 0.0]], np.float32)
        target = transform_points(shear, source)
        matrix, error = fit_affine(source, target)
        assert error == pytest.approx(0.0, abs=1e-3)

    def test_homography_needs_four_points(self):
        with pytest.raises(AlignmentError, match=">= 4"):
            fit_homography(np.zeros((3, 2), np.float32), np.zeros((3, 2), np.float32))

    def test_homography_recovers_a_perspective(self):
        source = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float32)
        target = np.array([[1, 2], [12, 0], [10, 13], [-1, 10]], np.float32)
        matrix, error = fit_homography(source, target)
        assert error == pytest.approx(0.0, abs=1e-2)

    def test_reprojection_error_measures_a_bad_fit(self):
        source = np.array([[0, 0], [10, 0], [0, 10]], np.float32)
        target = np.array([[0, 0], [10, 0], [0, 50]], np.float32)
        _, error = fit_similarity(source, target)
        assert error > 1.0

    def test_homography_error_uses_the_perspective_divide(self):
        # A pure translation must fit with a near zero error. Evaluating only
        # the first two rows of the homography would report a large error here.
        source = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float32)
        target = source + np.array([7, -3], np.float32)
        _, error = fit_homography(source, target)
        assert error == pytest.approx(0.0, abs=1e-2)


class TestAlignmentHelpers:
    def test_scale_matrix(self):
        matrix = scale_matrix(2.0, 5.0, -5.0)
        assert np.allclose(transform_points(matrix, [[1, 1]]), [[7, -3]])

    def test_translation_matrix(self):
        assert np.allclose(transform_points(translation_matrix(3, 4), [[0, 0]]), [[3, 4]])

    def test_compose_applies_inner_first(self):
        inner = translation_matrix(10, 0)
        outer = scale_matrix(2.0)
        combined = compose(outer, inner)
        # Scale by 2 after translating by 10 gives (0+10)*2 = 20.
        assert np.allclose(transform_points(combined, [[0, 0]]), [[20, 0]])

    def test_invert_round_trips(self):
        matrix = scale_matrix(2.0, 10.0, 20.0)
        point = np.array([[5.0, 7.0]], np.float32)
        moved = transform_points(matrix, point)
        restored = transform_points(invert(matrix), moved)
        assert np.allclose(restored, point, atol=1e-3)

    def test_transform_points_shape(self):
        assert transform_points(scale_matrix(1.0), np.zeros((68, 2), np.float32)).shape == (
            68,
            2,
        )


class TestAlignToFace:
    def test_aligns_identical_landmarks_onto_themselves(self):
        box = FaceBox(0, 0, 100, 100)
        landmarks = Landmarks(reference_layout(box), box)
        result = align_to_face(landmarks, landmarks, box)
        assert result.error == pytest.approx(0.0, abs=1e-3)
        assert result.is_reliable
        assert result.method == "similarity"

    def test_result_size_matches_the_target_box(self):
        source_box = FaceBox(0, 0, 80, 80)
        target_box = FaceBox(50, 50, 120, 140)
        result = align_to_face(
            Landmarks(reference_layout(source_box), source_box),
            Landmarks(reference_layout(target_box), target_box),
            target_box,
        )
        assert result.size == (120, 140)

    def test_scaled_target_produces_a_scaled_matrix(self):
        source_box = FaceBox(0, 0, 100, 100)
        target_box = FaceBox(0, 0, 200, 200)
        result = align_to_face(
            Landmarks(reference_layout(source_box), source_box),
            Landmarks(reference_layout(target_box), target_box),
            target_box,
        )
        # Uniform scale of 2, so the linear part must be 2 * identity.
        assert result.matrix[0, 0] == pytest.approx(2.0, abs=1e-3)
        assert result.matrix[1, 1] == pytest.approx(2.0, abs=1e-3)

    def test_unreliable_fit_is_flagged(self):
        # Shift a single point of an otherwise perfect fit by far more than the
        # reliability limit, which is 10% of the patch side (10 px here).
        box = FaceBox(0, 0, 100, 100)
        target_points = reference_layout(box).copy()
        target_points[30] += 60
        result = align_to_face(
            Landmarks(reference_layout(box), box),
            Landmarks(target_points),
            box,
        )
        assert result.is_reliable is False

    def test_tiny_target_box_raises(self):
        box = FaceBox(0, 0, 1, 1)
        with pytest.raises(AlignmentError, match="too small"):
            align_to_face(
                Landmarks(reference_layout(box), box),
                Landmarks(reference_layout(box), box),
                box,
            )

    def test_affine_and_homography_methods_run(self):
        box = FaceBox(0, 0, 100, 100)
        landmarks = Landmarks(reference_layout(box), box)
        assert align_to_face(landmarks, landmarks, box, method="affine").method == "affine"
        assert (
            align_to_face(landmarks, landmarks, box, method="homography").method
            == "homography"
        )

    def test_similarity_from_eyes_matches_align_to_face(self):
        box = FaceBox(0, 0, 100, 100)
        landmarks = Landmarks(reference_layout(box), box)
        direct, _ = similarity_from_eyes(landmarks, landmarks)
        via_helper = align_to_face(landmarks, landmarks, box).matrix
        assert np.allclose(direct, via_helper, atol=1e-4)


class TestAlignmentResult:
    def test_is_reliable_scales_with_the_patch(self):
        # A fixed 8 px error is unacceptable on a 20 px face but negligible on a
        # 400 px one, so the same error must be judged differently.
        small = AlignmentResult(scale_matrix(1.0), (20, 20), 8.0, "similarity")
        large = AlignmentResult(scale_matrix(1.0), (400, 400), 8.0, "similarity")
        assert small.is_reliable is False
        assert large.is_reliable is True

    def test_is_reliable_has_an_absolute_floor(self):
        # Tiny sub-pixel errors must be accepted even on a very small patch,
        # otherwise a correctly aligned thumbnail would always be rejected.
        result = AlignmentResult(scale_matrix(1.0), (10, 10), 1.0, "similarity")
        assert result.is_reliable is True


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


class TestFaceHullMask:
    def test_hull_is_filled(self):
        box = FaceBox(0, 0, 100, 100)
        mask = face_hull_mask(Landmarks(reference_layout(box), box), (100, 100))
        assert mask.shape == (100, 100)
        assert mask[50, 50] == 255  # the centre is inside the hull
        assert mask[0, 0] == 0  # a corner is outside

    def test_too_few_points_gives_an_empty_mask(self):
        mask = face_hull_mask(Landmarks(np.zeros((2, 2), np.float32)), (10, 10))
        assert np.count_nonzero(mask) == 0


class TestFaceTransformer:
    def _aligned(self, size: int = 100):
        box = FaceBox(0, 0, size, size)
        landmarks = Landmarks(reference_layout(box), box)
        return box, landmarks, align_to_face(landmarks, landmarks, box)

    def test_warp_produces_a_patch_matching_the_box(self):
        source = draw_synthetic_face(200, 200, cx=100, cy=100, face_width=90, face_height=110)
        box, landmarks, alignment = self._aligned()
        warped = FaceTransformer().warp(source, landmarks, alignment, box)
        assert warped.patch.shape == (100, 100, 3)
        assert warped.mask.shape == (100, 100)
        assert warped.coverage > 0

    def test_warped_patch_is_uint8(self):
        source = draw_synthetic_face(200, 200, cx=100, cy=100)
        box, landmarks, alignment = self._aligned()
        warped = FaceTransformer().warp(source, landmarks, alignment, box)
        assert warped.patch.dtype == np.uint8

    def test_empty_source_raises(self):
        box, landmarks, alignment = self._aligned()
        with pytest.raises(TransformError, match="empty"):
            FaceTransformer().warp(np.zeros((0, 0, 3), np.uint8), landmarks, alignment, box)

    def test_grayscale_source_is_converted(self):
        gray = cv2.cvtColor(draw_synthetic_face(200, 200, cx=100, cy=100), cv2.COLOR_BGR2GRAY)
        box, landmarks, alignment = self._aligned()
        warped = FaceTransformer().warp(gray, landmarks, alignment, box)
        assert warped.patch.ndim == 3 and warped.patch.shape[2] == 3

    def test_alpha_source_is_converted(self):
        rgba = np.dstack(
            [draw_synthetic_face(200, 200, cx=100, cy=100), np.full((200, 200), 255, np.uint8)]
        )
        box, landmarks, alignment = self._aligned()
        warped = FaceTransformer().warp(rgba, landmarks, alignment, box)
        assert warped.patch.shape[2] == 3

    def test_flip_source_is_applied(self):
        source = draw_synthetic_face(200, 200, cx=100, cy=100)
        box, landmarks, alignment = self._aligned()
        normal = FaceTransformer(TransformConfig(flip_source=False)).warp(
            source, landmarks, alignment, box
        )
        flipped = FaceTransformer(TransformConfig(flip_source=True)).warp(
            source, landmarks, alignment, box
        )
        assert not np.array_equal(normal.patch, flipped.patch)

    def test_warped_face_size_property(self):
        source = draw_synthetic_face(200, 200, cx=100, cy=100)
        box, landmarks, alignment = self._aligned()
        warped = FaceTransformer().warp(source, landmarks, alignment, box)
        assert warped.size == (100, 100)

    def test_match_color_shifts_the_mean_towards_the_reference(self):
        rng = np.random.default_rng(0)
        patch = rng.integers(40, 80, (32, 32, 3), dtype=np.uint8)
        reference = rng.integers(150, 200, (32, 32, 3), dtype=np.uint8)
        result = FaceTransformer(TransformConfig(color_match=1.0)).match_color(
            patch, reference
        )
        assert result.mean() > patch.mean()

    def test_match_color_with_zero_strength_is_a_no_op(self):
        patch = np.full((16, 16, 3), 100, np.uint8)
        reference = np.full((16, 16, 3), 200, np.uint8)
        result = FaceTransformer(TransformConfig(color_match=0.0)).match_color(
            patch, reference
        )
        assert np.array_equal(result, patch)

    def test_match_color_handles_a_flat_patch(self):
        patch = np.full((16, 16, 3), 100, np.uint8)
        reference = np.full((16, 16, 3), 200, np.uint8)
        # A zero standard deviation must not cause a division by zero.
        assert FaceTransformer().match_color(patch, reference) is not None

    def test_match_color_resizes_a_mismatched_reference(self):
        patch = np.full((16, 16, 3), 100, np.uint8)
        reference = np.full((64, 64, 3), 200, np.uint8)
        assert FaceTransformer().match_color(patch, reference).shape == (16, 16, 3)

    def test_match_color_ignores_a_tiny_mask(self):
        patch = np.full((16, 16, 3), 100, np.uint8)
        reference = np.full((16, 16, 3), 200, np.uint8)
        mask = np.zeros((16, 16), np.uint8)
        mask[0, 0] = 255
        result = FaceTransformer().match_color(patch, reference, mask)
        assert np.array_equal(result, patch)

    def test_match_histogram_changes_a_dark_patch(self):
        patch = np.full((32, 32, 3), 40, np.uint8)
        patch[:, :16] = 60
        reference = np.full((32, 32, 3), 200, np.uint8)
        reference[:, :16] = 220
        result = FaceTransformer().match_histogram(patch, reference)
        assert result.mean() > patch.mean()

    def test_sharpen_zero_is_a_no_op(self):
        patch = draw_synthetic_face(64, 64, cx=32, cy=32, face_width=40, face_height=48)
        assert np.array_equal(FaceTransformer(TransformConfig(sharpen=0.0)).sharpen(patch), patch)

    def test_sharpen_increases_local_contrast(self):
        patch = draw_synthetic_face(64, 64, cx=32, cy=32, face_width=40, face_height=48)
        sharpened = FaceTransformer(TransformConfig(sharpen=1.0)).sharpen(patch)
        assert sharpened.std() >= patch.std()

    def test_brightness_offset_is_applied(self):
        patch = np.full((8, 8, 3), 100, np.uint8)
        brighter = FaceTransformer(TransformConfig(brightness=0.2)).adjust_brightness(patch)
        assert brighter.mean() > patch.mean()

    def test_brightness_zero_is_a_no_op(self):
        patch = np.full((8, 8, 3), 100, np.uint8)
        assert np.array_equal(
            FaceTransformer(TransformConfig(brightness=0.0)).adjust_brightness(patch),
            patch,
        )

    def test_build_patch_combines_every_step(self):
        frame = draw_synthetic_face(200, 200, cx=100, cy=100, face_width=90, face_height=110)
        box = FaceBox(50, 50, 100, 100)
        landmarks = Landmarks(reference_layout(box), box)
        alignment = align_to_face(landmarks, landmarks, box)
        warped = FaceTransformer().build_patch(
            frame, landmarks, alignment, box, frame
        )
        assert warped.patch.shape == (100, 100, 3)
        assert warped.coverage > 0

    def test_sample_reference_returns_the_region(self):
        frame = draw_synthetic_face(200, 200, cx=100, cy=100)
        box = FaceBox(50, 50, 100, 100)
        landmarks = Landmarks(reference_layout(box), box)
        region = FaceTransformer().sample_reference(frame, landmarks, box)
        assert region is not None and region.shape[:2] == (100, 100)

    def test_sample_reference_outside_the_frame_returns_none(self):
        frame = draw_synthetic_face(64, 64, cx=32, cy=32)
        box = FaceBox(1000, 1000, 50, 50)
        landmarks = Landmarks(reference_layout(box), box)
        assert FaceTransformer().sample_reference(frame, landmarks, box) is None


# ---------------------------------------------------------------------------
# Blender
# ---------------------------------------------------------------------------


class TestFeatherMask:
    def test_hard_mask_without_feather(self):
        mask = np.full((20, 20), 255, np.uint8)
        alpha = feather_mask(mask, 0.0)
        assert alpha.max() == pytest.approx(1.0)
        assert alpha.min() == pytest.approx(1.0)

    def test_feather_softens_the_edge(self):
        mask = np.zeros((80, 80), np.uint8)
        mask[20:60, 20:60] = 255
        alpha = feather_mask(mask, 0.5)
        # The middle of the block stays fully opaque, the edge becomes partial.
        assert alpha[40, 40] == pytest.approx(1.0, abs=0.02)
        assert 0.0 < alpha[20, 40] < 1.0

    def test_feather_keeps_the_interior_opaque(self):
        mask = np.zeros((80, 80), np.uint8)
        mask[10:70, 10:70] = 255
        alpha = feather_mask(mask, 0.2)
        assert alpha[40, 40] == pytest.approx(1.0, abs=0.02)

    def test_empty_mask_returns_an_empty_alpha(self):
        assert feather_mask(np.zeros((0, 0), np.uint8), 0.5).size == 1

    def test_alpha_is_always_in_range(self):
        mask = np.zeros((30, 30), np.uint8)
        mask[5:25, 5:25] = 255
        alpha = feather_mask(mask, 1.0)
        assert alpha.min() >= 0.0 and alpha.max() <= 1.0


class TestBuildAlpha:
    def _warped(self, size: int = 80):
        box = FaceBox(0, 0, size, size)
        landmarks = Landmarks(reference_layout(box), box)
        alignment = align_to_face(landmarks, landmarks, box)
        source = draw_synthetic_face(size, size, cx=size // 2, cy=size // 2)
        warped = FaceTransformer().warp(source, landmarks, alignment, box)
        return box, landmarks, warped

    def test_alpha_shape_matches_the_patch(self):
        box, landmarks, warped = self._warped()
        alpha = build_alpha(warped, landmarks, box, BlendConfig())
        assert alpha.shape == warped.patch.shape[:2]

    def test_alpha_is_within_range(self):
        box, landmarks, warped = self._warped()
        alpha = build_alpha(warped, landmarks, box, BlendConfig())
        assert alpha.min() >= 0.0 and alpha.max() <= 1.0

    def test_opacity_scales_the_alpha(self):
        box, landmarks, warped = self._warped()
        full = build_alpha(warped, landmarks, box, BlendConfig(opacity=1.0))
        half = build_alpha(warped, landmarks, box, BlendConfig(opacity=0.5))
        assert half.max() <= full.max()
        assert half.max() <= 0.5 + 1e-6

    def test_hull_mask_can_be_disabled(self):
        box, landmarks, warped = self._warped()
        alpha = build_alpha(
            warped, landmarks, box, BlendConfig(use_hull_mask=False)
        )
        assert alpha.max() > 0


class TestFaceBlender:
    def _prepare(self, frame_size: int = 200, face_size: int = 90):
        frame = draw_synthetic_face(
            frame_size, frame_size, cx=frame_size // 2, cy=frame_size // 2,
            face_width=face_size, face_height=int(face_size * 1.2),
        )
        box = FaceBox(
            frame_size // 2 - face_size // 2,
            frame_size // 2 - face_size // 2,
            face_size, face_size,
        )
        landmarks = Landmarks(reference_layout(box), box)
        alignment = align_to_face(landmarks, landmarks, box)
        # A distinctly coloured source face, so a successful blend visibly
        # changes pixels. Using the frame itself would make the swap a no-op.
        source = draw_synthetic_face(
            frame_size, frame_size, cx=frame_size // 2, cy=frame_size // 2,
            face_width=face_size, face_height=int(face_size * 1.2),
            skin=(80, 200, 120),
        )
        warped = FaceTransformer().warp(source, landmarks, alignment, box)
        return frame, box, landmarks, warped

    def test_blend_returns_a_frame_of_the_same_shape(self):
        frame, _box, landmarks, warped = self._prepare()
        result, stats = FaceBlender().blend(frame, warped, landmarks)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8
        assert stats.coverage > 0

    def test_blend_does_not_mutate_the_input(self):
        frame, _box, landmarks, warped = self._prepare()
        before = frame.copy()
        FaceBlender().blend(frame, warped, landmarks)
        assert np.array_equal(frame, before)

    def test_blend_changes_pixels_inside_the_face(self):
        frame, box, landmarks, warped = self._prepare()
        result, _ = FaceBlender().blend(frame, warped, landmarks)
        centre = (box.y + box.height // 2, box.x + box.width // 2)
        assert not np.array_equal(
            result[centre[0] - 5 : centre[0] + 5, centre[1] - 5 : centre[1] + 5],
            frame[centre[0] - 5 : centre[0] + 5, centre[1] - 5 : centre[1] + 5],
        )

    def test_pixels_outside_the_face_are_untouched(self):
        frame, _box, landmarks, warped = self._prepare()
        result, _ = FaceBlender().blend(frame, warped, landmarks)
        assert np.array_equal(result[0:5, 0:5], frame[0:5, 0:5])

    def test_zero_opacity_leaves_the_frame_alone(self):
        frame, _box, landmarks, warped = self._prepare()
        result, stats = FaceBlender(BlendConfig(opacity=0.0)).blend(
            frame, warped, landmarks
        )
        assert np.array_equal(result, frame)
        assert stats.coverage == 0.0

    def test_empty_frame_raises(self):
        _frame, _box, landmarks, warped = self._prepare()
        with pytest.raises(BlendError, match="empty frame"):
            FaceBlender().blend(np.zeros((0, 0, 3), np.uint8), warped, landmarks)

    def test_tiny_box_raises(self):
        frame, _box, landmarks, warped = self._prepare()
        from LightSwapConverter.core.transformer import WarpedFace

        tiny = WarpedFace(
            warped.patch[:1, :1], warped.mask[:1, :1], FaceBox(0, 0, 1, 1), warped.matrix
        )
        with pytest.raises(BlendError, match="too small"):
            FaceBlender().blend(frame, tiny, landmarks)

    def test_blend_in_place_updates_the_frame(self):
        frame, _box, landmarks, warped = self._prepare()
        before = frame.copy()
        FaceBlender().blend_in_place(frame, warped, landmarks)
        assert not np.array_equal(frame, before)

    def test_seamless_mode_still_produces_a_frame(self):
        frame, _box, landmarks, warped = self._prepare()
        result, stats = FaceBlender(BlendConfig(seamless=True)).blend(
            frame, warped, landmarks
        )
        assert result.shape == frame.shape
        assert stats.method in {"seamless", "alpha"}

    def test_mask_preview_matches_the_box_region(self):
        frame, box, landmarks, warped = self._prepare()
        preview = mask_preview(frame, warped, landmarks)
        assert preview.shape == (box.height, box.width, 3)

    def test_module_level_composite_helper(self):
        from LightSwapConverter.core.blender import composite

        frame, _box, landmarks, warped = self._prepare()
        result, _ = composite(frame, warped, landmarks)
        assert result.shape == frame.shape
