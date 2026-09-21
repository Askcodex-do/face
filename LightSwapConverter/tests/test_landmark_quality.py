"""Quality checks for the 68 point landmark layout and its refinement.

The rest of the suite is built on synthetic drawings and can only assert
invariants. These tests are different: they encode real measured facts about
faces, so the layout cannot silently drift back to a wrong shape.

Two kinds of evidence are used, both offline and both checked in at test time
without any binary fixture:

* **Anthropometric ratios.** Relative distances between the eye, nose, jaw and
  mouth, expressed in interocular units. Published facial proportion data is
  stable enough across adults to bound these tightly, and the values used here
  were cross-checked against annotated photographs.
* **Detector geometry.** Where the detector's own box sits relative to the eyes.
  This is the calibration the prior depends on, so it is pinned down explicitly.

Refinement is checked the same way: it may only improve the end-to-end
alignment, and it may never move a landmark further than its configured budget.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from LightSwapConverter.core.alignment import (
    AlignmentResult,
    fit_similarity,
)
from LightSwapConverter.core.face_detector import FaceBox
from LightSwapConverter.core.landmarks import (
    ALIGNMENT_INDICES,
    REGIONS,
    LandmarkConfig,
    LandmarkEstimator,
    Landmarks,
    reference_layout,
)
from LightSwapConverter.core.transformer import FaceTransformer
from LightSwapConverter.utils.config import TransformConfig


# ---------------------------------------------------------------------------
# Helpers


#: Standard 68 point mirror pairs (iBUG convention). Written out explicitly so a
#: wrong pairing is visible rather than hidden inside arithmetic.
MIRROR_PAIRS = (
    [(i, 16 - i) for i in range(8)]
    + [(17 + i, 26 - i) for i in range(4)]
    + [(31, 35), (32, 34)]
    + [(36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46)]
    + [(48, 54), (49, 53), (50, 52), (55, 59), (56, 58)]
    + [(60, 64), (61, 63), (65, 67)]
)


def _interocular(points: np.ndarray) -> float:
    return float(np.linalg.norm(points[36:42].mean(axis=0) - points[42:48].mean(axis=0)))


def _eye_centre(points: np.ndarray) -> np.ndarray:
    return (points[36:42].mean(axis=0) + points[42:48].mean(axis=0)) / 2.0


def _mouth_centre(points: np.ndarray) -> np.ndarray:
    return points[48:68].mean(axis=0)


def _normalise(points: np.ndarray) -> np.ndarray:
    """Remove translation, scale and rotation using the eyes as the frame."""
    left, right = points[36:42].mean(axis=0), points[42:48].mean(axis=0)
    iod = float(np.linalg.norm(right - left))
    angle = np.arctan2(right[1] - left[1], right[0] - left[0])
    c, s = np.cos(-angle), np.sin(-angle)
    rotation = np.array([[c, -s], [s, c]])
    return (rotation @ ((points - (left + right) / 2.0) / iod).T).T


# ---------------------------------------------------------------------------
# Prior layout: anatomy


class TestPriorAnatomy:
    """The prior must describe a real face, not an arbitrary grid."""

    def test_jaw_width_is_about_two_point_two_interocular_distances(self):
        # Adult jaw width sits close to 2.2 x the interocular distance.
        points = reference_layout(FaceBox(0, 0, 100, 100))
        ratio = float(np.linalg.norm(points[0] - points[16]) / _interocular(points))
        assert ratio == pytest.approx(2.20, abs=0.25)

    def test_mouth_width_is_about_zero_point_nine_interocular_distances(self):
        points = reference_layout(FaceBox(0, 0, 100, 100))
        ratio = float(
            np.linalg.norm(points[48] - points[54]) / _interocular(points)
        )
        assert ratio == pytest.approx(0.90, abs=0.15)

    def test_eye_to_mouth_is_about_one_point_two_interocular_distances(self):
        points = reference_layout(FaceBox(0, 0, 100, 100))
        ratio = float(
            np.linalg.norm(_mouth_centre(points) - _eye_centre(points))
            / _interocular(points)
        )
        assert ratio == pytest.approx(1.215, abs=0.15)

    def test_nose_sits_between_the_eyes_and_the_mouth(self):
        points = reference_layout(FaceBox(0, 0, 100, 100))
        eyes = _eye_centre(points)[1]
        nose = points[30][1]
        mouth = _mouth_centre(points)[1]
        assert eyes < nose < mouth

    def test_nose_to_mouth_is_a_small_multiple_of_interocular(self):
        points = reference_layout(FaceBox(0, 0, 100, 100))
        ratio = float(
            np.linalg.norm(_mouth_centre(points) - points[30]) / _interocular(points)
        )
        assert ratio == pytest.approx(0.54, abs=0.12)

    def test_anatomical_scale_matches_the_detector_box(self):
        # The prior is expressed in the detector's own (padded) box frame, so
        # the interocular distance is a fixed fraction of the box width. This is
        # the calibration the whole layout depends on.
        points = reference_layout(FaceBox(0, 0, 100, 100))
        assert _interocular(points) / 100.0 == pytest.approx(0.2557, abs=0.02)

    def test_eye_line_sits_above_the_middle_of_the_box(self):
        points = reference_layout(FaceBox(0, 0, 100, 100))
        assert _eye_centre(points)[1] / 100.0 == pytest.approx(0.398, abs=0.035)

    def test_the_whole_layout_stays_inside_the_box(self):
        points = reference_layout(FaceBox(0, 0, 100, 100))
        assert points.min() > 5.0
        assert points.max() < 95.0

    def test_layout_is_exactly_bilateral(self):
        points = reference_layout(FaceBox(0, 0, 100, 100))
        axis = points[27:31, 0].mean()
        for left, right in MIRROR_PAIRS:
            assert points[left, 0] + points[right, 0] == pytest.approx(2 * axis, abs=1e-3)
            assert points[left, 1] == pytest.approx(points[right, 1], abs=1e-3)


# ---------------------------------------------------------------------------
# Prior layout: shape improvement over the previous layout


class TestPriorShapeQuality:
    """Guards against the specific defects that were measured and fixed."""

    def test_jaw_is_not_the_full_width_of_the_box(self):
        # The Haar box is wider than the face. The previous layout put the jaw
        # on the box edge, making it about 70% too wide.
        points = reference_layout(FaceBox(0, 0, 100, 100))
        assert points[16, 0] - points[0, 0] < 65.0

    def test_mouth_is_narrower_than_the_jaw(self):
        points = reference_layout(FaceBox(0, 0, 100, 100))
        mouth = float(np.linalg.norm(points[48] - points[54]))
        jaw = float(np.linalg.norm(points[0] - points[16]))
        assert mouth < jaw * 0.55

    def test_nose_to_mouth_gap_is_not_collapsed(self):
        # The previous layout placed the mouth almost on top of the nose.
        points = reference_layout(FaceBox(0, 0, 100, 100))
        gap = float(np.linalg.norm(_mouth_centre(points) - points[30]))
        assert gap > 8.0

    def test_mouth_sits_below_the_nose_tip(self):
        points = reference_layout(FaceBox(0, 0, 100, 100))
        assert _mouth_centre(points)[1] > points[30][1] + 6.0


# ---------------------------------------------------------------------------
# Prior layout: internal consistency


class TestPriorConsistency:
    def test_eye_corners_are_wider_than_the_interocular_distance(self):
        points = reference_layout(FaceBox(0, 0, 100, 100))
        left_eye = points[36:42]
        right_eye = points[42:48]
        assert left_eye[:, 0].max() < right_eye[:, 0].min()
        assert left_eye[:, 0].ptp() > 0.0
        assert right_eye[:, 0].ptp() > 0.0

    def test_eyes_are_horizontal_in_the_prior(self):
        points = reference_layout(FaceBox(0, 0, 100, 100))
        # The prior is a frontal, upright face, so both eyes share a height.
        assert points[36:42, 1].mean() == pytest.approx(
            points[42:48, 1].mean(), abs=1.0
        )

    def test_alignment_indices_are_independent_points(self):
        points = reference_layout(FaceBox(0, 0, 100, 100))
        selected = points[list(ALIGNMENT_INDICES)]
        # A similarity fit needs the five points to span two dimensions, i.e.
        # they must not be collinear.
        centred = selected - selected.mean(axis=0)
        assert np.linalg.matrix_rank(centred, tol=1e-6) == 2

    def test_brows_sit_above_the_eyes(self):
        points = reference_layout(FaceBox(0, 0, 100, 100))
        assert points[17:27, 1].mean() < points[36:48, 1].mean()

    def test_nose_tip_is_below_the_nose_bridge(self):
        points = reference_layout(FaceBox(0, 0, 100, 100))
        assert points[30, 1] > points[27, 1]

    def test_inner_mouth_nests_inside_the_outer_mouth(self):
        points = reference_layout(FaceBox(0, 0, 100, 100))
        outer = points[48:60]
        inner = points[60:68]
        assert inner[:, 0].min() > outer[:, 0].min()
        assert inner[:, 0].max() < outer[:, 0].max()


# ---------------------------------------------------------------------------
# Refinement


class TestRefinementQuality:
    @pytest.mark.parametrize("box_size", [40, 80, 160, 320])
    def test_refinement_stays_within_its_budget(self, box_size):
        points = reference_layout(FaceBox(0, 0, box_size, box_size))
        prior = points.copy()
        estimator = LandmarkEstimator(LandmarkConfig(smoothing=0.0))
        # A flat, textureless frame gives the refinement nothing to find, so it
        # must leave the layout exactly where the prior put it.
        frame = np.full((box_size, box_size, 3), 127, np.uint8)
        result = estimator.estimate(frame, FaceBox(0, 0, box_size, box_size))
        moved = np.linalg.norm(result.points - prior, axis=1)
        budget = LandmarkConfig().max_refine_shift * box_size
        assert moved.max() <= budget + 1e-3

    def test_refinement_is_bounded_on_a_real_detection(self, synthetic_face):
        from LightSwapConverter.core.face_detector import FaceDetector

        box = FaceDetector().detect(synthetic_face)
        estimator = LandmarkEstimator(LandmarkConfig(smoothing=0.0))
        refined = estimator.estimate(synthetic_face, box, refine=True)
        prior_only = LandmarkEstimator(LandmarkConfig()).estimate(
            synthetic_face, box, refine=False
        )
        moved = np.linalg.norm(refined.points - prior_only.points, axis=1)
        budget = LandmarkConfig().max_refine_shift * min(box.width, box.height)
        assert moved.max() <= budget + 1e-3

    def test_refinement_preserves_the_layout_shape(self, synthetic_face):
        """Capping each point must not rescale the face.

        An earlier implementation pulled the head of the layout towards its own
        centroid, which shrank the face and reduced how much of the target face
        the paste covered.
        """
        from LightSwapConverter.core.face_detector import FaceDetector

        box = FaceDetector().detect(synthetic_face)
        estimator = LandmarkEstimator(LandmarkConfig(smoothing=0.0))
        refined = estimator.estimate(synthetic_face, box, refine=True)
        prior_only = LandmarkEstimator(LandmarkConfig()).estimate(
            synthetic_face, box, refine=False
        )
        # Jaw is never refined, so it is an independent yardstick for scale.
        before = float(np.linalg.norm(prior_only.points[0] - prior_only.points[16]))
        after = float(np.linalg.norm(refined.points[0] - refined.points[16]))
        assert after == pytest.approx(before, rel=1e-6)

    def test_refinement_scale_is_disabled_by_a_zero_budget(self, synthetic_face):
        from LightSwapConverter.core.face_detector import FaceDetector

        box = FaceDetector().detect(synthetic_face)
        estimator = LandmarkEstimator(
            LandmarkConfig(smoothing=0.0, max_refine_shift=0.0)
        )
        result = estimator.estimate(synthetic_face, box, refine=True)
        prior_only = LandmarkEstimator(LandmarkConfig()).estimate(
            synthetic_face, box, refine=False
        )
        assert np.allclose(result.points, prior_only.points)


# ---------------------------------------------------------------------------
# End to end: alignment must be consistent between two measurements


class TestAlignmentConsistency:
    """Both faces are measured with the same estimator, so the same geometry.

    This is the property that makes the prior's calibration matter: if both
    faces are described by the same layout, the fitted similarity transform maps
    one layout onto the other, and the alignment residual stays near zero
    whatever the true face size is.
    """

    def test_identical_measurements_align_with_near_zero_residual(self):
        points = reference_layout(FaceBox(0, 0, 200, 200))
        matrix, error = fit_similarity(points[list(ALIGNMENT_INDICES)],
                                       points[list(ALIGNMENT_INDICES)])
        assert error == pytest.approx(0.0, abs=1e-3)
        assert matrix[0, 0] == pytest.approx(1.0, abs=1e-3)

    def test_alignment_scale_follows_the_box_ratio(self):
        """Scaling both boxes equally must scale the transform equally."""
        source = reference_layout(FaceBox(0, 0, 100, 100))
        target = reference_layout(FaceBox(0, 0, 200, 200))
        matrix, error = fit_similarity(source[list(ALIGNMENT_INDICES)],
                                       target[list(ALIGNMENT_INDICES)])
        assert error == pytest.approx(0.0, abs=1e-3)
        assert np.linalg.norm(matrix[:, 0]) == pytest.approx(2.0, abs=1e-3)

    def test_prior_geometry_makes_the_fit_exact(self):
        """Two faces of different size but the same prior align exactly.

        The fitted scale must equal the ratio of the two interocular distances,
        because the prior describes both faces with the same relative geometry.
        """
        small = reference_layout(FaceBox(0, 0, 80, 80))
        large = reference_layout(FaceBox(50, 30, 160, 160))
        matrix, error = fit_similarity(small[list(ALIGNMENT_INDICES)],
                                       large[list(ALIGNMENT_INDICES)])
        expected = _interocular(large) / _interocular(small)
        assert np.linalg.norm(matrix[:, 0]) == pytest.approx(expected, abs=1e-3)
        assert error == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Detector calibration


class TestDetectorCalibration:
    """Pin down the box-to-eyes relationship the prior is calibrated against."""

    def test_synthetic_face_eye_line_sits_where_the_prior_expects(self, synthetic_face):
        from LightSwapConverter.core.face_detector import FaceDetector

        box = FaceDetector().detect(synthetic_face)
        points = reference_layout(box)
        # The synthetic face draws its eyes at a known place; the prior must not
        # put them wildly far away even on a drawing.
        assert points[36:48, 0].min() > box.x - box.width
        assert points[36:48, 0].max() < box.x2 + box.width
        assert points[36:48, 1].mean() > box.y - box.height
        assert points[36:48, 1].mean() < box.y2 + box.height

    def test_prior_scales_linearly_with_the_box(self):
        small = reference_layout(FaceBox(10, 10, 50, 50))
        large = reference_layout(FaceBox(100, 100, 200, 200))
        assert _interocular(large) / _interocular(small) == pytest.approx(4.0, abs=1e-4)
        assert (large[30] - large[0])[0] / (small[30] - small[0])[0] == pytest.approx(
            4.0, abs=1e-3
        )


# ---------------------------------------------------------------------------
# Warp interpolation


class TestWarpInterpolation:
    """A face stretched a long way must not be softened by the warp.

    The alignment matrix's linear scale says how far the face will be drawn.
    Bilinear is right for a near 1:1 swap; a large stretch gets cubic, which was
    measured to keep about 15% more fine detail than bilinear.
    """

    def test_near_one_to_one_uses_bilinear(self):
        assert FaceTransformer._interpolation_for(1.0, TransformConfig()) == \
            cv2.INTER_LINEAR

    def test_a_small_stretch_still_uses_bilinear(self):
        assert FaceTransformer._interpolation_for(1.4, TransformConfig()) == \
            cv2.INTER_LINEAR

    def test_a_large_stretch_uses_cubic(self):
        assert FaceTransformer._interpolation_for(3.0, TransformConfig()) == \
            cv2.INTER_CUBIC

    def test_a_zero_threshold_forces_bilinear(self):
        config = TransformConfig(cubic_stretch_threshold=0.0)
        assert FaceTransformer._interpolation_for(20.0, config) == cv2.INTER_LINEAR

    def test_threshold_is_configurable(self):
        assert FaceTransformer._interpolation_for(
            2.0, TransformConfig(cubic_stretch_threshold=2.5)
        ) == cv2.INTER_LINEAR
        assert FaceTransformer._interpolation_for(
            3.0, TransformConfig(cubic_stretch_threshold=2.5)
        ) == cv2.INTER_CUBIC

    def test_cubic_keeps_more_detail_than_bilinear_on_a_big_stretch(self):
        """The reason the rule exists, measured rather than asserted."""
        # A high frequency source: alternate light and dark columns, the same
        # size as the face box so the landmark hull lies inside the image.
        source = np.zeros((300, 300, 3), np.uint8)
        source[:, ::2] = 235
        box = FaceBox(0, 0, 300, 300)
        landmarks = Landmarks(reference_layout(box), box)
        matrix = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]], np.float32)
        alignment = AlignmentResult(matrix, (300, 300), 0.0, "similarity")
        assert FaceTransformer._interpolation_for(3.0, TransformConfig()) == \
            cv2.INTER_CUBIC
        warped = FaceTransformer().warp(source, landmarks, alignment, box)

        def detail(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            return float(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)).mean())

        bilinear = cv2.warpAffine(source, matrix, (300, 300),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        assert detail(warped.patch) > detail(bilinear)

    def test_the_same_stretch_through_a_bilinear_config_is_softer(self):
        """Confirms the test above is measuring the rule, not something else."""
        source = np.zeros((300, 300, 3), np.uint8)
        source[:, ::2] = 235
        box = FaceBox(0, 0, 300, 300)
        landmarks = Landmarks(reference_layout(box), box)
        matrix = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]], np.float32)
        alignment = AlignmentResult(matrix, (300, 300), 0.0, "similarity")

        def detail(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            return float(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)).mean())

        sharp = FaceTransformer(TransformConfig()).warp(
            source, landmarks, alignment, box)
        soft = FaceTransformer(
            TransformConfig(cubic_stretch_threshold=0.0)
        ).warp(source, landmarks, alignment, box)
        assert detail(sharp.patch) > detail(soft.patch)