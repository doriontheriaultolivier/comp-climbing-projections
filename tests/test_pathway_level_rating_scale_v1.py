import unittest

import numpy as np

from scripts.pathway_level_rating_scale_v1 import (
    PathwayAnchors,
    PathwayScaleError,
    display_rating,
    half_probability_skill,
    skill_from_display_rating,
)


class PathwayLevelRatingScaleTests(unittest.TestCase):
    def setUp(self):
        self.anchors = PathwayAnchors(
            context="WC",
            semifinal_half_skill=1.5,
            win_half_skill=3.5,
            reference_definition="event-balanced frozen 2025 WC fields",
        )

    def test_two_anchors_have_exact_interpretation(self):
        result = display_rating(np.array([1.5, 3.5]), self.anchors)
        np.testing.assert_allclose(result, [2000.0, 3000.0])

    def test_transform_round_trip_and_order(self):
        skills = np.array([-1.0, 0.0, 1.0, 4.0])
        ratings = display_rating(skills, self.anchors)
        self.assertTrue(np.all(np.diff(ratings) > 0))
        np.testing.assert_allclose(skill_from_display_rating(ratings, self.anchors), skills)

    def test_half_probability_interpolates(self):
        value = half_probability_skill(
            np.array([0.0, 1.0, 2.0]), np.array([0.1, 0.4, 0.8])
        )
        self.assertAlmostEqual(value, 1.25)

    def test_unsupported_or_reversed_anchors_fail_closed(self):
        with self.assertRaises(PathwayScaleError):
            PathwayAnchors("OLYM", 2.0, 2.0, "one event").validate()
        with self.assertRaises(PathwayScaleError):
            half_probability_skill(
                np.array([0.0, 1.0]), np.array([0.1, 0.4])
            )


if __name__ == "__main__":
    unittest.main()
