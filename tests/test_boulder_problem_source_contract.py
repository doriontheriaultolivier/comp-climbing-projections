import unittest

from scripts.boulder_problem_source_contract import (
    CEC_CANADIAN_A_JR_SHARED_V1,
    COMBINED_BOULDER_ASCENTS,
    DIRECT_BOULDER_ASCENTS,
    IFSC_25_10_V1,
    IFSC_COMBINED_5_10_25_FLAGS_V1,
    LEGACY_EXPLICIT_FLAGS,
    BoulderSourceContractError,
    decode_problem_outcome,
    extract_boulder_ascent_groups,
    problem_identity,
    reviewed_cec_terrain_alias,
)


def decode(ascent, schema=IFSC_25_10_V1):
    source_lane = (
        COMBINED_BOULDER_ASCENTS
        if schema == IFSC_COMBINED_5_10_25_FLAGS_V1
        else DIRECT_BOULDER_ASCENTS
    )
    return decode_problem_outcome(
        ascent,
        problem_index=1,
        scoring_schema=schema,
        source_lane=source_lane,
    )


class ModernPointsFallbackTests(unittest.TestCase):
    def test_zero_is_no_zone_and_is_not_lost_by_truthiness(self):
        result = decode({"points": 0, "top_tries": 5, "zone_tries": 5})
        self.assertEqual(result.stage, "no_zone")
        self.assertEqual(result.points, 0.0)
        self.assertFalse(result.reached_zone)
        self.assertEqual(result.total_attempts, 5)

    def test_positive_through_exact_ten_is_zone_only(self):
        for points in (0.1, 9.8, 10.0):
            with self.subTest(points=points):
                result = decode(
                    {"points": points, "top_tries": 6, "zone_tries": 3}
                )
                self.assertEqual(result.stage, "zone_only")
                self.assertTrue(result.reached_zone)
                self.assertFalse(result.reached_top)
                self.assertEqual(result.attempts_to_zone, 3)
                self.assertIsNone(result.attempts_to_top)

    def test_above_ten_is_top(self):
        result = decode({"points": 24.8, "top_tries": 3, "zone_tries": 1})
        self.assertEqual(result.stage, "top")
        self.assertEqual(result.attempts_to_top, 3)
        self.assertEqual(result.total_attempts, 3)
        self.assertFalse(result.contradictions)

    def test_missing_points_and_flags_remains_unknown(self):
        result = decode({"top_tries": 4, "zone_tries": 4})
        self.assertEqual(result.stage, "unknown")
        self.assertIsNone(result.reached_zone)

    def test_points_above_modern_max_are_not_coerced(self):
        result = decode({"points": 36, "top_tries": 1, "zone_tries": 1})
        self.assertEqual(result.stage, "unknown")
        self.assertIn("modern_points_outside_0_to_25", result.contradictions)


class ExplicitAndOldEraTests(unittest.TestCase):
    def test_explicit_flags_are_authoritative_and_conflicts_are_retained(self):
        result = decode(
            {
                "top": False,
                "zone": True,
                "points": 24.9,
                "top_tries": 4,
                "zone_tries": 2,
            }
        )
        self.assertEqual(result.stage, "zone_only")
        self.assertEqual(result.marker_evidence, "explicit_api_flags")
        self.assertTrue(any("conflicts_with_points" in item for item in result.contradictions))

    def test_explicit_top_normalizes_zone_but_flags_source_contradiction(self):
        result = decode({"top": True, "zone": False, "top_tries": 2, "zone_tries": 1})
        self.assertEqual(result.stage, "top")
        self.assertIn("top_true_zone_false", result.contradictions[0])

    def test_legacy_numeric_points_are_never_guessed(self):
        result = decode(
            {"points": 25, "top_tries": 1, "zone_tries": 1},
            LEGACY_EXPLICIT_FLAGS,
        )
        self.assertEqual(result.stage, "unknown")
        self.assertIn("legacy_points_not_decoded", result.marker_evidence)

    def test_legacy_explicit_flags_still_decode(self):
        result = decode(
            {"top": True, "zone": True, "top_tries": 4, "zone_tries": 2},
            LEGACY_EXPLICIT_FLAGS,
        )
        self.assertEqual(result.stage, "top")

    def test_partial_explicit_flags_do_not_turn_missing_into_false(self):
        top_unknown = decode(
            {"zone": True, "zone_tries": 2},
            LEGACY_EXPLICIT_FLAGS,
        )
        self.assertEqual(top_unknown.stage, "unknown")
        self.assertIsNone(top_unknown.reached_top)
        self.assertTrue(top_unknown.reached_zone)

        zone_unknown = decode(
            {"top": False, "top_tries": 3},
            LEGACY_EXPLICIT_FLAGS,
        )
        self.assertEqual(zone_unknown.stage, "unknown")
        self.assertFalse(zone_unknown.reached_top)
        self.assertIsNone(zone_unknown.reached_zone)

        auxiliary_only = decode(
            {"low_zone": True, "low_zone_tries": 4},
            LEGACY_EXPLICIT_FLAGS,
        )
        self.assertEqual(auxiliary_only.stage, "unknown")
        self.assertTrue(auxiliary_only.reached_low_zone)
        self.assertIsNone(auxiliary_only.reached_zone)

    def test_lower_marker_false_safely_rules_out_higher_markers(self):
        result = decode(
            {"zone": False, "zone_tries": 3},
            LEGACY_EXPLICIT_FLAGS,
        )
        self.assertEqual(result.stage, "no_zone")
        self.assertFalse(result.reached_top)
        self.assertFalse(result.reached_zone)

    def test_undeclared_schema_is_rejected(self):
        with self.assertRaisesRegex(BoulderSourceContractError, "undeclared"):
            decode_problem_outcome(
                {},
                problem_index=1,
                scoring_schema="guess",
                source_lane=DIRECT_BOULDER_ASCENTS,
            )


class CombinedFiveTenTwentyFiveTests(unittest.TestCase):
    def test_numeric_points_alone_never_decode_combined_markers(self):
        for points in (0.0, 4.7, 5.0, 10.0, 24.9, 25.0):
            with self.subTest(points=points):
                result = decode(
                    {
                        "points": points,
                        "top_tries": 1,
                        "zone_tries": 1,
                        "low_zone_tries": 1,
                    },
                    IFSC_COMBINED_5_10_25_FLAGS_V1,
                )
                self.assertEqual(result.stage, "unknown")
                self.assertEqual(result.ordered_stage, "unknown")
                self.assertIsNone(result.reached_low_zone)
                self.assertIn("explicit_flags_required", result.marker_evidence)
                self.assertEqual(result.points, float(points))

    def test_explicit_low_zone_zone_and_top_form_ordered_stages(self):
        cases = (
            (
                {
                    "points": 4.7,
                    "top": False,
                    "zone": False,
                    "low_zone": True,
                    "top_tries": 0,
                    "zone_tries": 0,
                    "low_zone_tries": 4,
                },
                "low_zone_only",
                "no_zone",
                4,
            ),
            (
                {
                    "points": 9.9,
                    "top": False,
                    "zone": True,
                    "low_zone": True,
                    "top_tries": 0,
                    "zone_tries": 2,
                    "low_zone_tries": 1,
                },
                "zone_only",
                "zone_only",
                1,
            ),
            (
                {
                    "points": 24.7,
                    "top": True,
                    "zone": True,
                    "low_zone": True,
                    "top_tries": 4,
                    "zone_tries": 3,
                    "low_zone_tries": 2,
                },
                "top",
                "top",
                2,
            ),
        )
        for ascent, ordered_stage, stage, low_attempt in cases:
            with self.subTest(ordered_stage=ordered_stage):
                result = decode(ascent, IFSC_COMBINED_5_10_25_FLAGS_V1)
                self.assertEqual(result.ordered_stage, ordered_stage)
                self.assertEqual(result.stage, stage)
                self.assertEqual(result.attempts_to_low_zone, low_attempt)
                self.assertIsNone(result.total_attempts)

    def test_explicit_flags_override_points_without_inventing_thresholds(self):
        result = decode(
            {
                "points": 24.9,
                "top": False,
                "zone": False,
                "low_zone": True,
                "top_tries": 0,
                "zone_tries": 0,
                "low_zone_tries": 2,
            },
            IFSC_COMBINED_5_10_25_FLAGS_V1,
        )
        self.assertEqual(result.ordered_stage, "low_zone_only")
        self.assertFalse(result.reached_top)
        self.assertFalse(result.reached_zone)
        self.assertTrue(result.reached_low_zone)
        self.assertFalse(
            any("points_stage" in item for item in result.contradictions)
        )

    def test_partial_combined_flags_remain_unknown(self):
        result = decode(
            {"low_zone": True, "low_zone_tries": 2},
            IFSC_COMBINED_5_10_25_FLAGS_V1,
        )
        self.assertEqual(result.stage, "unknown")
        self.assertEqual(result.ordered_stage, "unknown")
        self.assertTrue(result.reached_low_zone)
        self.assertIsNone(result.reached_zone)
        self.assertIsNone(result.reached_top)

    def test_ordinary_25_10_fallback_is_a_different_declared_schema(self):
        ordinary = decode(
            {"points": 5.0, "top_tries": 5, "zone_tries": 2},
            IFSC_25_10_V1,
        )
        combined = decode(
            {
                "points": 5.0,
                "top_tries": 0,
                "zone_tries": 0,
                "low_zone_tries": 2,
            },
            IFSC_COMBINED_5_10_25_FLAGS_V1,
        )
        self.assertEqual(ordinary.stage, "zone_only")
        self.assertEqual(ordinary.marker_evidence, "derived_ifsc_25_10_points")
        self.assertEqual(combined.stage, "unknown")

    def test_combined_lane_cannot_be_decoded_by_ordinary_schema(self):
        with self.assertRaisesRegex(BoulderSourceContractError, "bound together"):
            decode_problem_outcome(
                {"points": 5.0},
                problem_index=1,
                scoring_schema=IFSC_25_10_V1,
                source_lane=COMBINED_BOULDER_ASCENTS,
            )
        with self.assertRaisesRegex(BoulderSourceContractError, "bound together"):
            decode_problem_outcome(
                {"low_zone": True},
                problem_index=1,
                scoring_schema=IFSC_COMBINED_5_10_25_FLAGS_V1,
                source_lane=DIRECT_BOULDER_ASCENTS,
            )


class CombinedStageExtractionTests(unittest.TestCase):
    def test_combined_boulder_stage_is_recovered_without_lead_leakage(self):
        groups = extract_boulder_ascent_groups(
            {
                "rank": 3,
                "score": "167.1",
                "combined_stages": [
                    {
                        "stage_name": "Boulder",
                        "stage_rank": 2,
                        "stage_score": 79.0,
                        "ascents": [{"route_id": 1, "points": 24.7}],
                    },
                    {
                        "stage_name": "Lead",
                        "stage_rank": 1,
                        "stage_score": 88.1,
                        "ascents": [{"route_id": 2, "score": "45+"}],
                    },
                ],
            },
            discipline="Boulder&Lead",
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].source_lane, "combined_stage_boulder_ascents")
        self.assertEqual(groups[0].stage_rank, 2)
        self.assertEqual(groups[0].ascents[0]["route_id"], 1)

    def test_direct_ascents_win_without_double_counting_dual_representation(self):
        groups = extract_boulder_ascent_groups(
            {
                "ascents": [{"route_id": 10}],
                "combined_stages": [
                    {
                        "stage_name": "Boulder",
                        "ascents": [{"route_id": 20}],
                    }
                ],
            },
            discipline="Boulder",
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].ascents[0]["route_id"], 10)
        self.assertIn("direct_and_combined", groups[0].contradictions[0])

    def test_missing_or_lead_only_evidence_is_not_fabricated(self):
        self.assertFalse(
            extract_boulder_ascent_groups({"score": "45+"}, discipline="Lead")
        )
        self.assertFalse(
            extract_boulder_ascent_groups(
                {
                    "combined_stages": [
                        {"stage_name": "Lead", "ascents": [{"route_id": 2}]}
                    ]
                },
                discipline="Lead",
            )
        )

    def test_speed_ascents_are_never_misread_as_boulder_problems(self):
        self.assertFalse(
            extract_boulder_ascent_groups(
                {"ascents": [{"route_id": 99, "time": 6.5}]},
                discipline="Speed",
            )
        )


class IdentityTests(unittest.TestCase):
    def test_route_id_is_primary_but_reviewed_terrain_alias_is_separate(self):
        outcome = decode(
            {
                "route_id": 9528,
                "top": True,
                "zone": True,
                "top_tries": 1,
                "zone_tries": 1,
            },
            LEGACY_EXPLICIT_FLAGS,
        )
        identity = problem_identity(
            source_scope="CEC",
            source_event_id=224,
            result_url="/api/v1/category_rounds/4455/results",
            outcome=outcome,
            reviewed_terrain_set_alias="CEC|224|final|women|AJR|B1",
        )
        self.assertEqual(identity.marker_key, "CEC|event:224|route:9528")
        self.assertEqual(identity.terrain_set_alias, "CEC|224|final|women|AJR|B1")

    def test_missing_route_id_falls_back_to_result_and_ordinal(self):
        outcome = decode(
            {"top": False, "zone": False, "top_tries": 5, "zone_tries": 5},
            LEGACY_EXPLICIT_FLAGS,
        )
        identity = problem_identity(
            source_scope="IFSC",
            source_event_id=1187,
            result_url="/api/v1/category_rounds/123/results",
            outcome=outcome,
        )
        self.assertIn("result:/api/v1/category_rounds/123/results|problem:1", identity.marker_key)
        self.assertIn("fallback", identity.identity_quality)

    def test_cec_u19_u21_share_reviewed_alias_but_keep_route_ids(self):
        aliases = [
            reviewed_cec_terrain_alias(
                sharing_rule=CEC_CANADIAN_A_JR_SHARED_V1,
                source_event_id=224,
                round_name="Semi-Finals",
                gender="Women",
                category=category,
                problem_index=1,
            )
            for category in ("U19 Female", "U21 Female")
        ]
        self.assertEqual(aliases[0], aliases[1])
        self.assertIn("women|semi_finals|problem:1", aliases[0])

    def test_cec_u17_cannot_enter_a_jr_alias(self):
        with self.assertRaisesRegex(BoulderSourceContractError, "only for U19/U21"):
            reviewed_cec_terrain_alias(
                sharing_rule=CEC_CANADIAN_A_JR_SHARED_V1,
                source_event_id=224,
                round_name="Finals",
                gender="Men",
                category="U17 Male",
                problem_index=1,
            )


if __name__ == "__main__":
    unittest.main()
