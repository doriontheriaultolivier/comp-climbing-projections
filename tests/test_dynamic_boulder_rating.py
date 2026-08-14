import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dynamic_boulder_rating import (  # noqa: E402
    DynamicBoulderRating,
    DynamicRatingConfig,
    RankedContest,
    RankedObservation,
    ranked_observations,
)


def update_for(result, athlete_id):
    return next(row for row in result.updates if row.athlete_id == athlete_id)


class DynamicBoulderRatingTests(unittest.TestCase):
    def anchored_model(self, **overrides):
        config = DynamicRatingConfig(**overrides)
        model = DynamicBoulderRating(config)
        model.seed_established("A", 2100, event_time=0)
        model.seed_established("B", 1950, event_time=0)
        model.seed_established("C", 1800, event_time=0)
        return model

    def test_newcomer_learns_but_cannot_move_established_athletes(self):
        model = self.anchored_model()
        control = self.anchored_model()
        result = model.update_event(
            "event-1",
            10,
            "wc+",
            ranked_observations(["NEW", "A", "B", "C"], [1, 2, 3, 4]),
        )
        newcomer = update_for(result, "NEW")
        self.assertGreater(newcomer.post_projection_mean, newcomer.pre_projection_mean)
        self.assertTrue(newcomer.provisional_after)
        control.update_event(
            "event-1",
            10,
            "wc+",
            ranked_observations(["A", "B", "C"], [1, 2, 3]),
        )
        for athlete_id in ("A", "B", "C"):
            # The established athletes may move against one another, but adding
            # a provisional winner contributes exactly no extra movement.
            self.assertAlmostEqual(
                model.states[athlete_id].skill_mean,
                control.states[athlete_id].skill_mean,
            )
            self.assertEqual(
                update_for(result, athlete_id).eligible_established_opponents, 2
            )

    def test_promotion_needs_distinct_chronological_anchored_events(self):
        model = self.anchored_model(
            promotion_min_anchored_events=3,
            promotion_min_anchored_comparisons=6,
            promotion_min_unique_opponents=6,
            promotion_min_effective_weight=9,
            promotion_max_skill_sd=400,
        )
        for athlete_id, mean in zip(("D", "E", "F"), (1750, 1700, 1650)):
            model.seed_established(athlete_id, mean, event_time=0)
        for event_number in range(1, 4):
            result = model.update_event(
                f"event-{event_number}",
                event_number * 10,
                "canada_senior_national",
                ranked_observations(
                    ["A", "NEW", "B", "C", "D", "E", "F"],
                    [1, 2, 3, 4, 5, 6, 7],
                ),
            )
            row = update_for(result, "NEW")
            self.assertEqual(row.anchored_events, event_number)
            self.assertEqual(row.provisional_after, event_number < 3)
        self.assertFalse(model.states["NEW"].provisional)
        with self.assertRaisesRegex(ValueError, "already processed"):
            model.update_event(
                "event-3",
                31,
                "canada_senior_national",
                ranked_observations(["A", "NEW"], [1, 2]),
            )

    def test_all_participants_use_one_frozen_state_independent_of_input_order(self):
        first = self.anchored_model()
        second = self.anchored_model()
        observations = [
            RankedObservation("A", 3, 0.45),
            RankedObservation("B", 1, 0.85),
            RankedObservation("C", 2, 0.62),
        ]
        first.update_event("event", 20, "wc+", observations)
        second.update_event("event", 20, "wc+", list(reversed(observations)))
        for athlete_id in ("A", "B", "C"):
            self.assertAlmostEqual(
                first.states[athlete_id].skill_mean,
                second.states[athlete_id].skill_mean,
                places=10,
            )
            self.assertAlmostEqual(
                first.states[athlete_id].form_mean,
                second.states[athlete_id].form_mean,
                places=10,
            )

    def test_multiple_rounds_share_one_freeze_and_never_compare_across_groups(self):
        model = self.anchored_model()
        result = model.update_competition(
            "competition",
            15,
            "wc+",
            (
                RankedContest(
                    "group-a",
                    ranked_observations(["NEW_A", "A", "B"], [1, 2, 3]),
                ),
                RankedContest(
                    "group-b",
                    ranked_observations(["NEW_B", "C"], [1, 2]),
                ),
            ),
        )
        new_a = update_for(result, "NEW_A")
        new_b = update_for(result, "NEW_B")
        self.assertEqual(result.contest_count, 2)
        self.assertEqual(new_a.eligible_established_opponents, 2)
        self.assertEqual(new_b.eligible_established_opponents, 1)
        self.assertNotIn("C", model.states["NEW_A"].anchored_opponent_ids)
        self.assertNotIn("A", model.states["NEW_B"].anchored_opponent_ids)

    def test_large_fields_have_capped_event_information(self):
        model = DynamicBoulderRating(
            DynamicRatingConfig(
                event_pair_weight_cap=5.0,
                maximum_projection_shift=320.0,
            )
        )
        for index in range(40):
            model.seed_established(f"E{index}", 1800 + index, event_time=0)
        result = model.update_event(
            "large",
            5,
            "wc+",
            ranked_observations(
                ["NEW", *[f"E{index}" for index in range(40)]],
                list(range(1, 42)),
                [1.0, *[0.0] * 40],
            ),
        )
        newcomer = update_for(result, "NEW")
        self.assertEqual(newcomer.eligible_established_opponents, 40)
        self.assertLessEqual(newcomer.effective_pair_weight, 5.0 + 1e-12)
        self.assertLessEqual(abs(newcomer.projection_shift), 320.0 + 1e-12)

    def test_no_round_recentering_forces_the_population_mean(self):
        model = self.anchored_model()
        model.states["C"].skill_variance = 180.0**2
        mean_before = sum(state.skill_mean for state in model.states.values()) / 3
        model.update_event(
            "event",
            4,
            "other",
            ranked_observations(["C", "B", "A"], [1, 2, 3]),
        )
        mean_after = sum(state.skill_mean for state in model.states.values()) / 3
        self.assertNotAlmostEqual(mean_before, mean_after, places=8)

    def test_target_offset_is_shrunk_and_domain_specific(self):
        model = self.anchored_model()
        model.update_event(
            "wc-event",
            10,
            "wc+",
            ranked_observations(["C", "B", "A"], [1, 2, 3]),
        )
        state = model.states["C"]
        self.assertGreater(state.target_offset_mean["wc+"], 0.0)
        self.assertEqual(state.target_offset_mean["canada_senior_national"], 0.0)
        wc_projection = model.projection("C", 10, "wc+")
        canada_projection = model.projection("C", 10, "canada_senior_national")
        self.assertGreater(wc_projection.mean, canada_projection.mean)

    def test_form_is_separate_and_mean_reverts_without_mutating_state(self):
        model = self.anchored_model(form_half_life_days=50.0)
        model.update_event(
            "event",
            10,
            "other",
            ranked_observations(["C", "B", "A"], [1, 2, 3]),
        )
        immediate = model.projection("C", 10, "other")
        future = model.projection("C", 110, "other")
        self.assertGreater(immediate.form_adjustment, 0.0)
        self.assertLess(abs(future.form_adjustment), abs(immediate.form_adjustment))
        self.assertAlmostEqual(
            model.states["C"].form_mean, immediate.form_adjustment, places=10
        )

    def test_score_gap_changes_information_not_the_observed_order(self):
        small = self.anchored_model()
        large = self.anchored_model()
        small_result = small.update_event(
            "small",
            3,
            "canada_senior_national",
            ranked_observations(["C", "B", "A"], [1, 2, 3], [0.51, 0.50, 0.49]),
        )
        large_result = large.update_event(
            "large",
            3,
            "canada_senior_national",
            ranked_observations(["C", "B", "A"], [1, 2, 3], [1.0, 0.5, 0.0]),
        )
        small_c = update_for(small_result, "C")
        large_c = update_for(large_result, "C")
        self.assertGreater(large_c.effective_pair_weight, small_c.effective_pair_weight)
        self.assertGreater(large_c.projection_shift, 0.0)
        self.assertGreater(small_c.projection_shift, 0.0)

    def test_uncertainty_is_explicit_in_head_to_head_probability(self):
        model = self.anchored_model()
        probability = model.head_to_head_probability("A", "C", 20, "wc+")
        self.assertGreater(probability, 0.5)
        projection = model.projection("A", 20, "wc+")
        self.assertGreater(projection.predictive_sd, projection.rating_sd)

    def test_events_must_be_chronological(self):
        model = self.anchored_model()
        model.update_event(
            "later", 20, "other", ranked_observations(["A", "B"], [1, 2])
        )
        with self.assertRaisesRegex(ValueError, "chronologically"):
            model.update_event(
                "earlier", 19, "other", ranked_observations(["A", "B"], [1, 2])
            )

    def test_reference_domain_has_exactly_zero_offset_and_no_offset_variance(self):
        model = self.anchored_model()
        before = model.projection("C", 0, "other")
        self.assertEqual(before.target_adjustment, 0.0)
        expected_variance = (
            model.states["C"].skill_variance
            + model.states["C"].form_variance
        )
        self.assertAlmostEqual(before.rating_sd**2, expected_variance, places=6)
        model.update_event(
            "reference-event",
            10,
            "other",
            ranked_observations(["C", "B", "A"], [1, 2, 3]),
        )
        state = model.states["C"]
        self.assertEqual(state.target_offset_mean["other"], 0.0)
        self.assertEqual(state.target_offset_variance["other"], 0.0)

    def test_default_newcomer_prior_is_broad(self):
        model = self.anchored_model()
        result = model.update_event(
            "broad",
            5,
            "wc+",
            ranked_observations(["NEW", "A", "B", "C"], [1, 2, 3, 4]),
        )
        self.assertGreater(update_for(result, "NEW").pre_rating_sd, 480.0)

    def test_repeated_opponents_alone_cannot_satisfy_unique_opponent_gate(self):
        model = self.anchored_model(
            promotion_min_anchored_events=2,
            promotion_min_anchored_comparisons=2,
            promotion_min_unique_opponents=4,
            promotion_min_effective_weight=2,
            promotion_max_skill_sd=650,
        )
        for number in range(1, 6):
            model.update_event(
                f"repeat-{number}",
                number,
                "other",
                ranked_observations(["NEW", "A", "B", "C"], [1, 2, 3, 4]),
            )
        self.assertTrue(model.states["NEW"].provisional)
        self.assertEqual(len(model.states["NEW"].anchored_opponent_ids), 3)

    def test_effective_information_gate_is_not_raw_pair_count(self):
        model = self.anchored_model(
            promotion_min_anchored_events=2,
            promotion_min_anchored_comparisons=2,
            promotion_min_unique_opponents=3,
            promotion_min_effective_weight=100,
            promotion_max_skill_sd=650,
        )
        for number in range(1, 4):
            model.update_event(
                f"low-information-{number}",
                number,
                "other",
                ranked_observations(["NEW", "A", "B", "C"], [1, 2, 3, 4]),
            )
        state = model.states["NEW"]
        self.assertGreaterEqual(state.anchored_comparisons, 6)
        self.assertLess(state.anchored_effective_weight, 100)
        self.assertTrue(state.provisional)

    def test_full_component_covariance_stays_positive_semidefinite(self):
        model = self.anchored_model()
        for event_number, domain in enumerate(
            ("wc+", "canada_senior_national", "nacs", "other"), start=1
        ):
            model.update_event(
                f"covariance-{event_number}",
                event_number * 20,
                domain,
                ranked_observations(["C", "B", "A"], [1, 2, 3]),
            )
        for athlete_id in ("A", "B", "C"):
            labels, covariance = model.covariance_matrix(athlete_id)
            self.assertTrue(model.states[athlete_id].component_covariance)
            self.assertTrue(np.allclose(covariance, covariance.T, atol=1e-10))
            self.assertGreaterEqual(float(np.linalg.eigvalsh(covariance).min()), -1e-8)
            projection = model.projection(athlete_id, 80, "wc+")
            design = model._readiness_design(labels, "wc+")
            self.assertAlmostEqual(
                projection.rating_sd**2,
                float(design @ covariance @ design),
                places=6,
            )
            self.assertGreaterEqual(
                model.states[athlete_id].form_variance,
                model.config.minimum_form_sd**2,
            )
            self.assertGreaterEqual(
                model.states[athlete_id].target_offset_variance["wc+"],
                model.config.minimum_target_offset_sd**2,
            )

    def test_form_and_target_components_can_be_disabled_exactly(self):
        model = DynamicBoulderRating(
            DynamicRatingConfig(enable_form=False, enable_target_offsets=False)
        )
        model.seed_established("A", 2050, event_time=0)
        model.seed_established("B", 1850, event_time=0)
        model.update_event(
            "disabled",
            30,
            "wc+",
            ranked_observations(["B", "A"], [1, 2]),
        )
        for athlete_id in ("A", "B"):
            state = model.states[athlete_id]
            labels, covariance = model.covariance_matrix(athlete_id)
            self.assertEqual(labels, ("skill",))
            self.assertEqual(covariance.shape, (1, 1))
            self.assertEqual(state.form_mean, 0.0)
            self.assertEqual(state.form_variance, 0.0)
            self.assertTrue(
                all(value == 0.0 for value in state.target_offset_mean.values())
            )
            self.assertTrue(
                all(value == 0.0 for value in state.target_offset_variance.values())
            )
            projection = model.projection(athlete_id, 30, "wc+")
            self.assertAlmostEqual(projection.mean, state.skill_mean)
            self.assertAlmostEqual(projection.rating_sd**2, state.skill_variance)

    def test_provisional_only_field_learns_order_but_not_absolute_location(self):
        model = DynamicBoulderRating()
        result = model.update_event(
            "provisional-only",
            0,
            "other",
            ranked_observations(["P1", "P2", "P3"], [1, 2, 3]),
        )
        projections = {
            athlete_id: model.projection(athlete_id, 0, "other")
            for athlete_id in ("P1", "P2", "P3")
        }
        self.assertGreater(projections["P1"].mean, projections["P2"].mean)
        self.assertGreater(projections["P2"].mean, projections["P3"].mean)
        self.assertAlmostEqual(
            np.mean([projection.mean for projection in projections.values()]),
            model.config.prior_mean,
            places=8,
        )
        for row in result.updates:
            self.assertEqual(row.eligible_established_opponents, 0)
            self.assertEqual(row.eligible_provisional_opponents, 2)
            self.assertEqual(row.anchored_events, 0)
            self.assertGreater(row.relative_pair_weight, 0.0)
            self.assertIn("absolute level unanchored", row.status)
            self.assertAlmostEqual(row.post_rating_sd, row.pre_rating_sd, places=8)
            self.assertTrue(row.provisional_after)

    def test_provisional_peer_evidence_does_not_move_established_athletes(self):
        enriched = self.anchored_model()
        control = self.anchored_model()
        result = enriched.update_event(
            "mixed",
            5,
            "other",
            ranked_observations(
                ["P1", "A", "P2", "B", "C"],
                [1, 2, 3, 4, 5],
            ),
        )
        control.update_event(
            "mixed",
            5,
            "other",
            ranked_observations(["A", "B", "C"], [1, 2, 3]),
        )
        for athlete_id in ("A", "B", "C"):
            self.assertAlmostEqual(
                enriched.states[athlete_id].skill_mean,
                control.states[athlete_id].skill_mean,
                places=10,
            )
            self.assertEqual(
                update_for(result, athlete_id).eligible_provisional_opponents,
                0,
            )
        self.assertGreater(
            update_for(result, "P1").eligible_provisional_opponents, 0
        )

    def test_larger_field_has_more_but_sublinear_information(self):
        def newcomer_result(opponent_count):
            model = DynamicBoulderRating()
            for index in range(opponent_count):
                model.seed_established(f"E{index}", 1800, event_time=0)
            result = model.update_event(
                f"field-{opponent_count}",
                1,
                "wc+",
                ranked_observations(
                    ["NEW", *[f"E{index}" for index in range(opponent_count)]],
                    list(range(1, opponent_count + 2)),
                ),
            )
            return model, update_for(result, "NEW")

        small_model, small = newcomer_result(7)
        large_model, large = newcomer_result(79)
        self.assertGreater(large.effective_pair_weight, small.effective_pair_weight)
        self.assertLess(large.post_rating_sd, small.post_rating_sd)
        self.assertLess(
            large.effective_pair_weight / small.effective_pair_weight,
            79.0 / 7.0,
        )
        self.assertAlmostEqual(
            small.effective_pair_weight,
            small_model._field_information_cap(7),
            places=8,
        )
        self.assertAlmostEqual(
            large.effective_pair_weight,
            large_model._field_information_cap(79),
            places=8,
        )


if __name__ == "__main__":
    unittest.main()
