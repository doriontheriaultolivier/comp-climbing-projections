import math
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np

from scripts.boulder_terrain_response import (
    AthleteTerrainParameters,
    FittedTerrainResponse,
    FrozenTargetItem,
    FrozenTerrainTarget,
    HurdleItemElo,
    ITEM_ELO_SEMANTICS,
    LeaveOneOutItemElo,
    ProblemOutcome,
    TerrainContractError,
    TerrainResponseConfig,
    _ordered_stage_probability_grid,
    _default_prior,
    _integrated_ordered_response_probabilities,
    _row_log_likelihood,
    calibrate_leave_one_out_item_elos,
    fit_terrain_response,
    hurdle_success_probability,
    project_terrain_target,
)


UTC = timezone.utc
START = datetime(2024, 5, 1, tzinfo=UTC)


def config(**changes):
    values = dict(
        development_end=datetime(2024, 12, 31, tzinfo=UTC),
        item_grid_size=61,
        coordinate_iterations=4,
        coordinate_points=13,
        projection_draws=500,
        minimum_athlete_items=6,
    )
    values.update(changes)
    return TerrainResponseConfig(**values)


def outcome(
    athlete,
    problem,
    zone,
    top,
    *,
    competition="C1",
    day=0,
    stable=1900.0,
    stable_sd=45.0,
    procedure="onsight",
    attempts_zone=None,
    attempts_top=None,
    post_zone_opportunities=None,
    post_zone_success_index=None,
    rank=None,
    score=None,
    pre_styles=None,
    post_styles=None,
    attempt_cap=8,
):
    start = START + timedelta(days=day)
    return ProblemOutcome(
        athlete_id=athlete,
        competition_id=competition,
        problem_id=problem,
        competition_start=start,
        frozen_at=start - timedelta(days=1),
        result_available_at=start + timedelta(days=1),
        stable_mean=stable,
        stable_sd=stable_sd,
        procedure=procedure,
        reached_zone=zone,
        reached_top=top,
        attempts_to_zone=attempts_zone,
        attempts_to_top=attempts_top,
        post_zone_opportunities=post_zone_opportunities,
        post_zone_success_index=post_zone_success_index,
        attempt_cap=attempt_cap,
        pre_zone_style_features=pre_styles or {},
        post_zone_style_features=post_styles or {},
        style_available_at=(
            start + timedelta(hours=12) if pre_styles or post_styles else None
        ),
        published_rank=rank,
        published_score=score,
    )


def hurdle(mean, sd=35.0, status="mixed_peer_outcomes_identified"):
    return HurdleItemElo(mean, sd, mean - 60.0, mean + 60.0, status, 4, 8)


def item_for(row, zone_mean, post_mean=3000.0, sd=35.0):
    return LeaveOneOutItemElo(
        competition_id=row.competition_id,
        problem_id=row.problem_id,
        procedure=row.confirmed_procedure,
        excluded_athlete_id=row.athlete_id,
        zone=hurdle(zone_mean, sd),
        post_zone=hurdle(post_mean, sd),
        top=hurdle(max(zone_mean, post_mean) + 90.0, sd),
        zone_post_covariance=0.0,
        peer_count=12,
        calibrated_at=row.result_available_at,
    )


def frozen_item(problem, zone, post, *, sd=0.0, pre_styles=None, post_styles=None):
    return FrozenTargetItem(
        problem_id=problem,
        zone_mean=zone,
        zone_sd=sd,
        post_zone_mean=post,
        post_zone_sd=sd,
        zone_post_correlation=0.3,
        attempt_cap=5,
        as_of=START - timedelta(days=2),
        provenance="historical same-procedure item library",
        pre_zone_style_features=pre_styles or {},
        post_zone_style_features=post_styles or {},
    )


def target(identifier, items, procedure="onsight", estimand="modern_25_10_achievement_without_attempt_penalty"):
    return FrozenTerrainTarget(
        target_id=identifier,
        competition_start=START,
        frozen_at=START - timedelta(days=1),
        procedure=procedure,
        items=tuple(items),
        projection_estimand=estimand,
    )


class ItemEloContractTests(unittest.TestCase):
    def test_default_item_prior_is_one_frozen_population_anchor(self):
        cfg = config(item_prior_reference_center=1975.0)
        self.assertEqual(
            _default_prior(cfg),
            (
                1975.0 + cfg.zone_prior_offset,
                cfg.item_prior_sd,
                1975.0 + cfg.top_prior_offset,
                cfg.item_prior_sd,
                0.35,
            ),
        )

    def test_shared_high_uncertainty_latent_is_integrated_jointly(self):
        grid = np.asarray([1900.0])
        top = _ordered_stage_probability_grid(
            1900.0, 350.0, grid, grid, 155.0, "top"
        )[0, 0]
        zone_only = _ordered_stage_probability_grid(
            1900.0, 350.0, grid, grid, 155.0, "zone_only"
        )[0, 0]
        no_zone = _ordered_stage_probability_grid(
            1900.0, 350.0, grid, grid, 155.0, "no_zone"
        )[0, 0]
        self.assertAlmostEqual(top, 0.360, delta=0.012)
        self.assertAlmostEqual(zone_only, 0.140, delta=0.012)
        self.assertAlmostEqual(no_zone + zone_only + top, 1.0, places=10)
        self.assertGreater(top, 0.25)
        self.assertLess(zone_only, 0.25)

    def test_zero_uncertainty_reduces_to_ordered_product_at_q50(self):
        grid = np.asarray([1900.0])
        top = _ordered_stage_probability_grid(
            1900.0, 0.0, grid, grid, 155.0, "top"
        )[0, 0]
        zone_only = _ordered_stage_probability_grid(
            1900.0, 0.0, grid, grid, 155.0, "zone_only"
        )[0, 0]
        self.assertAlmostEqual(top, 0.25, places=12)
        self.assertAlmostEqual(zone_only, 0.25, places=12)

    def test_marker_elo_is_event_q50_and_invariant_to_attempt_horizon(self):
        calibrated = {}
        for cap, competition in ((4, "M4"), (12, "M12")):
            rows = [
                outcome(
                    "held", "B1", False, False,
                    competition=competition, stable=1900, stable_sd=5, attempt_cap=cap,
                )
            ]
            for index in range(40):
                rows.append(
                    outcome(
                        f"p{index}",
                        "B1",
                        index < 20,
                        index < 20,
                        competition=competition,
                        stable=1900,
                        stable_sd=5,
                        attempt_cap=cap,
                    )
                )
            calibrated[cap] = calibrate_leave_one_out_item_elos(rows, config())[
                (competition, "B1", "held")
            ]
        self.assertAlmostEqual(calibrated[4].zone.mean, calibrated[12].zone.mean, places=12)
        self.assertAlmostEqual(calibrated[4].post_zone.mean, calibrated[12].post_zone.mean, places=12)
        self.assertLess(abs(calibrated[4].zone.mean - 1900.0), 30.0)
        self.assertAlmostEqual(
            hurdle_success_probability(
                calibrated[4].zone.mean,
                calibrated[4].zone.mean,
                config().response_scale,
            ),
            0.5,
            places=12,
        )

    def test_easy_post_zone_is_not_forced_above_zone_and_top_is_cumulative(self):
        rows = [outcome("held", "B1", False, False, stable_sd=5)]
        for index in range(40):
            rows.append(
                outcome(
                    f"p{index}",
                    "B1",
                    index < 20,
                    index < 20,
                    stable=1900,
                    stable_sd=5,
                )
            )
        item = calibrate_leave_one_out_item_elos(rows, config())[("C1", "B1", "held")]
        self.assertLess(item.post_zone.mean, item.zone.mean - 300.0)
        self.assertGreater(item.top.mean, item.zone.mean)
        zone_probability = hurdle_success_probability(
            item.top.mean, item.zone.mean, config().response_scale
        )
        post_probability = hurdle_success_probability(
            item.top.mean, item.post_zone.mean, config().response_scale
        )
        self.assertAlmostEqual(zone_probability * post_probability, 0.5, delta=0.04)

    def test_every_hurdle_is_leave_one_out_ordered_and_uncertain(self):
        rows = [
            outcome("held", "B1", True, True, stable=2600),
            outcome("a", "B1", True, True, stable=2050),
            outcome("b", "B1", True, False, stable=1950),
            outcome("c", "B1", False, False, stable=1800),
        ]
        fitted = calibrate_leave_one_out_item_elos(rows, config())
        held = fitted[("C1", "B1", "held")]
        self.assertEqual(held.excluded_athlete_id, "held")
        self.assertEqual(held.peer_count, 3)
        self.assertGreaterEqual(held.top.mean, held.zone.mean)
        self.assertGreater(held.zone.sd, 0.0)
        self.assertGreater(held.top.sd, 0.0)
        self.assertEqual(held.semantics, ITEM_ELO_SEMANTICS)

        # Changing every held-athlete field, including their outcome and frozen
        # Elo, cannot change the item posterior used to evaluate them.
        changed = [
            outcome("held", "B1", False, False, stable=900),
            *rows[1:],
        ]
        changed_held = calibrate_leave_one_out_item_elos(changed, config())[
            ("C1", "B1", "held")
        ]
        self.assertAlmostEqual(held.zone.mean, changed_held.zone.mean, places=12)
        self.assertAlmostEqual(held.top.mean, changed_held.top.mean, places=12)
        self.assertAlmostEqual(
            held.post_zone.mean, changed_held.post_zone.mean, places=12
        )
        self.assertAlmostEqual(held.zone.sd, changed_held.zone.sd, places=12)

    def test_sparse_extremes_are_censored_not_fabricated_as_precise(self):
        all_fail = [
            outcome("held", "B1", False, False),
            outcome("peer", "B1", False, False),
        ]
        sparse = calibrate_leave_one_out_item_elos(all_fail, config())[
            ("C1", "B1", "held")
        ]
        self.assertIn("lower_bound_only", sparse.zone.status)
        self.assertIn("prior_only", sparse.post_zone.status)
        self.assertIn("lower_bound_only", sparse.top.status)

        mixed = [outcome("held", "B1", True, False)]
        for index in range(12):
            mixed.append(
                outcome(
                    f"p{index}",
                    "B1",
                    index % 2 == 0,
                    index % 4 == 0,
                    stable=1750 + 25 * index,
                )
            )
        dense = calibrate_leave_one_out_item_elos(mixed, config())[
            ("C1", "B1", "held")
        ]
        self.assertEqual(dense.zone.status, "mixed_peer_outcomes_identified")
        self.assertLess(dense.zone.sd, sparse.zone.sd)

    def test_item_semantics_are_bound_to_one_rating_pool_and_reference_curve(self):
        rows = [
            outcome("a", "B1", True, True, stable=2050),
            outcome("b", "B1", True, False, stable=1900),
            outcome("c", "B1", False, False, stable=1750),
        ]
        cfg = config(response_scale=170.0)
        calibrated = calibrate_leave_one_out_item_elos(rows, cfg)
        item = calibrated[("C1", "B1", "a")]
        self.assertEqual(item.rating_pool_id, rows[0].rating_pool_id)
        self.assertEqual(item.reference_response_scale, 170.0)

        mixed = [*rows[:2], replace(rows[2], rating_pool_id="canadian_only_shadow")]
        with self.assertRaises(TerrainContractError):
            calibrate_leave_one_out_item_elos(mixed, cfg)

        mismatched = dict(calibrated)
        mismatched[("C1", "B1", "a")] = replace(
            item, rating_pool_id="canadian_only_shadow"
        )
        with self.assertRaises(TerrainContractError):
            fit_terrain_response(rows, mismatched, cfg)


class TerrainResponseTests(unittest.TestCase):
    def test_identical_flash_tops_are_censored_not_an_artificial_tie_or_equality(self):
        """A shared success is threshold evidence, never a pairwise draw.

        With no independent difficulty contrast, the terrain layer has no
        authority to overwrite the frozen stable-state ordering.  It must
        therefore keep the response adjustment at its pooled prior and carry
        the athletes' different pre-event means/uncertainties forward.
        """

        rows = [
            outcome(
                "higher-prior",
                "B1",
                True,
                True,
                stable=2100.0,
                stable_sd=35.0,
                procedure="flash",
                attempts_zone=1,
                attempts_top=1,
                rank=1,
                score=25,
            ),
            outcome(
                "lower-prior",
                "B1",
                True,
                True,
                stable=1750.0,
                stable_sd=95.0,
                procedure="flash",
                attempts_zone=1,
                attempts_top=1,
                rank=1,
                score=25,
            ),
            outcome(
                "peer",
                "B1",
                False,
                False,
                stable=1850.0,
                stable_sd=45.0,
                procedure="flash",
            ),
        ]
        cfg = config(
            minimum_difficulty_spread=1.0e9,
            projection_draws=1000,
        )
        items = calibrate_leave_one_out_item_elos(rows, cfg)
        fitted = fit_terrain_response(rows, items, cfg)
        self.assertEqual(
            fitted.athletes["higher-prior"].log_scale_ratio,
            fitted.athletes["lower-prior"].log_scale_ratio,
        )

        future_item = frozen_item("F1", 1900.0, 2025.0)
        future_target = replace(
            target("future", [future_item], "flash"),
            competition_start=START + timedelta(days=30),
            frozen_at=START + timedelta(days=29),
        )
        higher = project_terrain_target(
            "higher-prior", 2100.0, 35.0, future_target, fitted
        )
        lower = project_terrain_target(
            "lower-prior", 1750.0, 95.0, future_target, fitted
        )
        self.assertGreater(higher.zone_projection_elo, lower.zone_projection_elo + 250.0)
        self.assertGreater(lower.zone_projection_elo_sd, higher.zone_projection_elo_sd)

    def test_same_aggregate_score_on_different_item_difficulties_is_not_same_evidence(self):
        """One Top each is not exchangeable when the item thresholds differ."""

        easy_top = outcome("easy-pattern", "easy", True, True, competition="E")
        hard_fail = outcome("easy-pattern", "hard", False, False, competition="H")
        hard_top = outcome("hard-pattern", "hard", True, True, competition="H2")
        easy_fail = outcome("hard-pattern", "easy", False, False, competition="E2")
        parameters = np.zeros(3)
        cfg = config()
        easy_pattern_logp = _row_log_likelihood(
            easy_top, item_for(easy_top, 1600.0, 1700.0, sd=5.0), parameters, cfg, False
        ) + _row_log_likelihood(
            hard_fail, item_for(hard_fail, 2200.0, 2300.0, sd=5.0), parameters, cfg, False
        )
        hard_pattern_logp = _row_log_likelihood(
            hard_top, item_for(hard_top, 2200.0, 2300.0, sd=5.0), parameters, cfg, False
        ) + _row_log_likelihood(
            easy_fail, item_for(easy_fail, 1600.0, 1700.0, sd=5.0), parameters, cfg, False
        )
        self.assertGreater(easy_pattern_logp, hard_pattern_logp + 4.0)

    def test_easy_item_successes_saturate_instead_of_inflating_ability_linearly(self):
        """Extra strength above an already-easy threshold has diminishing value."""

        cfg = config()
        parameters = np.zeros(3)
        baseline = outcome(
            "a", "B", True, True, stable=1900.0, stable_sd=1.0e-6
        )
        improved = replace(baseline, stable_mean=2000.0)
        easy = item_for(baseline, 1200.0, 1200.0, sd=1.0e-6)
        threshold = item_for(baseline, 1900.0, 1900.0, sd=1.0e-6)
        easy_gain = _row_log_likelihood(
            improved, easy, parameters, cfg, False
        ) - _row_log_likelihood(baseline, easy, parameters, cfg, False)
        threshold_gain = _row_log_likelihood(
            improved, threshold, parameters, cfg, False
        ) - _row_log_likelihood(baseline, threshold, parameters, cfg, False)
        self.assertGreater(easy_gain, 0.0)
        self.assertGreater(threshold_gain, 10.0 * easy_gain)

    def test_only_exact_shared_problem_identity_connects_item_peers(self):
        """Same event labels alone do not make two category items identical."""

        shared_rows = [
            outcome("u15", "route-42", True, False, stable=1750.0),
            outcome("senior", "route-42", True, True, stable=2100.0),
        ]
        shared = calibrate_leave_one_out_item_elos(shared_rows, config())
        self.assertEqual(shared[("C1", "route-42", "u15")].peer_count, 1)
        self.assertGreater(shared[("C1", "route-42", "u15")].zone.sd, 0.0)

        separate_rows = [
            outcome("u15", "u15-route-42", True, False, stable=1750.0),
            outcome("senior", "senior-route-42", True, True, stable=2100.0),
        ]
        separate = calibrate_leave_one_out_item_elos(separate_rows, config())
        self.assertEqual(separate[("C1", "u15-route-42", "u15")].peer_count, 0)
        self.assertIn(
            "prior_only",
            separate[("C1", "u15-route-42", "u15")].zone.status,
        )

    def test_response_likelihood_integrates_shared_high_sd_state_jointly(self):
        top_row = outcome("a", "B1", True, True, stable_sd=350.0)
        zone_row = replace(top_row, reached_top=False)
        no_zone_row = replace(top_row, reached_zone=False, reached_top=False)
        item = item_for(top_row, 1800.0, 1800.0, sd=0.0)
        # item_for follows its row key only; align the q50 with the row.
        item = replace(
            item,
            zone=replace(item.zone, mean=top_row.stable_mean),
            post_zone=replace(item.post_zone, mean=top_row.stable_mean),
        )
        parameters = np.zeros(3)
        probabilities = [
            math.exp(_row_log_likelihood(row, item, parameters, config(), False))
            for row in (no_zone_row, zone_row, top_row)
        ]
        self.assertAlmostEqual(probabilities[2], 0.360, delta=0.012)
        self.assertAlmostEqual(probabilities[1], 0.140, delta=0.012)
        self.assertAlmostEqual(sum(probabilities), 1.0, places=9)

    def test_response_likelihood_retains_zone_post_item_covariance(self):
        row = outcome("a", "B1", True, True, stable_sd=1.0e-6)
        correlated = item_for(row, row.stable_mean, row.stable_mean, sd=350.0)
        correlated = replace(
            correlated,
            zone_post_covariance=350.0**2,
        )
        independent = replace(correlated, zone_post_covariance=0.0)
        correlated_top = _integrated_ordered_response_probabilities(
            row, correlated, 155.0, 0.0, 0.0
        )[1]
        independent_top = _integrated_ordered_response_probabilities(
            row, independent, 155.0, 0.0, 0.0
        )[1]
        self.assertAlmostEqual(correlated_top, 0.360, delta=0.012)
        self.assertAlmostEqual(independent_top, 0.25, delta=0.012)
        self.assertGreater(correlated_top, independent_top + 0.08)

    def test_post_zone_opportunity_horizon_requires_reaching_zone(self):
        with self.assertRaisesRegex(TerrainContractError, "achieved zone"):
            outcome(
                "a",
                "B1",
                False,
                False,
                post_zone_opportunities=4,
                attempt_cap=4,
            )

    def test_post_zone_efficiency_requires_explicit_conditional_opportunities(self):
        cfg = config()
        base = outcome(
            "a",
            "B1",
            True,
            True,
            attempts_zone=1,
            attempts_top=1,
        )
        item = item_for(base, 1850, 2050)
        parameters = np.zeros(3)
        aggregate_late = replace(base, attempts_to_top=8)
        self.assertEqual(
            _row_log_likelihood(base, item, parameters, cfg, True),
            _row_log_likelihood(aggregate_late, item, parameters, cfg, True),
        )

        explicit_fast = replace(
            base,
            post_zone_opportunities=8,
            post_zone_success_index=1,
        )
        explicit_late = replace(
            base,
            post_zone_opportunities=8,
            post_zone_success_index=8,
        )
        self.assertNotEqual(
            _row_log_likelihood(explicit_fast, item, parameters, cfg, True),
            _row_log_likelihood(explicit_late, item, parameters, cfg, True),
        )

    def test_response_curves_are_monotone_but_can_cross(self):
        difficulties = np.linspace(1500.0, 2300.0, 101)
        reliable_easy = hurdle_success_probability(1900.0, difficulties, 75.0)
        hard_tail = hurdle_success_probability(1980.0, difficulties, 260.0)
        self.assertTrue(np.all(np.diff(reliable_easy) < 0.0))
        self.assertTrue(np.all(np.diff(hard_tail) < 0.0))
        self.assertGreater(reliable_easy[10], hard_tail[10])
        self.assertLess(reliable_easy[-10], hard_tail[-10])

    def test_sparse_athlete_is_more_pooled_than_repeated_crossing_evidence(self):
        rows = []
        items = {}
        # Both athletes exhibit the same unusual flat pattern, but dense has
        # four times the independent item evidence.
        pattern = [False, True, False, True, False, True, True, False]
        difficulties = [1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300]
        for repeat in range(4):
            for index, (difficulty, success) in enumerate(zip(difficulties, pattern)):
                row = outcome(
                    "dense",
                    f"D{repeat}-{index}",
                    success,
                    False,
                    competition=f"CD{repeat}-{index}",
                    day=repeat * 10 + index,
                )
                rows.append(row)
                items[(row.competition_id, row.problem_id, row.athlete_id)] = item_for(
                    row, difficulty
                )
        for index, (difficulty, success) in enumerate(zip(difficulties, pattern)):
            row = outcome(
                "sparse",
                f"S{index}",
                success,
                False,
                competition=f"CS{index}",
                day=50 + index,
            )
            rows.append(row)
            items[(row.competition_id, row.problem_id, row.athlete_id)] = item_for(
                row, difficulty
            )
        fitted = fit_terrain_response(rows, items, config(minimum_athlete_items=12))
        dense = fitted.athletes["dense"]
        sparse = fitted.athletes["sparse"]
        self.assertEqual(dense.mode, "partially_pooled_athlete")
        self.assertEqual(sparse.mode, "strongly_pooled_sparse_athlete")
        self.assertGreater(abs(dense.log_scale_ratio), abs(sparse.log_scale_ratio))
        self.assertLess(dense.log_scale_sd, sparse.log_scale_sd)

    def test_rank_and_score_metadata_never_double_count_stage_evidence(self):
        base_rows = []
        items = {}
        for index, difficulty in enumerate((1650, 1750, 1850, 1950, 2050, 2150)):
            row = outcome(
                "a",
                f"B{index}",
                index < 4,
                index < 2,
                competition=f"C{index}",
                day=index,
                rank=1,
                score=9999,
            )
            base_rows.append(row)
            items[(row.competition_id, row.problem_id, "a")] = item_for(
                row, difficulty, difficulty + 150
            )
        altered = [
            ProblemOutcome(
                **{
                    **row.__dict__,
                    "published_rank": 999 - index,
                    "published_score": -9999 + index,
                }
            )
            for index, row in enumerate(base_rows)
        ]
        first = fit_terrain_response(base_rows, items, config()).athletes["a"]
        second = fit_terrain_response(altered, items, config()).athletes["a"]
        self.assertEqual(first.log_scale_ratio, second.log_scale_ratio)
        self.assertEqual(first.procedure_coefficients, second.procedure_coefficients)

    def test_attempts_only_enter_when_stage_has_a_peer(self):
        rows = []
        changed = []
        items = {}
        for index, difficulty in enumerate((1700, 1800, 1900, 2000, 2100, 2200)):
            row = outcome(
                "a",
                f"B{index}",
                True,
                True,
                competition=f"C{index}",
                day=index,
                attempts_zone=1,
                attempts_top=1,
                attempt_cap=12,
            )
            rows.append(row)
            changed.append(
                ProblemOutcome(**{**row.__dict__, "attempts_to_zone": 12, "attempts_to_top": 12})
            )
            items[(row.competition_id, row.problem_id, "a")] = item_for(
                row, difficulty, difficulty + 100
            )
        # There is no same-stage peer on any item, so attempt numbers are not
        # eligible and cannot move the fitted response.
        first = fit_terrain_response(rows, items, config()).athletes["a"]
        second = fit_terrain_response(changed, items, config()).athletes["a"]
        self.assertEqual(first.log_scale_ratio, second.log_scale_ratio)
        self.assertEqual(first.procedure_coefficients, second.procedure_coefficients)

    def test_one_procedure_cannot_become_a_disguised_free_intercept(self):
        rows = []
        items = {}
        for index, difficulty in enumerate((1650, 1750, 1850, 1950, 2050, 2150)):
            row = outcome(
                "a",
                f"B{index}",
                index < 4,
                index < 2,
                competition=f"C{index}",
                day=index,
                procedure="onsight",
            )
            rows.append(row)
            items[(row.competition_id, row.problem_id, "a")] = item_for(
                row, difficulty, difficulty + 120
            )
        athlete = fit_terrain_response(rows, items, config()).athletes["a"]
        self.assertEqual(athlete.procedure_coefficients, (0.0, 0.0))
        self.assertEqual(
            athlete.procedure_sds,
            (config().athlete_procedure_prior_sd,) * 2,
        )


class ChronologyAndProjectionTests(unittest.TestCase):
    def test_future_and_target_result_leakage_are_rejected(self):
        with self.assertRaisesRegex(TerrainContractError, "frozen"):
            ProblemOutcome(
                **{
                    **outcome("a", "B1", True, False).__dict__,
                    "frozen_at": START + timedelta(hours=1),
                }
            )
        with self.assertRaisesRegex(TerrainContractError, "target results"):
            FrozenTargetItem(
                problem_id="B1",
                zone_mean=1900,
                zone_sd=50,
                post_zone_mean=2050,
                post_zone_sd=50,
                zone_post_correlation=0.2,
                attempt_cap=5,
                as_of=START - timedelta(days=1),
                provenance="realized target result top rate",
            )
        late = outcome("a", "B1", True, False, day=300)
        item = item_for(late, 1900)
        with self.assertRaisesRegex(TerrainContractError, "locked/test"):
            fit_terrain_response(
                [late],
                {(late.competition_id, late.problem_id, "a"): item},
                config(development_end=datetime(2024, 6, 1, tzinfo=UTC)),
            )

    def fitted(self, log_ratio, *, proc=(0.0, 0.0), proc_sd=(0.001, 0.001)):
        cfg = config(projection_draws=800)
        athlete = AthleteTerrainParameters(
            athlete_id="a",
            log_scale_ratio=log_ratio,
            log_scale_sd=0.001,
            procedure_coefficients=proc,
            procedure_sds=proc_sd,
            pre_zone_style_coefficients=(),
            pre_zone_style_sds=(),
            post_zone_style_coefficients=(),
            post_zone_style_sds=(),
            item_count=30,
            competition_count=10,
            mode="partially_pooled_athlete",
        )
        return FittedTerrainResponse(cfg, {"a": athlete}, 30, 10, 0.0)

    def test_projection_elo_is_target_conditioned_for_crossing_response(self):
        fitted = self.fitted(math.log(75.0 / 155.0))
        easy = target(
            "easy",
            [frozen_item(f"E{i}", 1600 + i * 20, 1700 + i * 20) for i in range(4)],
        )
        hard = target(
            "hard",
            [frozen_item(f"H{i}", 2100 + i * 25, 2200 + i * 25) for i in range(4)],
        )
        easy_projection = project_terrain_target("a", 1900, 5, easy, fitted)
        hard_projection = project_terrain_target("a", 1900, 5, hard, fitted)
        self.assertGreater(easy_projection.projection_elo, 1900)
        self.assertLess(hard_projection.projection_elo, 1900)
        self.assertGreater(easy_projection.projection_elo, hard_projection.projection_elo + 80)
        self.assertEqual(easy_projection.projection_context.split(";")[0], "onsight")

    def test_projection_fails_closed_across_rating_pool_or_reference_curve(self):
        fitted = self.fitted(0.0)
        base_item = frozen_item("B1", 1900, 2020)
        other_pool_item = replace(base_item, rating_pool_id="canadian_only_shadow")
        other_pool_target = replace(
            target("other-pool", [base_item]),
            items=(other_pool_item,),
            rating_pool_id="canadian_only_shadow",
        )
        with self.assertRaisesRegex(TerrainContractError, "rating pools"):
            project_terrain_target("a", 1900, 5, other_pool_target, fitted)

        other_curve = target(
            "other-curve",
            [replace(base_item, reference_response_scale=170.0)],
        )
        with self.assertRaisesRegex(TerrainContractError, "response-scale"):
            project_terrain_target("a", 1900, 5, other_curve, fitted)

    def test_confirmed_procedure_specialist_changes_target_projection(self):
        fitted = self.fitted(0.0, proc=(90.0, -45.0))
        items = [frozen_item(f"B{i}", 1900 + i * 10, 2020 + i * 10) for i in range(4)]
        onsight = project_terrain_target("a", 1900, 5, target("on", items, "onsight"), fitted)
        flash = project_terrain_target("a", 1900, 5, target("fl", items, "flash"), fitted)
        self.assertGreater(onsight.projection_elo, flash.projection_elo + 100)

    def test_endpoint_vector_is_primary_and_overall_scalar_is_optional(self):
        fitted = self.fitted(0.0)
        items = [frozen_item(f"B{i}", 1900, 2020) for i in range(4)]
        projection = project_terrain_target(
            "a",
            1900,
            5,
            target("vector", items, estimand="endpoint_vector_only"),
            fitted,
        )
        self.assertIsNone(projection.projection_elo)
        self.assertIsNone(projection.projection_elo_sd)
        self.assertAlmostEqual(projection.zone_projection_elo, 1900, delta=5)
        self.assertAlmostEqual(projection.top_projection_elo, 1900, delta=5)
        self.assertGreater(projection.expected_zones, 0.0)
        self.assertGreater(projection.expected_tops, 0.0)

    def test_target_item_uncertainty_is_propagated_to_outcome_uncertainty(self):
        fitted = self.fitted(math.log(95.0 / 155.0))
        narrow = target(
            "narrow",
            [frozen_item(f"N{i}", 1950, 2075, sd=5.0) for i in range(4)],
        )
        wide = target(
            "wide",
            [frozen_item(f"W{i}", 1950, 2075, sd=220.0) for i in range(4)],
        )
        narrow_projection = project_terrain_target("a", 1900, 3, narrow, fitted)
        wide_projection = project_terrain_target("a", 1900, 3, wide, fitted)
        self.assertGreater(wide_projection.expected_tops_sd, narrow_projection.expected_tops_sd)
        self.assertGreater(wide_projection.expected_zones_sd, narrow_projection.expected_zones_sd)

    def test_reviewed_style_is_a_target_shift_not_an_extra_rating_ledger(self):
        cfg = config(style_names=("physical",), projection_draws=800)
        athlete = AthleteTerrainParameters(
            "a",
            0.0,
            0.001,
            (0.0, 0.0),
            (0.001, 0.001),
            (90.0,),
            (0.001,),
            (90.0,),
            (0.001,),
            30,
            10,
            "partially_pooled_athlete",
        )
        fitted = FittedTerrainResponse(cfg, {"a": athlete}, 30, 10, 0.0)
        low_items = [
            frozen_item(
                f"B{i}", 1900, 2020,
                pre_styles={"physical": 0.0},
                post_styles={"physical": 0.0},
            )
            for i in range(4)
        ]
        high_items = [
            frozen_item(
                f"B{i}", 1900, 2020,
                pre_styles={"physical": 3.0},
                post_styles={"physical": 3.0},
            )
            for i in range(4)
        ]
        low = project_terrain_target(
            "a", 1900, 5, target("low-style", low_items), fitted
        )
        high = project_terrain_target(
            "a", 1900, 5, target("high-style", high_items), fitted
        )
        self.assertGreater(high.projection_elo, low.projection_elo + 150)
        self.assertEqual(high.projection_model, low.projection_model)


if __name__ == "__main__":
    unittest.main()
