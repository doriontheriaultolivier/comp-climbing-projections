import json
import tempfile
import unittest
from pathlib import Path

from scripts.boulder_terrain_problem_adapter import (
    ADAPTER_SCHEMA,
    IdentityLookups,
    StableSnapshot,
    TerrainProblemAdapterError,
    is_hidden_or_test_event,
    normalize_round_row,
    to_problem_outcome,
    write_smoke_outputs,
)


def snapshot(scope="IFSC", event="1", url="/round/1", athlete="7"):
    state = StableSnapshot(
        athlete_id=f"PERSON:{athlete}",
        stable_mean=1930.0,
        stable_sd=45.0,
        confirmed_procedure="onsight",
        identity_status="retained",
        identity_provenance="exact_identity_safe_round_snapshot",
    )
    return IdentityLookups(
        exact_snapshots={(scope, event, url, athlete): state},
        source_node_ids={(scope, athlete): state.athlete_id},
        ambiguous_source_nodes=frozenset(),
    )


def round_row(*, scope="IFSC", event="1", url="/round/1", athlete="7", **updates):
    row = {
        "source_scope": scope,
        "source_event_id": event,
        "event_date": "2025-05-01",
        "event_name": "Real Boulder event",
        "source_url": url,
        "discipline": "Boulder",
        "gender": "Men",
        "category": "Men",
        "round_name": "Semifinal",
        "format_identifier": "boulder_one_group_one_boulder_set",
        "athlete_source_id": athlete,
        "boulder_outcomes_json": json.dumps(
            [
                {
                    "problem_index": 1,
                    "route_id": 9001,
                    "top": True,
                    "zone": True,
                    "low_zone": True,
                    "top_tries": 3,
                    "zone_tries": 1,
                    "points": None,
                }
            ]
        ),
    }
    row.update(updates)
    return row


class TerrainProblemAdapterTests(unittest.TestCase):
    def test_ordinary_row_preserves_route_and_withholds_ambiguous_attempt_exposure(self):
        evidence = normalize_round_row(round_row(), snapshot())[0]
        self.assertEqual(evidence.problem_id, "IFSC|event:1|route:9001")
        self.assertEqual(evidence.leaderboard_route_id, "9001")
        self.assertEqual(evidence.marker_chain, "ordinary_zone_top_2h")
        self.assertTrue(evidence.item_calibration_eligible)
        outcome = to_problem_outcome(evidence)
        self.assertEqual(outcome.competition_id, "IFSC|event:1")
        self.assertIsNone(outcome.attempts_to_zone)
        self.assertIsNone(outcome.attempts_to_top)
        self.assertIsNone(outcome.post_zone_opportunities)

    def test_combined_low_zone_only_is_retained_but_fails_closed_for_two_hurdles(self):
        row = round_row(
            format_identifier="bl_qualification",
            boulder_outcomes_json=json.dumps(
                [
                    {
                        "problem_index": 1,
                        "route_id": 42,
                        "points": 4.7,
                        "top": False,
                        "zone": False,
                        "low_zone": True,
                        "top_tries": 0,
                        "zone_tries": 0,
                        "low_zone_tries": 2,
                    }
                ]
            ),
        )
        evidence = normalize_round_row(row, snapshot())[0]
        self.assertEqual(evidence.marker_chain, "combined_low_zone_zone_top_3h")
        self.assertTrue(evidence.reached_low_zone)
        self.assertFalse(evidence.reached_zone)
        self.assertFalse(evidence.reached_top)
        self.assertFalse(evidence.item_calibration_eligible)
        with self.assertRaises(TerrainProblemAdapterError):
            to_problem_outcome(evidence)

    def test_ordinary_auxiliary_low_zone_is_not_promoted_to_primary_zone(self):
        row = round_row(
            boulder_outcomes_json=json.dumps(
                [
                    {
                        "problem_index": 1,
                        "route_id": 43,
                        "top": False,
                        "zone": False,
                        "low_zone": True,
                        "top_tries": 0,
                        "zone_tries": 0,
                        "low_zone_tries": 1,
                    }
                ]
            )
        )
        evidence = normalize_round_row(row, snapshot())[0]
        self.assertEqual(evidence.marker_chain, "ordinary_zone_top_2h")
        self.assertTrue(evidence.reached_low_zone)
        self.assertFalse(evidence.reached_zone)
        self.assertFalse(evidence.reached_top)
        self.assertTrue(evidence.item_calibration_eligible)

    def test_schema_is_bound_once_by_round_format_not_by_athlete_outcome(self):
        low_row = round_row(
            boulder_outcomes_json=json.dumps(
                [{"problem_index": 1, "route_id": 43, "top": False,
                  "zone": False, "low_zone": True, "low_zone_tries": 2}]
            )
        )
        top_row = round_row()
        low = normalize_round_row(low_row, snapshot())[0]
        top = normalize_round_row(top_row, snapshot())[0]
        self.assertEqual(low.marker_chain, top.marker_chain)
        self.assertEqual(low.marker_chain, "ordinary_zone_top_2h")
        self.assertEqual(low.source_marker_schema, top.source_marker_schema)
        self.assertEqual(
            low.source_marker_schema,
            "ordinary_explicit_zone_top_auxiliary_low_zone_retained",
        )

    def test_round_aggregate_attempts_never_replace_marker_success_ordinals(self):
        row = round_row(top_attempts="999", zone_attempts="777")
        evidence = normalize_round_row(row, snapshot())[0]
        self.assertEqual(evidence.source_attempts_to_zone, 1)
        self.assertEqual(evidence.source_attempts_to_top, 3)
        self.assertEqual(evidence.source_attempts_to_low_zone, None)
        self.assertIn("success_attempt", evidence.attempt_semantics)

    def test_partial_primary_flags_fail_closed_in_adapter(self):
        row = round_row(
            boulder_outcomes_json=json.dumps(
                [{"problem_index": 1, "route_id": 44, "top": False}]
            )
        )
        evidence = normalize_round_row(row, snapshot())[0]
        self.assertIsNone(evidence.reached_zone)
        self.assertFalse(evidence.reached_top)
        self.assertFalse(evidence.item_calibration_eligible)

    def test_reviewed_canadian_u19_u21_alias_does_not_replace_route_identity(self):
        lookup = snapshot(scope="CEC", event="224")
        base = round_row(scope="CEC", event="224")
        u19 = normalize_round_row({**base, "category": "U19 Men"}, lookup)[0]
        alternate = json.loads(base["boulder_outcomes_json"])
        alternate[0]["route_id"] = 9019
        u21 = normalize_round_row(
            {
                **base,
                "category": "U21 Men",
                "boulder_outcomes_json": json.dumps(alternate),
            },
            lookup,
        )[0]
        self.assertNotEqual(u19.problem_id, u21.problem_id)
        self.assertEqual(u19.terrain_set_alias, u21.terrain_set_alias)
        self.assertIn("terrain:A_JR", u19.terrain_set_alias)

    def test_cross_category_item_bridge_requires_common_leaderboard_identity(self):
        """A category label never blocks a real shared federation route ID.

        Conversely, a reviewed terrain-set alias alone remains insufficient:
        the previous test proves that different route IDs stay different
        items even when the wall/set is known to be shared.
        """

        lookup = snapshot(scope="CEC", event="224")
        base = round_row(scope="CEC", event="224")
        u15 = normalize_round_row({**base, "category": "U15 Men"}, lookup)[0]
        senior = normalize_round_row(
            {**base, "category": "Senior Men"}, lookup
        )[0]
        self.assertEqual(u15.leaderboard_route_id, senior.leaderboard_route_id)
        self.assertEqual(u15.problem_id, senior.problem_id)

    def test_ambiguous_identity_is_split_source_locally_and_not_model_ready(self):
        lookup = IdentityLookups(
            exact_snapshots={},
            source_node_ids={},
            ambiguous_source_nodes=frozenset({("IFSC", "7")}),
        )
        evidence = normalize_round_row(round_row(), lookup)[0]
        self.assertEqual(evidence.athlete_id, "RESEARCH-SOURCE:IFSC:7")
        self.assertEqual(
            evidence.identity_provenance,
            "ambiguous_identity_safe_source_node_split_locally",
        )
        self.assertFalse(evidence.item_calibration_eligible)

    def test_hidden_and_test_event_filter_is_deliberately_narrow(self):
        for name in ("Hidden TEST", "Boulder Test", "Pratique pour nouveau HJ"):
            self.assertTrue(is_hidden_or_test_event(name))
        self.assertFalse(is_hidden_or_test_event("Canadian Youth Boulder Nationals"))

    def test_smoke_manifest_binds_the_normalized_sample_hash_and_schema(self):
        evidence = normalize_round_row(round_row(), snapshot())[0]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_smoke_outputs(
                {"schema": ADAPTER_SCHEMA, "research_only": True},
                [evidence],
                output,
            )
            manifest = json.loads((output / "manifest.json").read_text("utf-8"))
            bound = manifest["output_files"]["normalized_sample.csv"]
            self.assertEqual(manifest["schema"], ADAPTER_SCHEMA)
            self.assertEqual(
                bound["bytes"], (output / "normalized_sample.csv").stat().st_size
            )
            self.assertRegex(bound["sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
