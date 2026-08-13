from __future__ import annotations

import inspect
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import comp_climbing_app as app_module
from comp_climbing_app import (
    ALL_RATINGS,
    RATING_ORDER,
    athlete_selector_frame,
    load_current_wc_projection_artifact,
    pool_scatter,
    progression_projection,
    quarantine_obvious_fixture_exposure,
    withhold_legacy_wc_without_direct_evidence,
    render_progression,
    render_wr_pool,
    selected_rows,
)


ROOT = Path(__file__).resolve().parents[1]


class CanadianPilotProjectionTests(unittest.TestCase):
    def test_legacy_wc_is_withheld_when_only_youth_world_evidence_exists(self) -> None:
        athletes = pd.DataFrame({
            "pool": ["Boulder_Women", "Boulder_Women"],
            "global_id": ["AMARI", "MADISON"],
            "WC+-ELO": [1942.8, 1978.8],
            "WC+-ELO evidence": [2.0, 12.0],
        })
        history = pd.DataFrame({
            "pool": ["Boulder_Women", "Boulder_Women"],
            "global_id": ["AMARI", "MADISON"],
            "source_scope": ["IFSC", "IFSC"],
            "source_event_id": ["1418", "1411"],
            "event_name": [
                "IFSC Youth World Championships Helsinki 2025",
                "IFSC World Cup Bern 2025",
            ],
        })

        safe, audit = withhold_legacy_wc_without_direct_evidence(athletes, history)

        amari = safe.loc[safe["global_id"].eq("AMARI")].iloc[0]
        madison = safe.loc[safe["global_id"].eq("MADISON")].iloc[0]
        self.assertTrue(pd.isna(amari["WC+-ELO"]))
        self.assertTrue(pd.isna(amari["WC+-ELO evidence"]))
        self.assertEqual(amari["direct_senior_wc_competitions"], 0)
        self.assertEqual(
            amari["legacy_wc_display_status"],
            "WITHHELD_NO_DIRECT_SENIOR_WC_EVIDENCE",
        )
        self.assertEqual(float(madison["WC+-ELO"]), 1978.8)
        self.assertEqual(madison["direct_senior_wc_competitions"], 1)
        self.assertEqual(int(audit.iloc[0]["legacy_wc_rows_withheld"]), 1)

    def test_deployed_amari_legacy_wc_is_not_publicly_presented(self) -> None:
        athletes = pd.read_parquet("data/boulder_overview_athletes.parquet")
        history = pd.read_parquet("data/boulder_overview_history.parquet")
        safe, _ = withhold_legacy_wc_without_direct_evidence(athletes, history)
        rows = safe.loc[
            safe["athlete_name"].astype(str).str.contains(
                "Bourbonnais Amari", case=False, na=False
            )
        ]
        self.assertGreaterEqual(len(rows), 1)
        self.assertTrue(rows["WC+-ELO"].isna().all())
        self.assertTrue(rows["WC+-ELO evidence"].isna().all())
        self.assertTrue(rows["direct_senior_wc_competitions"].eq(0).all())

    def test_public_read_hides_only_reviewed_amari_alias(self) -> None:
        app_module.read_data.clear()
        data = app_module.read_data()
        athletes = data["athletes"]
        history = data["history"]
        self.assertEqual(int(athletes["global_id"].eq("IFSC:14843").sum()), 1)
        self.assertEqual(int(athletes["global_id"].eq("IFSC:18545").sum()), 0)
        self.assertEqual(int(history["global_id"].eq("IFSC:18545").sum()), 2)
        audit = data["reviewed_identity_alias_audit"]
        self.assertEqual(audit["suppressed_profile_count"], 1)
        self.assertEqual(audit["history_rows_changed"], 0)
        self.assertFalse(audit["ratings_merged"])

    def test_current_bundle_exposes_wc_plus_not_nonexistent_wr_elo(self) -> None:
        athletes = pd.read_parquet("data/boulder_overview_athletes.parquet")
        self.assertIn("WC+-ELO", RATING_ORDER)
        self.assertIn("WC+-ELO", ALL_RATINGS)
        self.assertNotIn("WR-ELO", RATING_ORDER)
        self.assertNotIn("WR-ELO", ALL_RATINGS)
        self.assertIn("WC+-ELO", athletes.columns)
        self.assertGreater(int(athletes["WC+-ELO"].notna().sum()), 0)
        # Legacy schema columns remain for compatibility, but contain no ratings.
        self.assertEqual(int(athletes["WR-ELO"].notna().sum()), 0)

    def test_world_ranking_view_uses_current_wc_plus_rating_family(self) -> None:
        app = AppTest.from_file(
            ROOT / "comp_climbing_app.py", default_timeout=30
        ).run()
        app.segmented_control[2].set_value("WR Pool").run()
        self.assertEqual(len(app.exception), 0)
        rating_control = next(
            control
            for control in app.segmented_control
            if control.label == "Rating evidence"
        )
        self.assertEqual(rating_control.value, "WC+-ELO")
        self.assertIn("WC+-ELO", rating_control.options)
        self.assertNotIn("WR-ELO", rating_control.options)

    def test_duplicate_names_select_only_the_stable_requested_record(self) -> None:
        frame = pd.DataFrame(
            {
                "pool": ["Boulder_Men", "Boulder_Men"],
                "global_id": ["CEC:595", "VL-GLOBAL:18197"],
                "athlete_name": ["ALLORA KLINKER", "Allora Klinker"],
                "name_key": ["alloraklinker", "alloraklinker"],
                "country": ["CAN", ""],
                "nationality": ["CAN", ""],
                "Global-ELO": [1864.93, 1666.17],
                "Global-ELO uncertainty": [72.76, 260.14],
                "Global-ELO evidence": [18.0, 2.0],
                "Global-ELO status": ["Established", "Provisional"],
            }
        )
        selectors = athlete_selector_frame(frame)
        self.assertEqual(len(selectors), 2)
        self.assertTrue(selectors["_selection_label"].str.contains("CEC:595", regex=False).any())
        chosen = selected_rows(frame, ["Boulder_Men::CEC:595"])
        self.assertEqual(chosen["global_id"].tolist(), ["CEC:595"])
        self.assertEqual(selected_rows(frame, ["CEC:595"])["global_id"].tolist(), ["CEC:595"])

    def test_fixture_events_are_removed_without_dropping_legitimate_identity_rows(self) -> None:
        athletes = pd.DataFrame(
            {
                "global_id": ["CEC:1", "CEC:2"],
                "cnr_rank": [1.0, 2.0],
                "Global-ELO": [1900.0, 1800.0],
            }
        )
        history = pd.DataFrame(
            {
                "global_id": ["CEC:1", "CEC:1", "CEC:2"],
                "event_name": ["Bouldering Test", "Real Nationals", "Real Nationals"],
                "source_scope": ["CEC", "CEC", "CEC"],
                "source_event_id": ["fixture", "real", "real"],
                "rating_after": [1900.0, 1910.0, 1805.0],
            }
        )
        safe_athletes, safe_history, audit = quarantine_obvious_fixture_exposure(
            athletes, history
        )
        self.assertEqual(safe_athletes["global_id"].tolist(), ["CEC:1", "CEC:2"])
        self.assertEqual(safe_history["global_id"].unique().tolist(), ["CEC:1", "CEC:2"])
        self.assertTrue(pd.isna(safe_athletes.loc[0, "Global-ELO"]))
        self.assertEqual(float(safe_athletes.loc[1, "Global-ELO"]), 1800.0)
        self.assertTrue(
            safe_history.loc[safe_history["global_id"].eq("CEC:1"), "rating_after"]
            .isna()
            .all()
        )
        self.assertEqual(
            safe_history.loc[
                safe_history["global_id"].eq("CEC:2"), "rating_after"
            ].tolist(),
            [1805.0],
        )
        self.assertEqual(int(audit.iloc[0]["fixture_exposed_athlete_ids"]), 1)
        self.assertEqual(int(audit.iloc[0]["retained_canadian_identities"]), 1)

    def test_deployed_fixture_guard_has_exact_closed_effect(self) -> None:
        athletes = pd.read_parquet("data/boulder_overview_athletes.parquet")
        history = pd.read_parquet("data/boulder_overview_history.parquet")
        safe_athletes, safe_history, audit = quarantine_obvious_fixture_exposure(
            athletes, history
        )
        row = audit.iloc[0]
        self.assertEqual(int(row["fixture_event_rows"]), 2041)
        self.assertEqual(int(row["fixture_source_events"]), 94)
        self.assertEqual(int(row["fixture_pool_event_keys"]), 120)
        self.assertEqual(int(row["fixture_exposed_athlete_ids"]), 731)
        self.assertEqual(int(row["retained_canadian_identities"]), 4)
        self.assertFalse(
            safe_history["event_name"].astype(str).str.contains(
                r"(?i)\b(?:test|mock|demo|dummy|sandbox|hidden)\b", regex=True
            ).any()
        )
        self.assertEqual(len(athletes), len(safe_athletes))
        exposed = safe_athletes["legacy_fixture_exposed"]
        self.assertEqual(int(exposed.sum()), 731)
        self.assertTrue(safe_athletes.loc[exposed, "Global-ELO"].isna().all())
        exposed_history = safe_history["legacy_fixture_exposed"]
        self.assertTrue(
            safe_history.loc[
                exposed_history,
                [
                    "rating_after",
                    "rating_before",
                    "event_start_rating",
                    "performance_elo",
                ],
            ].isna().all().all()
        )

    def test_deployed_colliding_names_are_exact_stable_choices(self) -> None:
        athletes = pd.read_parquet("data/boulder_overview_athletes.parquet")
        for selection_id, expected_global_id in (
            ("Boulder_Women::CEC:595", "CEC:595"),
            ("Boulder_Men::CEC:20", "CEC:20"),
            ("Boulder_Men::USAC:610", "USAC:610"),
        ):
            chosen = selected_rows(athletes, [selection_id])
            self.assertEqual(chosen["global_id"].tolist(), [expected_global_id])

    def test_diagnostic_charts_never_apply_universal_outcome_thresholds(self) -> None:
        for function in (
            pool_scatter,
            render_wr_pool,
            render_progression,
            progression_projection,
        ):
            self.assertNotIn("add_outcome_thresholds(", inspect.getsource(function))

        for legacy_symbol in (
            "add_outcome_thresholds",
            "conditional_outcome_probability",
            "conditional_outcome_projection",
            "wc_semifinal_rating_evidence",
            "render_focus_hypotheses",
        ):
            self.assertFalse(hasattr(app_module, legacy_symbol), legacy_symbol)

    def test_current_pairwise_form_target_named_regression(self) -> None:
        projection = pd.read_csv(
            "data/canadian_current_wc_projection_v3_youth_world_complete.csv"
        )
        by_id = projection.set_index("athlete_id")
        values = {
            name: float(by_id.loc[athlete_id, "semifinal_probability_central"])
            for name, athlete_id in {
                "Oscar": "IFSC:11847",
                "Matthew": "IFSC:14842",
                "Hugo": "IFSC:1682",
                "Dylan": "IFSC:17188",
            }.items()
        }
        self.assertGreater(values["Oscar"], values["Matthew"])
        self.assertGreater(values["Matthew"], values["Hugo"])
        self.assertGreater(values["Hugo"], values["Dylan"])
        self.assertGreater(values["Oscar"], 0.35)
        self.assertLess(values["Dylan"], 0.02)
        dylan = by_id.loc["IFSC:17188"]
        self.assertEqual(
            dylan["score_route"],
            "wc_target_score_zero_prior_intercept_adjusted_link",
        )
        self.assertEqual(
            dylan["evidence_class"], "zero_prior_senior_open_wc_plus"
        )
        self.assertLess(float(dylan["form_adjustment_100d"]), 0.0)
        self.assertEqual(
            int(dylan["direct_senior_open_wc_plus_competitions"]), 0
        )
        self.assertTrue(
            np.isfinite(
                float(dylan["bridge_probability_evidence_class_sensitivity"])
            )
        )
        self.assertGreaterEqual(int(dylan["model_gate_anchored_events"]), 1)
        self.assertGreaterEqual(
            int(dylan["model_gate_unique_anchored_opponents"]), 1
        )

        available = projection.loc[
            projection["projection_status"].eq(
                "exploratory_current_reference_available"
            )
        ]
        senior_wc = pd.to_numeric(
            available["direct_senior_open_wc_plus_competitions"], errors="raise"
        )
        zero = available["score_route"].eq(
            "wc_target_score_zero_prior_intercept_adjusted_link"
        )
        one = available["score_route"].eq(
            "wc_target_score_one_prior_intercept_adjusted_link"
        )
        standard = available["score_route"].eq(
            "wc_target_score_standard_link"
        )
        self.assertTrue((zero | one | standard).all())
        self.assertTrue(senior_wc.loc[zero].eq(0).all())
        self.assertTrue(senior_wc.loc[one].eq(1).all())
        self.assertTrue(senior_wc.loc[standard].ge(2).all())
        expected_evidence = np.select(
            [senior_wc.eq(0), senior_wc.eq(1)],
            [
                "zero_prior_senior_open_wc_plus",
                "one_prior_senior_open_wc_plus",
            ],
            default="two_or_more_prior_senior_open_wc_plus",
        )
        self.assertTrue(
            np.array_equal(available["evidence_class"], expected_evidence)
        )
        governing_slope = pd.to_numeric(
            available["governing_calibration_slope_per_100"], errors="raise"
        )
        self.assertTrue(governing_slope.gt(0.0).all())
        self.assertAlmostEqual(
            float(governing_slope.loc[zero].iloc[0]),
            float(governing_slope.loc[standard].iloc[0]),
            places=12,
        )
        self.assertTrue(
            available["wc_projection_score_sd_source"]
            .eq("wc_latent_readiness_sd")
            .all()
        )
        self.assertTrue(
            pd.to_numeric(
                available["model_gate_anchored_events"], errors="raise"
            ).gt(0).all()
        )

    def test_current_projection_artifact_is_bound_and_tamper_closed(self) -> None:
        projection, metadata = load_current_wc_projection_artifact(Path("data"))
        self.assertTrue(metadata.get("verified"))
        self.assertFalse(projection.empty)
        self.assertTrue(metadata["calibration"]["event_clean_refit"])
        self.assertTrue(
            metadata["low_wc_evidence_calibration"]["event_clean_refit"]
        )
        self.assertEqual(
            metadata["low_wc_evidence_calibration"]["central_route"],
            "separate_k0_k1_intercept_adjusted_links",
        )
        self.assertGreater(
            metadata["low_wc_evidence_calibration"]["zero_prior"][
                "slope_per_100"
            ],
            0.0,
        )
        self.assertIsNone(metadata["model"]["initializer_warm_start_sha256"])
        routing = metadata["model"]["target_domain_routing"]
        self.assertEqual(routing["schema"], "ifsc-youth-world-separate-target-head-v1")
        self.assertEqual(routing["rows_routed"], 5752)
        self.assertEqual(routing["source_events_routed"], 11)
        self.assertEqual(routing["pool_events_routed"], 22)
        self.assertEqual(routing["youth_world_rows_in_senior_wc_target_state"], 0)
        self.assertTrue(routing["senior_open_world_major_preserved_in_wc_plus"])
        post_cutoff = routing["post_cutoff_replay"]
        self.assertEqual(post_cutoff["pool_events"], 6)
        self.assertEqual(post_cutoff["athlete_events"], 1111)
        self.assertEqual(post_cutoff["youth_world_events_in_wc_plus"], 0)
        self.assertEqual(
            post_cutoff["all_history_source_event_inventory_sha256"],
            "28fe20328b6eb6c6ed8893a045ff2eea66940f3a4cd47aa83d70ac6daef9005a",
        )
        self.assertTrue(
            metadata["claims"]["youth_world_shared_skill_graph_preserved"]
        )
        self.assertFalse(
            metadata["claims"]["youth_world_directly_updates_senior_wc_offset"]
        )
        for evidence_class in ("zero_prior", "one_prior"):
            self.assertEqual(
                metadata["low_wc_evidence_calibration"][evidence_class][
                    "slope_policy"
                ],
                "fixed_to_clean_base_slope",
            )
        self.assertFalse(
            metadata["claims"]["rating_state_sensitivity_uses_bridge_sd"]
        )
        raw_athletes = pd.read_parquet("data/boulder_overview_athletes.parquet")
        raw_history = pd.read_parquet("data/boulder_overview_history.parquet")
        retained, _, _ = quarantine_obvious_fixture_exposure(
            raw_athletes, raw_history
        )
        exposed_cnr_ids = set(
            retained.loc[
                retained["legacy_fixture_exposed"]
                & pd.to_numeric(retained["cnr_rank"], errors="coerce").notna(),
                "global_id",
            ].astype(str)
        )
        self.assertEqual(len(exposed_cnr_ids), 4)
        available_by_id = projection.set_index("athlete_id")["projection_status"]
        self.assertTrue(exposed_cnr_ids.issubset(set(available_by_id.index)))
        self.assertTrue(
            available_by_id.loc[sorted(exposed_cnr_ids)]
            .eq("exploratory_current_reference_available")
            .all()
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in (
                "canadian_current_wc_projection_v3_youth_world_complete.csv",
                "canadian_current_wc_projection_v3_youth_world_complete.metadata.json",
            ):
                shutil.copy2(Path("data") / name, root / name)
            with (
                root / "canadian_current_wc_projection_v3_youth_world_complete.csv"
            ).open(
                "a", encoding="utf-8"
            ) as destination:
                destination.write("\n")
            rejected, audit = load_current_wc_projection_artifact(root)
            self.assertTrue(rejected.empty)
            self.assertFalse(audit.get("verified"))
            self.assertIn("hash mismatch", str(audit.get("reason")))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_name = "canadian_current_wc_projection_v3_youth_world_complete.csv"
            metadata_name = (
                "canadian_current_wc_projection_v3_youth_world_complete.metadata.json"
            )
            shutil.copy2(Path("data") / csv_name, root / csv_name)
            payload = json.loads((Path("data") / metadata_name).read_text(encoding="utf-8"))
            payload["model"]["target_domain_routing"][
                "youth_world_rows_in_senior_wc_target_state"
            ] = 1
            (root / metadata_name).write_text(json.dumps(payload), encoding="utf-8")
            rejected, audit = load_current_wc_projection_artifact(root)
            self.assertTrue(rejected.empty)
            self.assertFalse(audit.get("verified"))
            self.assertIn("routing closure", str(audit.get("reason")))


if __name__ == "__main__":
    unittest.main()
