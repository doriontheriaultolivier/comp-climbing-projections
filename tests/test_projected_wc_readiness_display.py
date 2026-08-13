from __future__ import annotations

import unittest

import pandas as pd

from comp_climbing_app import projected_wc_readiness_display


class ProjectedWcReadinessDisplayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.athlete = pd.Series(
            {"global_id": "IFSC:999", "pool": "Boulder_Women", "WC+-ELO": 1943.0}
        )
        self.projection = pd.DataFrame(
            [
                {
                    "athlete_id": "IFSC:999",
                    "pool": "Boulder_Women",
                    "projection_status": "exploratory_current_reference_available",
                    "wc_projection_score": 1512.4,
                    "wc_projection_score_sd": 104.0,
                    "wc_projection_score_sd_source": "wc_latent_readiness_sd",
                    "semifinal_probability_central": 0.1234,
                    "direct_senior_open_wc_plus_competitions": 0,
                }
            ]
        )

    def test_verified_projection_is_distinct_from_direct_rating(self) -> None:
        displayed = projected_wc_readiness_display(
            self.athlete, self.projection, {"verified": True}
        )
        self.assertTrue(displayed["available"])
        self.assertEqual(displayed["value"], "12.3%")
        self.assertNotEqual(displayed["value"], "1943")
        self.assertIn("Latent-state SD 104", displayed["caption"])
        self.assertIn("P(semifinal | starts in the reference WC field)", displayed["caption"])
        self.assertIn("not a calibrated confidence or predictive", displayed["caption"])
        self.assertIn("not access/selection probability", displayed["caption"])
        self.assertIn("not a conservative lower bound", displayed["caption"])
        self.assertIn("No direct Senior/Open WC+ start", displayed["caption"])
        self.assertIn("0/78 reached a semifinal", displayed["caption"])
        self.assertIn("mean forecast was 5.4%", displayed["caption"])
        self.assertIn("never enters the direct-WC leaderboard", displayed["caption"])

    def test_evidence_class_context_is_continuous_not_a_truth_cliff(self) -> None:
        one = self.projection.assign(direct_senior_open_wc_plus_competitions=1)
        one_display = projected_wc_readiness_display(
            self.athlete, one, {"verified": True}
        )
        self.assertIn("6/58 reached a semifinal", one_display["caption"])
        self.assertIn("mean forecast was 13.7%", one_display["caption"])

        established = self.projection.assign(
            direct_senior_open_wc_plus_competitions=2
        )
        established_display = projected_wc_readiness_display(
            self.athlete, established, {"verified": True}
        )
        self.assertIn("established 2+ evidence class", established_display["caption"])

    def test_unverified_artifact_never_uses_legacy_wc_value(self) -> None:
        displayed = projected_wc_readiness_display(
            self.athlete, self.projection, {"verified": False}
        )
        self.assertFalse(displayed["available"])
        self.assertEqual(displayed["value"], "Unavailable")
        self.assertNotIn("1943", displayed["caption"])
        self.assertIn("no legacy WC value is substituted", displayed["caption"])

    def test_duplicate_or_wrong_provenance_fails_closed(self) -> None:
        duplicate = pd.concat([self.projection, self.projection], ignore_index=True)
        self.assertFalse(
            projected_wc_readiness_display(
                self.athlete, duplicate, {"verified": True}
            )["available"]
        )
        wrong = self.projection.assign(wc_projection_score_sd_source="display_sd")
        self.assertFalse(
            projected_wc_readiness_display(
                self.athlete, wrong, {"verified": True}
            )["available"]
        )


if __name__ == "__main__":
    unittest.main()
