import unittest

import pandas as pd

from scripts.audit_pathway_context_taxonomy_v1 import build_event_taxonomy


class PathwayContextTaxonomyTests(unittest.TestCase):
    def classify(self, source, tier, name, context="Senior / Open"):
        rows = pd.DataFrame(
            {
                "event_date": ["2025-01-01"],
                "event_name": [name],
                "source_scope": [source],
                "event_tier": [tier],
                "rating_context": [context],
                "pool": ["Boulder_Men"],
            }
        )
        return build_event_taxonomy(rows).iloc[0]

    def test_nacs_and_wc_are_direct_heads_but_never_input_filters(self):
        nacs = self.classify("CEC", "Continental / cross-border", "2025 NACS")
        wc = self.classify("IFSC", "World series", "IFSC World Cup Prague 2025")
        self.assertEqual(nacs.direct_context_head, "INTERFED:NORTH_AMERICA")
        self.assertEqual(wc.direct_context_head, "WC")
        self.assertTrue(nacs.all_results_update_shared_skill)
        self.assertFalse(nacs.direct_label_controls_input_eligibility)

    def test_youth_world_is_not_direct_adult_wc_evidence(self):
        row = self.classify(
            "IFSC",
            "World major youth",
            "IFSC Youth World Championships Helsinki 2025",
            "Youth",
        )
        self.assertEqual(row.direct_context_head, "IFSC_WORLD_YOUTH")

        inconsistent_tier = self.classify(
            "IFSC",
            "International other youth",
            "World Climbing Youth Championship Arco 2026",
            "Youth",
        )
        self.assertEqual(
            inconsistent_tier.direct_context_head,
            "IFSC_WORLD_YOUTH",
        )

    def test_ifsc_region_is_explicit_and_youth_separate(self):
        row = self.classify(
            "IFSC", "Continental series youth", "IFSC European Youth Cup Graz", "Youth"
        )
        self.assertEqual(row.direct_context_head, "CONT:EUROPE:YOUTH")

    def test_olympics_is_scenario_not_head(self):
        row = self.classify("IFSC", "World major", "Olympic Games Paris 2024")
        self.assertEqual(row.direct_context_head, "OLYM_SCENARIO_INPUT")

    def test_test_fixture_is_quarantined_before_nacs_rule(self):
        row = self.classify("USAC", "Continental / cross-border", "NACS Vail TEST")
        self.assertEqual(row.direct_context_head, "QUARANTINE")

    def test_ambiguous_continental_name_is_not_assigned(self):
        row = self.classify("IFSC", "Continental championship", "Continental Championship 2025")
        self.assertEqual(row.direct_context_head, "CONT:UNRESOLVED")

    def test_non_boulder_rows_are_out_of_scope(self):
        rows = pd.DataFrame({
            "event_date": ["2025-01-01", "2025-01-01"],
            "event_name": ["IFSC World Cup", "IFSC World Cup"],
            "source_scope": ["IFSC", "IFSC"],
            "event_tier": ["World series", "World series"],
            "rating_context": ["Senior / Open", "Senior / Open"],
            "pool": ["Boulder_Men", "Lead_Men"],
        })
        actual = build_event_taxonomy(rows)
        self.assertEqual(actual["pool"].tolist(), ["Boulder_Men"])


if __name__ == "__main__":
    unittest.main()
