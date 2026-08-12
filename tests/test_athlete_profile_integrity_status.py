import unittest

import numpy as np
import pandas as pd

import comp_climbing_app as app


class AthleteProfileIntegrityStatusTests(unittest.TestCase):
    def athletes(self):
        return pd.DataFrame(
            {
                "pool": ["Boulder_Women", "Boulder_Women", "Boulder_Women"],
                "global_id": ["IFSC:10", "IFSC:20", "CEC:30"],
                "name_key": ["sameathlete", "sameathlete", "otherathlete"],
                "athlete_name": ["SAME Athlete", "Same Athlete", "Other Athlete"],
                "Global-ELO": [2000.0, np.nan, np.nan],
                "IFSC-ELO": [1950.0, np.nan, np.nan],
                "WC+-ELO": [1900.0, np.nan, np.nan],
            }
        )

    def test_duplicate_name_is_visible_but_not_auto_merged(self):
        athletes = self.athletes()
        history = pd.DataFrame(
            {"pool": ["Boulder_Women"], "global_id": ["IFSC:10"]}
        )
        result = app.athlete_profile_integrity_status(
            athletes.iloc[0], athletes, history
        )
        self.assertEqual(result["status"], "RATING_EVIDENCE_AVAILABLE")
        self.assertTrue(result["same_name_identity_collision"])
        self.assertEqual(result["same_name_identity_ids"], ["IFSC:10", "IFSC:20"])

    def test_no_history_is_identity_status_not_low_ability(self):
        athletes = self.athletes()
        history = pd.DataFrame(columns=["pool", "global_id"])
        result = app.athlete_profile_integrity_status(
            athletes.iloc[2], athletes, history
        )
        self.assertEqual(result["status"], "NO_EXACT_HISTORY_LINK")
        self.assertIn("not a zero or low ability", result["explanation"])

    def test_linked_history_can_fail_minimum_evidence(self):
        athletes = self.athletes()
        history = pd.DataFrame(
            {"pool": ["Boulder_Women"], "global_id": ["IFSC:20"]}
        )
        result = app.athlete_profile_integrity_status(
            athletes.iloc[1], athletes, history
        )
        self.assertEqual(
            result["status"], "HISTORY_LINKED_MINIMUM_EVIDENCE_NOT_MET"
        )


if __name__ == "__main__":
    unittest.main()
