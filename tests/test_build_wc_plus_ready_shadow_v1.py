from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.build_wc_plus_ready_shadow_v1 import build_shadow


def test_zero_wc_starts_with_connected_graph_gets_readiness_not_direct_level() -> None:
    snapshot = pd.DataFrame(
        {
            "pool": ["Boulder_Women"],
            "global_id": ["IFSC:14843"],
            "snapshot_at_utc": ["2026-08-13T00:00:00Z"],
            "internal_graph_wc_central": [2108.6],
            "internal_graph_wc_state_sd": [189.3],
            "internal_graph_wc_predictive_sd": [244.6],
            "stable_skill_mean": [2155.7],
            "form_at_snapshot": [-47.1],
            "youth_world_offset_at_snapshot": [28.6],
        }
    )
    replay = pd.DataFrame(
        {
            "pool": ["Boulder_Women"],
            "athlete_id": ["IFSC:14843"],
            "event_index": [10],
            "event_id": ["IFSC|1531"],
            "target_domain": ["ifsc_youth_world"],
            "anchored_events": [10],
            "anchored_comparisons": [146],
            "unique_anchored_opponents": [62],
            "anchored_effective_weight": [34.1],
        }
    )
    direct = pd.DataFrame(columns=["pool", "athlete_id", "event_id"])
    initializer = pd.DataFrame(
        {
            "pool": ["Boulder_Women"],
            "athlete_id": ["IFSC:14843"],
            "component_anchored": [True],
            "effective_competitions": [1.0],
            "unique_opponents": [20],
        }
    )
    result = build_shadow(snapshot, replay, direct, initializer).iloc[0]
    assert result["wc_plus_ready"] == 2108.6
    assert not result["wc_plus_demonstrated_eligible"]
    assert "no direct WC+ start" in result["wc_plus_ready_status"]
    assert result["wc_plus_ready_evidence"] == "substantial connected indirect evidence"
    assert np.isnan(result["yw_ifsc_ready"])
    assert np.isclose(result["yw_ifsc_pooled_sensitivity"], 2137.2)
    assert result["yw_ifsc_ready_status"].startswith("withheld")
    assert np.isnan(result["reg_ifsc_ready"])
    assert result["direct_youth_ifsc_events"] == 1
    assert result["yw_ifsc_demonstrated_eligible"]


def test_unanchored_prior_is_not_estimable() -> None:
    snapshot = pd.DataFrame(
        {
            "pool": ["Boulder_Men"],
            "global_id": ["X"],
            "snapshot_at_utc": ["2026-08-13T00:00:00Z"],
            "internal_graph_wc_central": [1800.0],
            "internal_graph_wc_state_sd": [500.0],
            "internal_graph_wc_predictive_sd": [520.0],
            "stable_skill_mean": [1800.0],
            "form_at_snapshot": [0.0],
            "youth_world_offset_at_snapshot": [0.0],
        }
    )
    replay = pd.DataFrame(columns=[
        "pool", "athlete_id", "event_index", "event_id", "target_domain", "anchored_events",
        "anchored_comparisons", "unique_anchored_opponents", "anchored_effective_weight",
    ])
    direct = pd.DataFrame(columns=["pool", "athlete_id", "event_id"])
    initializer = pd.DataFrame(
        {
            "pool": ["Boulder_Men"],
            "athlete_id": ["X"],
            "component_anchored": [False],
            "effective_competitions": [0.0],
            "unique_opponents": [0],
        }
    )
    result = build_shadow(snapshot, replay, direct, initializer).iloc[0]
    assert np.isnan(result["wc_plus_ready"])
    assert result["wc_plus_ready_status"].startswith("not estimable")
