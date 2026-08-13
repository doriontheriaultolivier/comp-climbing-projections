from __future__ import annotations

import pandas as pd

from scripts.build_pathway_head_routing_v1 import build_routing, classify_event


def test_head_classifier_separates_youth_regional_from_senior_regional() -> None:
    assert classify_event("ifsc_youth_world", ("Youth",)) == "YW_IFSC"
    assert classify_event("ifsc_non_wc", ("Youth",)) == "YOUTH_REG_IFSC_GRAPH_ONLY"
    assert classify_event("ifsc_non_wc", ("Senior / Open",)) == "REG_IFSC"
    assert classify_event("wc+", ("Senior / Open",)) == "WC_PLUS"


def test_routing_never_labels_youth_regional_as_direct_reg_ifsc() -> None:
    plan = {
        "events": [
            {
                "event_id": "IFSC|1|Boulder_Women|2025-01-01",
                "event_date": "2025-01-01T00:00:00",
                "event_name": "Youth Regional",
                "pool": "Boulder_Women",
                "target_domain": "ifsc_non_wc",
            },
            {
                "event_id": "IFSC|2|Boulder_Women|2025-02-01",
                "event_date": "2025-02-01T00:00:00",
                "event_name": "Senior Regional",
                "pool": "Boulder_Women",
                "target_domain": "ifsc_non_wc",
            },
        ]
    }
    prepared = pd.DataFrame(
        {
            "event_date": ["2025-01-01", "2025-02-01"],
            "event_name": ["Youth Regional", "Senior Regional"],
            "pool": ["Boulder_Women", "Boulder_Women"],
            "age_class": ["Youth", "Senior / Open"],
        }
    )
    result = build_routing(plan, prepared).set_index("event_id")
    assert pd.isna(
        result.loc["IFSC|1|Boulder_Women|2025-01-01", "direct_demonstrated_head"]
    )
    assert (
        result.loc["IFSC|2|Boulder_Women|2025-02-01", "direct_demonstrated_head"]
        == "REG-IFSC"
    )


def test_distinct_plan_events_may_share_governed_metadata_key() -> None:
    plan = {
        "events": [
            {
                "event_id": f"SRC|{event_id}|Boulder_Men|2025-01-01",
                "event_date": "2025-01-01T00:00:00",
                "event_name": "Shared source event",
                "pool": "Boulder_Men",
                "target_domain": "ifsc_non_wc",
            }
            for event_id in (1, 2)
        ]
    }
    prepared = pd.DataFrame(
        {
            "event_date": ["2025-01-01"],
            "event_name": ["Shared source event"],
            "pool": ["Boulder_Men"],
            "age_class": ["Senior / Open"],
        }
    )
    result = build_routing(plan, prepared)
    assert len(result) == 2
    assert set(result["pathway_head"]) == {"REG_IFSC"}
