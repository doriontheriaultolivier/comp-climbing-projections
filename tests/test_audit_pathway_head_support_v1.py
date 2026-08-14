import pandas as pd

from scripts.audit_pathway_head_support_v1 import build_support


def test_support_counts_event_contexts_and_wc_bridges_without_binary_gate():
    rows = pd.DataFrame([
        {"global_id": "A", "pool": "Boulder_Men", "event_date": "2024-01-01", "event_name": "CEC Nationals", "source_scope": "CEC", "event_tier": "National championship", "rating_context": "Senior / Open", "round_group": "Qualification"},
        {"global_id": "A", "pool": "Boulder_Men", "event_date": "2024-01-01", "event_name": "CEC Nationals", "source_scope": "CEC", "event_tier": "National championship", "rating_context": "Senior / Open", "round_group": "Final"},
        {"global_id": "A", "pool": "Boulder_Men", "event_date": "2025-01-01", "event_name": "IFSC World Cup", "source_scope": "IFSC", "event_tier": "World series", "rating_context": "Senior / Open", "round_group": "Qualification"},
        {"global_id": "B", "pool": "Boulder_Men", "event_date": "2024-01-01", "event_name": "CEC Nationals", "source_scope": "CEC", "event_tier": "National championship", "rating_context": "Senior / Open", "round_group": "Qualification"},
        {"global_id": "C", "pool": "Lead_Men", "event_date": "2024-01-01", "event_name": "CEC Nationals", "source_scope": "CEC", "event_tier": "National championship", "rating_context": "Senior / Open", "round_group": "Qualification"},
    ])
    actual = build_support(rows).set_index("pathway_target_domain")
    assert actual.loc["fed_cec", "direct_event_contexts"] == 1
    assert actual.loc["fed_cec", "direct_athletes"] == 2
    assert actual.loc["fed_cec", "athletes_also_observed_in_wc"] == 1
    assert actual.loc["fed_cec", "wc_bridge_fraction"] == 0.5
    assert "binary" in actual.loc["fed_cec", "support_interpretation"]
