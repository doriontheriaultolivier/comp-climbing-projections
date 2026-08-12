import pandas as pd

from scripts.audit_boulder_pool_normalization_v1 import audit


def test_explicit_discipline_is_authoritative_and_youth_is_separate():
    rows = pd.DataFrame([
        {"discipline": "Boulder", "pool": "Boulder_Men", "age_class": "Youth", "category": "Youth B Male", "source_scope": "IFSC", "event_date": "1999-01-01", "event_name": "Youth World"},
        {"discipline": "Boulder", "pool": "Boulder_Women", "age_class": "Senior / Open", "category": "Women", "source_scope": "IFSC", "event_date": "2025-01-01", "event_name": "World Cup"},
    ])
    summary, unresolved, receipt = audit(rows)
    assert unresolved.empty
    assert receipt["youth_boulder_rows"] == 1
    assert set(summary["pool"]) == {"Boulder_Men", "Boulder_Women"}


def test_unknown_pool_and_cross_discipline_contamination_are_not_guessed():
    rows = pd.DataFrame([
        {"discipline": "Boulder", "pool": "Male Youth B", "age_class": "Youth", "category": "Youth B Male", "source_scope": "OLD", "event_date": "1995-01-01", "event_name": "Old event"},
        {"discipline": "Lead", "pool": "Boulder_Men", "age_class": "Youth", "category": "Youth B Male", "source_scope": "OLD", "event_date": "1995-01-01", "event_name": "Old event"},
    ])
    _, unresolved, receipt = audit(rows)
    assert len(unresolved) == 2
    assert receipt["status"] == "UNRESOLVED_HOLD"
