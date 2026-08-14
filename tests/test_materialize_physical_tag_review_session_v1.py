from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.materialize_physical_tag_review_session_v1 import (
    WAVE_A,
    audit_session_records,
    validate_session,
    verify_session_receipt,
)


ROOT = Path(__file__).parents[1]


def session() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "physical_tag_human_review_session_v1.csv")


def record(boulder_uid: str, reviewer: str, when: str, *, direction: str = "Up") -> dict[str, object]:
    return {
        "schema_version": "4.0",
        "submitted_at_utc": when,
        "boulder_uid": boulder_uid,
        "contributor": reviewer,
        "confidence": "High",
        "pre_zone_direction": direction,
        "post_zone_direction": "Diagonal",
        "pre_zone_physical_0_3": 2,
        "post_zone_physical_0_3": 3,
        "pre_zone_technical_0_3": 1,
        "post_zone_technical_0_3": 2,
        "pre_zone_coordination_0_3": 0,
        "post_zone_coordination_0_3": 1,
    }


def test_repository_session_and_receipt_are_exact() -> None:
    rows = validate_session(session())
    assert len(rows) == 30
    receipt = verify_session_receipt(
        ROOT / "data" / "physical_tag_human_review_session_v1.csv",
        ROOT / "data" / "physical_tag_human_review_session_v1.json",
    )
    assert receipt["operational_design"]["wave_a_tasks"] == 10


def test_empty_session_is_ready_without_fabricated_consensus() -> None:
    progress, directions, report = audit_session_records([], session())
    assert report["status"] == "READY_NO_REVIEWS"
    assert report["coverage"]["tasks_not_started"] == 30
    assert progress["independent_reviewers"].eq(0).all()
    assert directions.empty
    assert report["semantics"]["model_input_authorized"] is False


def test_latest_whole_record_and_direction_disagreement_are_preserved() -> None:
    uid = str(session().iloc[0]["boulder_uid"])
    records = [
        record(uid, "reviewer-a", "2026-08-14T00:00:00Z", direction="Sideways"),
        record(uid, "reviewer-a", "2026-08-14T01:00:00Z", direction="Up"),
        record(uid, "reviewer-b", "2026-08-14T02:00:00Z", direction="Diagonal"),
    ]
    progress, directions, report = audit_session_records(records, session())
    first = progress.iloc[0]
    assert first["independent_reviewers"] == 2
    assert first["review_status"] == "independently_double_reviewed"
    pre = directions.loc[
        directions["boulder_uid"].eq(uid) & directions["segment"].eq("pre_zone")
    ].iloc[0]
    assert pre["direction_counts_json"] == '{"Diagonal": 1, "Up": 1}'
    assert pre["modal_directions"] == "Diagonal|Up"
    assert not pre["unanimous"]
    assert report["coverage"]["latest_session_reviewer_boulder_records"] == 2


def test_wave_a_completion_is_operational_not_model_authority() -> None:
    rows = session().loc[session()["review_wave"].eq(WAVE_A)]
    records = []
    for index, uid in enumerate(rows["boulder_uid"].astype(str)):
        records.extend([
            record(uid, "reviewer-a", f"2026-08-14T01:{index:02d}:00Z"),
            record(uid, "reviewer-b", f"2026-08-14T02:{index:02d}:00Z"),
        ])
    _, _, report = audit_session_records(records, session())
    assert report["status"] == "RESEARCH_WAVE_A_COMPLETE_SESSION_IN_PROGRESS_NO_MODEL_INPUT"
    assert report["coverage"]["wave_a_tasks_double_reviewed"] == 10
    assert report["coverage"]["wave_b_tasks_double_reviewed"] == 0
    assert report["semantics"]["two_reviews_is_a_model_eligibility_cliff"] is False


def test_incomplete_latest_schema_record_fails_closed() -> None:
    uid = str(session().iloc[0]["boulder_uid"])
    bad = record(uid, "reviewer-a", "2026-08-14T00:00:00Z")
    del bad["post_zone_coordination_0_3"]
    with pytest.raises(ValueError, match="missing required fields"):
        audit_session_records([bad], session())
