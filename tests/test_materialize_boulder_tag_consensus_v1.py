from __future__ import annotations

import pandas as pd

from scripts.materialize_boulder_tag_consensus_v1 import load_records, materialize_consensus


def _record(reviewer: str, when: str, pre: int, post: int) -> dict[str, object]:
    return {
        "schema_version": "4.0",
        "submitted_at_utc": when,
        "boulder_uid": "round-0123456789abcdefabcd-b1",
        "problem_id": "problem-1",
        "contributor": reviewer,
        "confidence": "High",
        "pre_zone_physical_0_3": pre,
        "post_zone_physical_0_3": post,
    }


def _priority() -> pd.DataFrame:
    return pd.DataFrame([{
        "problem_id": "problem-1", "priority_rank": 1, "source_scope": "IFSC",
        "source_event_id": "1", "source_round_id": "11", "boulder_number": "1",
    }])


def _inventory() -> pd.DataFrame:
    return pd.DataFrame([{
        "boulder_uid": "round-0123456789abcdefabcd-b1", "source_scope": "IFSC",
        "source_event_id": "1", "source_round_ids": "11", "boulder_number": "1",
    }])


def test_latest_per_reviewer_and_segments_are_preserved() -> None:
    records = [
        _record("reviewer-a", "2026-01-01T00:00:00Z", 1, 2),
        _record("reviewer-a", "2026-01-02T00:00:00Z", 2, 3),
        _record("reviewer-b", "2026-01-03T00:00:00Z", 3, 3),
    ]
    summary, latest, report = materialize_consensus(records, _priority(), _inventory())
    assert len(latest) == 4
    pre = summary.loc[summary["segment"].eq("pre_zone")].iloc[0]
    assert pre["independent_reviewers"] == 2
    assert pre["mean_0_3"] == 2.5
    assert pre["reviewer_range_0_3"] == 1.0
    assert report["coverage"]["boulders_with_two_or_more_reviewers"] == 1


def test_one_review_is_retained_without_eligibility_cliff() -> None:
    summary, _, report = materialize_consensus(
        [_record("reviewer-a", "2026-01-01T00:00:00Z", 1, 2)], _priority(), _inventory()
    )
    assert len(summary) == 2
    assert summary["independent_reviewers"].eq(1).all()
    assert not summary["eligibility_threshold_applied"].any()
    assert not summary["model_input_authorized"].any()
    assert report["semantics"]["minimum_review_cliff"] is None


def test_empty_history_still_proves_inventory_mapping() -> None:
    summary, latest, report = materialize_consensus([], _priority(), _inventory())
    assert summary.empty and latest.empty
    assert report["status"] == "READY_NO_REVIEWS"
    assert report["coverage"]["priority_source_items"] == 1
    assert report["coverage"]["priority_items_resolved_to_boulder_uid"] == 1
    assert report["semantics"]["single_review_discarded"] is False


def test_json_download_accepts_utf8_bom(tmp_path) -> None:
    path = tmp_path / "records.json"
    path.write_text("[]", encoding="utf-8-sig")
    assert load_records(path) == []


def test_one_governed_boulder_fans_out_to_exact_source_items() -> None:
    priority = pd.concat(
        [
            _priority(),
            _priority().assign(problem_id="problem-2", source_round_id="12"),
        ],
        ignore_index=True,
    )
    inventory = _inventory().assign(source_round_ids="11|12")
    summary, _, report = materialize_consensus(
        [_record("reviewer-a", "2026-01-01T00:00:00Z", 1, 2)],
        priority,
        inventory,
    )
    assert set(summary["problem_id"]) == {"problem-1", "problem-2"}
    assert report["coverage"]["priority_items_resolved_to_boulder_uid"] == 2
    assert report["coverage"]["distinct_governed_boulder_tasks"] == 1


def test_bad_reviewer_or_tag_value_fails() -> None:
    bad = _record("email@example.com", "2026-01-01T00:00:00Z", 1, 2)
    try:
        materialize_consensus([bad], _priority(), _inventory())
    except ValueError as exc:
        assert "reviewer" in str(exc)
    else:
        raise AssertionError("identifying reviewer code was accepted")
    bad = _record("reviewer-a", "2026-01-01T00:00:00Z", 4, 2)
    try:
        materialize_consensus([bad], _priority(), _inventory())
    except ValueError as exc:
        assert "0-3" in str(exc)
    else:
        raise AssertionError("out-of-range tag was accepted")
