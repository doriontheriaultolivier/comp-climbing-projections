import pytest

from scripts.physical_transfer_chronology_contract_v1 import (
    ordered_outcome_code,
    validate_transfer_row,
)


BASE = {
    "analysis_mode": "pre_event_projection",
    "event_date": "2026-06-15",
    "board_flash_grade": 10,
    "board_flash_observed_at": "2026-05-01",
    "board_recent_grade": 12,
    "board_recent_observed_at": "2026-05-20",
    "physical_evidence_observed_at": "2026-04-01",
    "target_problem_tag_used": False,
    "outcome": "top",
}


def test_ordered_outcomes_preserve_none_zone_top_order() -> None:
    assert [ordered_outcome_code(value) for value in ("no_score", "zone", "top")] == [0, 1, 2]


def test_pre_event_row_accepts_only_prior_athlete_evidence() -> None:
    result = validate_transfer_row(BASE)
    assert result.board_flash_grade == 10
    assert result.problem_tag_temporal_role == "not_used_for_target_projection"
    with pytest.raises(ValueError, match="not available before the event"):
        validate_transfer_row({**BASE, "board_recent_observed_at": "2026-06-15"})


def test_pre_event_row_rejects_target_problem_tags() -> None:
    with pytest.raises(ValueError, match="cannot enter a pre-event projection"):
        validate_transfer_row(
            {**BASE, "target_problem_tag_used": True, "problem_tag_observed_at": "2026-06-16"}
        )


def test_post_event_row_accepts_retrospective_tag_without_backdating_it() -> None:
    result = validate_transfer_row(
        {
            **BASE,
            "analysis_mode": "post_event_coaching",
            "target_problem_tag_used": True,
            "problem_tag_observed_at": "2026-06-20",
        }
    )
    assert result.problem_tag_temporal_role == "retrospective_post_event_description"
    with pytest.raises(ValueError, match="predates the competition"):
        validate_transfer_row(
            {
                **BASE,
                "analysis_mode": "post_event_coaching",
                "target_problem_tag_used": True,
                "problem_tag_observed_at": "2026-06-14",
            }
        )


def test_missing_evidence_stays_missing_and_decimal_grade_is_rejected() -> None:
    result = validate_transfer_row(
        {
            **BASE,
            "board_flash_grade": None,
            "board_flash_observed_at": None,
            "physical_evidence_observed_at": None,
        }
    )
    assert result.board_flash_grade is None
    with pytest.raises(ValueError, match="not an average"):
        validate_transfer_row({**BASE, "board_flash_grade": 10.5})


def test_attempt_counter_requires_certified_semantics() -> None:
    with pytest.raises(ValueError, match="certified source semantics"):
        validate_transfer_row({**BASE, "attempt_value": 4})
    result = validate_transfer_row(
        {**BASE, "attempt_value": 4, "attempt_semantics_certified": True}
    )
    assert result.attempt_value == 4
