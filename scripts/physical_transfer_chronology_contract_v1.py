"""Fail-closed chronology contract for future physical-transfer panels.

This module intentionally does not fit a model.  It makes the pre-event versus
post-event boundary executable before private athlete/problem rows are joined.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Mapping


ANALYSIS_MODES = {"pre_event_projection", "post_event_coaching"}
OUTCOME_CODES = {"no_score": 0, "zone": 1, "top": 2}


def _day(value: object, *, field: str, allow_missing: bool = False) -> date | None:
    if value is None or value == "":
        if allow_missing:
            return None
        raise ValueError(f"{field} is required")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date") from exc


def ordered_outcome_code(value: str) -> int:
    try:
        return OUTCOME_CODES[value]
    except KeyError as exc:
        raise ValueError(f"unknown ordered competition outcome: {value}") from exc


def validate_ordinal_grade(value: object, *, field: str) -> int | None:
    """Accept a missing value or an explicit grade category, never a decimal mean."""

    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an ordinal grade category")
    if isinstance(value, int):
        grade = value
    elif isinstance(value, str) and value.strip().lstrip("-").isdigit():
        grade = int(value)
    else:
        raise ValueError(f"{field} must be an ordinal grade category, not an average")
    if not 0 <= grade <= 20:
        raise ValueError(f"{field} is outside the governed category range")
    return grade


@dataclass(frozen=True)
class ValidatedTransferRow:
    analysis_mode: str
    outcome_code: int
    board_flash_grade: int | None
    board_recent_grade: int | None
    attempt_value: int | None
    problem_tag_temporal_role: str


def validate_transfer_row(row: Mapping[str, object]) -> ValidatedTransferRow:
    """Validate one future panel row without imputing or interpreting evidence."""

    mode = str(row.get("analysis_mode", ""))
    if mode not in ANALYSIS_MODES:
        raise ValueError(f"analysis_mode must be one of {sorted(ANALYSIS_MODES)}")
    event_day = _day(row.get("event_date"), field="event_date")

    for value_field, date_field in (
        ("board_flash_grade", "board_flash_observed_at"),
        ("board_recent_grade", "board_recent_observed_at"),
    ):
        value = validate_ordinal_grade(row.get(value_field), field=value_field)
        observed = _day(
            row.get(date_field), field=date_field, allow_missing=value is None
        )
        if value is None and observed is not None:
            raise ValueError(f"{date_field} cannot exist without {value_field}")
        if value is not None and observed is not None and observed >= event_day:
            raise ValueError(f"{value_field} was not available before the event")

    physical_observed = _day(
        row.get("physical_evidence_observed_at"),
        field="physical_evidence_observed_at",
        allow_missing=True,
    )
    if physical_observed is not None and physical_observed >= event_day:
        raise ValueError("physical evidence was not available before the event")

    tag_observed = _day(
        row.get("problem_tag_observed_at"),
        field="problem_tag_observed_at",
        allow_missing=True,
    )
    target_tag_used = bool(row.get("target_problem_tag_used", False))
    if mode == "pre_event_projection":
        if target_tag_used:
            raise ValueError("target problem tags cannot enter a pre-event projection")
        temporal_role = "not_used_for_target_projection"
    else:
        if not target_tag_used or tag_observed is None:
            raise ValueError("post-event coaching requires an observed target problem tag")
        if tag_observed < event_day:
            raise ValueError("target problem tag timestamp predates the competition")
        temporal_role = "retrospective_post_event_description"

    attempt_value_raw = row.get("attempt_value")
    attempt_value: int | None = None
    if attempt_value_raw not in (None, ""):
        if not bool(row.get("attempt_semantics_certified", False)):
            raise ValueError("attempt counters require certified source semantics")
        if isinstance(attempt_value_raw, bool):
            raise ValueError("attempt_value must be a non-negative integer")
        try:
            attempt_value = int(attempt_value_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("attempt_value must be a non-negative integer") from exc
        if attempt_value < 0 or str(attempt_value) != str(attempt_value_raw).strip():
            raise ValueError("attempt_value must be a non-negative integer")

    return ValidatedTransferRow(
        analysis_mode=mode,
        outcome_code=ordered_outcome_code(str(row.get("outcome", ""))),
        board_flash_grade=validate_ordinal_grade(
            row.get("board_flash_grade"), field="board_flash_grade"
        ),
        board_recent_grade=validate_ordinal_grade(
            row.get("board_recent_grade"), field="board_recent_grade"
        ),
        attempt_value=attempt_value,
        problem_tag_temporal_role=temporal_role,
    )

