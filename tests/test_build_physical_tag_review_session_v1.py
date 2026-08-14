from __future__ import annotations

import pandas as pd
import pytest

from scripts.build_physical_tag_review_session_v1 import build_session


def _tasks(count: int = 35) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "boulder_uid": [f"round-x-b{index}" for index in range(1, count + 1)],
            "event_date": ["2026-01-01"] * count,
            "event_name": ["Governed event"] * count,
            "round_group": ["Qualification"] * count,
            "gender": ["Men"] * count,
            "boulder_number": list(range(1, count + 1)),
            "priority_rank": list(reversed(range(1, count + 1))),
            "coaching_unlock_rank": list(range(1, count + 1)),
            "coaching_unlock_source_items": [2] * count,
            "coaching_unlock_athletes": [8] * count,
            "coaching_unlock_observations": [145] * count,
            "coaching_unlock_physical_observations": [103] * count,
            "coaching_unlock_board_observations": [42] * count,
            "priority_top_given_zone_pairs": [4] * count,
            "priority_zone_pairs": [3] * count,
        }
    )


def test_session_is_deterministic_and_uses_continuous_coaching_order() -> None:
    result = build_session(_tasks().sample(frac=1, random_state=7), task_count=30)
    assert result["coaching_unlock_rank"].tolist() == list(range(1, 31))
    assert result["session_task_order"].tolist() == list(range(1, 31))
    assert result["review_wave"].value_counts().to_dict() == {
        "B_high_unlock_extension": 20,
        "A_same_tasks_independent_calibration": 10,
    }


def test_session_requires_two_independent_reviews_without_authorizing_model_use() -> None:
    result = build_session(_tasks(), task_count=30)
    assert result["requested_independent_reviewers"].eq(2).all()
    assert result["task_status"].eq("human_review_pending").all()
    assert not any("athlete_id" in column.lower() for column in result.columns)
    assert not any("athlete_name" in column.lower() for column in result.columns)
    assert not any("test_value" in column.lower() for column in result.columns)


def test_duplicate_governed_boulder_fails_closed() -> None:
    tasks = _tasks()
    tasks.loc[1, "boulder_uid"] = tasks.loc[0, "boulder_uid"]
    with pytest.raises(ValueError, match="boulder_uid"):
        build_session(tasks, task_count=30)
