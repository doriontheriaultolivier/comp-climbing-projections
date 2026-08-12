from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from comp_climbing_app import (
    athlete_selection_id,
    plain_key,
    render_target_event_scenario,
    simulate_target_event_scenario,
)


def athlete_fixture() -> pd.DataFrame:
    names = ["Focus Athlete", "Close Opponent", "Strong Opponent", "Missing Specialist"]
    return pd.DataFrame(
        {
            "pool": ["Boulder_Women"] * 4,
            "global_id": ["TEST:1", "TEST:2", "TEST:3", "TEST:4"],
            "athlete_name": names,
            "name_key": [plain_key(name) for name in names],
            "country": ["CAN", "USA", "FRA", "GER"],
            "nationality": ["CAN", "USA", "FRA", "GER"],
            "Global-ELO": [1800.0, 1775.0, 1950.0, 1700.0],
            "Global-ELO-Semies": [1810.0, 1785.0, 1960.0, np.nan],
            "Global-ELO-Finals": [1820.0, 1790.0, 1970.0, np.nan],
            "Global-ELO-Qualies": [1790.0, 1765.0, 1940.0, 1710.0],
            "Global-ELO-Qualies-Flash": [1795.0, 1770.0, 1945.0, np.nan],
            "Global-ELO-Qualies-Onsight": [1785.0, 1760.0, 1935.0, np.nan],
            "Global-ELO-Flash": [1805.0, 1780.0, 1955.0, np.nan],
            "Global-ELO-Onsight": [1795.0, 1770.0, 1945.0, np.nan],
            "Global-ELO-Scramble": [1815.0, 1790.0, 1965.0, np.nan],
            "Global-ELO uncertainty": [90.0, 95.0, 80.0, 120.0],
        }
    )


def selection_ids() -> list[str]:
    return [athlete_selection_id("Boulder_Women", f"TEST:{index}") for index in range(1, 5)]


def scenario_fixture_app(athletes, selected, history) -> None:
    from comp_climbing_app import render_target_event_scenario

    render_target_event_scenario(athletes, selected, history)


def test_joint_scenario_is_deterministic_coherent_and_monotone() -> None:
    athletes = athlete_fixture()
    ids = selection_ids()[:3]
    first = simulate_target_event_scenario(
        athletes,
        ids,
        ids[0],
        rating_column="Global-ELO",
        draws=3000,
    )
    second = simulate_target_event_scenario(
        athletes,
        ids,
        ids[0],
        rating_column="Global-ELO",
        draws=3000,
    )
    np.testing.assert_array_equal(
        first["placement_probabilities"], second["placement_probabilities"]
    )
    placement = first["placement_probabilities"]
    np.testing.assert_allclose(placement.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    summary = first["summary"]
    assert summary["P(win)"].le(summary["P(top 3)"]).all()
    assert summary["P(top 3)"].le(summary["P(top 8)"]).all()
    opponents = first["opponents"]
    np.testing.assert_allclose(
        opponents["Focus beats opponent"] + opponents["Opponent beats focus"],
        1.0,
        rtol=0.0,
        atol=1e-12,
    )
    assert first["seed"] == second["seed"]


def test_missing_specialist_is_withheld_not_replaced_by_global_rating() -> None:
    athletes = athlete_fixture()
    ids = selection_ids()
    result = simulate_target_event_scenario(
        athletes,
        ids,
        ids[0],
        rating_column="Global-ELO-Semies",
        draws=1000,
    )
    assert result["field_size"] == 3
    assert result["excluded"]["_selection_id"].tolist() == [ids[3]]
    with pytest.raises(ValueError, match="focus athlete lacks"):
        simulate_target_event_scenario(
            athletes,
            ids,
            ids[3],
            rating_column="Global-ELO-Semies",
            draws=1000,
        )


def test_target_scenario_ui_states_conditionality_and_limits() -> None:
    athletes = athlete_fixture().iloc[:3].copy()
    ids = selection_ids()[:3]
    history = pd.DataFrame({"event_date": ["2026-07-25"]})
    app = AppTest.from_function(
        scenario_fixture_app,
        default_timeout=30,
        args=(athletes, ids, history),
    ).run()
    assert not app.exception
    assert not app.error
    assert "Target event scenario" in [item.value for item in app.header]
    captions = [str(item.value) for item in app.caption]
    assert any("conditional on the manually selected field" in item for item in captions)
    assert any("same joint ranking draws" in item for item in captions)
    warnings = [str(item.value) for item in app.warning]
    assert any("does not predict attendance" in item for item in warnings)
    metrics = {item.label: item.value for item in app.metric}
    assert "Focus Athlete · P(1st)" in metrics
    assert metrics["P(top 2)"] != "100.0%"
    assert metrics["Field entries"] == "3"
    assert "P(top 3)" not in metrics
    opponent_tables = [
        item.value for item in app.dataframe if "Focus beats opponent" in item.value.columns
    ]
    assert len(opponent_tables) == 1
    assert len(opponent_tables[0]) == 2
