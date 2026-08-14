from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.build_compact_boulder_identity_successor_v1 import adapt_athletes, adapt_history


def test_athlete_adapter_preserves_release_schema_and_withholds_amari_wc() -> None:
    candidate = pd.DataFrame(
        {
            "pool": ["Boulder_Women"],
            "global_id": ["IFSC:14843"],
            "WC+-ELO": [np.nan],
            "age": [17.045],
            "birthday": ["2009-07-09"],
            "birth_date_analysis_value": ["2009-07-19"],
            "birth_date_uncertainty_days": [10.0],
            "Canada projection — all evidence": [1900.0],
        }
    )
    release_columns = [
        "pool",
        "global_id",
        "WC+-ELO",
        "WC+-ELO-Qualies-Flash",
        "WC+-ELO-Qualies-Flash evidence",
        "age",
        "birth_date_uncertainty_days",
        "canada_projection_all_evidence",
        "age_lower_years",
        "age_upper_years",
        "age_precision_status",
    ]
    output = adapt_athletes(candidate, release_columns)
    assert list(output.columns) == release_columns
    assert pd.isna(output.loc[0, "WC+-ELO"])
    assert pd.isna(output.loc[0, "WC+-ELO-Qualies-Flash"])
    assert output.loc[0, "canada_projection_all_evidence"] == 1900.0
    assert output.loc[0, "age_precision_status"] == "source_interval_plus_public_tenth"
    assert "birthday" not in output


def test_athlete_adapter_rejects_numeric_amari_wc() -> None:
    candidate = pd.DataFrame(
        {
            "pool": ["Boulder_Women"],
            "global_id": ["IFSC:14843"],
            "WC+-ELO": [1943.0],
            "age": [17.0],
            "birth_date_uncertainty_days": [0.0],
        }
    )
    with pytest.raises(ValueError, match="numeric direct WC"):
        adapt_athletes(
            candidate,
            [
                "pool",
                "global_id",
                "WC+-ELO",
                "WC+-ELO-Qualies-Flash",
                "WC+-ELO-Qualies-Flash evidence",
                "age",
                "birth_date_uncertainty_days",
                "age_lower_years",
                "age_upper_years",
                "age_precision_status",
            ],
        )


def test_history_adapter_requires_reviewed_25_row_chain() -> None:
    source = pd.DataFrame(
        {
            "global_id": ["IFSC:14843"] * 25,
            "event_name": [f"event-{index}" for index in range(25)],
        }
    )
    output = adapt_history(source, ["global_id", "event_name"])
    assert len(output) == 25
    with pytest.raises(ValueError, match="25 reviewed rows"):
        adapt_history(source.iloc[:-1], ["global_id", "event_name"])


def test_history_adapter_requires_joint_fields_when_release_does() -> None:
    source = pd.DataFrame(
        {
            "global_id": ["IFSC:14843"] * 25,
            "event_name": [f"event-{index}" for index in range(25)],
        }
    )
    with pytest.raises(ValueError, match="joint_ranking_performance_elo"):
        adapt_history(
            source,
            ["global_id", "event_name", "joint_ranking_performance_elo"],
        )
