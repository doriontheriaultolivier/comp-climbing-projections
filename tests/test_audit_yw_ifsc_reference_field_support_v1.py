from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.audit_yw_ifsc_reference_field_support_v1 import (
    BASE_COMPONENTS,
    FULL_COMPONENTS,
    component_projection,
    integrated_probability,
    reference_fields,
    transition_summaries,
)


def test_component_sum_uses_covariance_and_yw_offset() -> None:
    labels = ["skill", "form", "offset:ifsc_youth_world"]
    means = [1900.0, 20.0, 40.0]
    covariance = [[100.0, -10.0, 5.0], [-10.0, 64.0, 2.0], [5.0, 2.0, 36.0]]
    base_mean, base_sd = component_projection(
        json.dumps(labels), json.dumps(means), json.dumps(covariance), BASE_COMPONENTS
    )
    full_mean, full_sd = component_projection(
        json.dumps(labels), json.dumps(means), json.dumps(covariance), FULL_COMPONENTS
    )
    assert base_mean == 1920.0
    assert np.isclose(base_sd**2, 144.0)
    assert full_mean == 1960.0
    assert np.isclose(full_sd**2, 194.0)


def test_probability_is_symmetric_and_uncertainty_attenuates() -> None:
    certain = integrated_probability(2100.0, 1900.0, 0.0, 0.0)
    uncertain = integrated_probability(2100.0, 1900.0, 300.0, 300.0)
    reverse = integrated_probability(1900.0, 2100.0, 300.0, 300.0)
    assert 0.5 < uncertain < certain < 1.0
    assert np.isclose(uncertain + reverse, 1.0)


def test_transitions_do_not_create_category_aliases() -> None:
    entry = pd.DataFrame(
        {
            "event_date": ["2024-08-01", "2025-08-01"],
            "pool": ["Boulder_Women", "Boulder_Women"],
            "category": ["Youth A Female", "U19 Women"],
            "global_id": ["IFSC:1", "IFSC:1"],
        }
    )
    detail, summary = transition_summaries(entry)
    assert len(detail) == 1
    assert detail.iloc[0]["interpretation"] == "observed athlete transition; not a rule alias"
    assert summary.iloc[0]["semantic_status"].startswith("empirical continuity only")


def test_reference_fields_keep_legacy_and_u_categories_distinct() -> None:
    entry = pd.DataFrame(
        {
            "event_date": ["2024-08-01", "2025-08-01"],
            "event_id": ["IFSC|1", "IFSC|2"],
            "pool": ["Boulder_Men", "Boulder_Men"],
            "category": ["Youth A Male", "U19 Men"],
            "global_id": ["IFSC:1", "IFSC:1"],
            "age_at_event": [17.0, 18.0],
        }
    )
    result = reference_fields(entry).set_index("origin_year")
    assert "not auto-mapped" in result.loc[2024, "status"]
    assert "literal reference" in result.loc[2025, "status"]
    assert not result["current_readiness_publication_allowed"].any()
