from __future__ import annotations

from pathlib import Path

import pandas as pd

from future_vision_demo import (
    PATHWAY_LEVELS,
    PERSONAS,
    SYNTHETIC_MARK,
    development_references,
    select_default_model,
    synthetic_pathway,
)


def _percent(value: str) -> int:
    return int(value.rstrip("%"))


def test_personas_cover_complete_synthetic_pathways() -> None:
    assert PATHWAY_LEVELS == (
        "Y-NAT", "Y-REG", "YW-IFSC", "NAT", "REG", "WC+",
    )
    assert len(PERSONAS) >= 4
    assert "SYNTHETIC" in SYNTHETIC_MARK
    for persona in PERSONAS:
        pathway = synthetic_pathway(persona)
        assert tuple(pathway["Level"]) == PATHWAY_LEVELS
        assert pathway["Evidence route"].isin(("Direct", "Connected graph scenario")).all()
        assert all(
            _percent(row["Illustrative readiness F"])
            <= _percent(row["Illustrative readiness SF"])
            for _, row in pathway.iterrows()
        )


def test_future_vision_uses_governed_pathways_and_conditional_olympics() -> None:
    source = Path("future_vision_demo.py").read_text(encoding="utf-8")
    assert "Y-NAT → Y-REG → YW-IFSC" in source
    assert "NAT → REG → WC+" in source
    assert "OLY is not treated as a permanent rating rung" in source
    assert '"YW"' not in source
    assert '"WC"' not in source
    assert '"OLY"' not in source


def test_default_model_is_best_eligible_not_oldest() -> None:
    candidates = pd.DataFrame(
        (
            {"name": "Old", "eligible": True, "locked_loss": 0.50, "created_order": 1},
            {"name": "Best", "eligible": True, "locked_loss": 0.40, "created_order": 3},
            {"name": "Research", "eligible": False, "locked_loss": 0.30, "created_order": 2},
        )
    )
    assert select_default_model(candidates) == "Best"


def test_development_reference_bands_are_ordered() -> None:
    reference = development_references()
    assert (reference["peer_low"] < reference["peer_median"]).all()
    assert (reference["peer_median"] < reference["peer_high"]).all()
    assert (reference["elite_low"] < reference["elite_median"]).all()
    assert (reference["elite_median"] < reference["elite_high"]).all()
    assert (reference["elite_median"] > reference["peer_median"]).all()
