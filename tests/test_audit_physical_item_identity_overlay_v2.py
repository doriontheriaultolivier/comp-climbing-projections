from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.audit_physical_item_identity_overlay_v2 import load_governed_overlay
from scripts.boulder_terrain_problem_adapter import IdentityLookups, StableSnapshot


def _base() -> IdentityLookups:
    snapshot = StableSnapshot("IFSC:1", 1500.0, 100.0, "onsight", "ok", "exact")
    return IdentityLookups(
        exact_snapshots={("IFSC", "1", "/round/1", "1"): snapshot},
        source_node_ids={("IFSC", "1"): "IFSC:1", ("CEC", "9"): "CEC:9"},
        ambiguous_source_nodes=frozenset({("USAC", "3")}),
    )


def _links(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "source_scope": "CEC",
                "athlete_source_id": "9",
                "global_id": "IFSC:1",
                "bridge_status": "governed_research_join_allowed",
                "model_input_authorized": False,
                "canonical_identity_mutated": False,
            },
            {
                "source_scope": "USAC",
                "athlete_source_id": "3",
                "global_id": "IFSC:1",
                "bridge_status": "governed_research_join_allowed",
                "model_input_authorized": False,
                "canonical_identity_mutated": False,
            },
        ]
    ).to_csv(path, index=False)


def test_overlay_changes_nodes_but_never_exact_snapshots(tmp_path: Path) -> None:
    path = tmp_path / "links.csv"
    _links(path)
    result, targets, counts = load_governed_overlay(path, _base())
    assert result.exact_snapshots == _base().exact_snapshots
    assert result.source_node_ids[("CEC", "9")] == "IFSC:1"
    assert ("USAC", "3") not in result.ambiguous_source_nodes
    assert targets == {"IFSC:1"}
    assert counts["governed_overrides_of_old_node_assignment"] == 1


def test_overlay_rejects_model_authority(tmp_path: Path) -> None:
    path = tmp_path / "links.csv"
    _links(path)
    frame = pd.read_csv(path)
    frame.loc[0, "model_input_authorized"] = True
    frame.to_csv(path, index=False)
    try:
        load_governed_overlay(path, _base())
    except ValueError as exc:
        assert "cannot authorize model input" in str(exc)
    else:
        raise AssertionError("model-authorizing identity link was accepted")
