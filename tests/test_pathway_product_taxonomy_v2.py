from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.pathway_product_taxonomy_v2 import (
    OPEN_LADDER,
    YOUTH_LADDER,
    ProductTaxonomyError,
    product_route,
)


ROOT = Path(__file__).resolve().parents[1]


def test_user_facing_ladders_include_nat_and_y_nat() -> None:
    assert YOUTH_LADDER == ("Y-NAT", "Y-REG", "YW-IFSC")
    assert OPEN_LADDER == ("NAT", "REG", "WC+")
    assert product_route("FED:CEC").product_head == "NAT"
    assert product_route("FED:CEC:YOUTH").product_head == "Y-NAT"


def test_nacs_and_pan_american_share_reg_but_keep_subtypes() -> None:
    nacs = product_route("INTERFED:NORTH_AMERICA")
    panams = product_route("CONT:PAN_AMERICA")
    assert nacs.product_head == panams.product_head == "REG"
    assert nacs.event_subtype != panams.event_subtype
    assert nacs.readiness_label == "REG Ready"


def test_youth_and_open_direct_evidence_do_not_cross_fill() -> None:
    assert product_route("CONT:EUROPE:YOUTH").product_head == "Y-REG"
    assert product_route("CONT:EUROPE").product_head == "REG"
    assert product_route("IFSC_WORLD_YOUTH").product_head == "YW-IFSC"
    assert product_route("WC").product_head == "WC+"


def test_olympics_is_a_scenario_and_quarantine_fails_closed() -> None:
    route = product_route("OLYM_SCENARIO_INPUT")
    assert route.product_head == "OLY"
    assert route.scenario_only
    assert route.readiness_label is None
    with pytest.raises(ProductTaxonomyError, match="no product route"):
        product_route("QUARANTINE")


def test_every_current_governed_context_is_mapped() -> None:
    taxonomy = pd.read_csv(ROOT / "data/pathway_context_event_taxonomy_v1.csv")
    labels = set(taxonomy["direct_context_head"].dropna().astype(str))
    for label in labels - {"QUARANTINE"}:
        assert product_route(label).product_head


def test_machine_contract_remains_research_only() -> None:
    contract = json.loads(
        (ROOT / "data/pathway_product_taxonomy_v2.json").read_text(encoding="utf-8")
    )
    assert contract["ladders"]["youth"][0] == "Y-NAT"
    assert contract["ladders"]["open"][0] == "NAT"
    assert contract["regional_policy"]["nacs_and_pan_american_share_broad_head"]
    assert contract["evidence_policy"]["current_numeric_release_authorized"] is False
    assert contract["display_policy"]["two_anchor_scale_authorized_now"] is False
