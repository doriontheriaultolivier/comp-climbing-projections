from __future__ import annotations

import json
from pathlib import Path

from scripts.pathway_product_taxonomy_v3 import (
    OPEN_LADDER,
    YOUTH_LADDER,
    product_route,
)


ROOT = Path(__file__).resolve().parents[1]


def test_final_ladders_use_reg_ifsc() -> None:
    assert YOUTH_LADDER == ("Y-NAT", "Y-REG", "YW-IFSC")
    assert OPEN_LADDER == ("NAT", "REG-IFSC", "WC+")


def test_nacs_and_panams_share_reg_ifsc_but_keep_subtypes() -> None:
    nacs = product_route("INTERFED:NORTH_AMERICA")
    panams = product_route("CONT:PAN_AMERICA")
    assert nacs.product_head == panams.product_head == "REG-IFSC"
    assert nacs.event_subtype != panams.event_subtype
    assert nacs.readiness_label == "REG-IFSC Ready"


def test_machine_contract_remains_non_promoting() -> None:
    contract = json.loads(
        (ROOT / "data/pathway_product_taxonomy_v3.json").read_text(encoding="utf-8")
    )
    assert contract["ladders"]["open"] == ["NAT", "REG-IFSC", "WC+"]
    assert contract["regional_policy"]["broad_head"] == "REG-IFSC"
    assert contract["evidence_policy"]["current_numeric_release_authorized"] is False
