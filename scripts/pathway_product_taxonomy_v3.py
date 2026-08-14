"""Map governed pathway contexts to the final human-facing product heads.

V3 changes the ambiguous senior/open label ``REG`` to ``REG-IFSC``. This is a
product-vocabulary change only; it does not alter the shared graph or evidence.
"""

from __future__ import annotations

from dataclasses import dataclass


YOUTH_LADDER = ("Y-NAT", "Y-REG", "YW-IFSC")
OPEN_LADDER = ("NAT", "REG-IFSC", "WC+")
SCENARIO_HEAD = "OLY"


class ProductTaxonomyError(ValueError):
    pass


@dataclass(frozen=True)
class ProductRoute:
    product_head: str
    demonstrated_directly: bool
    internal_context: str
    event_subtype: str
    scenario_only: bool = False

    @property
    def readiness_label(self) -> str | None:
        if self.product_head in {"Overall", SCENARIO_HEAD}:
            return None
        return f"{self.product_head} Ready"


def product_route(direct_context_head: str) -> ProductRoute:
    label = str(direct_context_head).strip()
    if label == "QUARANTINE":
        raise ProductTaxonomyError("quarantined contexts have no product route")
    if label in {"SHARED_BRIDGE_ONLY", "CONT:UNRESOLVED"}:
        return ProductRoute("Overall", False, label, "shared graph only")
    if label == "OLYM_SCENARIO_INPUT":
        return ProductRoute(
            SCENARIO_HEAD,
            False,
            label,
            "Olympic field and format input",
            scenario_only=True,
        )
    if label == "IFSC_WORLD_YOUTH":
        return ProductRoute("YW-IFSC", True, label, "Youth World Championships")
    if label == "WC":
        return ProductRoute("WC+", True, label, "senior World Cup / World Championship")
    if label.startswith("FED:"):
        youth = label.endswith(":YOUTH")
        federation = label.split(":")[1]
        return ProductRoute(
            "Y-NAT" if youth else "NAT",
            True,
            label,
            f"{federation} {'youth ' if youth else ''}domestic",
        )
    if label.startswith("INTERFED:"):
        youth = label.endswith(":YOUTH")
        return ProductRoute(
            "Y-REG" if youth else "REG-IFSC",
            True,
            label,
            "North America Series / cross-federation regional",
        )
    if label.startswith("CONT:"):
        youth = label.endswith(":YOUTH")
        continent = label.split(":")[1].replace("_", " ").title()
        return ProductRoute(
            "Y-REG" if youth else "REG-IFSC",
            True,
            label,
            f"{continent} continental series or championship",
        )
    raise ProductTaxonomyError(f"unknown governed context: {label}")
