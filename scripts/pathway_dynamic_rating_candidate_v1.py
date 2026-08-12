"""Research-only pathway-context adapter for the dynamic Boulder model.

This module does not fit or promote a model. It converts the audited direct
context labels into a predeclared DynamicBoulderRating candidate in which every
valid competition updates shared skill and only governed direct contexts update
their own shrunk target offset.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from scripts.audit_pathway_context_taxonomy_v1 import classify_direct_context
from scripts.dynamic_boulder_rating import DynamicRatingConfig


# The existing audited model core deliberately requires ``other`` as its
# reference coordinate. Keep that stable contract; the taxonomy's
# SHARED_BRIDGE_ONLY label maps to it rather than renaming the model core.
REFERENCE_DOMAIN = "other"
DIRECT_CONTEXT_DOMAINS = (
    "fed_cec",
    "fed_cec_youth",
    "fed_usac",
    "fed_usac_youth",
    "fed_fasi",
    "fed_fasi_youth",
    "fed_sac_cas",
    "fed_sac_cas_youth",
    "fed_fedme",
    "fed_fedme_youth",
    "interfed_north_america",
    "interfed_north_america_youth",
    "cont_africa",
    "cont_africa_youth",
    "cont_asia",
    "cont_asia_youth",
    "cont_europe",
    "cont_europe_youth",
    "cont_oceania",
    "cont_oceania_youth",
    "cont_pan_america",
    "cont_pan_america_youth",
    "ifsc_world_youth",
    "wc",
)
TARGET_DOMAINS = (REFERENCE_DOMAIN, *DIRECT_CONTEXT_DOMAINS)

# Statistical shrinkage parents for the next challenger. These never exclude
# results or impose a hand-written result weight; sparse children borrow more
# from their parent and the amount must be selected chronologically.
DOMAIN_PARENT = {
    "fed_cec": "cont_pan_america",
    "fed_cec_youth": "cont_pan_america_youth",
    "fed_usac": "cont_pan_america",
    "fed_usac_youth": "cont_pan_america_youth",
    "interfed_north_america": "cont_pan_america",
    "interfed_north_america_youth": "cont_pan_america_youth",
    "fed_fasi": "cont_europe",
    "fed_fasi_youth": "cont_europe_youth",
    "fed_sac_cas": "cont_europe",
    "fed_sac_cas_youth": "cont_europe_youth",
    "fed_fedme": "cont_europe",
    "fed_fedme_youth": "cont_europe_youth",
    "cont_africa": REFERENCE_DOMAIN,
    "cont_africa_youth": "ifsc_world_youth",
    "cont_asia": REFERENCE_DOMAIN,
    "cont_asia_youth": "ifsc_world_youth",
    "cont_europe": REFERENCE_DOMAIN,
    "cont_europe_youth": "ifsc_world_youth",
    "cont_oceania": REFERENCE_DOMAIN,
    "cont_oceania_youth": "ifsc_world_youth",
    "cont_pan_america": REFERENCE_DOMAIN,
    "cont_pan_america_youth": "ifsc_world_youth",
    "ifsc_world_youth": REFERENCE_DOMAIN,
    "wc": REFERENCE_DOMAIN,
}


class PathwayCandidateError(ValueError):
    pass


def model_domain(direct_context_head: str) -> str:
    """Translate a frozen taxonomy label to one target-offset domain.

    Olympic/OQS and unresolved rows retain their shared-skill contribution but
    do not update a standalone target offset. Quarantined test fixtures must be
    removed before model input and therefore fail closed here.
    """

    label = str(direct_context_head).strip()
    if label == "QUARANTINE":
        raise PathwayCandidateError("quarantined event cannot enter model input")
    if label in {
        "SHARED_BRIDGE_ONLY",
        "OLYM_SCENARIO_INPUT",
        "CONT:UNRESOLVED",
    }:
        return REFERENCE_DOMAIN
    translated = label.casefold().replace(":", "_").replace("-", "_")
    if translated not in DIRECT_CONTEXT_DOMAINS:
        raise PathwayCandidateError(f"unknown direct context head: {label}")
    return translated


def pathway_candidate_config(
    base: DynamicRatingConfig | None = None,
) -> DynamicRatingConfig:
    """Return the exact context-head candidate over the existing model core."""

    source = base or DynamicRatingConfig()
    return replace(
        source,
        target_domains=TARGET_DOMAINS,
        reference_domain=REFERENCE_DOMAIN,
        enable_target_offsets=True,
    )


def attach_pathway_model_domains(rows: pd.DataFrame) -> pd.DataFrame:
    """Attach audited direct and model-domain labels without consulting outcomes.

    Quarantined fixtures remain in the returned audit frame with
    ``pathway_input_eligible=False``. Callers must filter on that explicit flag;
    no row is silently discarded.
    """

    required = {
        "source_scope",
        "event_tier",
        "event_name",
        "rating_context",
    }
    missing = required - set(rows.columns)
    if missing:
        raise PathwayCandidateError(f"rows missing taxonomy columns: {sorted(missing)}")
    output = rows.copy()
    classified = [classify_direct_context(row) for _, row in output.iterrows()]
    output["direct_context_head"] = [label for label, _ in classified]
    output["direct_context_reason"] = [reason for _, reason in classified]
    output["pathway_input_eligible"] = output["direct_context_head"].ne("QUARANTINE")
    output["pathway_target_domain"] = [
        model_domain(label) if label != "QUARANTINE" else None
        for label in output["direct_context_head"]
    ]
    return output


def pathway_ablation_configs(
    base: DynamicRatingConfig | None = None,
) -> dict[str, DynamicRatingConfig]:
    """Predeclare the minimum chronological comparison panel.

    The existing model core has independent Gaussian context offsets rather
    than a full hierarchical covariance. Consequently this is only the
    fixed-offset challenger. A later partially pooled hierarchy must be a
    separate candidate and cannot be claimed by this adapter.
    """

    source = base or DynamicRatingConfig()
    return {
        "fully_pooled_no_context_offsets": replace(
            source,
            target_domains=(REFERENCE_DOMAIN,),
            reference_domain=REFERENCE_DOMAIN,
            enable_target_offsets=False,
        ),
        "fixed_independent_shrunk_context_offsets": pathway_candidate_config(source),
    }


def continuous_hierarchical_parent_weight(
    direct_event_contexts: int,
    wc_bridge_fraction: float,
    *,
    event_half_saturation: float,
    bridge_half_saturation: float,
) -> float:
    """Illustrate continuous parent borrowing without a binary support gate."""

    if isinstance(direct_event_contexts, bool) or direct_event_contexts < 0:
        raise PathwayCandidateError("direct event contexts must be nonnegative")
    bridge = float(wc_bridge_fraction)
    event_half = float(event_half_saturation)
    bridge_half = float(bridge_half_saturation)
    if not 0.0 <= bridge <= 1.0:
        raise PathwayCandidateError("bridge fraction must be in [0, 1]")
    if event_half <= 0.0 or bridge_half <= 0.0:
        raise PathwayCandidateError("half-saturation values must be positive")
    event_support = direct_event_contexts / (direct_event_contexts + event_half)
    bridge_support = bridge / (bridge + bridge_half)
    child_information = event_support * bridge_support
    return float(1.0 - child_information)


def pathway_hierarchy_audit(
    support: pd.DataFrame,
    *,
    event_half_saturation: float = 20.0,
    bridge_half_saturation: float = 0.10,
) -> pd.DataFrame:
    """Describe the hierarchy; do not fit or update any athlete rating."""

    required = {
        "pathway_target_domain", "direct_event_contexts", "wc_bridge_fraction"
    }
    missing = required - set(support.columns)
    if missing:
        raise PathwayCandidateError(
            f"support table missing hierarchy columns: {sorted(missing)}"
        )
    rows: list[dict[str, object]] = []
    for record in support.to_dict("records"):
        child = str(record["pathway_target_domain"])
        if child == REFERENCE_DOMAIN:
            continue
        parent = DOMAIN_PARENT.get(child)
        if parent is None:
            raise PathwayCandidateError(f"hierarchy parent missing for {child}")
        rows.append({
            "pathway_target_domain": child,
            "parent_domain": parent,
            "direct_event_contexts": int(record["direct_event_contexts"]),
            "wc_bridge_fraction": float(record["wc_bridge_fraction"]),
            "illustrative_parent_weight": continuous_hierarchical_parent_weight(
                int(record["direct_event_contexts"]),
                float(record["wc_bridge_fraction"]),
                event_half_saturation=event_half_saturation,
                bridge_half_saturation=bridge_half_saturation,
            ),
            "selected_for_model": False,
        })
    return pd.DataFrame(rows).sort_values(
        ["illustrative_parent_weight", "pathway_target_domain"],
        ascending=[False, True],
    ).reset_index(drop=True)
