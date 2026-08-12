"""Research-only pathway-context adapter for the dynamic Boulder model.

This module does not fit or promote a model. It converts the audited direct
context labels into a predeclared DynamicBoulderRating candidate in which every
valid competition updates shared skill and only governed direct contexts update
their own shrunk target offset.
"""

from __future__ import annotations

from dataclasses import replace

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
    "nacs",
    "nacs_youth",
    "ifsc_reg_africa",
    "ifsc_reg_asia",
    "ifsc_reg_asia_youth",
    "ifsc_reg_europe",
    "ifsc_reg_europe_youth",
    "ifsc_reg_oceania",
    "ifsc_reg_oceania_youth",
    "ifsc_reg_pan_america",
    "ifsc_world_youth",
    "wc",
)
TARGET_DOMAINS = (REFERENCE_DOMAIN, *DIRECT_CONTEXT_DOMAINS)


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
        "IFSC_REG:UNRESOLVED",
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
