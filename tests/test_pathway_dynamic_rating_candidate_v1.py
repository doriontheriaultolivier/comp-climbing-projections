import pytest

from scripts.dynamic_boulder_rating import DynamicBoulderRating
from scripts.pathway_dynamic_rating_candidate_v1 import (
    DIRECT_CONTEXT_DOMAINS,
    PathwayCandidateError,
    model_domain,
    pathway_ablation_configs,
    pathway_candidate_config,
)


def test_taxonomy_translation_keeps_scenarios_shared_and_youth_world_separate():
    assert model_domain("FED:CEC") == "fed_cec"
    assert model_domain("FED:CEC:YOUTH") == "fed_cec_youth"
    assert model_domain("IFSC_WORLD_YOUTH") == "ifsc_world_youth"
    assert model_domain("WC") == "wc"
    assert model_domain("OLYM_SCENARIO_INPUT") == "other"
    assert model_domain("IFSC_REG:UNRESOLVED") == "other"


def test_quarantine_and_unknown_labels_fail_closed():
    with pytest.raises(PathwayCandidateError):
        model_domain("QUARANTINE")
    with pytest.raises(PathwayCandidateError):
        model_domain("FED:UNKNOWN")


def test_candidate_uses_existing_shared_skill_plus_declared_offsets():
    config = pathway_candidate_config()
    assert config.reference_domain == "other"
    assert set(DIRECT_CONTEXT_DOMAINS).issubset(config.target_domains)
    assert "olym" not in " ".join(config.target_domains)
    model = DynamicBoulderRating(config)
    model.seed_established("athlete", config.prior_mean)
    projection = model.projection("athlete", 1.0, "wc")
    assert projection.target_domain == "wc"
    assert projection.skill_mean == config.prior_mean


def test_ablation_panel_does_not_overclaim_hierarchical_pooling():
    panel = pathway_ablation_configs()
    assert set(panel) == {
        "fully_pooled_no_context_offsets",
        "fixed_independent_shrunk_context_offsets",
    }
    assert not panel["fully_pooled_no_context_offsets"].enable_target_offsets
    assert panel["fixed_independent_shrunk_context_offsets"].enable_target_offsets
