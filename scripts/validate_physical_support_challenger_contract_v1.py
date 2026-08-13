"""Validate the preregistered first physical-support challenger."""

from __future__ import annotations

import json
import re
from pathlib import Path


def validate_contract(path: Path) -> dict[str, object]:
    contract = json.loads(path.read_text(encoding="utf-8-sig"))
    if contract.get("schema") != "physical-support-challenger-contract-v1":
        raise ValueError("unexpected contract schema")
    if contract.get("status") != "PREREGISTERED_WAITING_FOR_HUMAN_TAG_EVIDENCE":
        raise ValueError("contract must remain waiting for human tag evidence")
    bindings = contract.get("evidence_bindings", {})
    if len(bindings) != 5 or any(
        re.fullmatch(r"[a-f0-9]{64}", str(value)) is None
        for value in bindings.values()
    ):
        raise ValueError("contract requires five exact SHA-256 evidence bindings")
    selection = contract.get("selection", {})
    if selection.get("outer_unit") != "whole_physical_competition":
        raise ValueError("selection must cluster by physical competition")
    if selection.get("binary_sample_gate") is not None:
        raise ValueError("binary sample gate is prohibited")
    candidates = contract.get("candidate_families", [])
    if len(candidates) != 2:
        raise ValueError("first challenger must remain a bounded two-family ladder")
    if candidates[0].get("grade_semantics") != "ordered_categories_not_interval_distances":
        raise ValueError("Kilter grades cannot be treated as interval distances")
    if "probabilistic_support_surface_not_absolute_ceiling" != candidates[1].get("shape"):
        raise ValueError("physical support cannot be labeled an absolute ceiling")
    deferred = set(contract.get("deferred_families", []))
    required_deferred = {
        "CES_substitution_surface",
        "soft_min_frontier",
        "attempt_hazard",
        "competition_context_or_stress_effect",
    }
    if not required_deferred.issubset(deferred):
        raise ValueError("complex sparse families must remain deferred")
    limits = contract.get("coaching_output_limits", {})
    if any(
        limits.get(key) is not False
        for key in (
            "training_prescription_authorized",
            "stress_or_anxiety_inference_authorized",
            "absolute_physical_ceiling_language_authorized",
        )
    ):
        raise ValueError("coaching authority is not established")
    activation = contract.get("activation_requirements", {})
    if activation.get("current_status") != "waiting_for_reviewed_tags":
        raise ValueError("activation status overclaims current evidence")
    return contract


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    validate_contract(ROOT / "data" / "physical_support_challenger_contract_v1.json")
    print("PASS")
