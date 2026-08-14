"""Validate the additive chronology clarification for the physical-support contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.validate_physical_support_challenger_contract_v1 import validate_contract


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_alignment(path: Path) -> dict[str, object]:
    alignment = json.loads(path.read_text(encoding="utf-8-sig"))
    if alignment.get("schema") != "physical-support-chronology-alignment-v1":
        raise ValueError("unexpected alignment schema")
    if alignment.get("status") != "PREREGISTERED_WAITING_FOR_HUMAN_TAG_EVIDENCE":
        raise ValueError("alignment must remain waiting for reviewed tags")
    if alignment.get("promotion_authority") is not False:
        raise ValueError("alignment has no promotion authority")

    contract_path = ROOT / "data/physical_support_challenger_contract_v1.json"
    review_path = ROOT / "data/physical_transfer_ceiling_design_review_v1.json"
    chronology_path = ROOT / "scripts/physical_transfer_chronology_contract_v1.py"
    validate_contract(contract_path)
    expected = {
        "physical_support_challenger_contract_sha256": _sha256(contract_path),
        "physical_transfer_design_review_sha256": _sha256(review_path),
        "physical_transfer_chronology_source_sha256": _sha256(chronology_path),
    }
    bindings = alignment.get("bindings", {})
    for key, value in expected.items():
        if bindings.get(key) != value:
            raise ValueError(f"alignment binding mismatch: {key}")

    adjudication = alignment.get("adjudication", {})
    if adjudication.get("model_ladder_changed") is not False:
        raise ValueError("chronology clarification cannot silently change the model ladder")
    if adjudication.get("target_event_tags_in_pre_event_projection") != "prohibited":
        raise ValueError("target-event tags must remain prohibited before the event")
    if adjudication.get("board_expression_before_physical_capacity") is not True:
        raise ValueError("board-expression baseline must precede the physical challenger")
    if adjudication.get("ceiling_frontier_before_board_baseline_passes") is not False:
        raise ValueError("a ceiling frontier is not authorized")
    if adjudication.get("stress_or_pressure_inference") != "not_authorized":
        raise ValueError("stress inference is not identified")

    gate = alignment.get("next_executable_gate", {})
    if gate.get("fit_model_now") is not False:
        raise ValueError("current evidence does not authorize a model fit")
    if any(
        gate.get(key) is not True
        for key in (
            "requires_human_confirmed_tag_rows",
            "requires_exact_item_identity",
            "requires_tag_submission_time",
            "requires_independent_reviewer_evidence",
        )
    ):
        raise ValueError("reviewed tag evidence requirements cannot be weakened")
    return alignment


if __name__ == "__main__":
    validate_alignment(ROOT / "data/physical_support_chronology_alignment_v1.json")
    print("PASS")
