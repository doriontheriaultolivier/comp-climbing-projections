from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_physical_support_challenger_contract_v1 import validate_contract


ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "data" / "physical_support_challenger_contract_v1.json"


def test_contract_is_preregistered_and_fail_closed() -> None:
    contract = validate_contract(CONTRACT)
    assert contract["selection"]["binary_sample_gate"] is None
    assert contract["activation_requirements"]["tag_consensus_receipt_required"]
    assert len(contract["evidence_bindings"]) == 5
    assert not contract["coaching_output_limits"]["training_prescription_authorized"]


def test_validator_rejects_interval_kilter_semantics(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["candidate_families"][0]["grade_semantics"] = "linear_grade_numbers"
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    try:
        validate_contract(path)
    except ValueError as exc:
        assert "Kilter" in str(exc)
    else:
        raise AssertionError("interval-grade contract was accepted")
