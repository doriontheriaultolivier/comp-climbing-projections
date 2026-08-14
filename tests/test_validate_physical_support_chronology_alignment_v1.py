from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate_physical_support_chronology_alignment_v1 import validate_alignment


ROOT = Path(__file__).resolve().parents[1]
ALIGNMENT = ROOT / "data/physical_support_chronology_alignment_v1.json"


def test_alignment_binds_both_evidence_chains_and_stays_fail_closed() -> None:
    result = validate_alignment(ALIGNMENT)
    assert result["status"] == "PREREGISTERED_WAITING_FOR_HUMAN_TAG_EVIDENCE"
    assert result["next_executable_gate"]["fit_model_now"] is False
    assert result["adjudication"]["model_ladder_changed"] is False


def test_alignment_rejects_backdated_target_tags(tmp_path: Path) -> None:
    payload = json.loads(ALIGNMENT.read_text(encoding="utf-8"))
    payload["adjudication"]["target_event_tags_in_pre_event_projection"] = "allowed"
    candidate = tmp_path / "alignment.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="prohibited"):
        validate_alignment(candidate)


def test_alignment_rejects_premature_model_fit(tmp_path: Path) -> None:
    payload = json.loads(ALIGNMENT.read_text(encoding="utf-8"))
    payload["next_executable_gate"]["fit_model_now"] = True
    candidate = tmp_path / "alignment.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="model fit"):
        validate_alignment(candidate)
