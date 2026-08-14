from pathlib import Path

import pytest

from scripts.build_streamlit_release_museum import required_state_ids, validate_captures


def test_release_museum_state_contract_is_complete() -> None:
    states = required_state_ids()
    assert len(states) == 13
    assert "vision--joe-bigbiceps--history" in states
    assert "vision--alex-latebloomer--development" in states
    assert "evidence--towards-olympics" in states
    assert len(states) == len(set(states))


def test_release_museum_rejects_partial_capture(tmp_path: Path) -> None:
    first = required_state_ids()[0]
    (tmp_path / f"{first}.png").write_bytes(b"not-a-real-image")
    with pytest.raises(ValueError, match="Missing required release captures"):
        validate_captures(tmp_path)
