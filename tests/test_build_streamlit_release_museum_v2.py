from pathlib import Path

from PIL import Image
import pytest

from scripts.build_streamlit_release_museum_v2 import (
    required_state_ids_v2,
    validate_full_page_captures,
)


def _png(path: Path, *, color: tuple[int, int, int], height: int = 1_200) -> None:
    Image.new("RGB", (1_100, height), color).save(path)


def test_v2_contract_covers_every_primary_state() -> None:
    states = required_state_ids_v2()
    assert len(states) == 14
    assert "vision--model-choice-expanded" in states
    assert "vision--alex-latebloomer--development" in states
    assert "evidence--towards-olympics" in states
    assert len(states) == len(set(states))


def test_v2_rejects_viewport_only_capture(tmp_path: Path) -> None:
    for index, state in enumerate(required_state_ids_v2()):
        _png(tmp_path / f"{state}.png", color=(index, 0, 0))
    first = tmp_path / f"{required_state_ids_v2()[0]}.png"
    _png(first, color=(255, 0, 0), height=720)
    with pytest.raises(ValueError, match="not full-page"):
        validate_full_page_captures(tmp_path)


def test_v2_rejects_duplicate_state_images(tmp_path: Path) -> None:
    for index, state in enumerate(required_state_ids_v2()):
        _png(tmp_path / f"{state}.png", color=(index, 0, 0))
    duplicate = (tmp_path / f"{required_state_ids_v2()[0]}.png").read_bytes()
    (tmp_path / f"{required_state_ids_v2()[1]}.png").write_bytes(duplicate)
    with pytest.raises(ValueError, match="Duplicate release captures"):
        validate_full_page_captures(tmp_path)
