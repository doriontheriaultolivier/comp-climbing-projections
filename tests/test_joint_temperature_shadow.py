from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from comp_climbing_app import load_joint_temperature_shadow


ROOT = Path(__file__).resolve().parents[1]


def render_shadow_fixture() -> None:
    from comp_climbing_app import render_joint_temperature_shadow

    render_joint_temperature_shadow()


def test_locked_shadow_loads_and_improves_both_metrics() -> None:
    value = load_joint_temperature_shadow()
    assert value is not None
    assert value["selected_temperature"] == 3.0
    assert [row["year"] for row in value["results"]] == [2025, 2026]
    for row in value["results"]:
        assert row["shadow_pair_log_loss"] < row["raw_pair_log_loss"]
        assert row["shadow_placement_rps"] < row["raw_placement_rps"]
        assert max(row["pair_delta_ci95"]) < 0
        assert max(row["placement_delta_ci95"]) < 0


def test_shadow_loader_fails_closed_on_coherent_claim_drift(tmp_path: Path) -> None:
    original = json.loads(
        (ROOT / "data" / "boulder_joint_temperature_shadow_v1.json").read_text(
            encoding="utf-8"
        )
    )
    original["results"][0]["shadow_pair_log_loss"] = 0.9
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(original), encoding="utf-8")
    assert load_joint_temperature_shadow(changed) is None


def test_shadow_render_is_visibly_research_only() -> None:
    app = AppTest.from_function(render_shadow_fixture).run(timeout=20)
    assert not app.exception
    text = "\n".join(item.value for item in [*app.subheader, *app.caption, *app.markdown])
    assert "Shadow probability calibration" in text
    assert "not yet applied to the current athlete cards" in text
    assert "frozen V4 family" in text
    assert "T=3.0" in text
