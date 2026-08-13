from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from comp_climbing_app import load_current_wc_model_status


ROOT = Path(__file__).resolve().parents[1]


def render_status_fixture() -> None:
    from comp_climbing_app import render_current_wc_model_status

    render_current_wc_model_status()


def test_status_loads_verified_baseline_and_unpromoted_challenger() -> None:
    value = load_current_wc_model_status()
    assert value is not None
    assert value["status"] == "RESEARCH_CHALLENGER_NOT_PROMOTED"
    assert value["validation_guard"]["scope"] == "CEC_vs_CEC"
    assert value["validation_guard"]["probability_of_harm"] == 1.0
    assert value["withholding"]["current_zero_or_one_wc_start_central_values_published"] is False


def test_status_loader_rejects_promotion_or_hidden_harm(tmp_path: Path) -> None:
    original = json.loads(
        (ROOT / "data" / "current_wc_model_validation_status_v1.json").read_text(
            encoding="utf-8"
        )
    )
    original["validation_guard"]["challenger_pair_log_loss"] = 0.60
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(original), encoding="utf-8")
    assert load_current_wc_model_status(changed) is None


def test_status_render_states_gain_harm_and_withholding() -> None:
    app = AppTest.from_function(render_status_fixture).run(timeout=20)
    assert not app.exception
    text = "\n".join(
        item.value
        for item in [*app.subheader, *app.caption, *app.warning, *app.markdown]
    )
    assert "Current probability-model validation" in text
    assert "not promoted" in text.lower()
    assert "CEC-vs-CEC" in text
    assert "zero or one prior" in text
    assert "prospective CEC context" in text
