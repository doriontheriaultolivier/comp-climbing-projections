from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from comp_climbing_app import load_probability_spectrum_shadow


ROOT = Path(__file__).resolve().parents[1]


def render_spectrum_fixture() -> None:
    from comp_climbing_app import render_probability_spectrum_shadow

    render_probability_spectrum_shadow()


def test_locked_probability_spectrum_loads_and_exposes_known_limits() -> None:
    value = load_probability_spectrum_shadow()
    assert value is not None
    assert value["evaluation"]["competition_fields"] == 3749
    assert value["evaluation"]["canonical_pairs"] == 1164644
    assert value["cnr"]["available_for_event_date_calibration"] is False
    cnr_subset = value["cnr"]["canadian_strict_prior_subset"]
    assert cnr_subset["rankable_pairs"] == 73
    assert cnr_subset["physical_events"] == 17
    assert cnr_subset["fixed_adjustment_supported"] is False
    assert [
        row["log_loss_delta"] < 0
        for row in cnr_subset["moderate_offset_diagnostics"]
    ] == [False, True, False]
    high_low = next(
        row
        for row in value["rating_diagnostics"]
        if row["comparison"] == "High-tertile vs low-tertile"
    )
    assert high_low["forecast_mean"] < 0.52
    assert high_low["observed_rate"] > 0.83
    assert all(row["log_loss_delta"] < 0 for row in value["prospective_temperature"])


def test_probability_spectrum_loader_rejects_incoherent_gap(tmp_path: Path) -> None:
    original = json.loads(
        (
            ROOT / "data" / "boulder_probability_spectrum_shadow_v1.json"
        ).read_text(encoding="utf-8")
    )
    original["rating_diagnostics"][0]["observed_minus_forecast"] = 0.9
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(original), encoding="utf-8")
    assert load_probability_spectrum_shadow(changed) is None


def test_probability_spectrum_render_is_clear_and_research_only() -> None:
    app = AppTest.from_function(render_spectrum_fixture).run(timeout=20)
    assert not app.exception
    text = "\n".join(
        str(item.value)
        for item in [*app.subheader, *app.caption, *app.warning, *app.markdown]
    )
    assert "Does the probability scale reflect demonstrated level?" in text
    assert "too compressed toward 50%" in text
    assert "aggregate model checks, not athlete labels" in text
    assert "CNR" in text
    assert "CNR residual direction reverses" in text
    assert len(app.dataframe) == 4
