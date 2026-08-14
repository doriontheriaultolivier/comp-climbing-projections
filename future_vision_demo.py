"""Synthetic, grant-facing product vision for Sport Performance Intelligence.

This module never reads the real athlete warehouse. Its fictional scenarios
exercise the intended product contract without making claims about real athletes
or current model validity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scripts.pathway_product_taxonomy_v3 import OPEN_LADDER, YOUTH_LADDER


SYNTHETIC_MARK = "SYNTHETIC SCENARIO - ILLUSTRATIVE OUTPUT ONLY"
PATHWAY_LEVELS = YOUTH_LADDER + OPEN_LADDER


@dataclass(frozen=True)
class SyntheticPersona:
    name: str
    birth_year: int
    archetype: str
    summary: str
    start_age: float
    current_age: float
    base: float
    annual_gain: float
    physical: int
    training_terrain: int
    competition_transfer: int
    pressure_access: int
    direct_levels: tuple[str, ...]

    @property
    def label(self) -> str:
        return f"{self.name} [SYN] {self.birth_year}"


PERSONAS = (
    SyntheticPersona(
        "Joe Bigbiceps", 2009, "Physical ceiling ahead of transfer",
        "Exceptional physical capacity; training-terrain and competition access lag behind it.",
        13.5, 17.3, 1260, 118, 94, 82, 61, 55,
        ("Y-NAT", "Y-REG", "YW-IFSC", "NAT"),
    ),
    SyntheticPersona(
        "Maya Transfergap", 2010, "Strong training output, uncertain competition transfer",
        "Training-board performance is progressing faster than results in unfamiliar competition settings.",
        12.8, 16.2, 1310, 104, 86, 90, 66, 62,
        ("Y-NAT", "Y-REG", "YW-IFSC"),
    ),
    SyntheticPersona(
        "Sam Pressureproof", 2008, "Competition accessibility above physical ceiling",
        "Competition outcomes repeatedly exceed what isolated physical benchmarks would suggest.",
        13.4, 18.1, 1290, 96, 73, 79, 88, 93,
        ("Y-NAT", "Y-REG", "YW-IFSC", "NAT", "REG-IFSC"),
    ),
    SyntheticPersona(
        "Alex Latebloomer", 2006, "Recent acceleration with limited historical support",
        "A sharp recent rise is promising, but the evidence window remains short.",
        15.2, 20.0, 1220, 145, 84, 81, 76, 74,
        ("Y-NAT", "NAT", "REG-IFSC"),
    ),
)


def persona_by_label(label: str) -> SyntheticPersona:
    return next(persona for persona in PERSONAS if persona.label == label)


def synthetic_history(persona: SyntheticPersona) -> pd.DataFrame:
    """Return deterministic scenario history, not an estimate of real dynamics."""
    ages = np.round(np.arange(persona.start_age, persona.current_age + 0.01, 0.25), 2)
    elapsed = ages - persona.start_age
    seasonal = 28 * np.sin(elapsed * 2.4 + len(persona.name) / 5)
    transfer_drag = (100 - persona.competition_transfer) * np.exp(-elapsed / 2.2)
    state = persona.base + persona.annual_gain * elapsed + seasonal - transfer_drag
    event_offset = 32 * np.sin(elapsed * 5.1 + 0.7)
    event_performance = state + event_offset
    round_cycle = np.array(("Qualification", "Qualification", "Semi-final", "Final"))
    return pd.DataFrame(
        {
            "age": ages,
            "historical_state": np.round(state, 1),
            "event_performance": np.round(event_performance, 1),
            "round": round_cycle[np.arange(len(ages)) % len(round_cycle)],
        }
    )


def synthetic_pathway(persona: SyntheticPersona) -> pd.DataFrame:
    """Build internally nested illustrative pathway outputs for one persona."""
    level_difficulty = {
        "Y-NAT": 1420, "Y-REG": 1580, "YW-IFSC": 1750,
        "NAT": 1640, "REG-IFSC": 1810, "WC+": 2040,
    }
    state = float(synthetic_history(persona).iloc[-1]["historical_state"])
    rows = []
    for index, level in enumerate(PATHWAY_LEVELS):
        gap = state - level_difficulty[level]
        transfer = persona.competition_transfer / 100
        if level == "WC+":
            transfer *= persona.pressure_access / 100
        semi = 1 / (1 + np.exp(-(gap * transfer) / 155))
        final = semi * (0.26 + 0.30 * transfer)
        evidence = 2 + ((len(persona.name) + index * 3) % 7)
        demonstrated = level in persona.direct_levels
        rows.append(
            {
                "Level": level,
                "Direct events": evidence if demonstrated else 0,
                "Demonstrated semifinal": f"{max(4, int(semi * 92))}%" if demonstrated else "-",
                "Demonstrated final": f"{max(1, int(final * 87))}%" if demonstrated else "-",
                "Illustrative readiness SF": f"{semi:.0%}",
                "Illustrative readiness F": f"{final:.0%}",
                "Evidence route": "Direct" if demonstrated else "Connected graph scenario",
            }
        )
    return pd.DataFrame(rows)


def development_references() -> pd.DataFrame:
    """Synthetic peer and future-elite survivor bands for the product mock-up."""
    ages = np.arange(13.0, 24.5, 0.5)
    peer_median = 1300 + 76 * (ages - 13) - 2.7 * (ages - 18) ** 2
    elite_median = peer_median + 245 + 14 * np.maximum(ages - 16, 0)
    return pd.DataFrame(
        {
            "age": ages,
            "peer_low": peer_median - 155,
            "peer_median": peer_median,
            "peer_high": peer_median + 155,
            "elite_low": elite_median - 110,
            "elite_median": elite_median,
            "elite_high": elite_median + 110,
        }
    )


def select_default_model(candidates: pd.DataFrame) -> str:
    """Choose the best eligible frozen model, independent of creation order."""
    eligible = candidates.loc[candidates["eligible"]].copy()
    if eligible.empty:
        raise ValueError("No eligible model is available")
    return str(eligible.sort_values(["locked_loss", "name"]).iloc[0]["name"])


def _mark(text: str) -> str:
    return f"{text} · {SYNTHETIC_MARK}"


def _render_hero() -> None:
    st.markdown(
        """
        <div class="vision-hero synthetic-surface">
          <div class="synthetic-badge">SYNTHETIC FUTURE VISION</div>
          <h2>From results to better development decisions</h2>
          <p>A working illustration of how Sport Performance Intelligence can connect
          development pathways, competition projections, physical testing and coaching questions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    items = (
        ("01", "See the pathway", "Y-NAT → Y-REG → YW-IFSC and NAT → REG-IFSC → WC+"),
        ("02", "Compare development", "Youth categories, senior open fields and future-elite references"),
        ("03", "Find the constraint", "Physical ceiling → training transfer → competition access"),
        ("04", "Test decisions", "Turn evidence gaps into hypotheses and coach questions"),
    )
    for column, (number, title, body) in zip(cols, items):
        with column:
            st.markdown(
                f"<div class='vision-card synthetic-surface'><b>{number} · {title}</b>"
                f"<p>{body}</p><small>{SYNTHETIC_MARK}</small></div>",
                unsafe_allow_html=True,
            )


def _render_history(persona: SyntheticPersona) -> None:
    history = synthetic_history(persona)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=history["age"], y=history["historical_state"], mode="lines",
            name="Illustrative historical state", line={"color": "#6C4AE0", "width": 4},
        )
    )
    symbols = {"Qualification": "circle", "Semi-final": "diamond", "Final": "star"}
    for round_name, rows in history.groupby("round"):
        figure.add_trace(
            go.Scatter(
                x=rows["age"], y=rows["event_performance"], mode="markers",
                name=round_name, marker={"symbol": symbols[round_name], "size": 9},
            )
        )
    figure.update_layout(
        title=_mark("Development history and event performances"),
        xaxis_title="Age", yaxis_title="Common illustrative performance coordinate",
        height=430, margin={"l": 35, "r": 20, "t": 70, "b": 40},
        legend={"orientation": "h", "y": -0.18},
    )
    figure.add_annotation(
        text=SYNTHETIC_MARK, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, opacity=0.13, font={"size": 24}, textangle=-18,
    )
    st.plotly_chart(figure, use_container_width=True, key=f"vision-history-{persona.name}")


def _render_development_chart(persona: SyntheticPersona) -> None:
    reference = development_references()
    history = synthetic_history(persona)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=reference["age"], y=reference["peer_high"], line={"width": 0}, showlegend=False))
    figure.add_trace(
        go.Scatter(
            x=reference["age"], y=reference["peer_low"], fill="tonexty",
            fillcolor="rgba(61,139,125,.18)", line={"width": 0}, name="All same-age peers · middle 50%",
        )
    )
    figure.add_trace(go.Scatter(x=reference["age"], y=reference["elite_high"], line={"width": 0}, showlegend=False))
    figure.add_trace(
        go.Scatter(
            x=reference["age"], y=reference["elite_low"], fill="tonexty",
            fillcolor="rgba(230,162,60,.24)", line={"width": 0},
            name="Later WORLD Top-40 survivors · middle 50%",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=history["age"], y=history["historical_state"], mode="lines+markers",
            line={"color": "#6C4AE0", "width": 4}, name=persona.name,
        )
    )
    figure.update_layout(
        title=_mark("Developmental references at the same age"), xaxis_title="Age",
        yaxis_title="Common illustrative performance coordinate", height=430,
        margin={"l": 35, "r": 20, "t": 70, "b": 40},
        legend={"orientation": "h", "y": -0.2},
    )
    figure.add_annotation(
        text=SYNTHETIC_MARK, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, opacity=0.13, font={"size": 24}, textangle=-18,
    )
    st.plotly_chart(figure, use_container_width=True, key=f"vision-development-{persona.name}")
    st.caption(
        "The survivor band describes athletes who eventually reached WORLD Top 40. "
        "It is not a required pathway and omits similar youth athletes who did not reach that level."
    )


def _render_transfer(persona: SyntheticPersona) -> None:
    st.subheader("Where might development be constrained?")
    stages = pd.DataFrame(
        {
            "Stage": ("Physical ceiling", "Training-terrain access", "Competition access", "Pressure access"),
            "Illustrative score": (
                persona.physical, persona.training_terrain,
                persona.competition_transfer, persona.pressure_access,
            ),
        }
    )
    figure = go.Figure(
        go.Bar(
            x=stages["Illustrative score"], y=stages["Stage"], orientation="h",
            marker={"color": ("#6C4AE0", "#4C83E0", "#31A58E", "#E6A23C")},
            text=[f"{value}/100" for value in stages["Illustrative score"]],
            textposition="inside",
        )
    )
    figure.update_layout(
        title=_mark("Physical → training terrain → competition"), xaxis={"range": [0, 100]},
        height=330, margin={"l": 20, "r": 20, "t": 70, "b": 30},
    )
    figure.add_annotation(
        text=SYNTHETIC_MARK, xref="paper", yref="paper", x=0.55, y=0.5,
        showarrow=False, opacity=0.14, font={"size": 20}, textangle=-16,
    )
    left, right = st.columns((1.35, 1))
    with left:
        st.plotly_chart(figure, use_container_width=True, key=f"vision-transfer-{persona.name}")
    with right:
        st.markdown("**Evidence**")
        st.write(persona.summary)
        st.markdown("**Discrepancy**")
        st.write(
            "The illustrative physical, training-terrain and competition layers do not "
            "provide the same estimate of accessible performance."
        )
        st.markdown("**Hypotheses to investigate**")
        st.write("Terrain transfer, competition format, pressure accessibility, or incomplete evidence.")
        st.markdown("**Questions for the coach**")
        st.write(
            "Does this pattern repeat across events? Which layer changed most recently? "
            "What observation would distinguish a physical ceiling from a transfer constraint?"
        )
        st.caption(SYNTHETIC_MARK)


def render_future_vision_demo() -> None:
    st.markdown(
        """
        <style>
        .vision-hero{padding:1.45rem 1.6rem;border-radius:22px;background:linear-gradient(120deg,#25184f,#6c4ae0 55%,#f08a4b);color:white;margin:.2rem 0 1rem}
        .vision-hero h2{color:white!important;font-size:2.05rem;margin:.3rem 0}
        .vision-hero p{max-width:850px;font-size:1.04rem;margin-bottom:.1rem}
        .synthetic-badge{font-size:.72rem;font-weight:800;letter-spacing:.12em}
        .vision-card{min-height:138px;border:1px solid #ded8f6;border-radius:16px;padding:1rem;background:#fbf9ff}
        .vision-card p{font-size:.88rem;margin:.55rem 0;color:#42514e}
        .vision-card small{font-size:.58rem;color:#7658c8;font-weight:700}
        .synthetic-surface{position:relative;overflow:hidden}
        .synthetic-surface:after{content:'SYNTHETIC';position:absolute;right:-18px;bottom:9px;transform:rotate(-20deg);font-size:1.2rem;font-weight:800;opacity:.08}
        </style>
        """,
        unsafe_allow_html=True,
    )
    _render_hero()
    st.markdown("### Explore a fictional athlete report")
    selected_label = st.radio(
        "Main fictional athlete", [persona.label for persona in PERSONAS],
        horizontal=True,
        key="future-vision-athlete",
    )
    persona = persona_by_label(selected_label)
    compare_options = [item.label for item in PERSONAS if item != persona]
    comparisons = st.multiselect(
        "Compare with", compare_options, max_selections=2,
        placeholder="Optional: add up to two fictional athletes",
        key="future-vision-comparisons",
    )
    st.markdown(
        f"<div class='vision-card synthetic-surface'><b>{persona.label}</b> · {persona.archetype}"
        f"<p>{persona.summary}</p><small>{SYNTHETIC_MARK}</small></div>",
        unsafe_allow_html=True,
    )
    if comparisons:
        st.caption("Comparison: " + " · ".join(comparisons) + " · " + SYNTHETIC_MARK)

    st.subheader("Development and competition pathways")
    youth, senior = st.columns(2)
    youth.info("**Youth pathway**  ·  Y-NAT  →  Y-REG  →  YW-IFSC")
    senior.info("**Open pathway**  ·  NAT  →  REG-IFSC  →  WC+")
    st.dataframe(
        synthetic_pathway(persona), use_container_width=True, hide_index=True,
        column_config={"Level": st.column_config.TextColumn(width="small")},
    )
    st.caption(
        "Demonstrated cells represent fictional direct events. Readiness cells are illustrative target behavior, "
        "not validation results or claims about a real model."
    )
    st.info(
        "**OLY scenario** · Olympic selection and performance would be projected as a "
        "conditional event scenario using the relevant format, field and qualification "
        "rules. OLY is not treated as a permanent rating rung."
    )

    history_tab, development_tab = st.tabs(("History and events", "Developmental references"))
    with history_tab:
        _render_history(persona)
    with development_tab:
        _render_development_chart(persona)
    _render_transfer(persona)

    with st.expander("How model choice would work", expanded=False):
        models = pd.DataFrame(
            (
                {"name": "Atlas", "eligible": True, "locked_loss": 0.42, "created_order": 2},
                {"name": "Bridge", "eligible": True, "locked_loss": 0.47, "created_order": 1},
                {"name": "Summit Research", "eligible": False, "locked_loss": 0.39, "created_order": 3},
            )
        )
        default = select_default_model(models)
        st.success(f"Default: {default} — best-performing eligible model on the frozen evaluation.")
        st.write(
            "Creation order never selects the default. Ineligible research models may be compared "
            "here, but cannot change the main athlete report."
        )
        st.caption(SYNTHETIC_MARK + " · model names and statuses are illustrative")

    st.markdown("### What the grant enables next")
    roadmap = st.columns(3)
    with roadmap[0]:
        st.markdown("**1 · Validate the shared scale**\n\nRepair youth/senior and Canadian-to-world transport.")
    with roadmap[1]:
        st.markdown("**2 · Unlock coaching outputs**\n\nTest readiness, physical ceilings and transfer hypotheses chronologically.")
    with roadmap[2]:
        st.markdown("**3 · Extend to another sport**\n\nReuse the evidence graph, validation gates and decision interface with a sport expert.")
    st.caption(SYNTHETIC_MARK)
