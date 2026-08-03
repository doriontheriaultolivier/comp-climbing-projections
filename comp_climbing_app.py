"""Boulder-first interface for Comp Climbing Projections.

The legacy product stays on its own release branch and URL.  This module loads
only the compact artifacts needed by the Overview so Streamlit Community Cloud
does not retain the full research warehouse in memory.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import base64
import hashlib
import io
import json
from pathlib import Path
import unicodedata
from urllib import error as urlerror
from urllib import request as urlrequest
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RATING_ORDER = ["Global-ELO", "IFSC-ELO", "WR-ELO"]
ROUND_OPTIONS = ["All rounds", "Qualification", "Semi-final", "Final"]
QUALIFICATION_FORMATS = ["Flash + Onsight", "Flash", "Onsight"]
ALL_RATINGS = [
    "Global-ELO", "Global-ELO-Qualies", "Global-ELO-Qualies-Flash",
    "Global-ELO-Qualies-Onsight", "Global-ELO-Semies", "Global-ELO-Finals",
    "Global-ELO-Onsight", "Global-ELO-Scramble", "Global-ELO-Flash",
    "WR-ELO", "WR-ELO-Qualies", "WR-ELO-Qualies-Flash",
    "WR-ELO-Qualies-Onsight", "WR-ELO-Semies", "WR-ELO-Finals",
    "IFSC-ELO", "IFSC-ELO-Qualies", "IFSC-ELO-Qualies-Flash",
    "IFSC-ELO-Qualies-Onsight", "IFSC-ELO-Semies", "IFSC-ELO-Finals",
]
DEFAULT_ATHLETES = ["Oscar Baudrand", "Matthew Rodriguez", "Colin Duffy"]
DISPLAY_OVERRIDES = {
    "baudrandoscar": "Oscar Baudrand",
    "matthewrodriguez": "Matthew Rodriguez",
    "colinduffy": "Colin Duffy",
    "madisonrichardson": "Madison Richardson",
    "babetteroy": "Babette Roy",
    "anniesanders": "Annie Sanders",
}
PALETTE = {
    "ink": "#102F2B",
    "teal": "#0B7A75",
    "coral": "#F26B5B",
    "gold": "#E6A23C",
    "blue": "#4285A9",
    "muted": "#71817E",
}
ATHLETE_COLORS = [
    "#0B7A75", "#F26B5B", "#4285A9", "#E6A23C", "#8E5AA7",
    "#2E8B57", "#D05A8A", "#6B7C93", "#A66A3F", "#4C9F9A",
]

STYLE_DEFINITIONS = {
    "physical": (
        "Force, power or body-tension demand. A move can be highly physical without "
        "being technical or coordinative."
    ),
    "technical": (
        "Precision, balance, body positioning or movement efficiency when linked timing "
        "and momentum are not the main problem."
    ),
    "coordination": (
        "Linked timing, momentum, redirection or multi-limb sequencing. A coordination "
        "move is not automatically high-physical or high-technical."
    ),
}

STYLE_TAG_GROUPS = {
    "Physical qualities": [
        ("explosiveness", "Explosiveness"),
        ("body_tension", "Body tension"),
        ("overall_strength", "Overall strength"),
    ],
    "Technical qualities": [
        ("slow_precision", "Slow precision"),
        ("curved_coordination", "Coordination curves"),
        ("reaction_time", "Reaction time"),
        ("proprioception", "Proprioception"),
    ],
    "Handholds": [
        ("hand_slopers", "Slopers"),
        ("hand_jugs", "Jugs"),
        ("hand_crimps", "Crimps"),
        ("hand_pinches", "Pinches"),
    ],
    "Footholds": [
        ("foot_small_incut", "Small incut feet"),
        ("foot_small_smeary", "Small smeary feet"),
        ("foot_volumes", "Volumes"),
        ("foot_juggy", "Juggy footholds"),
    ],
    "Move types": [
        ("move_blocked", "Blocked / constrained"),
        ("move_dyno", "Dyno"),
        ("move_run_jump", "Run-and-jump"),
        ("move_paddle", "Paddle"),
        ("move_deadpoint", "Deadpoint"),
        ("move_compression", "Compression"),
        ("move_press", "Press"),
        ("move_mantle", "Mantle"),
        ("move_toe_hook", "Toe hook"),
        ("move_heel_hook", "Heel hook"),
        ("move_drop_knee", "Drop-knee"),
        ("move_smear", "Smear"),
    ],
}


def transparent(color: str, alpha: float = 0.12) -> str:
    value = color.lstrip("#")
    red, green, blue = (int(value[index:index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def plain_key(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    plain = "".join(
        char.lower() if char.isalnum() else " "
        for char in text
        if not unicodedata.combining(char)
    )
    # IFSC commonly stores SURNAME Given while CEC/CNR stores Given SURNAME.
    # A token-sorted key reconciles that presentation difference; pool remains
    # part of every rating join so men/women and disciplines cannot cross-match.
    return "".join(sorted(plain.split()))


def friendly_name(value: object) -> str:
    text = str(value or "").strip()
    return DISPLAY_OVERRIDES.get(plain_key(text), text)


@st.cache_data(show_spinner=False, ttl=900, max_entries=2)
def read_data() -> dict[str, pd.DataFrame]:
    files = {
        "athletes": ("boulder_overview_athletes.parquet", "parquet"),
        "history": ("boulder_overview_history.parquet", "parquet"),
        "correlations": ("boulder_rating_correlations.csv", "csv"),
        "calibration": ("boulder_elo_calibration.csv", "csv"),
        "rosters": ("program_rosters.csv", "csv"),
        "physical_profiles": ("physical_test_profiles.csv", "csv"),
        "physical_associations": ("physical_test_associations.csv", "csv"),
        "physical_models": ("physical_model_summary.csv", "csv"),
        "physical_latest": ("physical_all_tests_latest.csv", "csv"),
        "physical_screen": ("physical_test_metric_screen.csv", "csv"),
        "physical_priorities": ("physical_athlete_priorities.csv", "csv"),
        "model_backtest": ("model_backtest_summary.csv", "csv"),
        "sensitivity_metrics": ("latent_volatility_challenger_metrics.csv", "csv"),
        "sensitivity_status": ("latent_volatility_challenger_status.csv", "csv"),
    }
    output: dict[str, pd.DataFrame] = {}
    for key, (filename, kind) in files.items():
        path = DATA / filename
        if not path.exists():
            output[key] = pd.DataFrame()
            continue
        output[key] = (
            pd.read_parquet(path)
            if kind == "parquet"
            else pd.read_csv(path, low_memory=False)
        )
    return output


def wide_athletes(ratings: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    if ratings.empty:
        return pd.DataFrame()
    index = ["pool", "global_id"]
    values = ratings.pivot_table(
        index=index,
        columns="rating_family",
        values="rating",
        aggfunc="first",
    ).reset_index()
    for family in ALL_RATINGS:
        if family not in values:
            values[family] = np.nan
    evidence = ratings.pivot_table(
        index=index,
        columns="rating_family",
        values="contests_seen",
        aggfunc="first",
    ).add_suffix(" evidence").reset_index()
    metadata_columns = [
        "pool", "global_id", "athlete_name", "nationality", "gender",
        "birthday", "birth_date_analysis_value", "birth_date_confidence",
        "birth_date_uncertainty_days", "days_since_last_result",
        "ifsc_athlete_id",
    ]
    available = [column for column in metadata_columns if column in ratings]
    metadata = (
        ratings.sort_values("last_result_date" if "last_result_date" in ratings else "rating_family")
        .drop_duplicates(index, keep="last")[available]
    )
    athletes = values.merge(evidence, on=index, how="left").merge(
        metadata, on=index, how="left"
    )
    birth = pd.to_datetime(
        athletes.get("birth_date_analysis_value", athletes.get("birthday")),
        errors="coerce",
    )
    as_of = (
        pd.to_datetime(history["event_date"], errors="coerce").max()
        if not history.empty else pd.Timestamp(date.today())
    )
    athletes["age"] = (as_of - birth).dt.days / 365.2425
    athletes["name_key"] = athletes["athlete_name"].map(plain_key)
    athletes["country"] = athletes.get("nationality", "").fillna("")

    athletes["momentum"] = np.nan
    if not history.empty:
        history = history.copy()
        history["event_date"] = pd.to_datetime(history["event_date"], errors="coerce")
        recent = history.loc[history["event_date"].ge(as_of - pd.Timedelta(days=365))]
        change = (
            recent.sort_values("event_date")
            .groupby(index)["rating_after"]
            .agg(["first", "last"])
        )
        change["momentum"] = change["last"] - change["first"]
        athletes = athletes.merge(
            change[["momentum"]].reset_index(), on=index, how="left",
            suffixes=("", "_recent"),
        )
        athletes["momentum"] = athletes["momentum_recent"].combine_first(
            athletes["momentum"]
        )
        athletes = athletes.drop(columns=["momentum_recent"])
    athletes["momentum"] = athletes["momentum"].fillna(0.0)
    athletes["gender"] = athletes.get("gender", athletes["pool"].str.rsplit("_", n=1).str[-1])
    return athletes


def attach_cnr(athletes: pd.DataFrame, cnr: pd.DataFrame) -> pd.DataFrame:
    if athletes.empty:
        return pd.DataFrame()
    if cnr.empty:
        athletes["cnr_rank"] = np.nan
        athletes["cnr_value"] = np.nan
        return athletes
    boulder = cnr.loc[cnr["discipline"].eq("Boulder")].copy()
    boulder["name_key"] = boulder["athlete_name"].map(plain_key)
    boulder = boulder.rename(columns={"athlete_name": "cnr_athlete_name"})
    columns = [
        "pool", "name_key", "cnr_athlete_name", "cnr_rank", "cnr_value",
        "cnr_as_of",
    ]
    boulder = boulder[columns].drop_duplicates(["pool", "name_key"])
    output = athletes.merge(boulder, on=["pool", "name_key"], how="outer")
    output["athlete_name"] = output["athlete_name"].combine_first(
        output["cnr_athlete_name"]
    )
    output["gender"] = output["gender"].combine_first(
        output["pool"].str.rsplit("_", n=1).str[-1]
    )
    output.loc[
        output["country"].isna() & output["cnr_rank"].notna(), "country"
    ] = "CAN"
    output["country"] = output["country"].fillna("")
    output["momentum"] = output["momentum"].fillna(0.0)
    return output.drop(columns=["cnr_athlete_name"])


def attach_world_ranking(athletes: pd.DataFrame, wr: pd.DataFrame) -> pd.DataFrame:
    if athletes.empty:
        return athletes
    if wr.empty:
        for column in ("world_event_rank", "starts_365", "top_40"):
            athletes[column] = np.nan
        return athletes
    boulder = wr.loc[wr["discipline"].eq("Boulder")].copy()
    boulder["name_key"] = boulder["athlete_name"].map(plain_key)
    keep = [
        "pool", "name_key", "world_event_rank", "starts_365", "top_40",
        "world_event_points_365", "points_per_start", "country",
    ]
    boulder = boulder[keep].sort_values("world_event_rank").drop_duplicates(
        ["pool", "name_key"]
    )
    output = athletes.merge(
        boulder, on=["pool", "name_key"], how="left", suffixes=("", "_wr")
    )
    output["country"] = output["country_wr"].combine_first(output["country"])
    return output.drop(columns=["country_wr"])


def enrich_public_ages(
    athletes: pd.DataFrame,
    profiles: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Fill only missing ages from the strongest public profile match."""

    if athletes.empty or profiles.empty:
        return athletes
    profiles = profiles.copy()
    profiles["name_key"] = profiles["athlete_name"].map(plain_key)
    profiles["birth_date_confidence_score"] = pd.to_numeric(
        profiles.get("birth_date_confidence_score"), errors="coerce"
    ).fillna(0)
    profiles = (
        profiles.sort_values(
            ["birth_date_confidence_score", "fetched_at_utc"],
            ascending=[False, False],
        )
        .drop_duplicates("name_key")
    )
    profile_dates = profiles[[
        "name_key", "birth_date_analysis_value", "birth_date_confidence",
        "birth_date_uncertainty_days",
    ]].rename(columns={
        "birth_date_analysis_value": "profile_birth_date",
        "birth_date_confidence": "profile_birth_confidence",
        "birth_date_uncertainty_days": "profile_birth_uncertainty_days",
    })
    output = athletes.merge(profile_dates, on="name_key", how="left")
    birth = pd.to_datetime(output["profile_birth_date"], errors="coerce")
    missing_age = output["age"].isna() & birth.notna()
    output.loc[missing_age, "age"] = (as_of - birth.loc[missing_age]).dt.days / 365.2425
    output["birth_date_confidence"] = output.get(
        "birth_date_confidence", pd.Series(index=output.index, dtype=object)
    ).combine_first(output["profile_birth_confidence"])
    output["birth_date_uncertainty_days"] = pd.to_numeric(
        output.get("birth_date_uncertainty_days"), errors="coerce"
    ).combine_first(pd.to_numeric(output["profile_birth_uncertainty_days"], errors="coerce"))
    return output.drop(columns=[
        "profile_birth_date", "profile_birth_confidence",
        "profile_birth_uncertainty_days",
    ])


def roster_names(
    mode: str,
    athletes: pd.DataFrame,
    history: pd.DataFrame,
    rosters: pd.DataFrame,
) -> list[str]:
    if mode == "EEQ" and not rosters.empty:
        return sorted(rosters.loc[rosters["organization"].eq("EEQ"), "athlete_name"].dropna().unique())
    if mode == "YNT Tier 1" and not history.empty:
        event = history.loc[
            history["event_name"].astype(str).str.contains(
                "Youth Championship Arco 2026", case=False, na=False
            )
            & history["source_scope"].eq("IFSC")
            & history["nationality"].astype(str).str.upper().eq("CAN")
        ]
        return sorted(event["athlete_name"].dropna().unique())
    if mode == "Canadian National Team proxy":
        return sorted(
            athletes.loc[athletes["cnr_rank"].le(15), "athlete_name"]
            .dropna().unique()
        )
    return []


def selected_rows(athletes: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    keys = {plain_key(name) for name in names}
    return athletes.loc[athletes["name_key"].isin(keys)].copy()


def rating_help() -> str:
    return (
        "Every displayed family uses the same anchor: 2000 means a fitted 50% "
        "chance of reaching a semifinal at a randomly sampled 2025 IFSC Open "
        "World Cup, within the athlete's gender pool. This shifts the scale for "
        "interpretation without changing athlete order or model updates. Dashed "
        "final, podium and win lines are fitted from the same frozen 2025 "
        "athlete-starts; they are historical references, not current 2026 odds. "
        "Global-ELO uses every de-duplicated Boulder result on one Open World-Cup "
        "readiness scale. IFSC-ELO uses IFSC results only. WR-ELO uses only events "
        "that award IFSC World Ranking points. Specialist ratings are shown only "
        "with at least two eligible rounds and enough athletes to calibrate the "
        "family; they shrink toward Global-ELO while evidence is limited. "
        "Performance-ELO is one round's isolated level, not "
        "the athlete's stable rating."
    )


def rating_transform_controls(key: str, default: str) -> str:
    columns = st.columns([1.3, 1, 1])
    family = columns[0].segmented_control(
        "Rating evidence", RATING_ORDER, default=default,
        help=rating_help(), key=f"{key}_family",
    )
    round_name = columns[1].selectbox(
        "Round evidence", ROUND_OPTIONS, index=0, key=f"{key}_round",
        help="Use all eligible rounds, or isolate qualification, semifinal or final evidence.",
    )
    procedure = columns[2].selectbox(
        "Qualification procedure", QUALIFICATION_FORMATS, index=0,
        disabled=round_name != "Qualification", key=f"{key}_procedure",
        help="Flash includes shared beta/video rounds; Onsight keeps athletes isolated from other attempts.",
    )
    suffix = {
        "All rounds": "", "Qualification": "-Qualies",
        "Semi-final": "-Semies", "Final": "-Finals",
    }[round_name]
    rating = f"{family}{suffix}"
    if round_name == "Qualification" and procedure != "Flash + Onsight":
        rating = f"{rating}-{procedure}"
    return rating


def correlation_note(
    correlations: pd.DataFrame, family: str, pool: str | None = None
) -> str:
    if correlations.empty or family == "WR-ELO":
        return "WR-ELO is the reference scale in this view."
    rows = correlations.loc[correlations["rating_family"].eq(family)]
    if pool:
        rows = rows.loc[rows["pool"].eq(pool)]
    rows = rows.dropna(subset=["spearman_correlation"])
    if rows.empty:
        return "Not enough paired evidence to report a stable relationship with WR-ELO."
    value = float(np.average(rows["spearman_correlation"], weights=rows["athletes"]))
    n = int(rows["athletes"].sum())
    return (
        f"Relationship with WR-ELO: {value:.2f} (rank correlation), using {n} athletes "
        "who have both ratings. A value near 1 means the two ratings order athletes "
        "similarly. It does not tell us why an athlete differs between them."
    )


def add_selected_highlight(
    figure: go.Figure,
    frame: pd.DataFrame,
    selected: list[str],
    x: str,
    y: str,
) -> None:
    focus = selected_rows(frame, selected).dropna(subset=[x, y])
    order = {plain_key(name): index for index, name in enumerate(selected)}
    focus["_selection_order"] = focus["athlete_name"].map(
        lambda name: order.get(plain_key(name), len(order))
    )
    focus = focus.sort_values("_selection_order")
    offsets = [(-58, -34), (-68, 18), (-54, 42)]
    for position, (_, row) in enumerate(focus.iterrows()):
        color = ATHLETE_COLORS[position % len(ATHLETE_COLORS)]
        figure.add_trace(
            go.Scatter(
                x=[row[x]], y=[row[y]], mode="markers",
                marker={
                    "size": 14, "color": "rgba(0,0,0,0)",
                    "line": {"width": 2, "color": color},
                    "symbol": "diamond" if row.get("gender") == "Women" else "circle",
                },
                hoverinfo="skip", name=friendly_name(row["athlete_name"]),
                legendgroup="Compared athletes", showlegend=True,
            )
        )
        if position >= 3:
            continue
        ax, ay = offsets[position % len(offsets)]
        figure.add_annotation(
            x=row[x], y=row[y], text=friendly_name(row["athlete_name"]),
            showarrow=True, arrowhead=0, arrowwidth=1,
            arrowcolor="rgba(54,82,78,0.52)", ax=ax, ay=ay,
            font={"size": 12, "color": PALETTE["ink"]},
            bgcolor="rgba(255,255,255,0.62)", borderpad=2,
        )


def displacement_lines(
    figure: go.Figure,
    frame: pd.DataFrame,
    stage: str,
    x: str,
) -> None:
    transitions = []
    if stage.startswith("Global-ELO-"):
        transitions.append(("Global-ELO", stage, "#9AA7A4"))
    if stage in {"IFSC-ELO", "WR-ELO"}:
        transitions.append(("Global-ELO", "IFSC-ELO", "#9AA7A4"))
    if stage == "WR-ELO":
        transitions.append(("IFSC-ELO", "WR-ELO", PALETTE["gold"]))
    for start, end, color in transitions:
        subset = frame.dropna(subset=[x, start, end])
        for _, row in subset.iterrows():
            figure.add_trace(
                go.Scatter(
                    x=[row[x], row[x]], y=[row[start], row[end]],
                    mode="lines", line={"width": 1, "color": color},
                    opacity=0.22, hoverinfo="skip", showlegend=False,
                )
            )


def outcome_threshold(
    calibration: pd.DataFrame, outcome: str, pool: str = "Boulder_All"
) -> float:
    column = f"display_elo_at_50pct_{outcome}"
    if calibration.empty or column not in calibration:
        return np.nan
    matched = calibration.loc[calibration["pool"].eq(pool), column]
    if matched.notna().any():
        return float(matched.dropna().iloc[0])
    pool_rows = calibration.loc[
        calibration["pool"].isin(["Boulder_Men", "Boulder_Women"]), column
    ]
    return float(pool_rows.mean()) if pool_rows.notna().any() else np.nan


def outcome_probability(
    rating: float, calibration: pd.DataFrame, outcome: str,
    pool: str = "Boulder_All",
) -> tuple[float, float]:
    """Return a fitted 2025 outcome probability and McFadden pseudo-R²."""

    if not np.isfinite(rating) or calibration.empty:
        return np.nan, np.nan
    rows = calibration.loc[calibration["pool"].eq(pool)]
    if rows.empty:
        rows = calibration.loc[calibration["pool"].eq("Boulder_All")]
    if rows.empty:
        return np.nan, np.nan
    row = rows.iloc[0]
    threshold = pd.to_numeric(pd.Series([
        row.get(f"display_elo_at_50pct_{outcome}")
    ]), errors="coerce").iloc[0]
    slope = pd.to_numeric(pd.Series([
        row.get(f"{outcome}_logistic_slope_per_100_native_elo")
    ]), errors="coerce").iloc[0]
    fit = pd.to_numeric(pd.Series([
        row.get(f"{outcome}_mcfadden_pseudo_r2")
    ]), errors="coerce").iloc[0]
    if not np.isfinite(threshold) or not np.isfinite(slope):
        return np.nan, float(fit) if np.isfinite(fit) else np.nan
    linear = np.clip((rating - threshold) / 100.0 * slope, -30, 30)
    return (
        float(1 / (1 + np.exp(-linear))),
        float(fit) if np.isfinite(fit) else np.nan,
    )


def rating_target(family: str) -> str:
    if "-Finals" in family:
        return "podium"
    if "-Semies" in family:
        return "final"
    return "semifinal"


def add_outcome_thresholds(
    figure: go.Figure, calibration: pd.DataFrame
) -> None:
    styles = [
        ("semifinal", "50% semifinal", "rgba(11,122,117,0.42)"),
        ("final", "50% final", "rgba(66,133,169,0.42)"),
        ("podium", "50% podium", "rgba(230,162,60,0.48)"),
        ("win", "50% win", "rgba(242,107,91,0.46)"),
    ]
    for outcome, label, color in styles:
        level = outcome_threshold(calibration, outcome)
        if not np.isfinite(level):
            continue
        figure.add_hline(
            y=level, line_dash="dash", line_width=1, line_color=color,
            annotation_text=f"{label} · 2025",
            annotation_position="top left",
            annotation_font={"size": 10, "color": color.replace("0.4", "0.9")},
        )


def pool_scatter(
    frame: pd.DataFrame,
    x: str,
    rating: str,
    selected: list[str],
    title: str,
    canadian_outline: bool = False,
    calibration: pd.DataFrame | None = None,
) -> go.Figure:
    if rating not in frame.columns:
        return go.Figure().update_layout(
            title=f"{rating} is not yet available in this data build"
        )
    plot = frame.dropna(subset=[x, rating]).copy()
    if plot.empty:
        return go.Figure().update_layout(title="No matched ratings for this view")
    if "display_name" not in plot:
        plot["display_name"] = plot["athlete_name"].map(friendly_name)
    country = (
        plot.get("country", pd.Series("", index=plot.index))
        .fillna("").astype(str).str.upper()
    )
    country_codes = {
        "CANADA": "CAN", "UNITED STATES": "USA", "FRANCE": "FRA",
        "AUSTRALIA": "AUS", "JAPAN": "JPN", "GREAT BRITAIN": "GBR",
        "UNITED KINGDOM": "GBR", "GERMANY": "GER", "ITALY": "ITA",
        "SLOVENIA": "SLO", "AUSTRIA": "AUT", "SWITZERLAND": "SUI",
    }
    plot["country_code"] = country.map(country_codes).fillna(country.str.slice(0, 3))
    age = pd.to_numeric(
        plot.get("age", pd.Series(np.nan, index=plot.index)), errors="coerce"
    )
    uncertainty_days = pd.to_numeric(
        plot.get(
            "birth_date_uncertainty_days", pd.Series(np.nan, index=plot.index)
        ),
        errors="coerce",
    )
    plot["age_hover"] = [
        f"{value:.1f} ± {uncertainty / 365.2425:.1f} years"
        if np.isfinite(value) and np.isfinite(uncertainty)
        else (f"{value:.1f} years" if np.isfinite(value) else "Unknown")
        for value, uncertainty in zip(age, uncertainty_days)
    ]
    days = pd.to_numeric(
        plot.get("days_since_last_result", pd.Series(np.nan, index=plot.index)),
        errors="coerce",
    )
    plot["days_hover"] = [
        (
            f"{'🔴' if value > 180 else '🟠' if value > 90 else '🟢'} "
            f"{int(value)} days"
        ) if np.isfinite(value) else "Unknown"
        for value in days
    ]
    evidence_column = f"{rating} evidence"
    evidence = pd.to_numeric(
        plot.get(evidence_column, pd.Series(np.nan, index=plot.index)), errors="coerce"
    )
    plot["rounds_hover"] = evidence.fillna(0).astype(int)
    plot["momentum_hover"] = plot["momentum"].map(
        lambda value: f"{value:+.1f} Global-ELO / 365d"
    )
    figure = px.scatter(
        plot,
        x=x,
        y=rating,
        color="momentum",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        symbol="gender",
        hover_name="display_name",
        custom_data=[
            "country_code", "momentum_hover", "age_hover", "days_hover",
            "rounds_hover", "display_name",
        ],
        title=title,
    )
    figure.update_traces(
        marker={"size": 9, "opacity": 0.72},
        hovertemplate=(
            "<b>%{customdata[5]} (%{customdata[0]})</b><br>"
            f"{rating}: %{{y:,.0f}}<br>"
            "Momentum: %{customdata[1]}<br>"
            "Age: %{customdata[2]}<br>"
            "Last result: %{customdata[3]}<br>"
            "Included rounds: %{customdata[4]}<extra></extra>"
        ),
    )
    if canadian_outline and "country" in plot:
        canadians = plot.loc[plot["country"].astype(str).str.upper().eq("CAN")]
        figure.add_trace(
            go.Scatter(
                x=canadians[x], y=canadians[rating], mode="markers",
                marker={"size": 13, "color": "rgba(0,0,0,0)", "line": {"width": 2, "color": "#111"}},
                hoverinfo="skip", name="Canada", showlegend=True,
            )
        )
    displacement_lines(figure, plot, rating, x)
    add_selected_highlight(figure, plot, selected, x, rating)
    add_outcome_thresholds(
        figure, calibration if calibration is not None else pd.DataFrame()
    )
    figure.update_layout(
        height=610,
        margin={"l": 88, "r": 36, "t": 76, "b": 138},
        legend={
            "title": {"text": "Gender"}, "orientation": "h",
            "x": 0, "xanchor": "left", "y": -0.18, "yanchor": "top",
        },
        coloraxis_colorbar={
            "title": {"text": "Momentum · Global-ELO / 365d", "side": "top"},
            "orientation": "h", "x": 1, "xanchor": "right",
            "y": -0.25, "yanchor": "top", "len": 0.46, "thickness": 12,
            "nticks": 3, "tickangle": 0,
        },
        hovermode="closest",
    )
    x_titles = {
        "cnr_rank": "CNR rank", "age": "Age (years)",
        "ifsc_rank": "IFSC-ELO rank", "world_event_rank": "IFSC World Ranking",
    }
    figure.update_xaxes(
        title={"text": x_titles.get(x, x.replace("_", " ").title()), "standoff": 16},
        tickformat=",.0f", separatethousands=True, automargin=True,
    )
    figure.update_yaxes(
        title={"text": rating, "standoff": 12}, gridcolor="#E5ECEA",
        tickformat=",.0f", separatethousands=True, automargin=True,
    )
    return figure


def compare_text(
    frame: pd.DataFrame,
    selected: list[str],
    rating: str,
    context: str = "",
) -> str:
    if rating not in frame.columns:
        return (
            f"{rating} is not present in this data build yet. The athlete selection "
            "is preserved while the specialist family is rebuilt."
        )
    focus = selected_rows(frame, selected).dropna(subset=[rating]).sort_values(rating, ascending=False)
    if focus.empty:
        return "The selected athletes do not yet have matched evidence in this pool."
    leader = focus.iloc[0]
    if len(focus) == 1:
        return f"{friendly_name(leader['athlete_name'])} is shown at {leader[rating]:.0f} {rating}."
    trailer = focus.iloc[-1]
    gap = leader[rating] - trailer[rating]
    base = (
        f"{friendly_name(leader['athlete_name'])} leads this comparison at {leader[rating]:.0f}. "
        f"The displayed gap to {friendly_name(trailer['athlete_name'])} is {gap:.0f} points. "
    )
    if context == "Canadian":
        rank = leader.get("cnr_rank", np.nan)
        detail = (
            f"The same athlete is CNR #{int(rank)}; disagreement between CNR and "
            "Global-ELO is the useful review signal. "
            if pd.notna(rank) else "Their CNR rank is not matched in this snapshot. "
        )
    elif context == "IFSC":
        transport = leader.get("IFSC-ELO", np.nan) - leader.get("Global-ELO", np.nan)
        detail = (
            f"Their IFSC-minus-Global gap is {transport:+.0f}; this describes how "
            "their IFSC evidence differs, without assigning the cause. "
            if np.isfinite(transport) else "Their IFSC-specific evidence is still incomplete. "
        )
    elif context == "WR":
        rank = leader.get("world_event_rank", np.nan)
        starts = leader.get("starts_365", np.nan)
        detail = (
            f"Current World Ranking: {int(rank) if pd.notna(rank) else 'not ranked'} "
            f"from {int(starts) if pd.notna(starts) else 0} eligible starts in 365 days. "
        )
    elif context == "Progression":
        detail = f"Their Global-ELO changed {leader.get('momentum', 0):+.0f} over the latest 365-day window. "
    else:
        detail = ""
    return base + detail + (
        "Read the gap with evidence count and last-result date; it is a projection "
        "difference, not a selection verdict."
    )


def render_canadian_pool(
    athletes: pd.DataFrame,
    selected: list[str],
    correlations: pd.DataFrame,
    calibration: pd.DataFrame,
) -> None:
    st.subheader("Canadian Pool")
    st.caption("Where every current Canadian CNR athlete sits, and what changes when the evidence becomes more competition-specific.")
    canadian = athletes.loc[athletes["cnr_rank"].notna()].copy()
    x_choice = st.segmented_control(
        "Horizontal axis", ["CNR rank", "Age"], default="CNR rank",
        help="CNR rank compares current Canadian ranking position. Age compares pathway timing.",
        key="canadian_x",
    )
    stage = rating_transform_controls("canadian", "Global-ELO")
    x = "cnr_rank" if x_choice == "CNR rank" else "age"
    figure = pool_scatter(
        canadian, x, stage, selected,
        f"Canadian CNR athletes — {stage}",
        calibration=calibration,
    )
    if x == "cnr_rank":
        figure.update_xaxes(autorange="reversed")
    st.plotly_chart(figure, width="stretch", theme=None)
    st.info(compare_text(canadian, selected, stage, "Canadian"), icon="↗️")
    st.caption(correlation_note(correlations, stage))
    missing = int(canadian[stage].isna().sum()) if stage in canadian else len(canadian)
    if missing:
        st.caption(f"{missing} CNR athlete(s) are retained in the pool but cannot be plotted on {stage} yet because eligible evidence is missing.")


def render_ifsc_pool(
    athletes: pd.DataFrame,
    history: pd.DataFrame,
    selected: list[str],
    correlations: pd.DataFrame,
    calibration: pd.DataFrame,
) -> None:
    st.subheader("IFSC Pool")
    st.caption("Canadians beside every 2025–2026 IFSC Boulder finalist; a black ring identifies Canada.")
    finalist_ids: set[str] = set()
    if not history.empty:
        dates = pd.to_datetime(history["event_date"], errors="coerce")
        finalist_ids = set(
            history.loc[
                history["source_scope"].eq("IFSC")
                & history["round_group"].eq("Final")
                & dates.dt.year.isin([2025, 2026]),
                "global_id",
            ].astype(str)
        )
    pool = athletes.loc[
        athletes["global_id"].astype(str).isin(finalist_ids)
        | athletes["cnr_rank"].notna()
    ].copy()
    pool["ifsc_rank"] = pool.groupby("pool")["IFSC-ELO"].rank(ascending=False, method="min")
    x_choice = st.segmented_control(
        "Horizontal axis", ["IFSC rank", "Age"], default="Age", key="ifsc_x"
    )
    stage = rating_transform_controls("ifsc", "IFSC-ELO")
    x = "ifsc_rank" if x_choice == "IFSC rank" else "age"
    figure = pool_scatter(
        pool, x, stage, selected,
        f"Canadian CNR athletes and recent IFSC finalists — {stage}",
        canadian_outline=True,
        calibration=calibration,
    )
    if x == "ifsc_rank":
        figure.update_xaxes(autorange="reversed")
    st.plotly_chart(figure, width="stretch", theme=None)
    st.info(compare_text(pool, selected, stage, "IFSC"), icon="↗️")
    st.caption(correlation_note(correlations, stage))


def render_wr_pool(
    athletes: pd.DataFrame,
    selected: list[str],
    correlations: pd.DataFrame,
    calibration: pd.DataFrame,
) -> None:
    st.subheader("WR Pool")
    st.caption("The current World Ranking top 40 plus every Canadian with a World Ranking start in the last 365 days.")
    pool = athletes.loc[
        athletes["world_event_rank"].le(40)
        | (
            athletes["country"].astype(str).str.upper().eq("CAN")
            & athletes["starts_365"].fillna(0).gt(0)
        )
    ].copy()
    stage = rating_transform_controls("wr", "WR-ELO")
    figure = pool_scatter(
        pool, "world_event_rank", stage, selected,
        f"World Ranking pool — {stage}", canadian_outline=True,
        calibration=calibration,
    )
    figure.update_xaxes(autorange="reversed", title="Current IFSC World Ranking")
    st.plotly_chart(figure, width="stretch", theme=None)
    st.info(compare_text(pool, selected, stage, "WR"), icon="↗️")
    st.caption(correlation_note(correlations, stage))

    available = sorted(
        country for country in pool["country"].dropna().astype(str).unique()
        if country and country.upper() != "CAN"
    )
    defaults = [country for country in ["FRA", "USA", "AUS"] if country in available]
    countries = st.multiselect(
        "Compare Canada with three countries",
        available,
        default=defaults,
        max_selections=3,
    )
    country_pool = pool.loc[pool["country"].isin(["CAN", *countries])].dropna(subset=["WR-ELO"])
    if not country_pool.empty:
        country_pool = country_pool.copy()
        country_pool["Included rounds"] = pd.to_numeric(
            country_pool.get("WR-ELO evidence"), errors="coerce"
        ).fillna(0).astype(int)
        comparison = px.strip(
            country_pool, x="country", y="WR-ELO", color="country",
            hover_name="display_name",
            hover_data={
                "world_event_rank": True, "starts_365": True,
                "Included rounds": True,
            },
            title="Actual WR-ELO distribution of current participants",
        )
        comparison.update_traces(marker={"size": 10, "opacity": 0.65})
        add_outcome_thresholds(comparison, calibration)
        comparison.update_layout(height=430, showlegend=False)
        st.plotly_chart(comparison, width="stretch", theme=None)


def age_group(age: float) -> str:
    if pd.isna(age):
        return "Age unknown"
    if age < 15:
        return "U15"
    if age < 17:
        return "U17"
    if age < 19:
        return "U19"
    if age < 21:
        return "U21"
    return "Senior"


def render_progression(
    athletes: pd.DataFrame,
    history: pd.DataFrame,
    selected: list[str],
    correlations: pd.DataFrame,
    calibration: pd.DataFrame,
) -> None:
    st.subheader("Global progression")
    st.caption("Compare athletes at the same age, then separate current level from the direction of travel.")
    cohort = athletes.loc[
        ((athletes["age"] < 21) & athletes["cnr_rank"].notna())
        | athletes["cnr_rank"].le(15)
    ].copy()
    if not history.empty:
        dates = pd.to_datetime(history["event_date"], errors="coerce")
        canadian_youth_event = (
            history["source_scope"].eq("CEC")
            & history["event_name"].astype(str).str.contains(
                "Youth National", case=False, na=False
            )
        )
        latest_canadian_youth = dates.loc[canadian_youth_event].max()
        finalists = history.loc[
            canadian_youth_event
            & history["round_group"].eq("Final")
            & dates.eq(latest_canadian_youth)
        ]
        cohort = pd.concat(
            [cohort, athletes.loc[athletes["global_id"].isin(finalists["global_id"])]],
            ignore_index=True,
        ).drop_duplicates(["pool", "global_id"])
    cohort["Age group"] = cohort["age"].map(age_group)
    plot = cohort.dropna(subset=["age", "Global-ELO"])
    plot = plot.copy()
    plot["Included rounds"] = pd.to_numeric(
        plot.get("Global-ELO evidence"), errors="coerce"
    ).fillna(0).astype(int)
    figure = px.scatter(
        plot, x="age", y="Global-ELO", color="Age group", symbol="gender",
        hover_name="display_name",
        hover_data={
            "cnr_rank": True, "momentum": ":.1f", "country": True,
            "Included rounds": True,
        },
        title="Current Canadian pathway — one Open World-readiness scale",
    )
    figure.update_traces(marker={"size": 10, "opacity": 0.65})
    add_selected_highlight(figure, plot, selected, "age", "Global-ELO")

    show_lines = st.toggle(
        "Show current World top-10 pathway",
        value=True,
        help=(
            "Shows how today's World Ranking top 10 were rated at each age. "
            "An age bin appears only when it contains several athletes."
        ),
    )
    if show_lines:
        top10 = athletes.loc[athletes["world_event_rank"].le(10), ["pool", "global_id"]]
        pathway = history.merge(top10, on=["pool", "global_id"], how="inner")
        pathway = pathway.dropna(subset=["age_at_event", "rating_after"]).copy()
        pathway["age_year"] = (
            pd.to_numeric(pathway["age_at_event"], errors="coerce") * 2
        ).round() / 2
        pathway = pathway.merge(
            athletes[["pool", "global_id", "gender"]],
            on=["pool", "global_id"], how="left",
        )
        grouped = pathway.groupby(["gender", "age_year"], as_index=False).agg(
            rating=("rating_after", "mean"), athletes=("global_id", "nunique"),
            rounds=("global_id", "size"),
        )
        grouped = grouped.loc[grouped["athletes"].ge(3)]
        if not grouped.empty:
            for gender, gender_rows in grouped.groupby("gender"):
                figure.add_trace(go.Scatter(
                    x=gender_rows["age_year"], y=gender_rows["rating"], mode="lines",
                    name=f"Current WR top-10 history — {gender}",
                    customdata=np.column_stack([
                        gender_rows["athletes"], gender_rows["rounds"],
                    ]),
                    hovertemplate=(
                        "Age: %{x:.1f}<br>Average Global-ELO: %{y:.0f}"
                        "<br>Athletes: %{customdata[0]:.0f}"
                        "<br>Included rounds: %{customdata[1]:.0f}<extra>%{fullData.name}</extra>"
                    ),
                    line={
                        "color": PALETTE["teal"] if gender == "Men" else PALETTE["blue"],
                        "width": 3, "dash": "dot",
                    },
                ))
    add_outcome_thresholds(figure, calibration)
    figure.update_layout(height=570, margin={"l": 20, "r": 20, "t": 70, "b": 20})
    st.plotly_chart(figure, width="stretch", theme=None)
    st.info(compare_text(cohort, selected, "Global-ELO", "Progression"), icon="↗️")

    projection_figure = progression_projection(
        athletes, history, selected, calibration
    )
    st.plotly_chart(projection_figure, width="stretch", theme=None)
    st.caption(
        "Projection rate = 65% of the athlete's bounded recent Global-ELO "
        "change + 35% of the median IFSC Performance-ELO change observed at "
        "the same age and gender. It assumes the trend continues; it is not a "
        "training-effect claim."
    )
    render_matchup_matrix(athletes, history, selected)
    render_focus_hypotheses(athletes, history, selected, calibration)
    st.caption(correlation_note(correlations, "Global-ELO"))


def progression_projection(
    athletes: pd.DataFrame,
    history: pd.DataFrame,
    selected: list[str],
    calibration: pd.DataFrame,
) -> go.Figure:
    figure = go.Figure()
    focus = selected_rows(athletes, selected)
    if history.empty or focus.empty:
        return figure.update_layout(title="Progression projection unavailable")
    as_of = pd.to_datetime(history["event_date"], errors="coerce").max()
    history = history.copy()
    history["event_date"] = pd.to_datetime(history["event_date"], errors="coerce")
    age_rates = typical_age_progression(history)
    colors = [ATHLETE_COLORS[index % len(ATHLETE_COLORS)] for index in range(len(focus))]
    for color, (_, athlete) in zip(colors, focus.iterrows()):
        rows = history.loc[
            history["global_id"].eq(athlete["global_id"])
            & history["pool"].eq(athlete["pool"])
        ].sort_values("event_date")
        if rows.empty:
            continue
        rows = rows.copy()
        rows["included_rounds"] = np.arange(1, len(rows) + 1)
        figure.add_trace(go.Scatter(
            x=rows["event_date"], y=rows["rating_after"], mode="lines",
            name=f"{athlete['athlete_name']} — observed", line={"color": color, "width": 3},
            customdata=np.column_stack([
                rows["event_name"], rows["round_group"], rows["included_rounds"],
            ]),
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>Global-ELO: %{y:.0f}"
                "<br>%{customdata[0]} · %{customdata[1]}"
                "<br>Included rounds: %{customdata[2]:.0f}<extra>%{fullData.name}</extra>"
            ),
        ))
        momentum = float(np.clip(athlete.get("momentum", 0.0), -150, 150))
        age_year = int(round(athlete.get("age", np.nan))) if pd.notna(athlete.get("age")) else -1
        typical = age_rates.get((athlete.get("pool"), age_year), 0.0)
        projected_rate = float(np.clip(0.65 * momentum + 0.35 * typical, -150, 150))
        future_dates = pd.date_range(as_of, periods=13, freq="MS")
        central = float(athlete["Global-ELO"]) + projected_rate * np.arange(13) / 12
        uncertainty = 35 + 4 * np.arange(13)
        current_evidence = int(pd.to_numeric(
            pd.Series([athlete.get("Global-ELO evidence", len(rows))]), errors="coerce"
        ).fillna(len(rows)).iloc[0])
        figure.add_trace(go.Scatter(
            x=future_dates, y=central, mode="lines",
            name=(
                f"{athlete['athlete_name']} — hypothesis "
                f"({projected_rate:+.0f}/year)"
            ),
            line={"color": color, "width": 3, "dash": "dash"},
            customdata=np.full((len(future_dates), 1), current_evidence),
            hovertemplate=(
                "%{x|%Y-%m}<br>Projected Global-ELO: %{y:.0f}"
                "<br>Current rating evidence: %{customdata[0]:.0f} rounds"
                "<extra>%{fullData.name}</extra>"
            ),
        ))
        figure.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(central + uncertainty) + list((central - uncertainty)[::-1]),
            fill="toself", fillcolor=transparent(color),
            line={"color": "rgba(0,0,0,0)"}, hoverinfo="skip", showlegend=False,
        ))
    figure.add_vline(x=as_of.timestamp() * 1000, line_dash="dash", line_color="#555", annotation_text="Now")
    add_outcome_thresholds(figure, calibration)
    figure.update_layout(
        title="Observed Global-ELO and a 12-month bounded-trend hypothesis",
        height=500, yaxis_title="Global-ELO", xaxis_title="Event date",
        hovermode="x unified",
    )
    return figure


def typical_age_progression(history: pd.DataFrame) -> dict[tuple[str, int], float]:
    """Median IFSC Performance-ELO change at each age; descriptive only."""

    required = {
        "pool", "global_id", "event_date", "performance_elo",
        "age_at_event", "source_scope",
    }
    if history.empty or not required.issubset(history.columns):
        return {}
    rows = history.loc[history["source_scope"].eq("IFSC")].dropna(
        subset=["age_at_event", "performance_elo", "event_date"]
    ).copy()
    rows["age_year"] = pd.to_numeric(rows["age_at_event"], errors="coerce").round()
    rows = rows.loc[rows["age_year"].between(12, 45)]
    rows = rows.sort_values("event_date")
    segments = (
        rows.groupby(["pool", "global_id", "age_year"], as_index=False)
        .agg(
            first_date=("event_date", "first"),
            last_date=("event_date", "last"),
            first_rating=("performance_elo", "first"),
            last_rating=("performance_elo", "last"),
            observations=("performance_elo", "size"),
        )
    )
    segments["years"] = (
        pd.to_datetime(segments["last_date"]) - pd.to_datetime(segments["first_date"])
    ).dt.days / 365.2425
    segments = segments.loc[segments["years"].ge(0.20) & segments["observations"].ge(2)]
    segments["annual_change"] = (
        (segments["last_rating"] - segments["first_rating"]) / segments["years"]
    ).clip(-300, 300)
    typical = segments.groupby(["pool", "age_year"])["annual_change"].median()
    return {
        (str(pool), int(age)): float(value)
        for (pool, age), value in typical.items()
        if pd.notna(age) and np.isfinite(value)
    }


def render_matchup_matrix(
    athletes: pd.DataFrame, history: pd.DataFrame, selected: list[str]
) -> None:
    """Pairwise Elo ordering now and under the same bounded 12-month hypothesis."""

    focus = selected_rows(athletes, selected).dropna(subset=["Global-ELO"]).copy()
    if focus.empty or len(focus) < 2:
        return
    focus["selection_order"] = focus["name_key"].map({
        plain_key(name): index for index, name in enumerate(selected)
    })
    focus = focus.sort_values("selection_order").head(12)
    age_rates = typical_age_progression(history)
    projected = {}
    for _, athlete in focus.iterrows():
        age_year = int(round(athlete["age"])) if pd.notna(athlete.get("age")) else -1
        typical = age_rates.get((athlete.get("pool"), age_year), 0.0)
        rate = float(np.clip(
            0.65 * float(np.clip(athlete.get("momentum", 0.0), -150, 150))
            + 0.35 * typical, -150, 150,
        ))
        projected[(athlete["pool"], athlete["global_id"])] = float(athlete["Global-ELO"]) + rate
    labels = [friendly_name(name) for name in focus["athlete_name"]]
    matrix = []
    for _, row in focus.iterrows():
        cells = []
        for _, opponent in focus.iterrows():
            if row["global_id"] == opponent["global_id"] and row["pool"] == opponent["pool"]:
                cells.append("—")
            elif row["pool"] != opponent["pool"]:
                cells.append("Different pool")
            else:
                now = 1 / (1 + 10 ** ((opponent["Global-ELO"] - row["Global-ELO"]) / 400))
                future = 1 / (1 + 10 ** ((
                    projected[(opponent["pool"], opponent["global_id"])]
                    - projected[(row["pool"], row["global_id"])]
                ) / 400))
                cells.append(f"{now:.0%} → {future:.0%}")
        matrix.append(cells)
    table = pd.DataFrame(matrix, columns=labels, index=labels)
    st.markdown("#### Pairwise rating comparison")
    st.caption(
        "Each cell is the row athlete's Elo-expected chance of placing ahead: now → "
        "the 12-month bounded-trend hypothesis. This is not a full competition simulation."
    )
    st.dataframe(table, width="stretch")


def render_focus_hypotheses(
    athletes: pd.DataFrame,
    history: pd.DataFrame,
    selected: list[str],
    calibration: pd.DataFrame,
) -> None:
    focus = selected_rows(athletes, selected)
    cards = []
    for _, athlete in focus.iterrows():
        global_elo = athlete.get("Global-ELO", np.nan)
        wr_elo = athlete.get("WR-ELO", np.nan)
        semi = outcome_threshold(calibration, "semifinal", athlete.get("pool"))
        wr_starts = athlete.get("starts_365", np.nan)
        momentum = athlete.get("momentum", 0.0)
        if pd.isna(global_elo):
            hypothesis = "Build a reliable competition baseline before choosing a pathway emphasis."
        elif np.isfinite(wr_elo) and global_elo - wr_elo >= 100:
            hypothesis = (
                "General performance is ahead of World-Ranking-specific performance. "
                "Prioritize WR-style simulations and selective WR starts; review onsight "
                "decision quality, setting specificity, travel and pressure response."
            )
        elif np.isfinite(semi) and global_elo >= semi - 100 and (pd.isna(wr_starts) or wr_starts < 3):
            hypothesis = "Test targeted WR competition exposure; readiness appears close enough for the experience to be informative."
        elif momentum > 35:
            hypothesis = "Protect the improving training process; add WR starts selectively rather than chasing participation volume."
        else:
            hypothesis = "Prioritize raising repeatable performance; choose competitions that answer a specific readiness question."
        analogy = exposure_analogy(history, athlete.get("pool"), global_elo, momentum)
        if analogy:
            hypothesis += " " + analogy
        cards.append((friendly_name(athlete["athlete_name"]), hypothesis))
    if cards:
        st.markdown("#### Working hypotheses")
        for start in range(0, len(cards), 3):
            columns = st.columns(min(3, len(cards) - start))
            for column, (name, hypothesis) in zip(columns, cards[start:start + 3]):
                with column.container(border=True):
                    st.markdown(f"**{name}**")
                    st.write(hypothesis)
        st.caption(
            "These are decision hypotheses from rating level, recent change and WR "
            "exposure—not causal training prescriptions. More starts can reveal a "
            "rating more clearly, but the current data do not prove that starts cause improvement."
        )


def exposure_analogy(
    history: pd.DataFrame, pool: str, level: float, momentum: float
) -> str:
    """Describe—not causally estimate—one-year outcomes for similar prior anchors."""

    if history.empty or not np.isfinite(level):
        return ""
    cohort = exposure_reference(history)
    if cohort.empty:
        return ""
    cohort = cohort.loc[cohort["pool"].eq(pool)]
    similar = cohort.loc[
        cohort["anchor_rating"].sub(level).abs().le(150)
        & cohort["prior_momentum"].sub(momentum).abs().le(75)
    ]
    if len(similar) < 8:
        return ""
    lower = similar.loc[similar["ifsc_starts"].lt(3), "next_change"]
    higher = similar.loc[similar["ifsc_starts"].ge(3), "next_change"]
    if len(lower) < 3 or len(higher) < 3:
        return ""
    difference = float(higher.median() - lower.median())
    return (
        f"Among {len(similar)} similar one-year historical cases, athletes with 3+ "
        f"IFSC starts changed {difference:+.0f} Elo more than those with 0–2. "
        "Treat this as an exposure analogy, not a causal effect."
    )


@st.cache_data(show_spinner=False, max_entries=1)
def exposure_reference(history: pd.DataFrame) -> pd.DataFrame:
    rows = history.copy()
    rows["event_date"] = pd.to_datetime(rows["event_date"], errors="coerce")
    as_of = pd.Timestamp(rows["event_date"].max())
    cutoff = as_of - pd.Timedelta(days=365)
    prior_cutoff = cutoff - pd.Timedelta(days=365)
    before = rows.loc[rows["event_date"].le(cutoff)].sort_values("event_date")
    if before.empty:
        return pd.DataFrame()
    keys = ["pool", "global_id"]
    anchor = before.groupby(keys, as_index=False).tail(1)[
        [*keys, "rating_after"]
    ].rename(columns={"rating_after": "anchor_rating"})
    prior = before.loc[before["event_date"].le(prior_cutoff)].groupby(
        keys, as_index=False
    ).tail(1)[[*keys, "rating_after"]].rename(
        columns={"rating_after": "prior_rating"}
    )
    future = rows.loc[rows["event_date"].gt(cutoff)].sort_values("event_date")
    end = future.groupby(keys, as_index=False).tail(1)[
        [*keys, "rating_after"]
    ].rename(columns={"rating_after": "end_rating"})
    ifsc_starts = (
        future.loc[future["source_scope"].eq("IFSC")]
        .groupby(keys)["source_event_id"].nunique()
        .rename("ifsc_starts")
        .reset_index()
    )
    cohort = anchor.merge(prior, on=keys).merge(end, on=keys).merge(
        ifsc_starts, on=keys, how="left"
    )
    cohort["ifsc_starts"] = cohort["ifsc_starts"].fillna(0)
    cohort["prior_momentum"] = cohort["anchor_rating"] - cohort["prior_rating"]
    cohort["next_change"] = cohort["end_rating"] - cohort["anchor_rating"]
    return cohort


def render_olympics(
    athletes: pd.DataFrame,
    selected: list[str],
    correlations: pd.DataFrame,
) -> None:
    st.subheader("Towards the Olympics")
    st.caption("A single view of World readiness, World Ranking access and evidence gaps. It does not claim qualification before the LA28 rules and field are modelled.")
    focus = selected_rows(athletes, selected)
    columns = st.columns(max(1, min(3, len(focus))))
    for column, (_, athlete) in zip(columns, focus.iterrows()):
        column.markdown(f"#### {friendly_name(athlete['athlete_name'])}")
        column.metric("Global-ELO", f"{athlete.get('Global-ELO', np.nan):.0f}" if pd.notna(athlete.get("Global-ELO")) else "—")
        column.metric("IFSC-ELO", f"{athlete.get('IFSC-ELO', np.nan):.0f}" if pd.notna(athlete.get("IFSC-ELO")) else "—")
        column.metric("WR-ELO", f"{athlete.get('WR-ELO', np.nan):.0f}" if pd.notna(athlete.get("WR-ELO")) else "—")
        rank = athlete.get("world_event_rank", np.nan)
        column.caption(f"Current World Ranking: {int(rank) if pd.notna(rank) else 'not ranked'} · starts/365d: {int(athlete.get('starts_365', 0) or 0)}")
    if not focus.empty:
        table = focus[[
            "athlete_name", "Global-ELO", "IFSC-ELO", "WR-ELO",
            "world_event_rank", "starts_365", "momentum",
        ]].copy()
        table["athlete_name"] = table["athlete_name"].map(friendly_name)
        table = table.rename(columns={
            "athlete_name": "Athlete", "world_event_rank": "World Ranking",
            "starts_365": "WR starts / 365d", "momentum": "Global-ELO change / 365d",
        })
        st.dataframe(table, hide_index=True, width="stretch")
    st.warning(
        "Olympic pathway probabilities remain a governed next step. Until the qualification rules, eligible fields and quota constraints are versioned, this page supports readiness discussion—not selection odds.",
        icon="⚠️",
    )
    st.caption(correlation_note(correlations, "Global-ELO"))


def top_ribbon(
    athletes: pd.DataFrame,
    history: pd.DataFrame,
    rosters: pd.DataFrame,
) -> tuple[list[str], str]:
    with st.container(border=True):
        top = st.columns([1.2, 1])
        mode = top[0].segmented_control(
            "Athlete set",
            ["Compare 3", "EEQ", "YNT Tier 1", "Canadian National Team proxy"],
            default="Compare 3",
            help=(
                "YNT Tier 1 means Canadian 2026 Youth Worlds participants. "
                "The national-team option is a clearly labelled CNR top-15 proxy; "
                "it is not presented as an official roster."
            ),
        )
        discipline = top[1].segmented_control(
            "Discipline", ["Boulder"], default="Boulder",
            help="Boulder is the governed release focus. Lead and Speed stay out of this interface until Boulder is excellent.",
        )
        names = sorted(athletes["athlete_name"].dropna().unique(), key=str.casefold)
        selected: list[str]
        if mode == "Compare 3":
            columns = st.columns(3)
            names_by_key = {plain_key(name): name for name in names}
            defaults = [
                names_by_key[plain_key(name)]
                for name in DEFAULT_ATHLETES
                if plain_key(name) in names_by_key
            ]
            while len(defaults) < 3 and names:
                candidate = names[min(len(defaults), len(names) - 1)]
                if candidate not in defaults:
                    defaults.append(candidate)
                else:
                    break
            selected = []
            for index, column in enumerate(columns):
                default = defaults[index] if index < len(defaults) else names[0]
                selected.append(column.selectbox(
                    "Main athlete" if index == 0 else f"Comparison {index + 1}",
                    names,
                    index=names.index(default),
                    format_func=friendly_name,
                    key=f"athlete_{index}",
                ))
        else:
            preset = roster_names(mode, athletes, history, rosters)
            matched = athletes.loc[athletes["name_key"].isin({plain_key(name) for name in preset}), "athlete_name"].dropna().unique().tolist()
            selected = st.multiselect(
                f"{mode} athletes",
                names,
                default=matched,
                help="All matched members start selected. Uncheck any athlete to simplify an individual graph.",
            )
            if mode == "Canadian National Team proxy":
                st.caption("Proxy only: current CNR top 15 by gender. Replace with the official roster when supplied.")
        return selected, discipline


def render_rating_detail(
    athletes: pd.DataFrame,
    history: pd.DataFrame,
    selected: list[str],
    calibration: pd.DataFrame,
) -> None:
    st.subheader("Compared athletes · all rating evidence")
    focus = selected_rows(athletes, selected)
    if focus.empty:
        st.caption("No matched rating evidence. Athletes with no competition yet remain in the roster.")
        return
    focus["selection_order"] = focus["name_key"].map({
        plain_key(name): index for index, name in enumerate(selected)
    })
    focus = focus.sort_values("selection_order")
    rows = []
    for athlete_index, (_, athlete) in enumerate(focus.iterrows()):
        for family in ALL_RATINGS:
            value = pd.to_numeric(pd.Series([athlete.get(family)]), errors="coerce").iloc[0]
            if not np.isfinite(value):
                continue
            target = rating_target(family)
            probability, fit = outcome_probability(
                value, calibration, target, athlete.get("pool", "Boulder_All")
            )
            target_label = {
                "semifinal": "Make semifinal", "final": "Make final",
                "podium": "Make podium",
            }[target]
            estimate = "Not enough validation data"
            if np.isfinite(probability):
                estimate = f"{target_label}: {probability:.0%}"
                if np.isfinite(fit):
                    estimate += f" (fit {fit:.0%})"
            rows.append({
                "Athlete": friendly_name(athlete["athlete_name"]),
                "Rating family": family,
                "Elo": round(float(value)),
                "Included rounds": athlete.get(f"{family} evidence"),
                "Historical outcome estimate": estimate,
            })
    detail = pd.DataFrame(rows)
    if detail.empty:
        st.info("These athletes are kept in the roster but do not yet have competition evidence.")
        return
    legend = " · ".join(
        f"<span style='color:{ATHLETE_COLORS[i % len(ATHLETE_COLORS)]}'>●</span> {friendly_name(name)}"
        for i, name in enumerate(selected)
        if plain_key(name) in set(focus["name_key"])
    )
    st.markdown(legend, unsafe_allow_html=True)
    st.dataframe(
        detail,
        hide_index=True,
        width="stretch",
        height=min(720, 38 * len(detail) + 40),
        column_config={
            "Elo": st.column_config.NumberColumn(format="%d"),
            "Included rounds": st.column_config.NumberColumn(format="%d"),
            "Historical outcome estimate": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(
        "Fit is McFadden pseudo-R² for the 2025 outcome curve: how much Elo improves "
        "that outcome model over using only the average success rate. It is not the "
        "athlete's certainty and it is not ordinary R²."
    )
    if not history.empty:
        ids = focus[["pool", "global_id", "athlete_name"]]
        recent = history.merge(ids, on=["pool", "global_id"], how="inner", suffixes=("", "_selected"))
        recent["event_date"] = pd.to_datetime(recent["event_date"], errors="coerce")
        recent = (
            recent.sort_values("event_date", ascending=False)
            .groupby(["pool", "global_id"], as_index=False, group_keys=False)
            .head(3)
        )
        if not recent.empty:
            recent["Athlete"] = recent["athlete_name_selected"].map(friendly_name)
            latest = recent[[
                "Athlete", "event_date", "event_name", "round_group",
                "confirmed_procedure", "performance_elo",
            ]].rename(columns={
                "event_date": "Event date", "event_name": "Competition",
                "round_group": "Round", "confirmed_procedure": "Procedure",
                "performance_elo": "Performance-ELO",
            })
            st.markdown("#### Latest isolated round performances")
            st.dataframe(
                latest, hide_index=True, width="stretch",
                column_config={
                    "Event date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    "Performance-ELO": st.column_config.NumberColumn(format="%d"),
                },
            )


def startup_status(data: dict[str, pd.DataFrame]) -> None:
    missing = [
        key for key in ("athletes", "history")
        if data[key].empty
    ]
    with st.expander("Data health", expanded=bool(missing)):
        if missing:
            st.error(
                "The overview is in safe fallback mode. Missing artifact(s): "
                + ", ".join(missing)
                + ". Run `python scripts/build_boulder_rating_families.py` before deployment."
            )
        else:
            latest = pd.to_datetime(data["history"]["event_date"], errors="coerce").max()
            st.success(
                f"Ready · {len(data['athletes']):,} Boulder athletes · results through {latest:%Y-%m-%d}."
            )
            calibration = data.get("calibration", pd.DataFrame())
            if not calibration.empty:
                combined = calibration.loc[calibration["pool"].eq("Boulder_All")]
                counted = combined if not combined.empty else calibration
                starts = int(counted["qualification_starts"].sum())
                events = int(counted["events"].sum())
                st.caption(
                    f"Display scale checked: 2000 is the fitted 50% semifinal level "
                    f"from {starts:,} pre-event athlete-starts across {events} "
                    "gender-pool World Cup samples in 2025."
                )
        st.caption("Specialist ratings appear only when the athlete has enough matching round evidence.")


def physical_transfer_figure(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    selected: list[str] | None = None,
) -> tuple[go.Figure, float, dict[str, float]]:
    """Plot Bayesian test-to-rating expectation and athlete transfer probabilities."""
    plot = frame.dropna(subset=[x, y]).copy()
    plot[x] = pd.to_numeric(plot[x], errors="coerce")
    plot[y] = pd.to_numeric(plot[y], errors="coerce")
    plot = plot.dropna(subset=[x, y])
    plot["name_key"] = plot["athlete_name"].map(plain_key)
    rho = float(plot[x].rank().corr(plot[y].rank())) if len(plot) >= 3 else np.nan
    figure = go.Figure()
    if plot.empty:
        return figure, rho, {
            "probability_positive": np.nan, "slope_low": np.nan,
            "slope_high": np.nan, "draws": 0,
        }

    # Conjugate Bayesian linear regression on standardized values. The weak
    # Normal-inverse-gamma prior prevents tiny samples from looking certain,
    # while preserving their direction instead of applying a binary gate.
    x_mean, x_sd = float(plot[x].mean()), float(plot[x].std(ddof=0))
    y_mean, y_sd = float(plot[y].mean()), float(plot[y].std(ddof=0))
    x_sd = x_sd if np.isfinite(x_sd) and x_sd > 0 else 1.0
    y_sd = y_sd if np.isfinite(y_sd) and y_sd > 0 else 1.0
    x_standard = (plot[x].to_numpy(float) - x_mean) / x_sd
    y_standard = (plot[y].to_numpy(float) - y_mean) / y_sd
    design = np.column_stack([np.ones(len(plot)), x_standard])
    prior_precision = np.diag([0.01, 0.04])
    posterior_precision = prior_precision + design.T @ design
    posterior_covariance = np.linalg.inv(posterior_precision)
    posterior_mean = posterior_covariance @ design.T @ y_standard
    prior_shape, prior_scale = 2.0, 1.0
    posterior_shape = prior_shape + len(plot) / 2
    scale_term = float(
        y_standard @ y_standard
        - posterior_mean @ posterior_precision @ posterior_mean
    )
    posterior_scale = max(1e-6, prior_scale + 0.5 * scale_term)
    rng = np.random.default_rng(20260803 + len(plot))
    draw_count = 1200
    precision_draws = rng.gamma(
        shape=posterior_shape, scale=1.0 / posterior_scale, size=draw_count
    )
    variance_draws = 1.0 / precision_draws
    normal_draws = rng.normal(size=(draw_count, 2))
    covariance_root = np.linalg.cholesky(posterior_covariance)
    beta_draws = posterior_mean + (
        normal_draws @ covariance_root.T
    ) * np.sqrt(variance_draws)[:, None]
    mean_draws = (design @ beta_draws.T).T * y_sd + y_mean
    fitted_mean = mean_draws.mean(axis=0)
    slope_draws = beta_draws[:, 1] * y_sd / x_sd
    probability_positive = float(np.mean(slope_draws > 0))
    slope_low, slope_high = np.quantile(slope_draws, [0.05, 0.95])

    plot["test_expected_rating"] = fitted_mean
    plot["transfer_residual"] = plot[y] - plot["test_expected_rating"]
    median_residual = float(plot["transfer_residual"].median())
    mad = float((plot["transfer_residual"] - median_residual).abs().median())
    residual_scale = 1.4826 * mad
    if not np.isfinite(residual_scale) or residual_scale < 1:
        residual_scale = float(plot["transfer_residual"].std(ddof=1))
    practical_margin = max(35.0, 0.35 * residual_scale) if np.isfinite(residual_scale) else 35.0
    actual = plot[y].to_numpy(float)[None, :]
    probability_opportunity = np.mean(actual > mean_draws + practical_margin, axis=0)
    probability_lower_transfer = np.mean(actual < mean_draws - practical_margin, axis=0)
    # A residual is only meaningful for training direction to the extent that
    # the population relationship itself is credibly positive.
    probability_opportunity *= probability_positive
    probability_lower_transfer *= probability_positive
    plot["probability_opportunity"] = probability_opportunity
    plot["probability_lower_transfer"] = probability_lower_transfer
    plot["posterior_direction_confidence"] = np.maximum(
        probability_opportunity, probability_lower_transfer
    )
    evidence_column = f"{y} evidence"
    plot["rating_rounds"] = pd.to_numeric(
        plot.get(evidence_column, pd.Series(0, index=plot.index)), errors="coerce"
    ).fillna(0).clip(lower=0)
    evidence_ceiling = float(plot["rating_rounds"].quantile(0.90))
    if evidence_ceiling > 0:
        plot["evidence_opacity"] = (
            0.30
            + 0.65
            * np.log1p(plot["rating_rounds"])
            / np.log1p(evidence_ceiling)
        ).clip(0.30, 0.95)
    else:
        plot["evidence_opacity"] = 0.72
    plot["transfer_reading"] = "Direction uncertain"
    plot.loc[
        plot["probability_opportunity"].gt(plot["probability_lower_transfer"]),
        "transfer_reading",
    ] = "Possible opportunity: performance ahead of test"
    plot.loc[
        plot["probability_lower_transfer"].gt(plot["probability_opportunity"]),
        "transfer_reading",
    ] = "Possible lower transfer: test ahead of performance"

    colors = {
        "Possible opportunity: performance ahead of test": PALETTE["coral"],
        "Possible lower transfer: test ahead of performance": PALETTE["blue"],
        "Direction uncertain": "#A2B5B1",
    }
    selected_keys = {plain_key(name) for name in (selected or [])}
    for reading, group in plot.groupby("transfer_reading", sort=False):
        labels = [
            friendly_name(name) if key in selected_keys else ""
            for name, key in zip(group["athlete_name"], group["name_key"])
        ]
        line_widths = [3 if key in selected_keys else 0 for key in group["name_key"]]
        opacities = group["evidence_opacity"].to_numpy(float)
        figure.add_trace(go.Scatter(
            x=group[x], y=group[y], mode="markers+text", text=labels,
            textposition="top center", name=reading,
            marker={
                "size": 11, "opacity": opacities, "color": colors.get(reading, "#A2B5B1"),
                "line": {"width": line_widths, "color": PALETTE["ink"]},
            },
            customdata=np.column_stack([
                group["athlete_name"], group["test_expected_rating"],
                group["transfer_residual"], 100 * group["probability_opportunity"],
                100 * group["probability_lower_transfer"], group["rating_rounds"],
            ]),
            hovertemplate=(
                "%{customdata[0]}<br>Test result: %{x:.2f}<br>Actual rating: %{y:.0f}"
                "<br>Rating expected from this test alone: %{customdata[1]:.0f}"
                "<br>Difference: %{customdata[2]:+.0f} Elo"
                "<br>Posterior P(physical opportunity): %{customdata[3]:.0f}%"
                "<br>Posterior P(lower transfer): %{customdata[4]:.0f}%"
                "<br>Included rating rounds: %{customdata[5]:.0f}"
                "<extra>%{fullData.name}</extra>"
            ),
        ))
    x_line = np.array([float(plot[x].min()), float(plot[x].max())])
    line_design = np.column_stack([np.ones(2), (x_line - x_mean) / x_sd])
    line_draws = (line_design @ beta_draws.T).T * y_sd + y_mean
    line_mean = line_draws.mean(axis=0)
    line_low, line_high = np.quantile(line_draws, [0.05, 0.95], axis=0)
    figure.add_trace(go.Scatter(
        x=np.concatenate([x_line, x_line[::-1]]),
        y=np.concatenate([line_high, line_low[::-1]]),
        fill="toself", fillcolor=transparent(PALETTE["ink"], 0.10),
        line={"color": "rgba(0,0,0,0)"}, hoverinfo="skip",
        name="90% credible interval",
    ))
    figure.add_trace(go.Scatter(
        x=x_line, y=line_mean, mode="lines",
        line={"color": PALETTE["ink"], "width": 2, "dash": "dash"},
        name="Rating expected from this test alone",
        hoverinfo="skip",
    ))
    figure.update_layout(
        title=title,
        xaxis_title=x.replace("_", " ").title(), yaxis_title=y,
        height=590, legend_title="Test-to-performance reading",
        margin={"l": 20, "r": 20, "t": 70, "b": 30},
    )
    return figure, rho, {
        "probability_positive": probability_positive,
        "slope_low": float(slope_low), "slope_high": float(slope_high),
        "draws": draw_count,
    }


def render_physical_strength(
    athletes: pd.DataFrame, selected: list[str], data: dict[str, pd.DataFrame]
) -> None:
    st.header("Physical Strength and Training Priorities")
    st.write(
        "Use the testing database for two decisions: which qualities deserve attention "
        "across the pathway, and which qualities are plausible priorities for one athlete."
    )
    view = st.segmented_control(
        "Physical analysis",
        ["Population priorities", "Athlete focus", "Explore all tests", "Climbing grades"],
        default="Population priorities", label_visibility="collapsed",
    )
    latest = data.get("physical_latest", pd.DataFrame())
    screen = data.get("physical_screen", pd.DataFrame())
    priorities = data.get("physical_priorities", pd.DataFrame())
    if not latest.empty:
        latest = latest.copy()
        latest["name_key"] = latest["athlete_name"].map(plain_key)
        evidence_columns = [
            column for column in [f"{family} evidence" for family in RATING_ORDER]
            if column in athletes
        ]
        if evidence_columns:
            evidence_lookup = (
                athletes[["pool", "name_key", *evidence_columns]]
                .sort_values(evidence_columns[0], ascending=False)
                .drop_duplicates(["pool", "name_key"])
            )
            latest = latest.merge(
                evidence_lookup, on=["pool", "name_key"], how="left"
            )

    if view == "Population priorities":
        if screen.empty:
            st.info("The all-test future-performance screen is being rebuilt.")
            return
        evidence = screen.copy()
        evidence["context_only"] = evidence["context_only"].astype(str).str.lower().eq("true")
        evidence = evidence.loc[~evidence["context_only"]].copy()
        evidence["future_adjusted_spearman"] = pd.to_numeric(
            evidence["future_adjusted_spearman"], errors="coerce"
        )
        evidence["future_bootstrap_low"] = pd.to_numeric(
            evidence["future_bootstrap_low"], errors="coerce"
        )
        evidence["future_bootstrap_high"] = pd.to_numeric(
            evidence["future_bootstrap_high"], errors="coerce"
        )
        evidence = evidence.sort_values(
            ["priority_weight", "future_adjusted_spearman"], ascending=False
        )
        promising = evidence.loc[evidence["evidence_tier"].eq("Promising predictive signal")]
        exploratory = evidence.loc[evidence["evidence_tier"].eq("Exploratory signal")]
        cards = st.columns(3)
        cards[0].metric("Tests screened", f"{len(evidence)}")
        cards[1].metric("Promising signals", f"{len(promising)}")
        cards[2].metric("Exploratory signals", f"{len(exploratory)}")
        if promising.empty:
            st.warning(
                "No physical test yet clears the full predictive gate. The highlighted "
                "tests are assessment priorities—not proven training prescriptions."
            )
        plot = evidence.dropna(subset=["future_adjusted_spearman"]).head(14).sort_values(
            "future_adjusted_spearman"
        )
        if not plot.empty:
            tier_colors = {
                "Promising predictive signal": PALETTE["teal"],
                "Exploratory signal": PALETTE["gold"],
                "Measured but unsettled": "#A8B6B3",
                "Insufficient future evidence": "#D9E0DE",
            }
            figure = go.Figure()
            for tier, group in plot.groupby("evidence_tier", sort=False):
                low = group["future_adjusted_spearman"] - group["future_bootstrap_low"]
                high = group["future_bootstrap_high"] - group["future_adjusted_spearman"]
                figure.add_trace(go.Scatter(
                    x=group["future_adjusted_spearman"], y=group["test_name"],
                    mode="markers", name=tier,
                    marker={"size": 11, "color": tier_colors.get(tier, "#A8B6B3")},
                    error_x={
                        "type": "data", "symmetric": False,
                        "array": high.clip(lower=0), "arrayminus": low.clip(lower=0),
                        "color": "#94A4A1", "thickness": 1,
                    },
                    customdata=np.column_stack([
                        group["future_athletes"], group["mae_improvement"],
                        group["metric_category"],
                    ]),
                    hovertemplate=(
                        "%{y}<br>Adjusted future relationship: %{x:.2f}"
                        "<br>Athletes: %{customdata[0]:.0f}"
                        "<br>Prediction MAE improvement: %{customdata[1]:.1f} Elo"
                        "<br>%{customdata[2]}<extra></extra>"
                    ),
                ))
            figure.add_vline(x=0, line_dash="dot", line_color="#82918E")
            figure.update_layout(
                title="Does the test add information about later competition performance?",
                xaxis_title="Age-, gender- and pre-event-Elo-adjusted rank relationship",
                yaxis_title="", height=max(520, 34 * len(plot)),
                legend_title="Evidence level", margin={"l": 20, "r": 20, "t": 70, "b": 30},
            )
            st.plotly_chart(figure, width="stretch", theme=None)
        st.caption(
            "Positive means higher test results tended to precede stronger Performance-ELO "
            "than pre-event Elo, age and gender alone expected. The line is the athlete-level "
            "bootstrap interval. Wide lines mean the database cannot yet separate signal from noise."
        )
        table = evidence[[
            "test_name", "metric_category", "future_athletes",
            "future_adjusted_spearman", "mae_improvement", "evidence_tier",
            "coach_reading",
        ]].rename(columns={
            "test_name": "Test", "metric_category": "Quality",
            "future_athletes": "Athletes with later results",
            "future_adjusted_spearman": "Added future signal",
            "mae_improvement": "Prediction error reduced (Elo)",
            "evidence_tier": "Evidence", "coach_reading": "Coaching use",
        })
        st.dataframe(table, hide_index=True, width="stretch")
        return

    if view == "Athlete focus":
        if priorities.empty:
            st.info("Athlete testing priorities are being rebuilt.")
            return
        priorities = priorities.copy()
        tested_options = sorted(
            priorities["athlete_name"].dropna().astype(str).unique(), key=str.casefold
        )
        options: list[str] = []
        seen_keys: set[str] = set()
        for name in [*selected, *tested_options]:
            key = plain_key(name)
            if key and key not in seen_keys:
                options.append(friendly_name(name))
                seen_keys.add(key)
        preferred_key = plain_key(selected[0]) if selected else plain_key(options[0])
        default_index = next(
            (index for index, option in enumerate(options) if plain_key(option) == preferred_key), 0
        )
        athlete_name = st.selectbox("Athlete", options, index=default_index)
        athlete = priorities.loc[
            priorities["athlete_name"].map(plain_key).eq(plain_key(athlete_name))
        ].copy()
        if athlete.empty:
            st.info(
                f"No physical-testing record is linked to {friendly_name(athlete_name)} yet. "
                "Choose another athlete or add a testing session before drawing a physical profile."
            )
            return
        physical = athlete.loc[~athlete["recommendation"].astype(str).str.startswith("Context")].copy()
        physical["peer_percentile"] = pd.to_numeric(physical["peer_percentile"], errors="coerce")
        physical["priority_score"] = pd.to_numeric(physical["priority_score"], errors="coerce")
        focus = physical.loc[physical["recommendation"].eq("Focus candidate")]
        strengths = physical.loc[physical["recommendation"].eq("Strength to protect")]
        cards = st.columns(3)
        cards[0].metric("Physical tests", f"{len(physical)}")
        cards[1].metric("Focus candidates", f"{len(focus)}")
        cards[2].metric("Strengths to protect", f"{len(strengths)}")
        if focus.empty:
            st.info(
                "No clear physical limiter can be defended from the current data for this "
                "athlete. Use the test history and climbing observations before changing training."
            )
        else:
            first = focus.sort_values("priority_score", ascending=False).iloc[0]
            st.success(
                f"First quality to investigate: {first['test_name']} ({first['metric_category']}). "
                f"This is a {first['certainty'].lower()} hypothesis, not a diagnosis."
            )
        ordered = pd.concat([
            focus.sort_values("priority_score", ascending=False),
            strengths.sort_values("peer_percentile", ascending=False),
            physical.loc[~physical.index.isin(focus.index.union(strengths.index))]
            .sort_values("test_date", ascending=False),
        ]).drop_duplicates("test_name").head(16).sort_values("peer_percentile")
        if not ordered.empty:
            recommendation_colors = {
                "Focus candidate": PALETTE["coral"],
                "Strength to protect": PALETTE["teal"],
                "Monitor; not a clear limiter": PALETTE["gold"],
                "Evidence too uncertain to prescribe": "#A8B6B3",
            }
            figure = go.Figure()
            for label, group in ordered.groupby("recommendation", sort=False):
                figure.add_trace(go.Bar(
                    x=100 * group["peer_percentile"], y=group["test_name"],
                    orientation="h", name=label,
                    marker_color=recommendation_colors.get(label, "#A8B6B3"),
                    customdata=np.column_stack([
                        group["value"], group["unit"], group["peer_athletes"],
                        group["test_date"], group["certainty"],
                    ]),
                    hovertemplate=(
                        "%{y}<br>Peer percentile: %{x:.0f}"
                        "<br>Result: %{customdata[0]} %{customdata[1]}"
                        "<br>Peer athletes: %{customdata[2]}"
                        "<br>Test date: %{customdata[3]}"
                        "<br>%{customdata[4]}<extra></extra>"
                    ),
                ))
            figure.add_vline(x=35, line_dash="dot", line_color=PALETTE["coral"])
            figure.add_vline(x=70, line_dash="dot", line_color=PALETTE["teal"])
            figure.update_layout(
                title=f"{friendly_name(athlete_name)} — physical profile against comparable peers",
                xaxis={"title": "Percentile among same-gender peers (age ±2 years when available)",
                       "range": [0, 100]},
                yaxis_title="", barmode="overlay", height=max(520, 34 * len(ordered)),
                margin={"l": 20, "r": 20, "t": 70, "b": 30},
            )
            st.plotly_chart(figure, width="stretch", theme=None)
        shown = physical[[
            "test_name", "metric_category", "test_date", "value", "unit",
            "peer_percentile", "peer_athletes", "recommendation", "certainty",
        ]].copy()
        shown["peer_percentile"] = (100 * shown["peer_percentile"]).round(0)
        shown = shown.rename(columns={
            "test_name": "Test", "metric_category": "Quality", "test_date": "Date",
            "value": "Result", "unit": "Unit", "peer_percentile": "Peer percentile",
            "peer_athletes": "Peer athletes", "recommendation": "Decision",
            "certainty": "Certainty",
        })
        st.dataframe(shown, hide_index=True, width="stretch")
        st.caption(
            "A focus candidate requires both a low peer result and a positive population signal. "
            "It should be checked against movement style, training history, injury risk and response "
            "to training before becoming a program priority."
        )
        return

    if view == "Explore all tests":
        if latest.empty:
            st.info("The all-test explorer is being rebuilt.")
            return
        latest = latest.copy()
        latest["value"] = pd.to_numeric(latest["value"], errors="coerce")
        tests = sorted(latest["test_name"].dropna().astype(str).unique(), key=str.casefold)
        test = st.selectbox("Test", tests)
        rating = st.selectbox("Rating", ["Global-ELO", "IFSC-ELO", "WR-ELO"])
        plot = latest.loc[latest["test_name"].eq(test)].copy()
        plot[rating] = pd.to_numeric(plot[rating], errors="coerce")
        plot = plot.dropna(subset=["value", rating])
        if plot.empty:
            st.info("No linked competition rating is available for this test yet.")
        else:
            figure, current_rho, transfer_evidence = physical_transfer_figure(
                plot, "value", rating, f"{test} and current {rating}", selected
            )
            st.plotly_chart(figure, width="stretch", theme=None)
            st.info(
                f"Current rank relationship: {current_rho:.2f}. Bayesian probability that the "
                f"population relationship is positive: "
                f"{100 * transfer_evidence['probability_positive']:.0f}% "
                f"(90% slope interval {transfer_evidence['slope_low']:.1f} to "
                f"{transfer_evidence['slope_high']:.1f} Elo per test unit). Athlete hover shows "
                "the posterior probability of each direction; marker density shows how many "
                "rounds support the displayed Elo."
            )
            st.caption(
                f"{plot['testing_person_key'].nunique()} linked athletes. Residual tags are screening "
                "hypotheses, not causal training prescriptions. Use Population priorities for the "
                "harder frozen future-performance test."
            )
        coverage = latest.groupby(["metric_category", "test_name"], as_index=False).agg(
            athletes=("testing_person_key", "nunique"),
            latest_date=("test_date", "max"),
        ).sort_values(["metric_category", "athletes"], ascending=[True, False])
        st.dataframe(coverage, hide_index=True, width="stretch")
        return

    # Self-reported climbing grades remain useful context, but are not physical
    # qualities and therefore never become training prescriptions by themselves.
    profiles = data.get("physical_profiles", pd.DataFrame())
    associations = data.get("physical_associations", pd.DataFrame())
    models = data.get("physical_models", pd.DataFrame())
    if profiles.empty:
        st.info(
            "The governed physical-testing artifact is being rebuilt. Athlete names remain "
            "available even when they have no test or competition evidence yet."
        )
        return
    profiles = profiles.copy()
    profiles["name_key"] = profiles["athlete_name"].map(plain_key)
    rating_columns = [
        "pool", "name_key", "Global-ELO", "IFSC-ELO", "WR-ELO",
        "Global-ELO evidence", "IFSC-ELO evidence", "WR-ELO evidence",
        "gender", "age",
    ]
    athlete_ratings = athletes[[
        column for column in rating_columns
        if column in athletes
    ]].copy()
    athlete_ratings = athlete_ratings.sort_values(
        "Global-ELO evidence" if "Global-ELO evidence" in athlete_ratings else "Global-ELO",
        ascending=False,
    ).drop_duplicates(["pool", "name_key"])
    profiles = profiles.merge(
        athlete_ratings,
        on=["pool", "name_key"], how="left",
    )
    tests = {
        "Self-reported 50% flash grade": "boulder_grade_50pct_flash_v",
        "Self-reported max grade in ≤3 physical sends (last 3 months)":
            "boulder_grade_3x_physical_sends_last_3_months_v",
    }
    test = st.selectbox("Test or self-reported grade", list(tests))
    value_column = tests[test]
    rating_candidates = [family for family in RATING_ORDER if family in profiles]
    rating = st.selectbox("Rating to explain", rating_candidates or ["Global-ELO"])
    plot = profiles.copy()
    if rating in plot and value_column in plot:
        plot[value_column] = pd.to_numeric(plot[value_column], errors="coerce")
        plot[rating] = pd.to_numeric(plot[rating], errors="coerce")
        plot = plot.dropna(subset=[value_column, rating])
        figure, grade_rho, grade_transfer_evidence = physical_transfer_figure(
            plot, value_column, rating, f"{test} and {rating}", selected
        )
        st.plotly_chart(figure, width="stretch", theme=None)
        st.info(
            f"Current rank relationship: {grade_rho:.2f}. Bayesian probability that the "
            f"population relationship is positive: "
            f"{100 * grade_transfer_evidence['probability_positive']:.0f}% "
            f"(90% slope interval {grade_transfer_evidence['slope_low']:.1f} to "
            f"{grade_transfer_evidence['slope_high']:.1f} Elo per grade). The dashed line and "
            "athlete probabilities identify questions to investigate, not proof of a limiter."
        )
        st.caption(
            f"{len(plot)} linked athletes are visible for this exact grade × rating pair. "
            "Grade values are converted to the common V-scale used in the testing source."
        )
    if not associations.empty:
        shown = associations.copy()
        grade_label = (
            "Boulder Grade 50% Flash" if value_column == "boulder_grade_50pct_flash_v"
            else "Boulder Grade 3x physical sends last 3 months"
        )
        shown = shown.loc[shown["grade_metric"].eq(grade_label)]
        elo_label = {
            "Global-ELO": "Current Global quality Elo",
            "IFSC-ELO": "Current production IFSC Elo",
            "WR-ELO": "Current WR Elo",
        }.get(rating)
        if elo_label and "elo_metric" in shown:
            shown = shown.loc[shown["elo_metric"].eq(elo_label)]
        st.markdown("#### Tested relationships")
        st.dataframe(shown, hide_index=True, width="stretch")
    if not models.empty:
        st.markdown("#### Combined models")
        st.dataframe(models, hide_index=True, width="stretch")
        st.caption(
            "Cross-validated means the model is judged on athletes it did not fit. "
            "A negative validation R² means the simple average would have predicted better."
        )


def render_maths_behind(
    athletes: pd.DataFrame, correlations: pd.DataFrame, calibration: pd.DataFrame,
    data: dict[str, pd.DataFrame],
) -> None:
    st.header("Maths behind")
    st.caption("What each model is for, what it can predict, and where it can fail.")
    comparison = pd.DataFrame([
        {
            "Model": "Global-ELO", "Best use": "Overall Open World-Cup readiness",
            "Evidence": "Local, national, youth and international rounds",
            "Strength": "Fast learning for athletes with little IFSC exposure",
            "Main caveat": "Local terrain and field strength must transport correctly",
        },
        {
            "Model": "IFSC-ELO", "Best use": "IFSC-specific readiness",
            "Evidence": "Non-para IFSC rounds",
            "Strength": "Closer competition environment and setting",
            "Main caveat": "Can lag for new or transformed athletes",
        },
        {
            "Model": "WR-ELO", "Best use": "World Ranking event performance",
            "Evidence": "Only WR-point events",
            "Strength": "Most specific to ranking access",
            "Main caveat": "Sparse and participation-selected evidence",
        },
        {
            "Model": "Performance-ELO", "Best use": "Describe one round",
            "Evidence": "One frozen pre-event field and observed result",
            "Strength": "Makes surprise and momentum visible",
            "Main caveat": "One round also contains terrain fit and ordinary noise",
        },
    ])
    st.dataframe(comparison, hide_index=True, width="stretch")
    backtest = data.get("model_backtest", pd.DataFrame())
    if not backtest.empty:
        st.markdown("#### Frozen next-competition backtest")
        st.dataframe(backtest, hide_index=True, width="stretch")
        st.caption(
            "Frozen means every prediction uses only information available before that "
            "competition. Higher rank correlation is better; lower log-loss and Brier "
            "error are better. This is the promotion test, not post-event fit."
        )
    sensitivity = data.get("sensitivity_metrics", pd.DataFrame())
    if not sensitivity.empty:
        boulder = sensitivity.loc[
            sensitivity.get("pool", pd.Series("", index=sensitivity.index))
            .astype(str).str.startswith("Boulder")
        ]
        if not boulder.empty:
            st.markdown("#### Faster-response challenger")
            st.dataframe(boulder, hide_index=True, width="stretch")
            st.caption(
                "The latent-volatility challenger reacts more to persistent Performance-ELO "
                "surprises. It slightly improved some ordering metrics but failed the "
                "predeclared probability-error gate, so it is research evidence—not production Elo."
            )
    if not calibration.empty:
        combined = calibration.loc[calibration["pool"].eq("Boulder_All")]
        shown = combined if not combined.empty else calibration.head(1)
        columns = [column for column in shown if (
            column.startswith("display_elo_at_50pct_") or column.endswith("mcfadden_pseudo_r2")
        )]
        if columns:
            st.markdown("#### 2025 outcome calibration")
            st.dataframe(shown[["pool", *columns]], hide_index=True, width="stretch")
    st.markdown("#### Why a large Performance-ELO surprise does not get full weight immediately")
    st.write(
        "A repeated surprise across independent competitions is stronger evidence of real "
        "change than several rounds at the same event. The production Elo stays zero-sum "
        "inside each event and lets uncertainty fall with evidence. A faster current-shape "
        "layer should be adopted only if it improves frozen next-event probability forecasts, "
        "not just because it follows the latest result more closely."
    )
    st.caption(correlation_note(correlations, "Global-ELO"))


def render_style_tagging(history: pd.DataFrame) -> None:
    """Public, structured boulder-demand tagging prototype.

    Community Cloud's local disk is temporary, so records are kept in the
    visitor's session and exported as a portable CSV/JSON bundle. This makes
    the schema usable now without pretending that an ephemeral file is a
    durable crowd database.
    """
    st.header("Boulder Style Tagging")
    st.write(
        "Turn a boulder image and a coach's reading into structured evidence. "
        "Score the visible demand—not the athlete who climbed it."
    )
    with st.expander("How to score 0–3", expanded=False):
        st.markdown(
            "- **0 — absent:** the demand is not meaningfully present.\n"
            "- **1 — present:** useful, but not a defining challenge.\n"
            "- **2 — important:** likely changes who succeeds.\n"
            "- **3 — dominant:** central to solving or executing the boulder.\n\n"
            "**Physical** is decomposed into explosiveness, body tension and "
            "overall strength. **Technical** is decomposed into slow precision, "
            "curved coordination, reaction time and proprioception. The separate "
            "**coordination** score describes how strongly linked movement timing "
            "and momentum determine success."
        )

    event_names: list[str] = []
    if not history.empty and "event_name" in history:
        dated = history.copy()
        dated["event_date"] = pd.to_datetime(dated["event_date"], errors="coerce")
        event_names = (
            dated.sort_values("event_date", ascending=False)["event_name"]
            .dropna().astype(str).drop_duplicates().head(80).tolist()
        )
    event_options = ["Custom competition", *event_names]

    image_file = st.file_uploader(
        "Boulder image", type=["png", "jpg", "jpeg", "webp"],
        help="Use a full-wall or close route image where the hold sequence is readable.",
    )
    if image_file is not None:
        st.image(image_file, caption=image_file.name, width="stretch")

    with st.form("style_tag_form", clear_on_submit=False):
        top = st.columns(4)
        event_choice = top[0].selectbox("Competition", event_options)
        custom_event = top[0].text_input(
            "Competition name", disabled=event_choice != "Custom competition"
        )
        round_name = top[1].selectbox(
            "Round", ["Qualification", "Semi-final", "Final", "Other"]
        )
        gender = top[2].selectbox("Gender terrain", ["Men", "Women", "Mixed / unknown"])
        boulder_number = top[3].text_input("Boulder", placeholder="e.g. M3 or W2")

        st.markdown("#### Main style profile")
        main = st.columns(4)
        physical = main[0].slider("Physical", 0, 3, 1)
        technical = main[1].slider("Technical", 0, 3, 1)
        coordination = main[2].slider("Coordination", 0, 3, 1)
        verticality = main[3].select_slider(
            "Wall angle", options=[0, 1, 2, 3], value=1,
            help="0 slab · 1 vertical · 2 overhang · 3 roof / very steep",
        )

        st.markdown("#### Why you gave those scores")
        physical_cols = st.columns(3)
        explosion = physical_cols[0].slider("Explosiveness", 0, 3, 0)
        tension = physical_cols[1].slider("Body tension", 0, 3, 0)
        strength = physical_cols[2].slider("Overall strength", 0, 3, 0)
        technical_cols = st.columns(4)
        slow_precision = technical_cols[0].slider("Slow precision", 0, 3, 0)
        curved_coordination = technical_cols[1].slider("Curved coordination", 0, 3, 0)
        reaction_time = technical_cols[2].slider("Reaction time", 0, 3, 0)
        proprioception = technical_cols[3].slider("Proprioception", 0, 3, 0)

        st.markdown("#### Movement sequence around the zone")
        sequence = st.columns(3)
        pre_zone = sequence[0].text_area(
            "Up to 3 moves before zone", placeholder="1. …\n2. …\n3. …"
        )
        zone_move = sequence[1].text_area(
            "Zone move / position", placeholder="How the zone is controlled"
        )
        post_zone = sequence[2].text_area(
            "Up to 3 moves after zone", placeholder="1. …\n2. …\n3. …"
        )
        tags = st.multiselect(
            "Movement and setting tags",
            [
                "Dyno", "Paddle", "Run-and-jump", "Deadpoint", "Toe hook",
                "Heel hook", "Knee bar", "Compression", "Press", "Mantle",
                "Rose move", "Drop-knee", "Flag", "Smear", "Volume walking",
                "Swing control", "Foot-first", "Coordination catch",
                "Low-percentage crux", "Complex beta", "Precision feet",
            ],
        )
        notes = st.text_area(
            "Coach notes (optional)",
            placeholder="Likely beta, alternative beta, deceptive feature, or uncertainty…",
        )
        confidence = st.select_slider(
            "Tag confidence", options=["Low", "Moderate", "High"], value="Moderate"
        )
        contributor = st.text_input("Contributor name or initials (optional)")
        submitted = st.form_submit_button("Add tag to this session", type="primary")

    if "style_tag_rows" not in st.session_state:
        st.session_state.style_tag_rows = []
    if "style_tag_images" not in st.session_state:
        st.session_state.style_tag_images = {}
    if submitted:
        final_event = custom_event.strip() if event_choice == "Custom competition" else event_choice
        image_bytes = image_file.getvalue() if image_file is not None else b""
        if len(image_bytes) > 10 * 1024 * 1024:
            st.error("Please use an image smaller than 10 MB.")
        elif not final_event or not boulder_number.strip():
            st.error("Competition and boulder number are required.")
        else:
            image_hash = hashlib.sha256(image_bytes).hexdigest() if image_bytes else ""
            suffix = Path(image_file.name).suffix.lower() if image_file is not None else ""
            stored_image = f"images/{image_hash}{suffix}" if image_hash else ""
            record = {
                "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
                "competition": final_event,
                "round": round_name,
                "gender_terrain": gender,
                "boulder": boulder_number.strip(),
                "image_name": image_file.name if image_file is not None else "",
                "image_sha256": image_hash,
                "image_in_bundle": stored_image,
                "physical_0_3": physical,
                "technical_0_3": technical,
                "coordination_0_3": coordination,
                "verticality_0_3": verticality,
                "explosiveness_0_3": explosion,
                "body_tension_0_3": tension,
                "overall_strength_0_3": strength,
                "slow_precision_0_3": slow_precision,
                "curved_coordination_0_3": curved_coordination,
                "reaction_time_0_3": reaction_time,
                "proprioception_0_3": proprioception,
                "moves_before_zone": pre_zone.strip(),
                "zone_move": zone_move.strip(),
                "moves_after_zone": post_zone.strip(),
                "tags": " | ".join(tags),
                "notes": notes.strip(),
                "confidence": confidence,
                "contributor": contributor.strip(),
            }
            st.session_state.style_tag_rows.append(record)
            if image_bytes and stored_image:
                st.session_state.style_tag_images[stored_image] = image_bytes
            st.success("Tag added. Export the session below to preserve it.")

    records = pd.DataFrame(st.session_state.style_tag_rows)
    if not records.empty:
        st.markdown("#### Tags in this session")
        st.dataframe(records, hide_index=True, width="stretch")
        csv_bytes = records.to_csv(index=False).encode("utf-8")
        bundle = json.dumps(records.to_dict("records"), ensure_ascii=False, indent=2)
        downloads = st.columns(2)
        downloads[0].download_button(
            "Download CSV", csv_bytes, "boulder_style_tags.csv", "text/csv",
        )
        downloads[1].download_button(
            "Download JSON", bundle, "boulder_style_tags.json", "application/json",
        )
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("boulder_style_tags.csv", csv_bytes)
            archive.writestr("boulder_style_tags.json", bundle.encode("utf-8"))
            for image_path, image_bytes in st.session_state.style_tag_images.items():
                archive.writestr(image_path, image_bytes)
        st.download_button(
            "Download complete bundle (tags + images)", zip_buffer.getvalue(),
            "boulder_style_tag_bundle.zip", "application/zip",
        )
    st.caption(
        "Public prototype: tags remain in this browser session until exported. "
        "A shared database will replace session storage after a durable write backend is connected."
    )
    with st.expander("How these tags improve the physical model"):
        st.write(
            "The next model will ask whether a test becomes more predictive as the tagged "
            "demand rises—for example, whether relative finger force matters more on high-"
            "strength or high-tension boulders. That interaction is more useful than asking "
            "whether one physical test predicts every style equally. It requires these tags "
            "plus problem-level tops, zones and attempts, and will be validated on athletes "
            "and competitions the model did not fit."
        )


def style_tag_backend_url() -> str:
    """Return the optional durable-write endpoint without requiring secrets locally."""
    try:
        return str(st.secrets.get("STYLE_TAG_WEBHOOK_URL", "")).strip()
    except (FileNotFoundError, KeyError):
        return ""


def save_style_tag_remotely(
    url: str, record: dict[str, object], image_bytes: bytes
) -> tuple[bool, str]:
    """Write one tag and its optional image to a Google Apps Script endpoint."""
    payload = {
        "record": record,
        "image_base64": base64.b64encode(image_bytes).decode("ascii") if image_bytes else "",
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urlrequest.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urlrequest.urlopen(request, timeout=25) as response:
            result = json.loads(response.read().decode("utf-8"))
        return bool(result.get("ok")), str(result.get("message", "Saved"))
    except (urlerror.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return False, f"Remote save failed: {exc}"


def render_style_tagging_v2(history: pd.DataFrame, standalone: bool = False) -> None:
    """Collect structured Zone/Top demand ratings for a boulder."""
    st.header("Boulder Style Tagging")
    st.write(
        "Score what the boulder demands, not the athlete who climbed it. "
        "Orange always describes the Top section; blue describes the Zone section."
    )
    st.markdown(
        """
        <style>
        div[data-baseweb="select"] > div{min-height:3rem}
        [role="listbox"]{min-width:min(950px,94vw)!important}
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Scoring guide and independent definitions", expanded=False):
        st.markdown(
            "- **0 - absent:** not meaningfully present.\n"
            "- **1 - present:** useful, but not defining.\n"
            "- **2 - important:** likely changes who succeeds.\n"
            "- **3 - dominant:** central to executing the section.\n\n"
            f"**Physical:** {STYLE_DEFINITIONS['physical']}\n\n"
            f"**Technical:** {STYLE_DEFINITIONS['technical']}\n\n"
            f"**Coordination:** {STYLE_DEFINITIONS['coordination']}"
        )

    event_catalog = pd.DataFrame(columns=["event_name", "event_date", "label"])
    if not history.empty and "event_name" in history:
        dated = history.copy()
        dated["event_name"] = dated["event_name"].fillna("").astype(str).str.strip()
        dated["event_date"] = pd.to_datetime(dated.get("event_date"), errors="coerce")
        event_catalog = (
            dated.loc[dated["event_name"].ne(""), ["event_name", "event_date"]]
            .drop_duplicates()
            .sort_values(["event_date", "event_name"], ascending=[False, True])
        )
        event_catalog["date_label"] = event_catalog["event_date"].dt.strftime("%Y-%m-%d")
        event_catalog["date_label"] = event_catalog["date_label"].fillna("Date unknown")
        event_catalog["label"] = event_catalog["date_label"] + " - " + event_catalog["event_name"]

    search = st.text_input(
        "Find a competition",
        placeholder="Type any part of its date or name, e.g. 2026-06 or Prague",
        help="Partial dates and names are accepted. Clear this field to see recent events.",
    ).strip()
    filtered_catalog = event_catalog
    if search and not event_catalog.empty:
        normalized = unicodedata.normalize("NFKD", search).encode("ascii", "ignore").decode().lower()
        search_values = event_catalog["label"].map(
            lambda value: unicodedata.normalize("NFKD", str(value))
            .encode("ascii", "ignore").decode().lower()
        )
        filtered_catalog = event_catalog.loc[search_values.str.contains(normalized, regex=False)]
    event_labels = filtered_catalog["label"].head(150).tolist()
    event_options = ["Custom competition", *event_labels]
    event_lookup = event_catalog.set_index("label")[["event_name", "event_date"]].to_dict("index")
    if search and not event_labels:
        st.info("No matching event. Choose Custom competition and enter it below.")

    image_file = st.file_uploader(
        "Boulder image", type=["png", "jpg", "jpeg", "webp"],
        help="Use a full-wall or close route image where the hold sequence is readable.",
    )
    if image_file is not None:
        st.image(image_file, caption=image_file.name, width="stretch")

    add_optional_tags = st.checkbox(
        "I would like to help further and identify tags",
        value=False,
        help="Optional: add hold, foothold, physical, technical and movement demands.",
    )

    def paired_sliders(container, label: str, key: str, help_text: str = "") -> tuple[int, int]:
        container.markdown(f"**{label}**")
        top_value = container.slider(
            f"🟠 Top - {label}", 0, 3, 0, key=f"style_top_{key}", help=help_text or None
        )
        zone_value = container.slider(
            f"🔵 Zone - {label}", 0, 3, 0, key=f"style_zone_{key}", help=help_text or None
        )
        return top_value, zone_value

    with st.form("style_tag_form_v2", clear_on_submit=False):
        event_choice = st.selectbox(
            "Competition", event_options,
            help="Results show the complete date and competition name.",
        )
        custom_event = st.text_input(
            "Custom competition name", disabled=event_choice != "Custom competition"
        )
        context = st.columns(3)
        round_name = context[0].selectbox(
            "Round", ["Qualification", "Semi-final", "Final", "Other"]
        )
        gender = context[1].selectbox(
            "Gender terrain", ["Men", "Women", "Mixed / unknown"]
        )
        boulder_number = context[2].text_input("Boulder", placeholder="e.g. M3 or W2")

        st.markdown("#### Core style profile")
        st.caption("Rate each factor separately for the section ending at Zone and the section ending at Top.")
        core_scores: dict[str, tuple[int, int]] = {}
        core_columns = st.columns(4)
        core_scores["physical"] = paired_sliders(
            core_columns[0], "Physical", "physical", STYLE_DEFINITIONS["physical"]
        )
        core_scores["technical"] = paired_sliders(
            core_columns[1], "Technical", "technical", STYLE_DEFINITIONS["technical"]
        )
        core_scores["coordination"] = paired_sliders(
            core_columns[2], "Coordination", "coordination", STYLE_DEFINITIONS["coordination"]
        )
        core_scores["verticality"] = paired_sliders(
            core_columns[3], "Wall angle", "verticality",
            "0 slab · 1 vertical · 2 overhang · 3 roof / very steep",
        )
        direction_columns = st.columns(2)
        top_direction = direction_columns[0].selectbox(
            "🟠 Top direction", ["Up", "Diagonal", "Sideways", "Mixed / unclear"]
        )
        zone_direction = direction_columns[1].selectbox(
            "🔵 Zone direction", ["Up", "Diagonal", "Sideways", "Mixed / unclear"]
        )

        optional_scores: dict[str, tuple[int, int]] = {}
        if add_optional_tags:
            st.markdown("#### Optional tags")
            st.caption("Leave any tag at 0 when it is absent or you are unsure.")
            for theme, items in STYLE_TAG_GROUPS.items():
                with st.expander(theme, expanded=theme in {"Physical qualities", "Handholds"}):
                    columns = st.columns(min(4, len(items)))
                    for index, (tag_key, tag_label) in enumerate(items):
                        optional_scores[tag_key] = paired_sliders(
                            columns[index % len(columns)], tag_label, tag_key
                        )

        descriptions = st.columns(2)
        zone_move = descriptions[0].text_area(
            "Zone move / position (optional)",
            placeholder="Only add information the standardized tags do not capture.",
        )
        top_move = descriptions[1].text_area(
            "Top move / position (optional)",
            placeholder="Only add information the standardized tags do not capture.",
        )
        notes = st.text_area(
            "Other notes (optional)",
            placeholder="Alternative beta, deceptive feature, image limitation, or uncertainty...",
        )
        confidence = st.select_slider(
            "Tag confidence", options=["Low", "Moderate", "High"], value="Moderate"
        )
        contributor = st.text_input("Contributor name or initials (optional)")
        submitted = st.form_submit_button("Save boulder tag", type="primary")

    if "style_tag_rows" not in st.session_state:
        st.session_state.style_tag_rows = []
    if "style_tag_images" not in st.session_state:
        st.session_state.style_tag_images = {}
    backend_url = style_tag_backend_url()
    if submitted:
        selected_event = event_lookup.get(event_choice, {})
        final_event = (
            custom_event.strip() if event_choice == "Custom competition"
            else str(selected_event.get("event_name", event_choice))
        )
        selected_date = selected_event.get("event_date")
        final_date = pd.Timestamp(selected_date).date().isoformat() if pd.notna(selected_date) else ""
        image_bytes = image_file.getvalue() if image_file is not None else b""
        if len(image_bytes) > 10 * 1024 * 1024:
            st.error("Please use an image smaller than 10 MB.")
        elif not final_event or not boulder_number.strip():
            st.error("Competition and boulder number are required.")
        else:
            image_hash = hashlib.sha256(image_bytes).hexdigest() if image_bytes else ""
            suffix = Path(image_file.name).suffix.lower() if image_file is not None else ""
            stored_image = f"images/{image_hash}{suffix}" if image_hash else ""
            record: dict[str, object] = {
                "schema_version": "2.0",
                "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
                "competition": final_event,
                "competition_date": final_date,
                "round": round_name,
                "gender_terrain": gender,
                "boulder": boulder_number.strip(),
                "image_name": image_file.name if image_file is not None else "",
                "image_sha256": image_hash,
                "image_in_bundle": stored_image,
                "top_direction": top_direction,
                "zone_direction": zone_direction,
                "optional_tags_completed": add_optional_tags,
                "zone_move": zone_move.strip(),
                "top_move": top_move.strip(),
                "notes": notes.strip(),
                "confidence": confidence,
                "contributor": contributor.strip(),
            }
            for score_key, (top_value, zone_value) in {**core_scores, **optional_scores}.items():
                record[f"top_{score_key}_0_3"] = top_value
                record[f"zone_{score_key}_0_3"] = zone_value
            st.session_state.style_tag_rows.append(record)
            if image_bytes and stored_image:
                st.session_state.style_tag_images[stored_image] = image_bytes
            if backend_url:
                saved, message = save_style_tag_remotely(backend_url, record, image_bytes)
                if saved:
                    st.success("Saved to the shared tagging database. A session backup is also available below.")
                else:
                    st.warning(f"{message}. Download the session backup below; your entry is not lost.")
            else:
                st.success("Added to this session. Export it below to preserve the entry.")

    records = pd.DataFrame(st.session_state.style_tag_rows)
    if not records.empty:
        st.markdown("#### Tags in this session")
        st.dataframe(records, hide_index=True, width="stretch")
        csv_bytes = records.to_csv(index=False).encode("utf-8")
        bundle = json.dumps(records.to_dict("records"), ensure_ascii=False, indent=2)
        downloads = st.columns(2)
        downloads[0].download_button(
            "Download CSV", csv_bytes, "boulder_style_tags_v2.csv", "text/csv",
        )
        downloads[1].download_button(
            "Download JSON", bundle, "boulder_style_tags_v2.json", "application/json",
        )
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("boulder_style_tags_v2.csv", csv_bytes)
            archive.writestr("boulder_style_tags_v2.json", bundle.encode("utf-8"))
            for image_path, saved_image_bytes in st.session_state.style_tag_images.items():
                archive.writestr(image_path, saved_image_bytes)
        st.download_button(
            "Download complete backup (tags + images)", zip_buffer.getvalue(),
            "boulder_style_tag_bundle_v2.zip", "application/zip",
        )
    if backend_url:
        st.caption("Shared database connection active. Session exports remain available as a safety copy.")
    else:
        st.caption(
            "Session storage is active. The standalone public app is ready to use a Google "
            "Drive/Sheet write endpoint as soon as its deployment URL is configured."
        )
    if standalone:
        st.info(
            "This separate annotator keeps public tagging and image traffic away from the "
            "performance dashboard."
        )


def main() -> None:
    st.set_page_config(
        page_title="Comp Climbing Projections",
        page_icon="🧗",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        .block-container{max-width:1500px;padding-top:1.2rem;padding-bottom:4rem}
        h1,h2,h3{color:#102F2B;letter-spacing:-.025em}
        [data-testid="stMetric"]{background:#F4F8F7;border:1px solid #DCE7E4;border-radius:14px;padding:14px}
        [data-testid="stHorizontalBlock"]{gap:.8rem}
        .stCaption{color:#627571}
        @media(max-width:640px){
          .block-container{padding:.7rem .75rem 3rem}
          h1{font-size:2rem!important}
          [data-testid="stHorizontalBlock"]{flex-wrap:wrap}
          [data-testid="column"]{min-width:100%!important}
          [data-testid="stSegmentedControl"] [role="radiogroup"]{flex-wrap:wrap!important}
          .js-plotly-plot{min-height:420px}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Comp Climbing Projections")
    st.markdown("### Level, depth, physicality and progression of Canadian climbers: from local comps to the Olympics")
    st.caption("Boulder release · model evidence supports coaching and governance judgment; it does not replace it.")

    data = read_data()
    startup_status(data)
    if data["athletes"].empty:
        st.stop()
    athletes = data["athletes"].copy()
    athletes["display_name"] = athletes["athlete_name"].map(friendly_name)
    selected, _ = top_ribbon(athletes, data["history"], data["rosters"])
    if not selected:
        st.info("Select at least one athlete to begin.")
        st.stop()
    workspace = st.segmented_control(
        "Workspace", ["Overview", "Physical Strength", "Tag Boulder Styles", "Maths behind"],
        default="Overview", label_visibility="collapsed",
    )
    if workspace == "Overview":
        render_rating_detail(athletes, data["history"], selected, data["calibration"])
        st.header("Overview")
        section = st.segmented_control(
            "Overview section",
            ["Canadian Pool", "IFSC Pool", "WR Pool", "Global progression", "Towards Olympics"],
            default="Canadian Pool",
            label_visibility="collapsed",
        )
        renderers = {
            "Canadian Pool": lambda: render_canadian_pool(
                athletes, selected, data["correlations"], data["calibration"]
            ),
            "IFSC Pool": lambda: render_ifsc_pool(
                athletes, data["history"], selected, data["correlations"],
                data["calibration"],
            ),
            "WR Pool": lambda: render_wr_pool(
                athletes, selected, data["correlations"], data["calibration"]
            ),
            "Global progression": lambda: render_progression(
                athletes, data["history"], selected, data["correlations"],
                data["calibration"],
            ),
            "Towards Olympics": lambda: render_olympics(
                athletes, selected, data["correlations"]
            ),
        }
        renderers[section]()
    elif workspace == "Physical Strength":
        render_physical_strength(athletes, selected, data)
    elif workspace == "Tag Boulder Styles":
        render_style_tagging_v2(data["history"])
    else:
        render_maths_behind(
            athletes, data["correlations"], data["calibration"], data
        )

    with st.expander("Rating glossary and model contract"):
        st.write(rating_help())
        st.markdown(
            "- **Global-ELO-Onsight / Scramble / Flash:** only rounds with a confirmed procedure.\n"
            "- **WR-ELO-Qualies / Semies / Finals:** only the named round of World Ranking events.\n"
            "- **IFSC-ELO-Qualies / Semies / Finals:** only the named round of non-para IFSC events.\n"
            "- **Performance-ELO:** the isolated level shown in one round, calculated from ratings frozen before the event."
        )
        st.markdown(
            "**Why one exceptional Performance-ELO does not automatically accelerate Elo.** "
            "One result can reflect real improvement, but also terrain fit and ordinary event noise. "
            "The tested one-state dynamic challenger became slightly better at ordering fields but "
            "worse at forecasting advancement probabilities after multi-round and cross-discipline "
            "dependence were corrected. Production therefore keeps the uncertainty-sensitive, "
            "zero-sum Elo. A future current-shape layer must first show repeated surprise across "
            "independent competitions and improve frozen next-event forecasts."
        )


if __name__ == "__main__":
    main()
