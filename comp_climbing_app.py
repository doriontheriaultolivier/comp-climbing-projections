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
import math
from pathlib import Path
import re
import unicodedata
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlencode
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RATING_ORDER = ["Global-ELO", "IFSC-ELO", "WC+-ELO"]
ROUND_OPTIONS = ["All rounds", "Qualification", "Semi-final", "Final"]
QUALIFICATION_FORMATS = ["Flash + Onsight", "Flash", "Onsight"]
ALL_RATINGS = [
    "Global-ELO", "Global-ELO-Open", "Global-ELO-Qualies", "Global-ELO-Qualies-Flash",
    "Global-ELO-Qualies-Onsight", "Global-ELO-Semies", "Global-ELO-Finals",
    "Global-ELO-Onsight", "Global-ELO-Scramble", "Global-ELO-Flash",
    "WC+-ELO", "WC+-ELO-Open", "WC+-ELO-Qualies", "WC+-ELO-Qualies-Flash",
    "WC+-ELO-Qualies-Onsight", "WC+-ELO-Semies", "WC+-ELO-Finals",
    "IFSC-ELO", "IFSC-ELO-Open", "IFSC-ELO-Qualies", "IFSC-ELO-Qualies-Flash",
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

STYLE_TAG_TAXONOMY_VERSION = "2026-08-03.1"

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
        ("hand_crimps_12_30mm", "Crimps / edges · 12–30 mm"),
        ("hand_crimps_under_12mm", "Small crimps / edges · <12 mm"),
        ("hand_pinches", "Pinches"),
    ],
    "Footholds": [
        ("foot_small_incut", "Small incut feet"),
        ("foot_small_smeary", "Small smeary feet"),
        ("foot_no_texture", "No-texture footholds"),
        ("foot_volumes", "Volumes"),
        ("foot_juggy", "Juggy footholds"),
    ],
    "Move types · Dynamic": [
        ("move_dyno", "Dyno"),
        ("move_run_jump", "Run-and-jump"),
        ("move_paddle", "Paddle"),
        ("move_deadpoint", "Deadpoint"),
        ("move_one_arm_catch", "One-arm catch"),
    ],
    "Move types · Press, pull & opposition": [
        ("move_no_hand", "No-hand balance / movement"),
        ("move_fight_barndoor", "Fight a barn door"),
        ("move_press", "Press"),
        ("move_overhead_press", "Overhead press"),
        ("move_mantle", "Mantle"),
        ("move_gaston", "Gaston"),
        ("move_small_sideways_compression", "Small sideways compression"),
        ("move_large_sideways_compression", "Large sideways compression"),
        ("move_small_sideways_opposition", "Small sideways opposition"),
        ("move_large_sideways_opposition", "Large sideways opposition"),
        ("move_undercling_press", "Undercling press"),
        ("move_bicep_undercling", "Bicep undercling"),
    ],
    "Move types · Reading & constraints": [
        ("move_blocked_holds", "Blocked holds"),
        ("move_hidden_holds", "Hidden holds"),
    ],
    "Move types · Hooks & feet": [
        ("move_far_toe_hook", "Far toe hook"),
        ("move_close_toe_hook", "Close toe hook"),
        ("move_incut_heel_hook", "Incut heel hook"),
        ("move_smeary_heel_hook", "Smeary heel hook"),
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


def wc_plus_event_mask(frame: pd.DataFrame) -> pd.Series:
    names = frame["event_name"].astype(str).map(plain_key)
    return names.str.contains(
        r"world\s*(?:climbing\s*)?(?:cup|series)|"
        r"world\s*(?:youth\s*)?championship|youth\s*world\s*championship|"
        r"olympic|oqs|qualifier\s*series",
        regex=True,
        na=False,
    )


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
        "country_entry": ("boulder_country_entry_progression.csv", "csv"),
        "cuwr_history": ("cuwr_top40_history.csv", "csv"),
        "context_benchmarks": ("boulder_context_benchmarks.csv", "csv"),
        "prediction_backtest": ("boulder_prediction_backtest_summary.csv", "csv"),
        "program_backtest": ("boulder_program_backtest_summary.csv", "csv"),
        "rating_v4_backtest": ("boulder_rating_model_v4_backtests.csv", "csv"),
        "pairwise_calibration": ("boulder_pairwise_probability_calibration.csv", "csv"),
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
    global_rows = ratings.loc[
        ratings["rating_family"].eq("Global-ELO")
    ].drop_duplicates(index, keep="last")
    target_columns = {
        "cec_projected_rating": "Canada projection — all evidence",
        "cec_context_offset": "Canada context adjustment",
        "rating_uncertainty": "Global-ELO uncertainty",
        "rating_status": "Global-ELO status",
    }
    available_targets = [
        column for column in target_columns if column in global_rows
    ]
    if available_targets:
        target = global_rows[index + available_targets].rename(
            columns={column: target_columns[column] for column in available_targets}
        )
        athletes = athletes.merge(target, on=index, how="left")
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


def selected_rows(athletes: pd.DataFrame, selections: list[str]) -> pd.DataFrame:
    """Resolve stable global IDs, with name fallback for saved legacy sessions."""

    tokens = {str(value) for value in selections}
    ids = athletes.get(
        "global_id", pd.Series("", index=athletes.index)
    ).astype(str)
    keys = {plain_key(value) for value in selections if str(value) not in set(ids)}
    return athletes.loc[
        ids.isin(tokens) | athletes["name_key"].isin(keys)
    ].copy()


def selection_order(row: pd.Series, selections: list[str]) -> int:
    global_id = str(row.get("global_id", ""))
    if global_id in selections:
        return selections.index(global_id)
    key = plain_key(row.get("athlete_name", ""))
    legacy = [plain_key(value) for value in selections]
    return legacy.index(key) if key in legacy else len(selections)


def selection_labels(athletes: pd.DataFrame, selections: list[str]) -> list[str]:
    focus = selected_rows(athletes, selections).copy()
    if focus.empty:
        return []
    focus["_selection_order"] = focus.apply(
        lambda row: selection_order(row, selections), axis=1
    )
    return focus.sort_values("_selection_order")["athlete_name"].map(
        friendly_name
    ).tolist()


def rating_help() -> str:
    return (
        "Every displayed family uses the same anchor: 2000 means a fitted 50% "
        "chance of reaching a semifinal at a randomly sampled 2025 IFSC Open "
        "World Cup, within the athlete's gender pool. This shifts the scale for "
        "interpretation without changing athlete order or model updates. Dashed "
        "final, podium and win lines are fitted from the same frozen 2025 "
        "athlete-starts; they are historical references, not current 2026 odds. "
        "Global-ELO uses every de-duplicated Boulder result on one Open World-Cup "
        "readiness scale. Youth evidence is included in these three main families. "
        "The -Open variants exclude youth rounds. IFSC-ELO uses IFSC results only. "
        "WC+-ELO uses World Cups/World Climbing Series, World Championships (including "
        "Youth), Olympic qualification events and the Olympics. Specialist ratings are shown only "
        "with at least two eligible rounds and enough athletes to calibrate the "
        "family; they shrink toward Global-ELO while evidence is limited. "
        "Performance-ELO is the mean WC-level rating left plausible after combining "
        "the athlete's frozen Cumulative-ELO prior with every beat/lost-to pairing "
        "in that round. WC+ uses the full result likelihood; lower-level evidence is "
        "tempered when translated to WC terrain. Posterior uncertainty and the "
        "unregularized estimate remain visible in the evidence audit."
    )


def rating_transform_controls(key: str, default: str) -> str:
    columns = st.columns([1.3, 1, 1, 1])
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
    population = columns[3].selectbox(
        "Age evidence", ["Youth + Open", "Open only"], index=0,
        disabled=round_name != "All rounds", key=f"{key}_population",
        help=(
            "The main families include youth and senior evidence on one readiness scale. "
            "Open only removes every youth round and is currently available for the overall family."
        ),
    )
    suffix = {
        "All rounds": "", "Qualification": "-Qualies",
        "Semi-final": "-Semies", "Final": "-Finals",
    }[round_name]
    rating = f"{family}{suffix}"
    if round_name == "All rounds" and population == "Open only":
        rating = f"{family}-Open"
    if round_name == "Qualification" and procedure != "Flash + Onsight":
        rating = f"{rating}-{procedure}"
    return rating


def correlation_note(
    correlations: pd.DataFrame, family: str, pool: str | None = None
) -> str:
    if correlations.empty or family == "WC+-ELO":
        return "WC+-ELO is the highest-level circuit reference in this view."
    rows = correlations.loc[correlations["rating_family"].eq(family)]
    if pool:
        rows = rows.loc[rows["pool"].eq(pool)]
    rows = rows.dropna(subset=["spearman_correlation"])
    if rows.empty:
        return "Not enough paired evidence to report a stable relationship with WC+-ELO."
    value = float(np.average(rows["spearman_correlation"], weights=rows["athletes"]))
    n = int(rows["athletes"].sum())
    return (
        f"Relationship with WC+-ELO: {value:.2f} (rank correlation), using {n} athletes "
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
    focus["_selection_order"] = focus.apply(
        lambda row: selection_order(row, selected), axis=1
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
    if stage in {"IFSC-ELO", "WC+-ELO"}:
        transitions.append(("Global-ELO", "IFSC-ELO", "#9AA7A4"))
    if stage == "WC+-ELO":
        transitions.append(("IFSC-ELO", "WC+-ELO", PALETTE["gold"]))
    if stage in {"IFSC-ELO-Open", "WC+-ELO-Open"}:
        transitions.append(("Global-ELO-Open", "IFSC-ELO-Open", "#9AA7A4"))
    if stage == "WC+-ELO-Open":
        transitions.append(("IFSC-ELO-Open", "WC+-ELO-Open", PALETTE["gold"]))
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
    elif context == "WC+":
        rank = leader.get("world_event_rank", np.nan)
        starts = leader.get("starts_365", np.nan)
        detail = (
            f"Current reconstructed CUWR: {int(rank) if pd.notna(rank) else 'not ranked'} "
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
    country_entry: pd.DataFrame,
    cuwr_history: pd.DataFrame,
) -> None:
    st.subheader("WC+ / CUWR Pool")
    st.caption(
        "The current CUWR top 40 plus every Canadian with a counting World-event "
        "start. WC+ Elo is narrower: World Cups/Series, World Championships "
        "(including Youth), Olympic qualification and the Olympics."
    )
    pool = athletes.loc[
        athletes["world_event_rank"].le(40)
        | (
            athletes["country"].astype(str).str.upper().eq("CAN")
            & athletes["starts_365"].fillna(0).gt(0)
        )
    ].copy()
    stage = rating_transform_controls("wc_plus", "WC+-ELO")
    figure = pool_scatter(
        pool, "world_event_rank", stage, selected,
        f"CUWR access pool — {stage}", canadian_outline=True,
        calibration=calibration,
    )
    figure.update_xaxes(autorange="reversed", title="Current IFSC World Ranking")
    st.plotly_chart(figure, width="stretch", theme=None)
    st.info(compare_text(pool, selected, stage, "WC+"), icon="↗️")
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
    country_pool = pool.loc[pool["country"].isin(["CAN", *countries])].dropna(subset=["WC+-ELO"])
    if not country_pool.empty:
        country_pool = country_pool.copy()
        country_pool["Included rounds"] = pd.to_numeric(
            country_pool.get("WC+-ELO evidence"), errors="coerce"
        ).fillna(0).astype(int)
        comparison = px.strip(
            country_pool, x="country", y="WC+-ELO", color="country",
            hover_name="display_name",
            hover_data={
                "world_event_rank": True, "starts_365": True,
                "Included rounds": True,
            },
            title="Actual WC+-ELO distribution of current participants",
        )
        comparison.update_traces(marker={"size": 10, "opacity": 0.65})
        add_outcome_thresholds(comparison, calibration)
        comparison.update_layout(height=430, showlegend=False)
        st.plotly_chart(comparison, width="stretch", theme=None)
    render_country_entry_progression(country_entry, ["CAN", *countries])
    render_cuwr_cycle(cuwr_history)


def _render_country_entry_progression_legacy(
    evidence: pd.DataFrame, countries: list[str]
) -> None:
    st.markdown("#### Entry level and early circuit adaptation")
    if evidence.empty:
        st.caption("The chronological country-entry artifact is being rebuilt.")
        return
    frame = evidence.loc[evidence["country"].isin(countries)].copy()
    if frame.empty:
        st.caption("No matched entry histories for these countries.")
        return
    pools = sorted(frame["pool"].dropna().unique())
    chosen_pool = st.selectbox(
        "Competition pool", pools,
        format_func=lambda value: str(value).replace("Boulder_", ""),
        key="country_entry_pool",
    )
    frame = frame.loc[frame["pool"].eq(chosen_pool)]
    statistic = st.segmented_control(
        "Country summary", ["Median", "Average"], default="Median",
        help="Median is less affected by one unusual athlete; Average answers the federation-level mean question directly.",
        key="country_entry_statistic",
    )
    entry_column = "median_entry_elo" if statistic == "Median" else "mean_entry_elo"
    change_column = (
        "median_change_first_3" if statistic == "Median" else "mean_change_first_3"
    )
    labels = {
        "Global-ELO": "Senior/Open debut",
        "IFSC-ELO": "IFSC debut",
        "WC+-ELO": "WC+ debut",
    }
    frame["Evidence scope"] = frame["rating_family"].map(labels).fillna(
        frame["rating_family"]
    )
    left, right = st.columns(2)
    entry = px.bar(
        frame, x="country", y=entry_column, color="Evidence scope",
        barmode="group", title=f"{statistic} pre-event Global-ELO at each circuit entry",
        hover_data={"athletes": True, entry_column: ":.0f"},
    )
    entry.update_yaxes(title="Frozen pre-event Global-ELO", tickformat=",.0f")
    entry.update_layout(height=420, legend_title="")
    left.plotly_chart(entry, width="stretch", theme=None)
    progress = px.bar(
        frame, x="country", y=change_column, color="Evidence scope",
        barmode="group", title=f"{statistic} Global-ELO change across the first three events",
        hover_data={"athletes": True, change_column: ":+.0f"},
    )
    progress.add_hline(y=0, line_color="rgba(16,47,43,.45)", line_width=1)
    progress.update_yaxes(title="Elo change", tickformat="+,.0f")
    progress.update_layout(height=420, legend_title="")
    right.plotly_chart(progress, width="stretch", theme=None)
    st.caption(
        "Entry is Global-ELO frozen before an athlete's first event in that circuit. "
        "It measures how strongly a country prepares and selects its entrants—not a "
        "new circuit-specific starting value. Early change is measured after up to "
        "three distinct competitions in that circuit. "
        "This describes selection and adaptation together; it does not prove that "
        "international starts caused improvement."
    )


def render_country_entry_progression(
    evidence: pd.DataFrame, countries: list[str]
) -> None:
    st.markdown("#### Entry level and early circuit adaptation")
    if evidence.empty:
        st.caption("The chronological country-entry artifact is being rebuilt.")
        return
    frame = evidence.loc[evidence["country"].isin(countries)].copy()
    if frame.empty:
        st.caption("No matched entry histories for these countries.")
        return
    pools = sorted(frame["pool"].dropna().unique())
    chosen_pool = st.selectbox(
        "Competition pool", pools,
        format_func=lambda value: str(value).replace("Boulder_", ""),
        key="country_entry_pool_v2",
    )
    frame = frame.loc[frame["pool"].eq(chosen_pool)]
    labels = {
        "Global-ELO": "Senior/Open debut",
        "IFSC-ELO": "IFSC debut",
        "WC+-ELO": "WC+ debut",
    }
    frame["Evidence scope"] = frame["rating_family"].map(labels).fillna(frame["rating_family"])
    chosen_family = st.segmented_control(
        "Entry comparison", list(labels.values()), default="WC+ debut",
        key="country_entry_family_v2",
    ) or "WC+ debut"
    shown = frame.loc[frame["Evidence scope"].eq(chosen_family)].copy()
    view = st.segmented_control(
        "Question", ["Level at entry", "Change in first three events"],
        default="Level at entry", key="country_entry_question_v2",
    ) or "Level at entry"
    if view == "Level at entry":
        value, low, high = "median_entry_elo", "entry_q25", "entry_q75"
        title = f"Established Global-ELO before first {chosen_family.lower()}"
        axis_title = "Frozen pre-event Global-ELO"
    else:
        value, low, high = "median_change_first_3", "change_q25", "change_q75"
        title = f"Global-ELO change across first three {chosen_family.lower()} events"
        axis_title = "Global-ELO change"
    if shown.empty:
        st.caption("No established pre-entry histories for this selection.")
        return
    shown = shown.sort_values(value)
    error_plus = (
        pd.to_numeric(shown[high], errors="coerce") - pd.to_numeric(shown[value], errors="coerce")
        if high in shown else None
    )
    error_minus = (
        pd.to_numeric(shown[value], errors="coerce") - pd.to_numeric(shown[low], errors="coerce")
        if low in shown else None
    )
    figure = go.Figure(go.Scatter(
        x=shown[value], y=shown["country"], mode="markers+text",
        text=shown["athletes"].map(lambda count: f"n={int(count)}"),
        textposition="middle right",
        marker={"size": 13, "color": PALETTE["teal"], "line": {"color": "white", "width": 1}},
        error_x={
            "type": "data", "array": error_plus, "arrayminus": error_minus,
            "visible": error_plus is not None, "color": "rgba(16,47,43,.38)",
        },
        customdata=np.column_stack([
            shown["athletes"],
            shown.get("median_prior_competitions", pd.Series(np.nan, index=shown.index)),
            shown.get("median_observed_events", pd.Series(np.nan, index=shown.index)),
        ]),
        hovertemplate=(
            "<b>%{y}</b><br>Median: %{x:.0f}<br>Athletes: %{customdata[0]:.0f}"
            "<br>Prior competitions at entry: %{customdata[1]:.1f}"
            "<br>Observed circuit events: %{customdata[2]:.1f}<extra></extra>"
        ),
    ))
    if view != "Level at entry":
        figure.add_vline(x=0, line_color="rgba(16,47,43,.45)", line_width=1)
    figure.update_layout(
        title=title, height=max(360, 62 * len(shown) + 130),
        margin={"l": 55, "r": 75, "t": 65, "b": 55}, showlegend=False,
    )
    figure.update_xaxes(
        title=axis_title, tickformat="+,.0f" if view != "Level at entry" else ",.0f"
    )
    figure.update_yaxes(title="")
    st.plotly_chart(figure, width="stretch", theme=None)
    st.caption(
        "The old chart mostly compared shared cold-start priors. This version requires at "
        "least two earlier independent competitions, then shows the median and middle 50% "
        "of athletes. Early change still mixes selection, adaptation and model correction; "
        "it is not proof that starts caused improvement."
    )


def _render_cuwr_cycle_legacy(history: pd.DataFrame) -> None:
    st.markdown("#### Top-40 CUWR pressure through the Olympic cycle")
    if history.empty:
        st.caption("The historical top-40 reconstruction is being rebuilt.")
        return
    frame = history.copy()
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    pools = sorted(frame["pool"].dropna().unique())
    chosen_pool = st.selectbox(
        "CUWR history pool", pools,
        format_func=lambda value: str(value).replace("Boulder_", ""),
        key="cuwr_history_pool",
    )
    frame = frame.loc[frame["pool"].eq(chosen_pool)].dropna(subset=["snapshot_date"])
    left, right = st.columns(2)
    points = px.line(
        frame, x="snapshot_date", y="rank40_points", markers=True,
        color="cycle_phase", title="Points held by reconstructed rank 40",
        hover_data={"events_365": True, "ranked_athletes": True},
    )
    points.update_yaxes(title="Best-six points")
    points.update_layout(height=420, legend_title="Cycle phase")
    left.plotly_chart(points, width="stretch", theme=None)
    elo = px.line(
        frame, x="snapshot_date", y="rank40_global_elo", markers=True,
        color="cycle_phase", title="Global-Elo of reconstructed rank 40",
        hover_data={"rank40_points": True, "events_365": True},
    )
    elo.add_hline(
        y=2000, line_dash="dash", line_color="rgba(11,122,117,.5)",
        annotation_text="50% 2025 WC semifinal reference",
    )
    elo.update_yaxes(title="Global-Elo", tickformat=",.0f")
    elo.update_layout(height=420, legend_title="Cycle phase")
    right.plotly_chart(elo, width="stretch", theme=None)
    st.info(
        "Governance use: when the top-40 level is rising, protect development time for "
        "athletes still well below WC+ semifinal readiness. Give targeted WC+ starts to "
        "athletes near the level whose next start can answer a selection question or build "
        "a realistic top-40 campaign. Reassess that balance as the rolling 365-day window "
        "approaches quota-setting dates.",
        icon="🎯",
    )
    st.dataframe(
        pd.DataFrame([
            {
                "Cycle phase": "Early cycle",
                "Priority athlete": "High-upside youth and emerging seniors",
                "Competition use": "A few diagnostic starts; preserve long training blocks",
            },
            {
                "Cycle phase": "Build phase",
                "Priority athlete": "Athletes approaching WC+ semifinal readiness",
                "Competition use": "Progressively specific fields and venues",
            },
            {
                "Cycle phase": "Qualification pressure",
                "Priority athlete": "Realistic top-40 / qualification contributors",
                "Competition use": "Target counting opportunities and protect recovery",
            },
            {
                "Cycle phase": "Olympic year",
                "Priority athlete": "Qualified or near-qualified athletes",
                "Competition use": "Outcome-specific preparation; avoid participation for its own sake",
            },
        ]),
        hide_index=True,
        width="stretch",
        column_config={
            "Cycle phase": st.column_config.TextColumn(width="small"),
            "Priority athlete": st.column_config.TextColumn(width="medium"),
            "Competition use": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(
        "This is a transparent reconstruction of the World-event component, using the "
        "best six results in a rolling 365-day window. The official CUWR also contains "
        "designated lower-status events and annual factors, so the point line is a policy "
        "signal—not an official historical ranking table."
    )


def render_cuwr_cycle(history: pd.DataFrame) -> None:
    st.markdown("#### Top-40 CUWR pressure through the Olympic cycle")
    if history.empty:
        st.caption("The historical top-40 reconstruction is being rebuilt.")
        return
    frame = history.copy()
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    pools = sorted(frame["pool"].dropna().unique())
    chosen_pool = st.selectbox(
        "CUWR history pool", pools,
        format_func=lambda value: str(value).replace("Boulder_", ""),
        key="cuwr_history_pool_v2",
    )
    frame = frame.loc[frame["pool"].eq(chosen_pool)].dropna(subset=["snapshot_date"])
    window = st.segmented_control(
        "History window", ["Current Olympic cycle", "Last two cycles", "All"],
        default="Last two cycles", key="cuwr_window",
    ) or "Last two cycles"
    if window == "Current Olympic cycle":
        frame = frame.loc[frame["snapshot_date"].ge(pd.Timestamp("2024-08-12"))]
    elif window == "Last two cycles":
        frame = frame.loc[frame["snapshot_date"].ge(pd.Timestamp("2017-01-01"))]
    measure = st.segmented_control(
        "Top-40 threshold", ["CUWR points", "Global-ELO"],
        default="CUWR points", key="cuwr_measure",
    ) or "CUWR points"
    y = "rank40_points" if measure == "CUWR points" else "rank40_global_elo"
    shown = frame.sort_values("snapshot_date").copy()
    shown["Smoothed threshold"] = pd.to_numeric(shown[y], errors="coerce").rolling(
        3, min_periods=1, center=True
    ).median()
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=shown["snapshot_date"], y=shown[y], mode="markers",
        name="Observed reconstruction", marker={
            "size": 7, "color": "rgba(70,91,88,.28)",
        },
        customdata=np.column_stack([
            shown["cycle_phase"], shown["events_365"], shown["ranked_athletes"],
        ]),
        hovertemplate=(
            "%{x|%Y-%m-%d}<br>Observed: %{y:.0f}"
            "<br>Phase: %{customdata[0]}<br>Events in 365d: %{customdata[1]:.0f}"
            "<br>Ranked athletes: %{customdata[2]:.0f}<extra></extra>"
        ),
    ))
    figure.add_trace(go.Scatter(
        x=shown["snapshot_date"], y=shown["Smoothed threshold"], mode="lines",
        name="Three-snapshot median", line={"color": PALETTE["teal"], "width": 4},
        hovertemplate="%{x|%Y-%m-%d}<br>Typical threshold: %{y:.0f}<extra></extra>",
    ))
    if measure == "Global-ELO":
        figure.add_hline(
            y=2000, line_dash="dash", line_color="rgba(11,122,117,.45)",
            annotation_text="50% WC semifinal",
        )
    figure.update_layout(
        title=(
            "Reconstructed rank-40 points threshold"
            if measure == "CUWR points" else "World-readiness of reconstructed rank 40"
        ),
        height=465, margin={"l": 65, "r": 35, "t": 65, "b": 55},
        legend={"orientation": "h", "y": -0.18},
    )
    figure.update_yaxes(
        title="Best-six points" if measure == "CUWR points" else "Global-ELO",
        tickformat=",.0f",
    )
    figure.update_xaxes(title="Ranking snapshot")
    st.plotly_chart(figure, width="stretch", theme=None)
    latest = shown.dropna(subset=[y]).tail(1)
    if not latest.empty:
        latest_value = float(latest.iloc[0][y])
        st.info(
            f"Current reconstructed rank-40 threshold: {latest_value:,.0f} "
            f"{'points' if measure == 'CUWR points' else 'Global-ELO'}. "
            "Use targeted starts for athletes close enough that one result can change access; "
            "protect longer development blocks for athletes still far below semifinal readiness.",
            icon="🎯",
        )
    st.caption(
        "Faint points are event-date reconstructions; the dark line is a three-snapshot median, "
        "which removes the artificial spikes created by joining separate cycle-phase series. "
        "This World-event proxy is not the official historical CUWR."
    )


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
        ).drop_duplicates(["global_id"])
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

    reference = st.selectbox(
        "Pathway reference cohort",
        [
            "Current WC+ semifinal-ready athletes",
            "Ever WC+ semifinal-ready athletes",
            "Past WC+ finalists",
            "Combined: ever ready or finalist",
            "None",
        ],
        help=(
            "Semifinal-ready means WC+-ELO at or above the fitted 50% semifinal "
            "reference. The shaded interval shows the central 80% of observed ratings; "
            "marker density grows with athletes and rounds in that age bin."
        ),
    )
    show_reference_range = st.toggle(
        "Show the central 80% reference range",
        value=False,
        help="Off by default because the range can hide the athletes you are trying to compare.",
        key="pathway_reference_range",
    )
    if reference != "None":
        wc_mask = wc_plus_event_mask(history) & history["source_scope"].eq("IFSC")
        finalists = set(history.loc[
            wc_mask & history["round_group"].eq("Final"), "global_id"
        ].astype(str))
        wc_rating = pd.to_numeric(
            athletes["WC+-ELO"] if "WC+-ELO" in athletes else pd.Series(np.nan, index=athletes.index),
            errors="coerce",
        )
        current_ready = set(athletes.loc[
            wc_rating.ge(2000),
            "global_id",
        ].astype(str))
        ever_ready = set(history.loc[
            wc_mask & pd.to_numeric(history["rating_after"], errors="coerce").ge(2000),
            "global_id",
        ].astype(str))
        if reference == "Current WC+ semifinal-ready athletes":
            reference_ids = current_ready
        elif reference == "Ever WC+ semifinal-ready athletes":
            reference_ids = ever_ready
        elif reference == "Past WC+ finalists":
            reference_ids = finalists
        else:
            reference_ids = ever_ready | finalists
        reference_rows = athletes.loc[
            athletes["global_id"].astype(str).isin(reference_ids),
            ["pool", "global_id"],
        ].drop_duplicates()
        pathway = history.merge(reference_rows, on=["pool", "global_id"], how="inner")
        pathway = pathway.dropna(subset=["age_at_event", "rating_after"]).copy()
        pathway["age_year"] = (
            pd.to_numeric(pathway["age_at_event"], errors="coerce") * 2
        ).round() / 2
        pathway = pathway.merge(
            athletes[["pool", "global_id", "gender"]],
            on=["pool", "global_id"], how="left",
        )
        grouped = pathway.groupby(["gender", "age_year"], as_index=False).agg(
            rating=("rating_after", "median"),
            low=("rating_after", lambda values: values.quantile(0.10)),
            high=("rating_after", lambda values: values.quantile(0.90)),
            athletes=("global_id", "nunique"), rounds=("global_id", "size"),
        )
        grouped = grouped.loc[grouped["athletes"].ge(3)].sort_values("age_year")
        if not grouped.empty:
            for gender, gender_rows in grouped.groupby("gender"):
                gender_rows = gender_rows.sort_values("age_year").copy()
                for column in ("rating", "low", "high"):
                    gender_rows[column] = gender_rows[column].rolling(
                        3, min_periods=1, center=True
                    ).median()
                color = PALETTE["teal"] if gender == "Men" else PALETTE["blue"]
                if show_reference_range:
                    figure.add_trace(go.Scatter(
                        x=[*gender_rows["age_year"], *gender_rows["age_year"].iloc[::-1]],
                        y=[*gender_rows["high"], *gender_rows["low"].iloc[::-1]],
                        fill="toself", fillcolor=transparent(color, 0.08),
                        line={"width": 0}, hoverinfo="skip", showlegend=False,
                        legendgroup=f"reference-{gender}",
                    ))
                figure.add_trace(go.Scatter(
                    x=gender_rows["age_year"], y=gender_rows["rating"],
                    mode="lines", name=f"{reference} — {gender}",
                    customdata=np.column_stack([
                        gender_rows["athletes"], gender_rows["rounds"],
                    ]),
                    hovertemplate=(
                        "Age: %{x:.1f}<br>Median Global-ELO: %{y:.0f}"
                        "<br>Athletes: %{customdata[0]:.0f}"
                        "<br>Included rounds: %{customdata[1]:.0f}<extra>%{fullData.name}</extra>"
                    ),
                    line={
                        "color": color,
                        "width": 3, "dash": "dot", "shape": "spline",
                    },
                    legendgroup=f"reference-{gender}",
                ))
    add_outcome_thresholds(figure, calibration)
    figure.update_yaxes(tickformat=",.0f", title="Global-ELO")
    figure.update_xaxes(title="Age")
    figure.update_layout(height=570, margin={"l": 80, "r": 25, "t": 70, "b": 65})
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
    focus["selection_order"] = focus.apply(
        lambda row: selection_order(row, selected), axis=1
    )
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
        wr_elo = athlete.get("WC+-ELO", np.nan)
        semi = outcome_threshold(calibration, "semifinal", athlete.get("pool"))
        wr_starts = athlete.get("starts_365", np.nan)
        momentum = athlete.get("momentum", 0.0)
        if pd.isna(global_elo):
            hypothesis = "Build a reliable competition baseline before choosing a pathway emphasis."
        elif np.isfinite(wr_elo) and global_elo - wr_elo >= 100:
            hypothesis = (
                "General performance is ahead of WC+-specific performance. "
                "Prioritize WC+-style simulations and selective international starts; review onsight "
                "decision quality, setting specificity, travel and pressure response."
            )
        elif np.isfinite(semi) and global_elo >= semi - 100 and (pd.isna(wr_starts) or wr_starts < 3):
            hypothesis = "Test targeted WC+ competition exposure; readiness appears close enough for the experience to be informative."
        elif momentum > 35:
            hypothesis = "Protect the improving training process; add WC+ starts selectively rather than chasing participation volume."
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
            "These are decision hypotheses from rating level, recent change and WC+ "
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
    cutoff = as_of - pd.Timedelta(365, unit="D")
    prior_cutoff = cutoff - pd.Timedelta(365, unit="D")
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
        column.metric("WC+-ELO", f"{athlete.get('WC+-ELO', np.nan):.0f}" if pd.notna(athlete.get("WC+-ELO")) else "—")
        rank = athlete.get("world_event_rank", np.nan)
        column.caption(f"Current World Ranking: {int(rank) if pd.notna(rank) else 'not ranked'} · starts/365d: {int(athlete.get('starts_365', 0) or 0)}")
    if not focus.empty:
        table = focus[[
            "athlete_name", "Global-ELO", "IFSC-ELO", "WC+-ELO",
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
        selector = athletes.dropna(subset=["Global-ELO", "global_id", "athlete_name"]).copy()
        selector["_evidence"] = pd.to_numeric(
            selector.get("Global-ELO evidence"), errors="coerce"
        ).fillna(0)
        selector = selector.sort_values(
            ["athlete_name", "_evidence"], ascending=[True, False]
        ).drop_duplicates(["pool", "global_id"])
        selector["_label"] = selector.apply(
            lambda row: (
                f"{friendly_name(row['athlete_name'])} "
                f"({str(row.get('country') or row.get('nationality') or '—').upper()}) "
                f"· {row['Global-ELO']:.0f}"
            ),
            axis=1,
        )
        option_ids = selector["global_id"].astype(str).tolist()
        labels = dict(zip(option_ids, selector["_label"]))
        selected: list[str]
        if mode == "Compare 3":
            columns = st.columns(3)
            ids_by_key = {
                plain_key(row["athlete_name"]): str(row["global_id"])
                for _, row in selector.sort_values("_evidence").iterrows()
            }
            defaults = [
                ids_by_key[plain_key(name)]
                for name in DEFAULT_ATHLETES
                if plain_key(name) in ids_by_key
            ]
            while len(defaults) < 3 and option_ids:
                candidate = option_ids[min(len(defaults), len(option_ids) - 1)]
                if candidate not in defaults:
                    defaults.append(candidate)
                else:
                    break
            selected = []
            for index, column in enumerate(columns):
                default = defaults[index] if index < len(defaults) else option_ids[0]
                selected.append(column.selectbox(
                    "Main athlete" if index == 0 else f"Comparison {index + 1}",
                    option_ids,
                    index=option_ids.index(default),
                    format_func=lambda global_id: labels.get(global_id, global_id),
                    key=f"athlete_{index}",
                ))
        else:
            preset = roster_names(mode, athletes, history, rosters)
            matched = selector.loc[
                selector["name_key"].isin({plain_key(name) for name in preset}),
                "global_id",
            ].astype(str).unique().tolist()
            selected = st.multiselect(
                f"{mode} athletes",
                option_ids,
                default=matched,
                format_func=lambda global_id: labels.get(global_id, global_id),
                help="All matched members start selected. Uncheck any athlete to simplify an individual graph.",
            )
            if mode == "Canadian National Team proxy":
                st.caption("Proxy only: current CNR top 15 by gender. Replace with the official roster when supplied.")
        return selected, discipline


def relevant_rating_benchmarks(
    focus: pd.DataFrame, calibration: pd.DataFrame, contexts: pd.DataFrame,
) -> list[dict[str, object]]:
    """Choose outcome references that answer the selected athletes' next questions."""
    pools = focus.get("pool", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
    benchmarks: list[dict[str, object]] = []
    for outcome, label, color in (
        ("semifinal", "50% WC semifinal", PALETTE["teal"]),
        ("final", "50% WC final", PALETTE["blue"]),
    ):
        values = [outcome_threshold(calibration, outcome, pool) for pool in pools]
        values = [value for value in values if np.isfinite(value)]
        if values:
            benchmarks.append({"label": label, "elo": float(np.mean(values)), "color": color})
    if contexts.empty:
        return benchmarks
    matched = contexts.loc[
        contexts["pool"].isin(pools)
        & contexts["context"].eq("Canadian Senior Nationals")
    ]
    values = pd.to_numeric(matched.get("threshold_elo"), errors="coerce").dropna()
    if not values.empty:
        method = "fitted" if matched["method"].astype(str).str.startswith("fitted").any() else "median achiever"
        benchmarks.append({
            "label": f"Canadian Senior Nationals final ({method})",
            "elo": float(values.mean()), "color": PALETTE["gold"],
        })
    youth_categories: set[str] = set()
    for age in pd.to_numeric(focus.get("age"), errors="coerce").dropna():
        next_age = float(age) + 1.0
        if next_age < 17:
            youth_categories.add("U17")
        elif next_age < 19:
            youth_categories.add("U19")
        elif next_age < 21:
            youth_categories.add("U21")
    if youth_categories:
        youth = contexts.loc[
            contexts["pool"].isin(pools)
            & contexts["context"].eq("Youth World Championships")
            & contexts["category_group"].isin(youth_categories)
        ]
        for category, rows in youth.groupby("category_group"):
            values = pd.to_numeric(rows["threshold_elo"], errors="coerce").dropna()
            if not values.empty:
                benchmarks.append({
                    "label": f"50% Youth Worlds semifinal - next {category}",
                    "elo": float(values.mean()), "color": PALETTE["coral"],
                })
    return benchmarks


def rating_radar_figure(
    detail: pd.DataFrame, athlete_order: list[str], view: str,
    benchmarks: list[dict[str, object]] | None = None,
) -> go.Figure:
    """Show rating shape while keeping evidence volume visible in hover/markers."""
    profiles = {
        "All-competition profile": [
            ("Global-ELO", "Overall"),
            ("Global-ELO-Qualies", "Qualifying"),
            ("Global-ELO-Semies", "Semifinal"),
            ("Global-ELO-Finals", "Final"),
        ],
        "World-circuit profile": [
            ("WC+-ELO", "Overall"),
            ("WC+-ELO-Qualies", "Qualifying"),
            ("WC+-ELO-Semies", "Semifinal"),
            ("WC+-ELO-Finals", "Final"),
        ],
    }
    axes = profiles[view]
    families = [family for family, _ in axes]
    labels = [label for _, label in axes]
    visible = detail.loc[detail["Rating family"].isin(families)].copy()
    comparable_families = {
        family for profile_axes in profiles.values() for family, _ in profile_axes
    }
    values = pd.to_numeric(
        detail.loc[detail["Rating family"].isin(comparable_families), "Elo"],
        errors="coerce",
    ).dropna()
    benchmarks = benchmarks or [{"label": "50% WC semifinal", "elo": 2000.0, "color": PALETTE["teal"]}]
    benchmark_values = [
        float(item["elo"]) for item in benchmarks
        if np.isfinite(item.get("elo", np.nan))
    ]
    scale_values = [*values.tolist(), *benchmark_values]
    low = float(min(scale_values) if scale_values else 1900)
    high = float(max(scale_values) if scale_values else 2100)
    pad = max(25.0, (high - low) * 0.08)
    radial_min = 25 * np.floor((low - pad) / 25)
    radial_max = 25 * np.ceil((high + pad) / 25)
    radial_tick = max(50.0, 25.0 * np.ceil((radial_max - radial_min) / 125.0))
    figure = go.Figure()
    closed_labels = [*labels, labels[0]]
    for benchmark in benchmarks:
        value = float(benchmark["elo"])
        figure.add_trace(go.Scatterpolar(
            r=[value] * len(closed_labels), theta=closed_labels,
            mode="lines", name=f"{value:.0f} - {benchmark['label']}",
            line={
                "color": transparent(str(benchmark["color"]), 0.55),
                "width": 1.4, "dash": "dash",
            },
            hovertemplate=f"{benchmark['label']}: {value:.0f} Elo<extra></extra>",
        ))
    for athlete_index, athlete in enumerate(athlete_order):
        athlete_rows = visible.loc[visible["Athlete"].eq(athlete)].set_index("Rating family")
        if athlete_rows.empty:
            continue
        ratings: list[float | None] = []
        evidence: list[float] = []
        outcomes: list[str] = []
        deltas: list[str] = []
        all_comp_rows = detail.loc[detail["Athlete"].eq(athlete)].set_index(
            "Rating family"
        )
        counterpart = {
            "WC+-ELO": "Global-ELO",
            "WC+-ELO-Qualies": "Global-ELO-Qualies",
            "WC+-ELO-Semies": "Global-ELO-Semies",
            "WC+-ELO-Finals": "Global-ELO-Finals",
        }
        for family in families:
            if family not in athlete_rows.index:
                ratings.append(None)
                evidence.append(0.0)
                outcomes.append("No rating evidence")
                deltas.append("—")
                continue
            item = athlete_rows.loc[family]
            if isinstance(item, pd.DataFrame):
                item = item.iloc[0]
            rating = pd.to_numeric(pd.Series([item["Elo"]]), errors="coerce").iloc[0]
            rounds = pd.to_numeric(pd.Series([item["Included rounds"]]), errors="coerce").iloc[0]
            ratings.append(float(rating) if np.isfinite(rating) else None)
            evidence.append(float(rounds) if np.isfinite(rounds) else 0.0)
            outcomes.append(str(item["Historical outcome estimate"]))
            baseline_family = counterpart.get(family)
            baseline = np.nan
            if baseline_family and baseline_family in all_comp_rows.index:
                baseline_item = all_comp_rows.loc[baseline_family]
                if isinstance(baseline_item, pd.DataFrame):
                    baseline_item = baseline_item.iloc[0]
                baseline = pd.to_numeric(
                    pd.Series([baseline_item["Elo"]]), errors="coerce"
                ).iloc[0]
            deltas.append(
                f"{float(rating - baseline):+.0f} Elo"
                if np.isfinite(rating) and np.isfinite(baseline) else "—"
            )
        if not any(value is not None and np.isfinite(value) for value in ratings):
            continue
        athlete_color = ATHLETE_COLORS[athlete_index % len(ATHLETE_COLORS)]
        if view == "World-circuit profile":
            baseline_ratings: list[float | None] = []
            for family in families:
                baseline_family = counterpart.get(family)
                baseline = np.nan
                if baseline_family and baseline_family in all_comp_rows.index:
                    baseline_item = all_comp_rows.loc[baseline_family]
                    if isinstance(baseline_item, pd.DataFrame):
                        baseline_item = baseline_item.iloc[0]
                    baseline = pd.to_numeric(
                        pd.Series([baseline_item["Elo"]]), errors="coerce"
                    ).iloc[0]
                baseline_ratings.append(
                    float(baseline) if np.isfinite(baseline) else None
                )
            if any(value is not None for value in baseline_ratings):
                figure.add_trace(go.Scatterpolar(
                    r=[*baseline_ratings, baseline_ratings[0]], theta=closed_labels,
                    mode="lines+markers",
                    name=f"{athlete} — all-competition reference",
                    legendgroup=athlete, showlegend=False,
                    line={"color": transparent(athlete_color, 0.42), "width": 1.5, "dash": "dot"},
                    marker={
                        "color": "white", "size": 7,
                        "line": {"color": athlete_color, "width": 1.2},
                    },
                    hovertemplate=(
                        f"<b>{athlete}</b><br>%{{theta}}"
                        "<br>All-competition Elo: %{r:.0f}<extra></extra>"
                    ),
                    connectgaps=False,
                ))
                for label, baseline, world in zip(labels, baseline_ratings, ratings):
                    if baseline is None or world is None:
                        continue
                    figure.add_trace(go.Scatterpolar(
                        r=[baseline, world], theta=[label, label], mode="lines",
                        legendgroup=athlete, showlegend=False,
                        line={"color": transparent(athlete_color, 0.55), "width": 2},
                        hoverinfo="skip",
                    ))
        sizes = [min(17.0, 7.0 + 2.4 * np.log1p(max(value, 0.0))) for value in evidence]
        custom = [
            [families[index], evidence[index], outcomes[index], deltas[index]]
            for index in range(len(families))
        ]
        figure.add_trace(go.Scatterpolar(
            r=[*ratings, ratings[0]], theta=closed_labels,
            mode="lines+markers", name=athlete,
            legendgroup=athlete,
            line={"color": athlete_color, "width": 3},
            marker={
                "color": athlete_color,
                "size": [*sizes, sizes[0]], "line": {"color": "white", "width": 1},
            },
            fill="toself", fillcolor=transparent(
                athlete_color, 0.08
            ),
            customdata=[*custom, custom[0]],
            hovertemplate=(
                "<b>%{fullData.name}</b><br>%{theta}<br>Elo: %{r:.0f}"
                "<br>Included rounds: %{customdata[1]:.0f}"
                "<br>%{customdata[2]}"
                "<br>Change from all competitions: %{customdata[3]}"
                "<extra></extra>"
            ),
            connectgaps=False,
        ))
    figure.update_layout(
        height=575,
        margin={"l": 55, "r": 55, "t": 45, "b": 95},
        polar={
            "radialaxis": {
                "range": [radial_min, radial_max], "tickformat": ",.0f",
                "dtick": radial_tick,
                "gridcolor": "rgba(113,129,126,.22)", "angle": 45,
            },
            "angularaxis": {"gridcolor": "rgba(113,129,126,.18)"},
            "bgcolor": "rgba(244,248,247,.65)",
        },
        legend={"orientation": "h", "y": -0.16, "x": 0.5, "xanchor": "center"},
        paper_bgcolor="white",
    )
    return figure


def render_rating_detail(
    athletes: pd.DataFrame,
    history: pd.DataFrame,
    selected: list[str],
    calibration: pd.DataFrame,
    context_benchmarks: pd.DataFrame,
) -> None:
    st.subheader("Compared athletes · all rating evidence")
    focus = selected_rows(athletes, selected)
    if focus.empty:
        st.caption("No matched rating evidence. Athletes with no competition yet remain in the roster.")
        return
    focus["selection_order"] = focus.apply(
        lambda row: selection_order(row, selected), axis=1
    )
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
        f"<span style='color:{ATHLETE_COLORS[i % len(ATHLETE_COLORS)]}'>●</span> "
        f"{friendly_name(row['athlete_name'])}"
        for i, (_, row) in enumerate(focus.iterrows())
    )
    st.markdown(legend, unsafe_allow_html=True)
    athlete_order = focus["athlete_name"].map(friendly_name).tolist()
    benchmarks = relevant_rating_benchmarks(focus, calibration, context_benchmarks)
    if 2 <= len(athlete_order) <= 5:
        radar_view = st.segmented_control(
            "Rating radar", ["All-competition profile", "World-circuit profile"],
            default="All-competition profile",
            help=(
                "The radar compares rating shape, not certainty. Marker size represents the "
                "number of included rounds; exact evidence remains in the table."
            ),
        ) or "All-competition profile"
        st.plotly_chart(
            rating_radar_figure(detail, athlete_order, radar_view, benchmarks),
            width="stretch", config={"displayModeBar": False},
        )
        st.caption(
            "Further from the centre = higher Elo. Dashed rings are selected for the athletes' "
            "current pathway; a median-achiever ring is labelled when a 50% fit is too sparse. "
            "Larger markers mean more included rounds. The two views keep identical axes, and "
            "WC+ hover text reports the change from the matching all-competition axis."
        )
        delta_rows: list[dict[str, object]] = []
        axis_pairs = {
            "Overall": ("Global-ELO", "WC+-ELO"),
            "Qualifying": ("Global-ELO-Qualies", "WC+-ELO-Qualies"),
            "Semifinal": ("Global-ELO-Semies", "WC+-ELO-Semies"),
            "Final": ("Global-ELO-Finals", "WC+-ELO-Finals"),
        }
        for athlete_name in athlete_order:
            athlete_rows = detail.loc[detail["Athlete"].eq(athlete_name)].set_index(
                "Rating family"
            )
            record: dict[str, object] = {"Athlete": athlete_name}
            for label, (global_family, wc_family) in axis_pairs.items():
                global_value = pd.to_numeric(
                    pd.Series([athlete_rows.at[global_family, "Elo"]])
                    if global_family in athlete_rows.index else pd.Series([np.nan]),
                    errors="coerce",
                ).iloc[0]
                wc_value = pd.to_numeric(
                    pd.Series([athlete_rows.at[wc_family, "Elo"]])
                    if wc_family in athlete_rows.index else pd.Series([np.nan]),
                    errors="coerce",
                ).iloc[0]
                record[f"{label} Δ"] = (
                    float(wc_value - global_value)
                    if np.isfinite(global_value) and np.isfinite(wc_value) else np.nan
                )
            delta_rows.append(record)
        st.dataframe(
            pd.DataFrame(delta_rows), hide_index=True, width="stretch",
            column_config={
                f"{label} Δ": st.column_config.NumberColumn(
                    f"{label} WC+ − all comps", format="%+.0f"
                )
                for label in axis_pairs
            },
        )
    elif len(athlete_order) > 5:
        st.caption("Radar hidden for more than five athletes to preserve readability; uncheck athletes to compare profiles.")
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
    target_rows: list[dict[str, object]] = []
    for _, athlete in focus.iterrows():
        global_elo = pd.to_numeric(athlete.get("Global-ELO"), errors="coerce")
        canada = pd.to_numeric(
            athlete.get("Canada projection — all evidence"), errors="coerce"
        )
        target_rows.append({
            "Athlete": friendly_name(athlete["athlete_name"]),
            "Open WC projection": global_elo,
            "Canadian-event projection": canada,
            "Canada context change": (
                canada - global_elo
                if np.isfinite(canada) and np.isfinite(global_elo) else np.nan
            ),
            "Current rating status": athlete.get(
                "Global-ELO status", "Established"
            ),
            "Uncertainty (± Elo)": pd.to_numeric(
                athlete.get("Global-ELO uncertainty"), errors="coerce"
            ),
        })
    target_frame = pd.DataFrame(target_rows)
    if target_frame["Canadian-event projection"].notna().any():
        st.markdown("#### Same evidence, different target environment")
        st.dataframe(
            target_frame, hide_index=True, width="stretch",
            column_config={
                "Open WC projection": st.column_config.NumberColumn(format="%.0f"),
                "Canadian-event projection": st.column_config.NumberColumn(format="%.0f"),
                "Canada context change": st.column_config.NumberColumn(format="%+.0f"),
                "Uncertainty (± Elo)": st.column_config.NumberColumn(format="%.0f"),
            },
        )
        st.caption(
            "One shared ability estimate is translated to the target environment. The Canada "
            "adjustment is learned chronologically from Canadian results; it is not a separate "
            "athlete identity or an extra independent Elo ledger."
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
            latest_columns = [
                "Athlete", "event_date", "event_name", "round_group",
                "confirmed_procedure", "performance_elo",
            ]
            if "raw_performance_elo" in recent:
                latest_columns.append("raw_performance_elo")
            if "performance_elo_uncertainty" in recent:
                latest_columns.insert(-1, "performance_elo_uncertainty")
            latest = recent[latest_columns].rename(columns={
                "event_date": "Event date", "event_name": "Competition",
                "round_group": "Round", "confirmed_procedure": "Procedure",
                "performance_elo": "Performance-ELO",
                "performance_elo_uncertainty": "Posterior uncertainty (SD)",
                "raw_performance_elo": "Raw round estimate",
            })
            st.markdown("#### Latest round-performance signals")
            st.dataframe(
                latest, hide_index=True, width="stretch",
                column_config={
                    "Event date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    "Performance-ELO": st.column_config.NumberColumn(format="%d"),
                    "Posterior uncertainty (SD)": st.column_config.NumberColumn(format="%d"),
                    "Raw round estimate": st.column_config.NumberColumn(format="%d"),
                },
            )
        with st.expander("Inspect every competition round behind these ratings"):
            evidence = history.merge(
                ids, on=["pool", "global_id"], how="inner", suffixes=("", "_selected")
            )
            evidence["event_date"] = pd.to_datetime(evidence["event_date"], errors="coerce")
            evidence["Athlete"] = evidence["athlete_name_selected"].map(friendly_name)
            evidence["Global-ELO change"] = (
                pd.to_numeric(evidence["rating_after"], errors="coerce")
                - pd.to_numeric(evidence["rating_before"], errors="coerce")
            )
            if "n_athletes" not in evidence:
                evidence["n_athletes"] = evidence.groupby(
                    ["source_event_id", "pool", "round_group"]
                )["global_id"].transform("nunique")
            chosen_athlete = st.selectbox(
                "Athlete evidence", athlete_order, key="rating_evidence_athlete"
            )
            scopes = ["All competitions", "IFSC", "WC+"]
            chosen_scope = st.segmented_control(
                "Competition evidence", scopes, default="All competitions",
                key="rating_evidence_scope",
            ) or "All competitions"
            shown = evidence.loc[evidence["Athlete"].eq(chosen_athlete)].copy()
            if chosen_scope == "IFSC":
                shown = shown.loc[shown["source_scope"].eq("IFSC")]
            elif chosen_scope == "WC+":
                shown = shown.loc[
                    shown["source_scope"].eq("IFSC") & wc_plus_event_mask(shown)
                ]
            shown = shown.sort_values("event_date", ascending=False)
            shown_columns = [
                "event_date", "event_name", "round_group", "confirmed_procedure",
                "rank_numeric", "n_athletes", "performance_elo", "rating_before",
                "rating_after", "Global-ELO change",
            ]
            if "raw_performance_elo" in shown:
                shown_columns.insert(7, "raw_performance_elo")
            if "performance_elo_uncertainty" in shown:
                shown_columns.insert(7, "performance_elo_uncertainty")
            shown = shown[shown_columns].rename(columns={
                "event_date": "Event date", "event_name": "Competition",
                "round_group": "Round", "confirmed_procedure": "Format",
                "rank_numeric": "Place", "n_athletes": "Field size",
                "performance_elo": "Performance-ELO", "rating_before": "Global-ELO before",
                "performance_elo_uncertainty": "Posterior uncertainty (SD)",
                "raw_performance_elo": "Raw round estimate",
                "rating_after": "Global-ELO after",
            })
            st.dataframe(
                shown, hide_index=True, width="stretch", height=430,
                column_config={
                    "Event date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    "Competition": st.column_config.TextColumn(width="large"),
                    "Performance-ELO": st.column_config.NumberColumn(format="%.0f"),
                    "Posterior uncertainty (SD)": st.column_config.NumberColumn(format="%.0f"),
                    "Raw round estimate": st.column_config.NumberColumn(format="%.0f"),
                    "Global-ELO before": st.column_config.NumberColumn(format="%.0f"),
                    "Global-ELO after": st.column_config.NumberColumn(format="%.0f"),
                    "Global-ELO change": st.column_config.NumberColumn(format="%+.0f"),
                },
            )
            st.caption(
                "Performance-ELO is the mean of the WC-rating probability distribution "
                "remaining after the round. SD shows its uncertainty; the unregularized "
                "estimate is retained for audit. "
                "Global-ELO change is the "
                "zero-sum update after the round; it is not a claim that one event changed ability."
            )


def current_form_signal(history: pd.DataFrame, athlete: pd.Series) -> tuple[float, int]:
    """Previous-three independent-event surprise, matching the frozen backtest."""
    rows = history.loc[
        history["pool"].eq(athlete["pool"])
        & history["global_id"].astype(str).eq(str(athlete["global_id"]))
    ].copy()
    if rows.empty:
        return 0.0, 0
    rows["event_date"] = pd.to_datetime(rows["event_date"], errors="coerce")
    baseline = pd.to_numeric(
        rows.get(
            "event_start_global_rating",
            rows.get("event_start_rating", rows["rating_before"]),
        ),
        errors="coerce",
    )
    rows["surprise"] = pd.to_numeric(rows["performance_elo"], errors="coerce") - baseline
    event = (
        rows.groupby(["source_scope", "source_event_id", "event_date"], as_index=False)
        .agg(surprise=("surprise", "median"))
        .dropna(subset=["event_date", "surprise"])
        .sort_values("event_date")
        .tail(3)
    )
    if event.empty:
        return 0.0, 0
    latest = event["event_date"].max()
    ages = (latest - event["event_date"]).dt.days.to_numpy(float)
    weights = np.power(0.5, np.maximum(ages, 0) / 180.0)
    raw = float(np.average(event["surprise"].to_numpy(float), weights=weights))
    evidence = len(event)
    return float(np.clip(raw * evidence / (evidence + 2.0), -250, 250)), evidence


def competition_level_figure(
    history: pd.DataFrame, athlete: pd.Series, calibration: pd.DataFrame,
) -> go.Figure:
    """Combine stable level, isolated performances and field level without mixing them."""
    athlete_rows = history.loc[
        history["pool"].eq(athlete["pool"])
        & history["global_id"].astype(str).eq(str(athlete["global_id"]))
    ].copy()
    athlete_rows["event_date"] = pd.to_datetime(athlete_rows["event_date"], errors="coerce")
    athlete_rows = athlete_rows.dropna(subset=["event_date"]).sort_values("event_date")
    figure = go.Figure()
    if athlete_rows.empty:
        return figure
    figure.add_trace(go.Scatter(
        x=athlete_rows["event_date"], y=athlete_rows["rating_after"],
        mode="lines", name="Global-ELO after each round",
        line={"color": PALETTE["ink"], "width": 3},
        customdata=np.column_stack([athlete_rows["event_name"], athlete_rows["round_group"]]),
        hovertemplate=(
            "%{x|%Y-%m-%d}<br>%{customdata[0]} - %{customdata[1]}"
            "<br>Global-ELO: %{y:.0f}<extra></extra>"
        ),
    ))
    round_colors = {
        "Qualification": PALETTE["blue"], "Semi-final": PALETTE["gold"],
        "Final": PALETTE["coral"],
    }
    for round_group, rows in athlete_rows.groupby("round_group"):
        rows = rows.sort_values("event_date").copy()
        field_sizes = rows.get("n_athletes", pd.Series(np.nan, index=rows.index))
        figure.add_trace(go.Scatter(
            x=rows["event_date"], y=rows["performance_elo"], mode="markers",
            name=f"{round_group} Performance-ELO",
            marker={
                "color": round_colors.get(str(round_group), PALETTE["teal"]),
                "size": np.clip(
                    7 + np.log1p(pd.to_numeric(field_sizes, errors="coerce").fillna(1)),
                    8, 15,
                ),
                "opacity": 0.68, "line": {"color": "white", "width": 1},
            },
            customdata=np.column_stack([
                rows["event_name"], rows["rank_numeric"], field_sizes,
                pd.to_numeric(
                    rows.get(
                        "performance_elo_uncertainty",
                        pd.Series(np.nan, index=rows.index),
                    ),
                    errors="coerce",
                ),
            ]),
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>%{customdata[0]}"
                "<br>Place: %{customdata[1]:.0f} / %{customdata[2]:.0f}"
                "<br>Performance-ELO: %{y:.0f} ± %{customdata[3]:.0f}"
                "<extra>%{fullData.name}</extra>"
            ),
        ))
        performances = pd.to_numeric(rows["performance_elo"], errors="coerce")
        if performances.notna().sum() >= 2:
            trend = performances.ewm(span=4, adjust=False, min_periods=2).mean()
            figure.add_trace(go.Scatter(
                x=rows["event_date"], y=trend, mode="lines",
                name=f"{round_group} recent-performance trend",
                line={
                    "color": round_colors.get(str(round_group), PALETTE["teal"]),
                    "width": 2, "dash": "dot",
                },
                opacity=0.78,
                hovertemplate=(
                    f"%{{x|%Y-%m-%d}}<br>Recent weighted {str(round_group).lower()} "
                    "level: %{y:.0f}<extra></extra>"
                ),
            ))
    event_keys = athlete_rows[["source_event_id", "pool", "round_group"]].drop_duplicates()
    field_rows = history.merge(
        event_keys, on=["source_event_id", "pool", "round_group"], how="inner"
    )
    field_rows["event_date"] = pd.to_datetime(field_rows["event_date"], errors="coerce")
    field = field_rows.groupby(
        ["source_event_id", "pool", "round_group", "event_date", "event_name"],
        as_index=False,
    ).agg(field_level=("rating_before", "median"), field_size=("global_id", "nunique"))
    figure.add_trace(go.Scatter(
        x=field["event_date"], y=field["field_level"], mode="markers",
        name="Median pre-round field Elo",
        marker={
            "symbol": "line-ew", "size": 17,
            "color": "rgba(70,91,88,.72)",
            "line": {"color": "rgba(70,91,88,.72)", "width": 2.3},
        },
        customdata=np.column_stack([field["event_name"], field["round_group"], field["field_size"]]),
        hovertemplate=(
            "%{x|%Y-%m-%d}<br>%{customdata[0]} - %{customdata[1]}"
            "<br>Median field: %{y:.0f}<br>Field: %{customdata[2]:.0f}<extra></extra>"
        ),
    ))
    latest_date = athlete_rows["event_date"].max()
    latest_rating = pd.to_numeric(athlete_rows.iloc[-1]["rating_after"], errors="coerce")
    annual_momentum = pd.to_numeric(athlete.get("momentum"), errors="coerce")
    if np.isfinite(latest_rating) and np.isfinite(annual_momentum):
        annual_momentum = float(np.clip(annual_momentum, -150, 150))
        future = latest_date + pd.Timedelta(days=365)
        projected = float(latest_rating + 0.65 * annual_momentum)
        figure.add_vline(x=latest_date, line_dash="dash", line_color="rgba(16,47,43,.55)")
        figure.add_trace(go.Scatter(
            x=[latest_date, future], y=[latest_rating, projected], mode="lines",
            name="Bounded trend hypothesis",
            line={"color": PALETTE["teal"], "dash": "dash", "width": 3},
            hovertemplate="%{x|%Y-%m-%d}<br>Hypothesis: %{y:.0f}<extra></extra>",
        ))
        figure.add_trace(go.Scatter(
            x=[latest_date, future, future, latest_date],
            y=[latest_rating - 35, projected - 90, projected + 90, latest_rating + 35],
            fill="toself", fillcolor=transparent(PALETTE["teal"], 0.10),
            line={"width": 0}, hoverinfo="skip", showlegend=False,
        ))
    add_outcome_thresholds(figure, calibration)
    figure.update_layout(
        title=None,
        height=565, hovermode="closest", legend={"orientation": "h", "y": -0.22},
        margin={"l": 65, "r": 25, "t": 65, "b": 110},
    )
    figure.update_yaxes(title="Elo on the shared World-readiness scale", tickformat=",.0f")
    figure.update_xaxes(title="Competition date")
    return figure


def render_actionable_analysis(
    athletes: pd.DataFrame, history: pd.DataFrame, selected: list[str],
    calibration: pd.DataFrame,
) -> None:
    st.header("Actionable Analysis")
    st.caption("A decision desk: what the evidence changes for the next training block or competition choice.")
    focus = selected_rows(athletes, selected).copy()
    if focus.empty:
        st.info("Select at least one athlete with results.")
        return
    focus["selection_order"] = focus.apply(lambda row: selection_order(row, selected), axis=1)
    focus = focus.sort_values("selection_order")
    decision_rows = []
    for _, athlete in focus.iterrows():
        global_elo = pd.to_numeric(athlete.get("Global-ELO"), errors="coerce")
        world_elo = pd.to_numeric(athlete.get("WC+-ELO"), errors="coerce")
        momentum = pd.to_numeric(athlete.get("momentum"), errors="coerce")
        evidence = pd.to_numeric(athlete.get("WC+-ELO evidence"), errors="coerce")
        form_signal, form_events = current_form_signal(history, athlete)
        next_start = (
            float(world_elo + 0.50 * form_signal)
            if np.isfinite(world_elo) else np.nan
        )
        gap = world_elo - global_elo if np.isfinite(world_elo) and np.isfinite(global_elo) else np.nan
        if np.isfinite(gap) and gap < -45:
            action = "Build WC-specific transfer: onsight simulations, unfamiliar setting and selected WC+ starts."
            why = f"WC+-ELO is {abs(gap):.0f} below all-competition Elo."
        elif np.isfinite(gap) and gap > 45:
            action = "Protect the process that transfers well; use starts selectively, not for volume alone."
            why = f"WC+-ELO is {gap:.0f} above all-competition Elo."
        elif form_events >= 2 and form_signal > 45:
            action = "Retest upward quickly: recent independent events are ahead of the steady estimate."
            why = f"Backtested next-start form signal is +{form_signal:.0f} Elo."
        else:
            action = "Use the next start to answer a specific terrain or round question; keep the main training block intact."
            why = "No large, well-supported transfer or momentum gap is visible."
        confidence = "higher" if np.isfinite(evidence) and evidence >= 10 else "limited"
        decision_rows.append({
            "Athlete": friendly_name(athlete["athlete_name"]),
            "What to do next": action, "Why": why,
            "Evidence confidence": f"{confidence} ({int(evidence) if np.isfinite(evidence) else 0} WC+ rounds)",
            "Next-start projection": (
                f"{next_start:.0f} Elo" if np.isfinite(next_start) else "Not available"
            ),
            "Form evidence": f"{form_events} independent events",
        })
    for decision in decision_rows:
        with st.container(border=True):
            st.markdown(f"**{decision['Athlete']}**")
            st.write(decision["What to do next"])
            st.caption(
                f"{decision['Why']} Next WC+ start: {decision['Next-start projection']} "
                f"({decision['Form evidence']}). Stable-rating evidence: "
                f"{decision['Evidence confidence']}."
            )
    athlete_options = focus["global_id"].astype(str).tolist()
    labels = dict(zip(focus["global_id"].astype(str), focus["athlete_name"].map(friendly_name)))
    chosen = st.selectbox(
        "Competition-level timeline", athlete_options,
        format_func=lambda value: labels[value], key="actionable_timeline_athlete",
    )
    athlete = focus.loc[focus["global_id"].astype(str).eq(str(chosen))].iloc[0]
    st.markdown(
        f"#### {friendly_name(athlete['athlete_name'])}: level, field and isolated performances"
    )
    st.plotly_chart(
        competition_level_figure(history, athlete, calibration),
        width="stretch", theme=None,
    )
    st.caption(
        "Dots are noisy single-round performances; the dark line is the steadier cumulative estimate. "
        "Grey marks describe the field faced. The future line is a bounded continuation of recent "
        "direction, not a promised training response; its range widens because plans and health change."
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


def _physical_gender(frame: pd.DataFrame) -> pd.Series:
    """Return a stable analysis sex label without assuming the label is complete."""
    if "gender" in frame:
        labels = frame["gender"].astype(str).str.title()
    else:
        labels = frame.get("pool", pd.Series("", index=frame.index)).astype(str)
        labels = labels.str.extract(r"_(Men|Women)$", expand=False).fillna("Unspecified")
    return labels.where(labels.isin(["Men", "Women"]), "Unspecified")


def _rank_corr(x_values: object, y_values: object) -> float:
    x_rank = pd.Series(x_values).rank()
    y_rank = pd.Series(y_values).rank()
    if x_rank.nunique(dropna=True) < 2 or y_rank.nunique(dropna=True) < 2:
        return np.nan
    return float(x_rank.corr(y_rank))


def _fit_bayesian_saturation(
    x_values: np.ndarray, y_values: np.ndarray,
) -> dict[str, object] | None:
    """Fit a model-averaged linear-then-plateau relationship.

    Each candidate threshold is a conjugate Bayesian regression using
    min(standardized_test, threshold) as its only slope term. Marginal
    likelihood weights average across thresholds instead of pretending that
    one estimated cut point is certain.
    """
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_values, y_values = x_values[valid], y_values[valid]
    if len(x_values) < 5:
        return None
    x_mean, x_sd = float(x_values.mean()), float(x_values.std(ddof=0))
    y_mean, y_sd = float(y_values.mean()), float(y_values.std(ddof=0))
    if not np.isfinite(x_sd) or x_sd <= 1e-9 or not np.isfinite(y_sd) or y_sd <= 1e-9:
        return None
    x_z = (x_values - x_mean) / x_sd
    y_z = (y_values - y_mean) / y_sd
    tau_grid = np.unique(np.concatenate([
        np.quantile(x_z, np.linspace(0.35, 0.95, 13)),
        [float(x_z.max() + 0.60)],  # a near-linear candidate
    ]))
    prior_mean = np.zeros(2)
    prior_covariance = np.diag([4.0, 2.0])
    prior_precision = np.linalg.inv(prior_covariance)
    prior_shape, prior_scale = 2.0, 1.0
    _, prior_logdet = np.linalg.slogdet(prior_covariance)
    candidates: list[dict[str, object]] = []
    for tau in tau_grid:
        design = np.column_stack([np.ones(len(x_z)), np.minimum(x_z, tau)])
        posterior_precision = prior_precision + design.T @ design
        posterior_covariance = np.linalg.inv(posterior_precision)
        posterior_mean = posterior_covariance @ (design.T @ y_z)
        posterior_shape = prior_shape + len(x_z) / 2
        posterior_scale = max(1e-9, prior_scale + 0.5 * float(
            y_z @ y_z
            + prior_mean @ prior_precision @ prior_mean
            - posterior_mean @ posterior_precision @ posterior_mean
        ))
        _, posterior_logdet = np.linalg.slogdet(posterior_covariance)
        log_marginal = (
            math.lgamma(posterior_shape) - math.lgamma(prior_shape)
            + prior_shape * np.log(prior_scale)
            - posterior_shape * np.log(posterior_scale)
            + 0.5 * (posterior_logdet - prior_logdet)
            - len(x_z) / 2 * np.log(np.pi)
        )
        candidates.append({
            "tau": float(tau), "mean": posterior_mean,
            "covariance": posterior_covariance, "shape": posterior_shape,
            "scale": posterior_scale, "log_marginal": float(log_marginal),
        })
    log_weights = np.array([item["log_marginal"] for item in candidates], dtype=float)
    weights = np.exp(log_weights - np.max(log_weights))
    weights /= weights.sum()
    return {
        "candidates": candidates, "weights": weights,
        "x_mean": x_mean, "x_sd": x_sd, "y_mean": y_mean, "y_sd": y_sd,
        "n": len(x_values),
    }


def _draw_bayesian_saturation(
    model: dict[str, object], x_evaluate: np.ndarray,
    draw_count: int = 1200, seed: int = 20260803,
) -> dict[str, np.ndarray]:
    """Draw model-averaged curves, slopes and sufficiency thresholds."""
    rng = np.random.default_rng(seed)
    candidates = model["candidates"]
    weights = np.asarray(model["weights"], dtype=float)
    model_indices = rng.choice(len(candidates), size=draw_count, p=weights)
    betas = np.empty((draw_count, 2))
    tau = np.empty(draw_count)
    for index in np.unique(model_indices):
        positions = np.flatnonzero(model_indices == index)
        candidate = candidates[int(index)]
        variance = 1.0 / rng.gamma(
            shape=float(candidate["shape"]),
            scale=1.0 / float(candidate["scale"]), size=len(positions),
        )
        root = np.linalg.cholesky(np.asarray(candidate["covariance"], dtype=float))
        noise = rng.normal(size=(len(positions), 2)) @ root.T
        betas[positions] = np.asarray(candidate["mean"], dtype=float) + (
            noise * np.sqrt(variance)[:, None]
        )
        tau[positions] = float(candidate["tau"])
    x_z = (
        np.asarray(x_evaluate, dtype=float) - float(model["x_mean"])
    ) / float(model["x_sd"])
    capped = np.minimum(x_z[None, :], tau[:, None])
    predicted_z = betas[:, [0]] + betas[:, [1]] * capped
    predictions = predicted_z * float(model["y_sd"]) + float(model["y_mean"])
    return {
        "predictions": predictions,
        "slope": betas[:, 1] * float(model["y_sd"]) / float(model["x_sd"]),
        "threshold": tau * float(model["x_sd"]) + float(model["x_mean"]),
    }


def _mean_saturation_prediction(
    model: dict[str, object], x_evaluate: np.ndarray,
) -> np.ndarray:
    x_z = (
        np.asarray(x_evaluate, dtype=float) - float(model["x_mean"])
    ) / float(model["x_sd"])
    predictions = np.zeros(len(x_z))
    for weight, candidate in zip(model["weights"], model["candidates"]):
        mean = np.asarray(candidate["mean"], dtype=float)
        predictions += float(weight) * (
            mean[0] + mean[1] * np.minimum(x_z, float(candidate["tau"]))
        )
    return predictions * float(model["y_sd"]) + float(model["y_mean"])


def _saturation_cv_comparison(
    frame: pd.DataFrame, x: str, y: str,
) -> dict[str, float | str]:
    """Compare pooled and sex-specific saturation using leave-one-athlete-out error."""
    plot = frame.dropna(subset=[x, y]).copy()
    plot["analysis_gender"] = _physical_gender(plot)
    plot = plot.loc[plot["analysis_gender"].isin(["Men", "Women"])].reset_index(drop=True)
    if len(plot) < 9:
        return {
            "choice": "Pooled", "pooled_rmse": np.nan, "gender_rmse": np.nan,
            "reason": "Too few athletes to compare held-out curves",
        }
    pooled_predictions, gender_predictions, actual = [], [], []
    for index, row in plot.iterrows():
        train = plot.drop(index=index)
        pooled = _fit_bayesian_saturation(train[x].to_numpy(), train[y].to_numpy())
        same = train.loc[train["analysis_gender"].eq(row["analysis_gender"])]
        gender_model = _fit_bayesian_saturation(same[x].to_numpy(), same[y].to_numpy())
        if pooled is None:
            continue
        pooled_predictions.append(float(_mean_saturation_prediction(pooled, [row[x]])[0]))
        chosen_gender = gender_model if gender_model is not None else pooled
        gender_predictions.append(float(_mean_saturation_prediction(chosen_gender, [row[x]])[0]))
        actual.append(float(row[y]))
    if not actual:
        return {
            "choice": "Pooled", "pooled_rmse": np.nan, "gender_rmse": np.nan,
            "reason": "Too few valid held-out predictions",
        }
    actual_values = np.asarray(actual)
    pooled_rmse = float(np.sqrt(np.mean((actual_values - pooled_predictions) ** 2)))
    gender_rmse = float(np.sqrt(np.mean((actual_values - gender_predictions) ** 2)))
    # Prefer the simpler pooled grade model unless sex-specific curves reduce
    # genuine held-out error by a practical amount.
    group_rho = []
    for _, group in plot.groupby("analysis_gender"):
        if len(group) >= 5:
            value = _rank_corr(group[x], group[y])
            if np.isfinite(value):
                group_rho.append(value)
    directional_heterogeneity = (
        len(group_rho) >= 2
        and np.nanmin(group_rho) < -0.15
        and np.nanmax(group_rho) > 0.15
    )
    if gender_rmse + 5 < pooled_rmse * 0.98:
        choice = "Gender-specific"
        reason = "Lower held-out prediction error"
    elif directional_heterogeneity:
        choice = "Gender-specific"
        reason = "Pooled line hides opposite group directions"
    else:
        choice = "Pooled"
        reason = "Separate curves did not add stable held-out value"
    return {
        "choice": choice, "pooled_rmse": pooled_rmse,
        "gender_rmse": gender_rmse, "reason": reason,
    }


def _rank_relationship_table(frame: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Return pooled and sex-specific rank relationships with bootstrap uncertainty."""
    plot = frame.dropna(subset=[x, y]).copy()
    plot["analysis_gender"] = _physical_gender(plot)
    rng = np.random.default_rng(20260803)
    rows = []
    for label, group in [("Pooled", plot), *list(plot.groupby("analysis_gender"))]:
        if label == "Unspecified" or len(group) < 5:
            continue
        rho = _rank_corr(group[x], group[y])
        draws = []
        values = group[[x, y]].to_numpy(float)
        for _ in range(500):
            sample = values[rng.integers(0, len(values), len(values))]
            draw = _rank_corr(sample[:, 0], sample[:, 1])
            if np.isfinite(draw):
                draws.append(float(draw))
        low, high = np.quantile(draws, [0.05, 0.95]) if draws else (np.nan, np.nan)
        rows.append({
            "Group": label, "Athletes": len(group), "Rank relationship": rho,
            "90% interval low": low, "90% interval high": high,
            "P(positive)": float(np.mean(np.asarray(draws) > 0)) if draws else np.nan,
        })
    return pd.DataFrame(rows)


def physical_transfer_figure(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    selected: list[str] | None = None,
    gender_specific: bool = True,
) -> tuple[go.Figure, float, dict[str, object]]:
    """Plot Bayesian saturation thresholds, separately by sex when supported."""
    plot = frame.dropna(subset=[x, y]).copy()
    plot[x] = pd.to_numeric(plot[x], errors="coerce")
    plot[y] = pd.to_numeric(plot[y], errors="coerce")
    plot = plot.dropna(subset=[x, y])
    plot["name_key"] = plot["athlete_name"].map(plain_key)
    plot["analysis_gender"] = _physical_gender(plot)
    rho = _rank_corr(plot[x], plot[y]) if len(plot) >= 3 else np.nan
    figure = go.Figure()
    if plot.empty:
        return figure, rho, {
            "probability_positive": np.nan, "slope_low": np.nan,
            "slope_high": np.nan, "draws": 0,
        }

    if gender_specific and plot["analysis_gender"].nunique() > 1:
        model_groups = [(label, group.copy()) for label, group in plot.groupby("analysis_gender")]
    else:
        model_groups = [("Pooled", plot.copy())]
    group_evidence: list[dict[str, object]] = []
    fitted_groups: list[pd.DataFrame] = []
    curve_colors = {"Men": PALETTE["blue"], "Women": PALETTE["coral"], "Pooled": PALETTE["ink"]}
    for group_index, (group_label, group) in enumerate(model_groups):
        model = _fit_bayesian_saturation(group[x].to_numpy(), group[y].to_numpy())
        if model is None:
            continue
        draws = _draw_bayesian_saturation(
            model, group[x].to_numpy(), seed=20260803 + group_index + len(group)
        )
        positive = draws["slope"] > 0
        probability_positive = float(positive.mean())
        if positive.any():
            group["probability_sufficient"] = np.mean(
                group[x].to_numpy()[None, :] >= draws["threshold"][positive, None], axis=0
            )
            group["probability_below"] = 1 - group["probability_sufficient"]
        else:
            group["probability_sufficient"] = np.nan
            group["probability_below"] = np.nan
        group["test_expected_rating"] = draws["predictions"].mean(axis=0)
        group["model_group"] = group_label
        group["sufficiency_reading"] = "Threshold unresolved"
        if probability_positive < 0.70:
            group["sufficiency_reading"] = "No stable positive test relationship"
        else:
            group.loc[group["probability_sufficient"].ge(0.75), "sufficiency_reading"] = (
                "Likely sufficient on this test"
            )
            group.loc[group["probability_below"].ge(0.75), "sufficiency_reading"] = (
                "Possibly below estimated sufficiency"
            )
        x_line = np.linspace(float(group[x].min()), float(group[x].max()), 80)
        curve_draws = _draw_bayesian_saturation(
            model, x_line, seed=20260853 + group_index + len(group)
        )["predictions"]
        curve_mean = curve_draws.mean(axis=0)
        curve_low, curve_high = np.quantile(curve_draws, [0.05, 0.95], axis=0)
        color = curve_colors.get(group_label, PALETTE["ink"])
        figure.add_trace(go.Scatter(
            x=np.concatenate([x_line, x_line[::-1]]),
            y=np.concatenate([curve_high, curve_low[::-1]]),
            fill="toself", fillcolor=transparent(color, 0.10),
            line={"color": "rgba(0,0,0,0)"}, hoverinfo="skip",
            name=f"{group_label} 90% interval", legendgroup=group_label,
            showlegend=False,
        ))
        figure.add_trace(go.Scatter(
            x=x_line, y=curve_mean, mode="lines",
            line={"color": color, "width": 2, "dash": "dash"},
            name=f"{group_label} diminishing-return estimate", legendgroup=group_label,
            hoverinfo="skip",
        ))
        slope_low, slope_high = np.quantile(draws["slope"], [0.05, 0.95])
        threshold_low, threshold_mid, threshold_high = np.quantile(
            draws["threshold"][positive] if positive.any() else draws["threshold"],
            [0.05, 0.50, 0.95],
        )
        group_evidence.append({
            "group": group_label, "athletes": len(group),
            "probability_positive": probability_positive,
            "slope_low": float(slope_low), "slope_high": float(slope_high),
            "threshold_low": float(threshold_low), "threshold": float(threshold_mid),
            "threshold_high": float(threshold_high),
        })
        fitted_groups.append(group)
    if not fitted_groups:
        return figure, rho, {
            "probability_positive": np.nan, "slope_low": np.nan,
            "slope_high": np.nan, "draws": 0, "groups": [],
        }
    plot = pd.concat(fitted_groups, ignore_index=True)
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
    colors = {
        "Likely sufficient on this test": PALETTE["teal"],
        "Possibly below estimated sufficiency": PALETTE["coral"],
        "Threshold unresolved": PALETTE["gold"],
        "No stable positive test relationship": "#A2B5B1",
    }
    selected_ids = {str(value) for value in (selected or [])}
    selected_frame = selected_rows(plot, list(selected or []))
    selected_keys = set(selected_frame["name_key"].dropna().astype(str))
    if not selected_keys:
        # Keeps older saved URLs, which used athlete names, working.
        selected_keys = {plain_key(value) for value in (selected or [])}
    for reading, group in plot.groupby("sufficiency_reading", sort=False):
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
                100 * group["probability_sufficient"],
                100 * group["probability_below"], group["rating_rounds"],
                group["model_group"],
            ]),
            hovertemplate=(
                "%{customdata[0]}<br>Test result: %{x:.2f}<br>Actual rating: %{y:.0f}"
                "<br>Saturating-model rating: %{customdata[1]:.0f}"
                "<br>P(test is at/above plateau): %{customdata[2]:.0f}%"
                "<br>P(test is below plateau): %{customdata[3]:.0f}%"
                "<br>Included rating rounds: %{customdata[4]:.0f}"
                "<br>Calibration group: %{customdata[5]}"
                "<extra>%{fullData.name}</extra>"
            ),
        ))
    figure.update_layout(
        title=title,
        xaxis_title=x.replace("_", " ").title(), yaxis_title=y,
        height=590, legend_title="Sufficiency reading",
        margin={"l": 20, "r": 20, "t": 70, "b": 30},
    )
    probabilities = [float(item["probability_positive"]) for item in group_evidence]
    return figure, rho, {
        "probability_positive": float(np.mean(probabilities)),
        "slope_low": float(min(item["slope_low"] for item in group_evidence)),
        "slope_high": float(max(item["slope_high"] for item in group_evidence)),
        "draws": 1200, "groups": group_evidence,
    }


def physical_sufficiency_table(
    latest: pd.DataFrame, athlete_name: str, rating: str = "Global-ELO",
) -> pd.DataFrame:
    """Screen test sufficiency without turning one test into a causal limiter."""
    if latest.empty or rating not in latest:
        return pd.DataFrame()
    source = latest.copy()
    source["analysis_gender"] = _physical_gender(source)
    source["value"] = pd.to_numeric(source["value"], errors="coerce")
    source[rating] = pd.to_numeric(source[rating], errors="coerce")
    context = source.get("context_only", pd.Series(False, index=source.index))
    context = context.astype(str).str.lower().isin(["true", "1", "yes"])
    source = source.loc[~context].copy()
    target = source.loc[source["athlete_name"].map(plain_key).eq(plain_key(athlete_name))]
    if target.empty:
        return pd.DataFrame()
    rows = []
    today = pd.Timestamp.now().normalize()
    for _, athlete_test in target.sort_values("test_date").drop_duplicates(
        "test_name", keep="last"
    ).iterrows():
        peers = source.loc[
            source["test_name"].eq(athlete_test["test_name"])
            & source["analysis_gender"].eq(athlete_test["analysis_gender"])
        ].dropna(subset=["value", rating])
        value = float(athlete_test["value"])
        percentile = float((peers["value"] <= value).mean()) if len(peers) else np.nan
        probability_positive = probability_sufficient = probability_below = np.nan
        threshold_low = threshold_mid = threshold_high = np.nan
        model = _fit_bayesian_saturation(peers["value"].to_numpy(), peers[rating].to_numpy())
        if model is not None:
            draws = _draw_bayesian_saturation(
                model, np.array([value]), draw_count=1000,
                seed=20260803 + len(peers) + len(str(athlete_test["test_name"])),
            )
            positive = draws["slope"] > 0
            probability_positive = float(positive.mean())
            if positive.any():
                probability_sufficient = float(
                    np.mean(value >= draws["threshold"][positive])
                )
                probability_below = 1 - probability_sufficient
                threshold_low, threshold_mid, threshold_high = np.quantile(
                    draws["threshold"][positive], [0.05, 0.50, 0.95]
                )
        tested = pd.to_datetime(athlete_test.get("test_date"), errors="coerce")
        days_old = int((today - tested).days) if pd.notna(tested) else np.nan
        if (
            np.isfinite(probability_sufficient) and probability_sufficient >= 0.75
            and probability_positive >= 0.70
        ):
            status = "Likely sufficient—not the first limiter"
        elif percentile >= 0.80:
            status = "High peer capacity—no deficit signal"
        elif (
            np.isfinite(probability_below) and probability_below >= 0.75
            and probability_positive >= 0.70 and (not np.isfinite(days_old) or days_old <= 540)
        ):
            status = "Capacity candidate—verify on matching boulders"
        else:
            status = "Unresolved—not a training prescription"
        rows.append({
            "Test": athlete_test["test_name"], "Quality": athlete_test["metric_category"],
            "Date": tested, "Result": value, "Unit": athlete_test["unit"],
            "Same-sex peers": len(peers), "Peer percentile": 100 * percentile,
            "P(positive relation)": 100 * probability_positive,
            "P(at/above sufficiency)": 100 * probability_sufficient,
            "Estimated sufficiency": threshold_mid,
            "Sufficiency 90% low": threshold_low, "Sufficiency 90% high": threshold_high,
            "Days since test": days_old, "Current reading": status,
        })
    return pd.DataFrame(rows)


def grade_evidence_summary(
    profiles: pd.DataFrame, athletes: pd.DataFrame,
) -> pd.DataFrame:
    """Build transparent pooled/sex-specific grade relationships for the maths page."""
    if profiles.empty:
        return pd.DataFrame()
    joined = profiles.copy()
    joined["name_key"] = joined["athlete_name"].map(plain_key)
    columns = [
        column for column in [
            "pool", "name_key", "gender", "Global-ELO", "IFSC-ELO", "WC+-ELO",
            "Global-ELO evidence", "IFSC-ELO evidence", "WC+-ELO evidence",
        ] if column in athletes
    ]
    if not {"pool", "name_key"}.issubset(columns):
        return pd.DataFrame()
    ratings = athletes[columns].copy()
    evidence = "Global-ELO evidence" if "Global-ELO evidence" in ratings else "Global-ELO"
    ratings = ratings.sort_values(evidence, ascending=False).drop_duplicates(["pool", "name_key"])
    joined = joined.merge(ratings, on=["pool", "name_key"], how="left")
    grade_metrics = {
        "50% flash grade": "boulder_grade_50pct_flash_v",
        "Max ≤3-send physical grade": "boulder_grade_3x_physical_sends_last_3_months_v",
    }
    rows = []
    for grade_label, grade_column in grade_metrics.items():
        if grade_column not in joined:
            continue
        for rating in RATING_ORDER:
            if rating not in joined:
                continue
            plot = joined.copy()
            plot[grade_column] = pd.to_numeric(plot[grade_column], errors="coerce")
            plot[rating] = pd.to_numeric(plot[rating], errors="coerce")
            plot = plot.dropna(subset=[grade_column, rating])
            if len(plot) < 5:
                continue
            comparison = _saturation_cv_comparison(plot, grade_column, rating)
            relationships = _rank_relationship_table(plot, grade_column, rating)
            for _, relationship in relationships.iterrows():
                rows.append({
                    "Grade": grade_label, "Rating": rating,
                    "Group": relationship["Group"], "Athletes": relationship["Athletes"],
                    "Rank relationship": relationship["Rank relationship"],
                    "90% low": relationship["90% interval low"],
                    "90% high": relationship["90% interval high"],
                    "P(positive)": relationship["P(positive)"],
                    "Held-out curve": comparison["choice"],
                    "Model reason": comparison["reason"],
                    "Pooled RMSE": comparison["pooled_rmse"],
                    "Sex-specific RMSE": comparison["gender_rmse"],
                })
    return pd.DataFrame(rows)


def _selected_priority_table(
    athletes: pd.DataFrame,
    selected: list[str],
    profiles: pd.DataFrame,
    latest: pd.DataFrame,
) -> pd.DataFrame:
    """Turn grade, Elo and test evidence into cautious athlete-level hypotheses.

    The percentages are an *analysis allocation*: where the coach should look
    next. They are deliberately not presented as weekly training volume.
    """
    if not selected:
        return pd.DataFrame()
    current = athletes.copy()
    current["name_key"] = current["athlete_name"].map(plain_key)
    current = current.sort_values(
        "Global-ELO evidence" if "Global-ELO evidence" in current else "Global-ELO",
        ascending=False,
    ).drop_duplicates("name_key")
    profile = profiles.copy()
    if not profile.empty:
        profile["name_key"] = profile["athlete_name"].map(plain_key)
        rating_columns = [
            column for column in [
                "pool", "name_key", "gender", "Global-ELO", "Global-ELO evidence",
            ] if column in current
        ]
        profile = profile.merge(
            current[rating_columns].drop_duplicates("name_key"),
            on=[column for column in ["pool", "name_key"] if column in rating_columns],
            how="left",
        )
        profile["analysis_gender"] = _physical_gender(profile)

    grade_columns = {
        "50% flash": "boulder_grade_50pct_flash_v",
        "max <=3 sends": "boulder_grade_3x_physical_sends_last_3_months_v",
    }
    grade_models: dict[tuple[str, str], dict[str, object]] = {}
    if not profile.empty:
        for label, column in grade_columns.items():
            plot = profile.dropna(subset=[column, "Global-ELO"]).copy()
            plot[column] = pd.to_numeric(plot[column], errors="coerce")
            plot["Global-ELO"] = pd.to_numeric(plot["Global-ELO"], errors="coerce")
            comparison = _saturation_cv_comparison(plot, column, "Global-ELO")
            if comparison["choice"] == "Gender-specific":
                for gender, group in plot.groupby("analysis_gender"):
                    model = _fit_bayesian_saturation(
                        group[column].to_numpy(), group["Global-ELO"].to_numpy()
                    )
                    if model is not None:
                        grade_models[(label, gender)] = model
            pooled = _fit_bayesian_saturation(
                plot[column].to_numpy(), plot["Global-ELO"].to_numpy()
            )
            if pooled is not None:
                grade_models[(label, "Pooled")] = pooled

    rows: list[dict[str, object]] = []
    for selected_token in selected:
        resolved = selected_rows(current, [selected_token])
        if resolved.empty:
            continue
        rating_row = resolved.sort_values(
            "Global-ELO evidence" if "Global-ELO evidence" in resolved else "Global-ELO",
            ascending=False,
        ).iloc[0]
        selected_name = str(rating_row.get("athlete_name", selected_token))
        key = plain_key(selected_name)
        global_elo = pd.to_numeric(rating_row.get("Global-ELO"), errors="coerce")
        evidence_rounds = pd.to_numeric(
            rating_row.get("Global-ELO evidence"), errors="coerce"
        )
        person = profile.loc[profile["name_key"].eq(key)].tail(1) if not profile.empty else pd.DataFrame()
        gender = (
            str(person.iloc[0].get("analysis_gender", "Unspecified"))
            if not person.empty else str(rating_row.get("gender", "Unspecified"))
        )
        grade_values: dict[str, float] = {}
        grade_predictions: list[float] = []
        positive_probabilities: list[float] = []
        if not person.empty:
            for label, column in grade_columns.items():
                value = pd.to_numeric(person.iloc[0].get(column), errors="coerce")
                if not np.isfinite(value):
                    continue
                grade_values[label] = float(value)
                model = grade_models.get((label, gender)) or grade_models.get((label, "Pooled"))
                if model is None:
                    continue
                draws = _draw_bayesian_saturation(
                    model, np.array([value]), draw_count=800,
                    seed=20260803 + len(key) + len(label),
                )
                grade_predictions.append(float(np.median(draws["predictions"][:, 0])))
                positive_probabilities.append(float(np.mean(draws["slope"] > 0)))

        sufficiency = physical_sufficiency_table(latest, selected_name) if not latest.empty else pd.DataFrame()
        candidates = (
            sufficiency.loc[sufficiency["Current reading"].str.startswith("Capacity candidate")]
            if not sufficiency.empty else pd.DataFrame()
        )
        supported = (
            sufficiency.loc[sufficiency["Current reading"].str.contains(
                "Likely sufficient|High peer capacity", regex=True
            )] if not sufficiency.empty else pd.DataFrame()
        )
        if not candidates.empty:
            ordered_candidates = candidates.sort_values(
                ["P(at/above sufficiency)", "Peer percentile"]
            )
            main = ordered_candidates.iloc[0]
            secondary = ordered_candidates.iloc[1] if len(ordered_candidates) > 1 else None
            main_potential = f"Verify {main['Test']} ({main['Quality']})"
            secondary_potential = (
                f"Verify {secondary['Test']} ({secondary['Quality']})"
                if secondary is not None else "No second physical deficit supported"
            )
            physical_weight = 46.0
            physical_certainty = float(np.nanmean([
                main.get("P(positive relation)", np.nan),
                main.get("P(at/above sufficiency)", np.nan),
            ]))
        else:
            strongest = (
                supported.sort_values("Peer percentile", ascending=False).iloc[0]
                if not supported.empty else None
            )
            main_potential = (
                f"Maintain {strongest['Test']}; no deficit signal"
                if strongest is not None else "No supported physical limiter yet"
            )
            secondary_potential = "Collect/refresh tests, then verify on matching boulders"
            physical_weight = 24.0 if strongest is not None else 32.0
            physical_certainty = (
                float(strongest.get("P(at/above sufficiency)", np.nan))
                if strongest is not None else np.nan
            )

        grade_prediction = float(np.mean(grade_predictions)) if grade_predictions else np.nan
        transfer_gap = (
            float(global_elo - grade_prediction)
            if np.isfinite(global_elo) and np.isfinite(grade_prediction) else np.nan
        )
        flash_gap = (
            grade_values.get("max <=3 sends", np.nan) - grade_values.get("50% flash", np.nan)
        )
        # A large project-to-flash gap points to conversion work, but cannot tell
        # technical and coordinative causes apart until boulders are tagged.
        technical_weight = 38.0
        coordination_weight = 38.0
        if np.isfinite(flash_gap):
            conversion_signal = float(np.clip((flash_gap - 1.0) * 6.0, -5.0, 12.0))
            technical_weight += 0.55 * conversion_signal
            coordination_weight += 0.45 * conversion_signal
        if np.isfinite(transfer_gap) and transfer_gap < -40:
            technical_weight += 5.0
            coordination_weight += 5.0
        total = physical_weight + technical_weight + coordination_weight
        physical_pct = int(round(100 * physical_weight / total))
        technical_pct = int(round(100 * technical_weight / total))
        coordination_pct = 100 - physical_pct - technical_pct

        grade_confidence = (
            float(np.mean(positive_probabilities)) if positive_probabilities else np.nan
        )
        evidence_factor = min(1.0, float(evidence_rounds) / 12.0) if np.isfinite(evidence_rounds) else 0.0
        data_factor = min(1.0, (len(grade_values) + min(len(sufficiency), 3)) / 5.0)
        certainty_score = np.nanmean([
            grade_confidence if np.isfinite(grade_confidence) else 0.25,
            evidence_factor, data_factor,
        ])
        certainty = "high" if certainty_score >= 0.72 else "moderate" if certainty_score >= 0.48 else "low"
        grade_text = (
            ", ".join(f"{name} V{value:.1f}" for name, value in grade_values.items())
            if grade_values else "no linked grade report"
        )
        if np.isfinite(transfer_gap):
            if transfer_gap < -40:
                transfer_text = (
                    f"Grades imply about {abs(transfer_gap):.0f} more Elo than current results; "
                    "prioritize competition conversion and matching-style evidence."
                )
            elif transfer_gap > 40:
                transfer_text = (
                    f"Competition Elo is about {transfer_gap:.0f} above the grade-only estimate; "
                    "do not reduce the athlete to gym grades."
                )
            else:
                transfer_text = "Grade-only and competition estimates are broadly aligned."
        else:
            transfer_text = "Grade-to-Elo transfer cannot yet be estimated."
        rows.append({
            "Athlete": friendly_name(selected_name),
            "Main physical potential": f"{main_potential} ({certainty})",
            "Secondary physical potential": f"{secondary_potential} ({certainty})",
            "Physical focus": f"{physical_pct}%",
            "Technical focus": f"{technical_pct}%",
            "Coordination focus": f"{coordination_pct}%",
            "Key underperformance tags": "Pending matched tagged boulders + athlete outcomes (low)",
            "Explanation": (
                f"Global-ELO {global_elo:.0f} from {evidence_rounds:.0f} rounds; {grade_text}. "
                f"{transfer_text} Percentages allocate the next analysis, not training volume. "
                f"The physical candidate still needs terrain-demand and observed-failure checks ({certainty})."
                if np.isfinite(global_elo) else
                f"No linked Global-ELO; {grade_text}. Collect competition and matched-boulder evidence before prescribing ({certainty})."
            ),
        })
    return pd.DataFrame(rows)


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
        live_rating_columns = [
            column for column in [
                *RATING_ORDER, *[f"{family} evidence" for family in RATING_ORDER]
            ] if column in athletes
        ]
        if live_rating_columns:
            latest = latest.drop(
                columns=[
                    column for column in ["global_id", *live_rating_columns]
                    if column in latest
                ],
                errors="ignore",
            )
            evidence_lookup = (
                athletes[["pool", "name_key", "global_id", *live_rating_columns]]
                .sort_values(
                    "Global-ELO evidence"
                    if "Global-ELO evidence" in live_rating_columns
                    else "Global-ELO",
                    ascending=False,
                )
                .drop_duplicates(["pool", "name_key"])
            )
            latest = latest.merge(
                evidence_lookup, on=["pool", "name_key"], how="left"
            )

    profiles = data.get("physical_profiles", pd.DataFrame())
    priority_table = _selected_priority_table(athletes, selected, profiles, latest)
    if not priority_table.empty:
        st.markdown("#### Selected athletes - working priorities")
        st.caption(
            "Kilter-equivalent grades use the reported conversion already supplied in the "
            "testing database (Tension +1 grade; Moon +1.5). Percentages show where to "
            "investigate next, not a weekly training-volume prescription."
        )
        st.dataframe(
            priority_table, hide_index=True, width="stretch",
            height=min(470, 76 + 46 * len(priority_table)),
            column_config={
                "Explanation": st.column_config.TextColumn(width="large"),
                "Key underperformance tags": st.column_config.TextColumn(width="medium"),
            },
        )
        with st.expander("How to read this table"):
            st.write(
                "A physical test can identify a capacity worth checking; it cannot, by itself, "
                "prove the athlete failed because of that capacity. Style and movement tags "
                "will replace the pending column only after the same boulder is linked to the "
                "athlete's result or attempts. Certainty reflects grade coverage, rating rounds "
                "and same-sex test evidence."
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
        sufficiency = physical_sufficiency_table(latest, athlete_name)
        if sufficiency.empty:
            st.info("No current same-sex sufficiency screen is available for this athlete.")
            return
        likely_sufficient = sufficiency.loc[
            sufficiency["Current reading"].str.contains(
                "Likely sufficient|High peer capacity", regex=True
            )
        ]
        candidates = sufficiency.loc[
            sufficiency["Current reading"].str.startswith("Capacity candidate")
        ]
        cards = st.columns(3)
        cards[0].metric("Physical tests", f"{len(sufficiency)}")
        cards[1].metric("No deficit signal", f"{len(likely_sufficient)}")
        cards[2].metric("Capacity candidates", f"{len(candidates)}")
        if candidates.empty:
            st.info(
                "No physical limiter can currently be defended. A sufficient test can rule out "
                "one simple explanation; it cannot prove what is limiting performance."
            )
        else:
            first = candidates.sort_values("P(at/above sufficiency)").iloc[0]
            st.warning(
                f"First capacity to verify: {first['Test']} ({first['Quality']}). It becomes a "
                "limiter only if matching boulders demand it and the athlete's attempts show it "
                "is the first constraint reached."
            )
        if plain_key(athlete_name) == plain_key("Louka Boivin"):
            louka_20mm = sufficiency.loc[
                sufficiency["Test"].astype(str).str.fullmatch(
                    "20mm HC Semi-Isolé", case=False, na=False
                )
            ].sort_values("Date").tail(1)
            if not louka_20mm.empty:
                item = louka_20mm.iloc[0]
                st.success(
                    f"Louka sanity check: {item['Result']:.1f} {item['Unit']} on {item['Test']} "
                    f"is around the {item['Peer percentile']:.0f}th percentile of comparable men. "
                    "The evidence argues against maximal 20 mm strength as his first limiter. "
                    "Maintain it; investigate terrain transfer, movement, tactics and other "
                    "capacities before adding more finger-strength emphasis."
                )
        ordered = sufficiency.sort_values(
            ["Current reading", "Peer percentile"], ascending=[True, True]
        ).head(18).sort_values("Peer percentile")
        if not ordered.empty:
            recommendation_colors = {
                "Capacity candidate—verify on matching boulders": PALETTE["coral"],
                "Likely sufficient—not the first limiter": PALETTE["teal"],
                "High peer capacity—no deficit signal": PALETTE["blue"],
                "Unresolved—not a training prescription": PALETTE["gold"],
            }
            figure = go.Figure()
            for label, group in ordered.groupby("Current reading", sort=False):
                figure.add_trace(go.Bar(
                    x=group["Peer percentile"], y=group["Test"],
                    orientation="h", name=label,
                    marker_color=recommendation_colors.get(label, "#A8B6B3"),
                    customdata=np.column_stack([
                        group["Result"], group["Unit"], group["Same-sex peers"],
                        group["Date"], group["P(at/above sufficiency)"],
                    ]),
                    hovertemplate=(
                        "%{y}<br>Peer percentile: %{x:.0f}"
                        "<br>Result: %{customdata[0]} %{customdata[1]}"
                        "<br>Peer athletes: %{customdata[2]}"
                        "<br>Test date: %{customdata[3]}"
                        "<br>P(at/above estimated sufficiency): %{customdata[4]:.0f}%"
                        "<extra></extra>"
                    ),
                ))
            figure.add_vline(x=35, line_dash="dot", line_color=PALETTE["coral"])
            figure.add_vline(x=70, line_dash="dot", line_color=PALETTE["teal"])
            figure.update_layout(
                title=f"{friendly_name(athlete_name)} — physical profile against comparable peers",
                xaxis={"title": "Percentile among same-sex peers",
                       "range": [0, 100]},
                yaxis_title="", barmode="overlay", height=max(520, 34 * len(ordered)),
                margin={"l": 20, "r": 20, "t": 70, "b": 30},
            )
            st.plotly_chart(figure, width="stretch", theme=None)
        shown = sufficiency[[
            "Test", "Quality", "Date", "Result", "Unit", "Peer percentile",
            "Same-sex peers", "P(positive relation)", "P(at/above sufficiency)",
            "Estimated sufficiency", "Days since test", "Current reading",
        ]].sort_values(["Current reading", "Peer percentile"])
        st.dataframe(shown, hide_index=True, width="stretch")
        st.markdown("#### The first-limiter check")
        st.dataframe(pd.DataFrame([
            {"Gate": "Capacity", "Question": "Is the fresh test probably below sufficiency?",
             "Current evidence": "Estimated above; uncertainty is shown per test."},
            {"Gate": "Demand", "Question": "Did the target boulder strongly require this quality?",
             "Current evidence": "Pending structured boulder-style tags."},
            {"Gate": "Observed failure", "Question": "Did attempts fail first for this reason?",
             "Current evidence": "Requires matching on-wall observations."},
            {"Gate": "Actionability", "Question": "Can training improve it safely in the available time?",
             "Current evidence": "Coach judgement, health and training response required."},
        ]), hide_index=True, width="stretch")
        st.caption(
            "Only converging gates justify a training priority. The model deliberately calls a "
            "below-threshold result a capacity candidate—not a limiter or a prescription."
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
        rating = st.selectbox("Rating", ["Global-ELO", "IFSC-ELO", "WC+-ELO"])
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
            group_table = pd.DataFrame(transfer_evidence.get("groups", [])).rename(columns={
                "group": "Group", "athletes": "Athletes",
                "probability_positive": "P(positive relation)",
                "threshold": "Estimated sufficiency",
                "threshold_low": "Sufficiency 90% low",
                "threshold_high": "Sufficiency 90% high",
            })
            if not group_table.empty:
                group_table["P(positive relation)"] *= 100
                st.dataframe(group_table[[
                    "Group", "Athletes", "P(positive relation)",
                    "Estimated sufficiency", "Sufficiency 90% low",
                    "Sufficiency 90% high",
                ]], hide_index=True, width="stretch")
            st.info(
                "Each sex is calibrated separately. The curve rises below an uncertain "
                "sufficiency point and then flattens: more capacity past that point is not "
                "assumed to buy the same Elo gain. Hover shows threshold probabilities and "
                "how many rounds support the displayed Elo."
            )
            st.caption(
                f"{plot['testing_person_key'].nunique()} linked athletes. A below-threshold result "
                "is only a capacity candidate. Terrain demand and the first observed cause of "
                "failure must agree before it becomes a limiter hypothesis."
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
        "pool", "name_key", "global_id", "Global-ELO", "IFSC-ELO", "WC+-ELO",
        "Global-ELO evidence", "IFSC-ELO evidence", "WC+-ELO evidence",
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
        model_comparison = _saturation_cv_comparison(plot, value_column, rating)
        use_gender_model = model_comparison["choice"] == "Gender-specific"
        figure, grade_rho, grade_transfer_evidence = physical_transfer_figure(
            plot, value_column, rating, f"{test} and {rating}", selected,
            gender_specific=use_gender_model,
        )
        st.plotly_chart(figure, width="stretch", theme=None)
        relationship_table = _rank_relationship_table(plot, value_column, rating)
        if not relationship_table.empty:
            st.markdown("#### Grade relationships worth retaining")
            relationship_shown = relationship_table.copy()
            relationship_shown["P(positive)"] *= 100
            relationship_shown = relationship_shown.rename(
                columns={"P(positive)": "P(positive), %"}
            )
            st.dataframe(relationship_shown, hide_index=True, width="stretch")
        st.info(
            f"Held-out model choice: {model_comparison['choice']}. Pooled saturation RMSE: "
            f"{model_comparison['pooled_rmse']:.0f} Elo; sex-specific saturation RMSE: "
            f"{model_comparison['gender_rmse']:.0f} Elo. Reason: {model_comparison['reason']}. "
            "Grades are pooled unless separate "
            "curves materially predict unseen athletes better. Physical tests remain "
            "sex-specific by default."
        )
        st.caption(
            f"{len(plot)} linked athletes are visible for this exact grade × rating pair. "
            "Rank relationships retain non-linear ordering; the curve allows diminishing "
            "returns. Neither establishes a causal training effect."
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
            "WC+-ELO": "Current WC+ Elo",
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


def render_prediction_backtest(summary: pd.DataFrame) -> None:
    st.markdown("#### Which score best predicts the next competition?")
    if summary.empty:
        st.caption("The frozen chronological comparison is being rebuilt.")
        return
    scopes = summary["comparison_scope"].dropna().astype(str).unique().tolist()
    scope = st.selectbox("Backtest comparison", scopes, key="prediction_backtest_scope")
    shown = summary.loc[summary["comparison_scope"].eq(scope)].copy()
    shown = shown.loc[pd.to_numeric(shown["events"], errors="coerce").gt(0)]
    if shown.empty:
        st.caption("No eligible future competitions for this comparison.")
        return
    shown["Field ordering"] = 100 * pd.to_numeric(shown["mean_rank_correlation"], errors="coerce")
    shown["Head-to-head ordering"] = 100 * pd.to_numeric(shown["pairwise_accuracy"], errors="coerce")
    shown["Top-8 probability error"] = pd.to_numeric(shown["top8_brier"], errors="coerce")
    shown["Momentum weight"] = pd.to_numeric(shown["momentum_gain"], errors="coerce")
    chart = px.bar(
        shown.sort_values("Field ordering"), x="Field ordering", y="model",
        orientation="h", color="Field ordering", color_continuous_scale="Teal",
        range_color=[
            float(shown["Field ordering"].min()),
            float(shown["Field ordering"].max()),
        ],
        hover_data={
            "events": True, "athlete_starts": True,
            "Head-to-head ordering": ":.1f",
            "Top-8 probability error": ":.3f",
            "Momentum weight": ":.2f",
        },
        title="Frozen-before-event field ordering",
    )
    chart.update_layout(
        height=max(350, 48 * len(shown) + 130), showlegend=False,
        coloraxis_showscale=False, margin={"l": 145, "r": 30, "t": 65, "b": 55},
    )
    chart.update_xaxes(
        title="Average rank correlation with the next result (%)", ticksuffix="%"
    )
    chart.update_yaxes(title="")
    st.plotly_chart(chart, width="stretch", theme=None)
    table = shown[[
        "model", "events", "athlete_starts", "Field ordering",
        "Head-to-head ordering", "Top-8 probability error", "Momentum weight",
    ]].rename(columns={
        "model": "Model", "events": "Competitions",
        "athlete_starts": "Athlete-starts",
    })
    st.dataframe(
        table.sort_values("Field ordering", ascending=False), hide_index=True, width="stretch",
        column_config={
            "Field ordering": st.column_config.NumberColumn(format="%.1f%%"),
            "Head-to-head ordering": st.column_config.NumberColumn(format="%.1f%%"),
            "Top-8 probability error": st.column_config.NumberColumn(format="%.3f"),
            "Momentum weight": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    best_order = shown.loc[shown["Field ordering"].idxmax()]
    probability_candidates = shown.dropna(subset=["Top-8 probability error"])
    if not probability_candidates.empty:
        best_probability = probability_candidates.loc[
            probability_candidates["Top-8 probability error"].idxmin()
        ]
        st.success(
            f"Best field ordering here: {best_order['model']} "
            f"({best_order['Field ordering']:.1f}%). Lowest top-8 probability error: "
            f"{best_probability['model']} ({best_probability['Top-8 probability error']:.3f}). "
            "Those can differ: ordering a field and calibrating advancement probabilities are "
            "related, but not identical jobs."
        )
    else:
        st.info(
            f"Best field ordering in this published evidence set: {best_order['model']} "
            f"({best_order['Field ordering']:.1f}%). Probability scores were not built for "
            "this domestic comparison."
        )
    st.caption(
        "Every score is frozen before the competition. Momentum weights are selected only "
        "on pre-2025 events, then judged from 2025 onward. Higher ordering is better; lower "
        "probability error is better. CNR and CUWR comparisons use only athlete-fields where "
        "both the ranking score and Elo existed before the result."
    )


def render_performance_elo_dependence_audit(summary: pd.DataFrame) -> None:
    """Explain and quantify the within-round dependence limitation."""

    st.markdown("#### Does Performance-ELO count one round too many times?")
    st.write(
        "One placing creates many beat/lost-to comparisons, but those comparisons are "
        "connected: the athlete produced one round, not dozens of independent tests. We "
        "therefore tested a second calculation that treats the entire ordered field as one "
        "joint ranking. Ties are kept as tied groups."
    )
    if summary.empty:
        st.caption("The dependence comparison is being rebuilt.")
        return
    target = summary.loc[
        summary["comparison_scope"].astype(str).eq(
            "WC+ next competition · all entrants"
        )
        & summary["model"].isin([
            "Global-ELO + momentum",
            "Global-ELO + joint-ranking momentum",
        ])
    ].copy()
    if target.empty:
        st.caption("The dependence comparison is not present in this data release.")
        return
    target["Method"] = target["model"].map({
        "Global-ELO + momentum": "Responsive beat/lost-to posterior (production)",
        "Global-ELO + joint-ranking momentum": "One joint ranking per round (challenger)",
    })
    shown = target.rename(columns={
        "events": "Competitions",
        "athlete_starts": "Athlete-starts",
        "mean_rank_correlation": "Field-order agreement",
        "pairwise_accuracy": "Head-to-head accuracy",
        "top8_brier": "Top-8 probability error",
        "winner_log_loss": "Winner probability error",
        "momentum_gain": "Tuned momentum weight",
    })
    st.dataframe(
        shown[[
            "Method", "Competitions", "Athlete-starts", "Field-order agreement",
            "Head-to-head accuracy", "Top-8 probability error",
            "Winner probability error", "Tuned momentum weight",
        ]],
        hide_index=True,
        width="stretch",
        column_config={
            "Field-order agreement": st.column_config.NumberColumn(format="%.3f"),
            "Head-to-head accuracy": st.column_config.NumberColumn(format="%.3f"),
            "Top-8 probability error": st.column_config.NumberColumn(format="%.3f"),
            "Winner probability error": st.column_config.NumberColumn(format="%.3f"),
            "Tuned momentum weight": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    production = target.loc[target["model"].eq("Global-ELO + momentum")]
    challenger = target.loc[
        target["model"].eq("Global-ELO + joint-ranking momentum")
    ]
    if len(production) == 1 and len(challenger) == 1:
        prod = production.iloc[0]
        joint = challenger.iloc[0]
        all_three = (
            float(prod["mean_rank_correlation"]) > float(joint["mean_rank_correlation"])
            and float(prod["pairwise_accuracy"]) > float(joint["pairwise_accuracy"])
            and float(prod["top8_brier"]) < float(joint["top8_brier"])
        )
        if all_three:
            st.success(
                "Decision: keep the responsive posterior in production. The joint-ranking "
                "challenger is mathematically cleaner about within-round dependence, but it "
                "was slightly worse on every frozen next-WC+ measure. We do not trade useful "
                "sensitivity for a theoretical improvement that did not improve prediction."
            )
        else:
            st.warning(
                "The two methods split the prediction measures. Neither earns an automatic "
                "promotion; retain both in shadow evaluation on new competitions."
            )
    with st.expander("What the improved caveat means · ?"):
        st.markdown(
            "- **What is reliable:** the posterior mean is a responsive description of the "
            "WC-level performance implied by whom the athlete beat and lost to.\n"
            "- **What is overstated:** its displayed SD conditions on one-dimensional Elo, "
            "fixed opponent ratings and the composite pair likelihood. It is **not** a "
            "calibrated 90% or 95% interval for the athlete's true ability.\n"
            "- **What the challenger fixes:** it counts the ordering as one joint ranking and "
            "handles ties together.\n"
            "- **What the challenger still assumes:** one latent skill and the "
            "Plackett–Luce independence-of-irrelevant-alternatives structure. Terrain fit, "
            "health, tactics and shared event conditions remain outside the likelihood."
        )
        st.markdown(
            "Method reference: [Turner et al., *Modelling rankings in R: the "
            "PlackettLuce package*](https://arxiv.org/abs/1810.12068)."
        )


def render_program_backtest(summary: pd.DataFrame) -> None:
    """Show frozen forecast quality for the current EEQ and CNR top-15 cohorts."""

    st.markdown("#### Does the model work for the athletes we actually support?")
    if summary.empty:
        st.caption("The EEQ and Canadian-team cohort checks are being rebuilt.")
        return
    cohorts = summary["cohort"].dropna().astype(str).unique().tolist()
    if not cohorts:
        return
    cohort = st.selectbox("Athlete group", cohorts, key="program_backtest_cohort")
    available_scopes = summary.loc[
        summary["cohort"].eq(cohort), "scope"
    ].dropna().astype(str).unique().tolist()
    preferred = (
        available_scopes.index("All accessible competitions")
        if "All accessible competitions" in available_scopes else 0
    )
    scope = st.selectbox(
        "Competitions used for the check",
        available_scopes,
        index=preferred,
        key="program_backtest_scope",
    )
    shown = summary.loc[
        summary["cohort"].eq(cohort) & summary["scope"].eq(scope)
    ].copy()
    shown = shown.loc[pd.to_numeric(shown["athlete_starts"], errors="coerce").gt(0)]
    if shown.empty:
        st.caption("No eligible starts for this group and competition scope.")
        return
    shown["Probability error"] = pd.to_numeric(
        shown["pairwise_brier"], errors="coerce"
    )
    shown["Correct head-to-head order"] = 100 * pd.to_numeric(
        shown["pairwise_accuracy"], errors="coerce"
    )
    shown["Expected-place percentile error"] = 100 * pd.to_numeric(
        shown["placement_percentile_mae"], errors="coerce"
    )
    order = [
        "Equal-athlete baseline", "Global-ELO",
        "Global-ELO + joint-ranking momentum",
        "Global-ELO + pairwise momentum",
    ]
    shown["_order"] = shown["model"].map({name: index for index, name in enumerate(order)})
    shown = shown.sort_values("_order")
    shown["Chart label"] = shown["model"].map({
        "Equal-athlete baseline": "Equal baseline",
        "Global-ELO": "Global-ELO",
        "Global-ELO + joint-ranking momentum": "+ joint momentum",
        "Global-ELO + pairwise momentum": "+ responsive momentum",
    }).fillna(shown["model"])
    chart = px.bar(
        shown,
        x="Probability error",
        y="Chart label",
        orientation="h",
        color="model",
        color_discrete_map={
            "Equal-athlete baseline": "#94a3b8",
            "Global-ELO": "#2563eb",
            "Global-ELO + joint-ranking momentum": "#f59e0b",
            "Global-ELO + pairwise momentum": "#0f766e",
        },
        hover_data={
            "athletes": True,
            "competitions": True,
            "athlete_starts": True,
            "opponent_pairs": True,
            "Correct head-to-head order": ":.1f",
            "Expected-place percentile error": ":.1f",
        },
        title="Head-to-head probability error (lower is better)",
    )
    chart.update_layout(
        height=390,
        showlegend=False,
        margin={"l": 145, "r": 20, "t": 65, "b": 50},
    )
    chart.update_yaxes(title="")
    st.plotly_chart(chart, width="stretch", theme=None)
    table = shown[[
        "model", "athletes", "competitions", "athlete_starts", "opponent_pairs",
        "Probability error", "pairwise_brier_ci90_low", "pairwise_brier_ci90_high",
        "Correct head-to-head order", "Expected-place percentile error",
        "mean_probability_bias", "evidence_grade",
    ]].rename(columns={
        "model": "Model", "athletes": "Athletes", "competitions": "Competitions",
        "athlete_starts": "Athlete-starts", "opponent_pairs": "Head-to-head cases",
        "pairwise_brier_ci90_low": "Error 90% low",
        "pairwise_brier_ci90_high": "Error 90% high",
        "mean_probability_bias": "Mean probability bias",
        "evidence_grade": "Evidence coverage",
    })
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Probability error": st.column_config.NumberColumn(format="%.3f"),
            "Error 90% low": st.column_config.NumberColumn(format="%.3f"),
            "Error 90% high": st.column_config.NumberColumn(format="%.3f"),
            "Correct head-to-head order": st.column_config.NumberColumn(format="%.1f%%"),
            "Expected-place percentile error": st.column_config.NumberColumn(format="%.1f%%"),
            "Mean probability bias": st.column_config.NumberColumn(format="%+.3f"),
        },
    )
    best = shown.loc[shown["Probability error"].idxmin()]
    st.info(
        f"Best probability forecast in this view: **{best['model']}** "
        f"(error {best['Probability error']:.3f}; "
        f"{best['Correct head-to-head order']:.1f}% of head-to-head orders correct)."
    )
    st.warning(
        "Important cohort caveat: this is an honest frozen replay for athletes who are in "
        "today's group, not a reconstruction of who belonged to the group at the time. EEQ "
        "uses the 2025-2026 roster. The Canadian National Team view is a transparent proxy—"
        "today's CNR top 15—not an official historical team list. Selection and survivorship "
        "therefore affect the cohort; the comparison evaluates forecast usefulness for these "
        "athletes, not the causal effect of the program."
    )
    if scope == "WC+ competitions" and (
        pd.to_numeric(shown["athlete_starts"], errors="coerce").max() < 40
    ):
        st.caption(
            "WC+ evidence is sparse for this group. Use the result as a directional audit, "
            "not a selection threshold; the broader competition view is more stable."
        )


def render_maths_behind(
    athletes: pd.DataFrame, correlations: pd.DataFrame, calibration: pd.DataFrame,
    data: dict[str, pd.DataFrame],
) -> None:
    st.header("Maths behind")
    st.caption("What each model is for, what it can predict, and where it can fail.")
    render_prediction_backtest(data.get("prediction_backtest", pd.DataFrame()))
    render_performance_elo_dependence_audit(
        data.get("prediction_backtest", pd.DataFrame())
    )
    render_program_backtest(data.get("program_backtest", pd.DataFrame()))
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
            "Model": "WC+-ELO", "Best use": "World Cups/Series and harder-event performance",
            "Evidence": "Only WC+ events",
            "Strength": "Most specific to ranking access",
            "Main caveat": "Sparse and participation-selected evidence",
        },
        {
            "Model": "Performance-ELO", "Best use": "Describe one round",
            "Evidence": "One frozen pre-event field and observed result",
            "Strength": "Makes surprise and momentum visible",
            "Main caveat": "Responsive composite likelihood; its SD is conditional, not a full ability interval",
        },
    ])
    st.dataframe(comparison, hide_index=True, width="stretch")
    st.markdown("#### Physical capacity: why the straight line was rejected")
    st.write(
        "A straight line says that every extra kilogram on a test buys the same expected Elo. "
        "That is implausible for a multi-constraint sport. Too little capacity can make a move "
        "impossible; enough capacity removes that constraint; additional capacity may save time "
        "or attempts, but usually with diminishing returns. The app now fits physical tests "
        "separately for men and women with an uncertain linear-then-plateau curve."
    )
    st.latex(r"Elo_i = \alpha_g + \beta_g\,\min(z_i,\tau_g) + \varepsilon_i")
    st.caption(
        "g is the sex-specific pool, z is the standardized test, and τ is the unknown "
        "sufficiency point. The Bayesian fit averages many possible τ values. It reports "
        "P(at/above sufficiency), not a fake exact cutoff. A pooled curve remains available "
        "for self-reported grades and is selected unless sex-specific curves predict held-out "
        "athletes materially better."
    )
    st.dataframe(pd.DataFrame([
        {
            "Option": "Straight line", "What it assumes": "Constant Elo return per test unit",
            "Why keep it": "Simple diagnostic baseline",
            "Why not use it for decisions": "Extrapolates endless benefit and turns residuals into false deficits",
            "Status": "Rejected as decision model",
        },
        {
            "Option": "Bayesian linear-to-plateau", "What it assumes": "Benefit below an uncertain sufficiency point, then diminishing return",
            "Why keep it": "Interpretable with small samples; threshold uncertainty is explicit",
            "Why not use it for decisions": "A population plateau is not a boulder-specific requirement",
            "Status": "Current screening model",
        },
        {
            "Option": "Hill / logistic curve", "What it assumes": "Smooth S-shaped response",
            "Why keep it": "Physiologically plausible when low values have little effect and middle values matter most",
            "Why not use it for decisions": "More weakly identified parameters at current sample sizes",
            "Status": "Backtest challenger",
        },
        {
            "Option": "Isotonic or spline model", "What it assumes": "Monotonic shape learned from data",
            "Why keep it": "Can discover several bends without choosing one threshold",
            "Why not use it for decisions": "Can follow noise and is unstable outside observed values",
            "Status": "Use when the database grows",
        },
        {
            "Option": "Hierarchical sex-specific model", "What it assumes": "Separate curves share information instead of being fully pooled or split",
            "Why keep it": "Best compromise for unequal male/female samples",
            "Why not use it for decisions": "Needs more repeated tests and careful athlete-level validation",
            "Status": "Preferred next upgrade",
        },
        {
            "Option": "Boulder-demand first-limiter model", "What it assumes": "The smallest capacity-minus-demand margin controls success",
            "Why keep it": "Matches the coaching question: what failed first on this terrain?",
            "Why not use it for decisions": "Requires style tags and attempt-level observations",
            "Status": "Target model",
        },
    ]), hide_index=True, width="stretch")
    st.markdown("#### The proposed first-limiter model")
    st.write(
        "A competition boulder is not an average of finger strength, power, technique and "
        "coordination. The first demand that exceeds the athlete's usable capacity often "
        "controls the attempt. The next model will combine athlete capacities with each "
        "tagged boulder's demands through a smooth minimum:"
    )
    st.latex(
        r"m_{ij}=\operatorname{softmin}_k(C_{ik}-D_{jk}),\qquad "
        r"P(\mathrm{success}_{ij})=\operatorname{logit}^{-1}(a+b\,m_{ij})"
    )
    st.caption(
        "C is athlete capacity, D is boulder demand, and the smallest capacity-minus-demand "
        "margin dominates. This is not yet a production causal model: it requires style tags, "
        "attempt-level failure evidence and enough repeated boulders. Until then the app says "
        "capacity candidate, likely sufficient, or unresolved—never 'train this' from one residual."
    )
    st.markdown("#### Style Elo: each boulder is a mini-round")
    st.write(
        "Physical-, Technical- and Coordination-Elo will be separate ratings, not relabelled "
        "Global-Elo. Every boulder becomes one mini-round, divided into start→zone and "
        "zone→top. Missing the zone is not counted as failure after the zone: that second "
        "segment is unobserved for the athlete. This keeps an early failure from falsely "
        "diagnosing the top section."
    )
    st.latex(
        r"\Delta R^s_{i,e}=\frac{q_eK_{i,e}}{B_e}\sum_{b=1}^{B_e}"
        r"\left[w_Zd^s_{b,Z}r_{i,b,Z}+w_Td^s_{b,T}r_{i,b,T}\right],"
        r"\qquad w_Z+w_T=1"
    )
    st.caption(
        "s is physical, technical or coordination; B is the source-confirmed boulder count; "
        "d is the 0–3 demand tag converted to 0–1; r is observed minus expected evidence; "
        "and q is event quality. Dividing by B prevents a six-boulder round from carrying 50% "
        "more total weight than a four-boulder round. Paired athlete updates are symmetric, "
        "so the field remains zero-sum. Start→zone uses every starter. Zone→top is conditional "
        "on reaching zone. Initial segment weights are ½ and ½; frozen backtests may change "
        "them, but their sum stays one."
    )
    st.info(
        "Production gate: publish style Elo only after boulder outcomes are joined to stable "
        "boulder IDs, independent taggers agree sufficiently, and frozen next-event tests show "
        "that the style rating adds predictive value beyond Global-Elo. Until then it is a "
        "research layer, not a selection score."
    )
    st.markdown("#### How this fits the climbing literature")
    st.markdown(
        "- A randomized trial improved dynamic and isometric finger strength without a "
        "detectable improvement in bouldering performance. That is direct evidence against "
        "treating strength change as automatic performance transfer "
        "([Saeterbakken et al., 2024](https://pubmed.ncbi.nlm.nih.gov/39450143/)).\n"
        "- In a 67-climber bouldering competition, finger strength, experience and climbing "
        "frequency jointly predicted performance; no single test explained the full result "
        "([Stefan et al., 2022](https://pubmed.ncbi.nlm.nih.gov/35235904/)).\n"
        "- Test-performance relationships differed between female and male groups in an "
        "intermittent finger-endurance study, supporting sex-specific physical calibration "
        "rather than one forced line "
        "([Augste et al., 2022](https://pubmed.ncbi.nlm.nih.gov/35677360/)).\n"
        "- Force on a deep edge did not predict force on very shallow edges in experienced "
        "climbers, showing that even 'finger strength' is hold-specific "
        "([Bourne et al., 2011](https://pubmed.ncbi.nlm.nih.gov/21451181/))."
    )
    st.warning(
        "The literature supports specificity, multiple constraints and imperfect transfer. "
        "It does not prove this app's exact plateau or first-limiter equation. Those are "
        "testable modelling hypotheses and must earn promotion through frozen future-event "
        "validation."
    )
    grade_summary = grade_evidence_summary(
        data.get("physical_profiles", pd.DataFrame()), athletes
    )
    if not grade_summary.empty:
        st.markdown("#### Valuable self-reported grade relationships")
        grade_shown = grade_summary.copy()
        grade_shown["P(positive)"] *= 100
        grade_shown = grade_shown.rename(columns={"P(positive)": "P(positive), %"})
        st.dataframe(
            grade_shown.sort_values(
                ["Grade", "Rating", "Group"]
            ), hide_index=True, width="stretch",
        )
        eligible = grade_summary.loc[grade_summary["Athletes"].ge(8)].copy()
        if not eligible.empty:
            strongest = eligible.sort_values("P(positive)", ascending=False).iloc[0]
            st.info(
                f"Strongest current positive direction: {strongest['Grade']} versus "
                f"{strongest['Rating']} in {strongest['Group']} (rank relationship "
                f"{strongest['Rank relationship']:.2f}; {strongest['Athletes']:.0f} athletes; "
                f"P(positive) {100 * strongest['P(positive)']:.0f}%). Its 90% interval is "
                f"{strongest['90% low']:.2f} to {strongest['90% high']:.2f}, so this remains "
                "a useful direction with visible uncertainty—not a settled benchmark."
            )
        if grade_summary["Model reason"].eq(
            "Pooled line hides opposite group directions"
        ).any():
            st.warning(
                "At least one pooled grade relationship reverses or hides the group-specific "
                "directions. The app shows men and women separately there instead of turning a "
                "Simpson's-paradox pattern into a coaching conclusion."
            )
        st.caption(
            "Rank relationship preserves ordering without forcing linearity. P(positive) and "
            "the 90% interval keep uncertain relationships useful without presenting them as "
            "certain. Held-out RMSE chooses pooled versus sex-specific saturation curves."
        )
    backtest = data.get("model_backtest", pd.DataFrame())
    if not backtest.empty:
        st.markdown("#### Frozen next-competition backtest")
        st.dataframe(backtest, hide_index=True, width="stretch")
        st.caption(
            "Frozen means every prediction uses only information available before that "
            "competition. Higher rank correlation is better; lower log-loss and Brier "
            "error are better. This is the promotion test, not post-event fit."
        )
    v4_backtest = data.get("rating_v4_backtest", pd.DataFrame())
    if not v4_backtest.empty:
        model_names = {
            "quality_a0.50_transfer_0.20": "Former cold-start model",
            "hier_source_balanced_anchored_e4_sd100": "Current v4 model",
        }
        locked = v4_backtest.loc[
            v4_backtest["model"].isin(model_names)
            & v4_backtest["evaluation_window"].eq("2025+ locked evaluation")
            & v4_backtest["evaluation_domain"].isin(
                ["Open WC+", "CEC national", "CEC provincial / local"]
            )
        ].copy()
        if not locked.empty:
            locked["Model"] = locked["model"].map(model_names)
            locked = locked.rename(columns={
                "evaluation_domain": "Future competition type",
                "spearman": "Field-order accuracy",
                "pairwise_brier": "Pairwise probability error",
                "top_8_brier": "Top-8 probability error",
                "top_3_brier": "Top-3 probability error",
            })
            st.markdown("#### Global-ELO v4: locked 2025+ comparison")
            st.dataframe(
                locked[[
                    "Model", "Future competition type", "Field-order accuracy",
                    "Pairwise probability error", "Top-8 probability error",
                    "Top-3 probability error",
                ]].sort_values(["Future competition type", "Model"]),
                hide_index=True,
                width="stretch",
                column_config={
                    "Field-order accuracy": st.column_config.NumberColumn(format="%.3f"),
                    "Pairwise probability error": st.column_config.NumberColumn(format="%.3f"),
                    "Top-8 probability error": st.column_config.NumberColumn(format="%.3f"),
                    "Top-3 probability error": st.column_config.NumberColumn(format="%.3f"),
                },
            )
            st.caption(
                "Predictions were tuned on 2022–2024 and then frozen for 2025+. "
                "Higher field-order accuracy is better; lower probability error is better. "
                "V4 removes cold-start compression and improves domestic/advancement decisions, "
                "while the former model still orders the middle of WC+ fields slightly better."
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
        "Performance-ELO is the posterior mean after combining frozen Cumulative-ELO with "
        "all beat/lost-to outcomes; WC+ uses the full likelihood and lower events use a "
        "tempered likelihood. The raw value remains in the evidence audit. Repeated surprise "
        "across independent competitions is "
        "stronger evidence of real change than several rounds at one event, so cumulative Elo "
        "still waits for repeated evidence instead of copying one result."
    )
    st.caption(
        "Measured limit: beat/lost-to pairings from one round are related rather than fully "
        "independent. A joint-ranking challenger counted each field once and was slightly less "
        "accurate on frozen 2025+ WC+ forecasts, so the responsive mean remains in production. "
        "Its SD is conditional on fixed opponent ratings and a one-dimensional Elo model; do "
        "not read it as a complete interval for true ability or future result volatility."
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


@st.cache_data(show_spinner=False, ttl=60, max_entries=2)
def fetch_style_tags_remotely(url: str, limit: int = 1500) -> list[dict[str, object]]:
    """Read shared tags when the optional Apps Script endpoint supports listing."""
    if not url:
        return []
    separator = "&" if "?" in url else "?"
    request_url = f"{url}{separator}{urlencode({'action': 'list', 'limit': limit})}"
    try:
        with urlrequest.urlopen(request_url, timeout=18) as response:
            result = json.loads(response.read().decode("utf-8"))
        return list(result.get("records", [])) if result.get("ok") else []
    except (urlerror.URLError, TimeoutError, json.JSONDecodeError, TypeError):
        return []


def _render_style_tagging_v2_legacy(history: pd.DataFrame, standalone: bool = False) -> None:
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
                with st.expander(theme, expanded=theme in {"Physical qualities", "Move types · Dynamic"}):
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


def _tag_round(value: object) -> str:
    text = plain_key(value)
    if "qual" in text or "clasific" in text:
        return "Qualification"
    if "semi" in text:
        return "Semi-final"
    if "final" in text:
        return "Final"
    return ""


def _tag_boulder_number(value: object) -> int | None:
    text = str(value or "")
    match = re.search(r"\bB\s*(\d+)", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    fallback = re.search(r"\d+", text)
    return int(fallback.group(0)) if fallback else None


def _tag_uid(*values: object) -> str:
    """Return a stable, compact ID for a tagged terrain entity."""
    canonical = "|".join(plain_key(value) for value in values)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def _governed_boulder_count(
    round_rows: pd.DataFrame, recorded_counts: pd.Series | None = None,
    round_name: str = "",
) -> dict[str, object]:
    """Resolve a round count without presenting an assumption as observed data."""
    if round_rows.empty or "boulder_count" not in round_rows:
        counts = pd.Series(dtype=float)
        statuses: set[str] = set()
        sources: list[str] = []
    else:
        counts = pd.to_numeric(round_rows["boulder_count"], errors="coerce").dropna()
        counts = counts.loc[counts.gt(0)].astype(int)
        statuses = set(round_rows.get(
            "boulder_count_status", pd.Series("unknown", index=round_rows.index)
        ).dropna().astype(str))
        sources = sorted(set(round_rows.get(
            "boulder_count_source", pd.Series("", index=round_rows.index)
        ).dropna().astype(str)) - {""})
    distinct = sorted(counts.unique().tolist())
    if len(distinct) == 1 and statuses == {"source-confirmed"}:
        return {
            "count": distinct[0], "status": "source-confirmed", "editable": False,
            "source": "; ".join(sources) or "normalized results metadata",
            "candidates": distinct,
        }
    if len(distinct) > 1 or "source-conflict" in statuses:
        return {
            "count": max(distinct) if distinct else 4, "status": "source-conflict",
            "editable": True, "source": "; ".join(sources) or "conflicting source metadata",
            "candidates": distinct,
        }
    if distinct:
        return {
            "count": distinct[0], "status": "format-assumption", "editable": True,
            "source": "; ".join(sources) or "round-format assumption", "candidates": distinct,
        }
    if recorded_counts is not None:
        recorded = pd.to_numeric(recorded_counts, errors="coerce").dropna()
        recorded = recorded.loc[recorded.gt(0)].astype(int)
        if not recorded.empty:
            values = sorted(recorded.unique().tolist())
            return {
                "count": max(values), "status": "contributor-proposed", "editable": True,
                "source": "prior contributor response", "candidates": values,
            }
    return {
        "count": 5 if round_name == "Qualification" else 4,
        "status": "unknown", "editable": True,
        "source": "no source count available", "candidates": [],
    }


def _tag_records_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=[
            "competition", "competition_date", "round", "gender_terrain",
            "terrain_group", "boulder", "record_type", "optional_tags_completed",
            "expected_boulders",
        ])
    defaults = {
        "terrain_group": "Open / Senior", "record_type": "style",
        "optional_tags_completed": False, "expected_boulders": np.nan,
    }
    for column, value in defaults.items():
        if column not in frame:
            frame[column] = value
        else:
            frame[column] = frame[column].fillna(value)
    return frame


def _tag_context_mask(
    frame: pd.DataFrame, event: str, round_name: str, gender: str, terrain_group: str,
) -> pd.Series:
    if frame.empty:
        return pd.Series(False, index=frame.index)
    return (
        frame["competition"].astype(str).eq(str(event))
        & frame["round"].astype(str).eq(str(round_name))
        & frame["gender_terrain"].astype(str).eq(str(gender))
        & frame["terrain_group"].astype(str).eq(str(terrain_group))
        & ~frame["record_type"].astype(str).eq("flag")
    )


def render_style_tagging_v2(history: pd.DataFrame, standalone: bool = False) -> None:
    """Fast public workflow for event/round/boulder demand annotation."""
    st.header("Boulder Style Tagging")
    st.caption(
        "Choose the terrain once, score start→zone and zone→top separately, then move directly "
        "to the next boulder. Each boulder becomes one mini-round for future style Elo."
    )
    st.markdown(
        """
        <style>
        .zt-key{display:flex;gap:1.2rem;align-items:center;margin:.2rem 0 .7rem}
        .zt-zone{color:#287DB2;font-weight:800}.zt-top{color:#D87921;font-weight:800}
        div[data-baseweb="select"]>div{min-height:3rem}
        [role="listbox"]{min-width:min(980px,94vw)!important}
        </style>
        """, unsafe_allow_html=True,
    )
    with st.expander("0-3 scoring guide", expanded=False):
        st.markdown(
            "**0 absent · 1 present · 2 important · 3 dominant.** Score the demand, "
            "not whether a particular athlete succeeded. Physical, technical and "
            "coordination are independent dimensions."
        )
        st.write(f"Physical: {STYLE_DEFINITIONS['physical']}")
        st.write(f"Technical: {STYLE_DEFINITIONS['technical']}")
        st.write(f"Coordination: {STYLE_DEFINITIONS['coordination']}")

    if "style_tag_rows" not in st.session_state:
        st.session_state.style_tag_rows = []
    if "style_tag_images" not in st.session_state:
        st.session_state.style_tag_images = {}
    pending_navigation = st.session_state.pop("tag_pending_navigation", None)
    if isinstance(pending_navigation, dict):
        for state_key, state_value in pending_navigation.items():
            if state_value is None:
                st.session_state.pop(state_key, None)
            else:
                st.session_state[state_key] = state_value
    backend_url = style_tag_backend_url()
    remote_rows = fetch_style_tags_remotely(backend_url) if backend_url else []
    all_rows = [*remote_rows, *st.session_state.style_tag_rows]
    records = _tag_records_frame(all_rows)

    if not records.empty and records["competition"].astype(str).ne("").any():
        coverage_rows = []
        valid = records.loc[
            records["competition"].astype(str).ne("")
            & ~records["record_type"].astype(str).eq("flag")
        ].copy()
        for keys, group in valid.groupby(
            ["competition", "competition_date", "round", "gender_terrain", "terrain_group"],
            dropna=False,
        ):
            boulders = group["boulder"].map(_tag_boulder_number).dropna().astype(int)
            expected = pd.to_numeric(group["expected_boulders"], errors="coerce").dropna()
            expected_count = int(expected.max()) if not expected.empty else (
                int(boulders.max()) if not boulders.empty else 0
            )
            style_done = boulders.nunique()
            tag_done = group.loc[
                group["optional_tags_completed"].astype(str).str.lower().isin(["true", "1"]),
                "boulder",
            ].map(_tag_boulder_number).dropna().nunique()
            coverage_rows.append({
                "Competition": keys[0], "Date": keys[1], "Round": keys[2],
                "Gender": keys[3], "Terrain": keys[4],
                "Style": f"{style_done}/{expected_count or '?'}",
                "Tags": f"{tag_done}/{expected_count or '?'}",
                "Responses": len(group),
            })
        if coverage_rows:
            with st.expander("Completed and in-progress rounds", expanded=True):
                st.dataframe(
                    pd.DataFrame(coverage_rows).sort_values(
                        ["Date", "Competition"], ascending=[False, True]
                    ),
                    hide_index=True, width="stretch", height=250,
                )

    event_catalog = pd.DataFrame()
    if not history.empty and "event_name" in history:
        event_catalog = history.copy()
        event_catalog["event_name"] = event_catalog["event_name"].fillna("").astype(str).str.strip()
        event_catalog["event_date"] = pd.to_datetime(event_catalog.get("event_date"), errors="coerce")
        if "confirmed_procedure" in event_catalog:
            event_catalog = event_catalog.loc[
                ~event_catalog["confirmed_procedure"].astype(str).str.casefold().eq("scramble")
            ]
        event_catalog["round_clean"] = event_catalog.get(
            "round_group", pd.Series("", index=event_catalog.index)
        ).map(_tag_round)
        event_catalog = event_catalog.loc[event_catalog["event_name"].ne("")]
        event_catalog["date_label"] = event_catalog["event_date"].dt.strftime("%Y-%m-%d").fillna("Date unknown")
        event_catalog["label"] = event_catalog["date_label"] + " - " + event_catalog["event_name"]

    search = st.text_input(
        "Competition search", placeholder="Start typing a date or name: 2026-06, Prague, Nationals...",
        help="Suggestions come from the results.info event database; partial dates work.",
    ).strip()
    suggestions = event_catalog
    if search and not suggestions.empty:
        query = unicodedata.normalize("NFKD", search).encode("ascii", "ignore").decode().casefold()
        normalized = suggestions["label"].map(
            lambda value: unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().casefold()
        )
        suggestions = suggestions.loc[normalized.str.contains(query, regex=False)]
    option_rows = suggestions.sort_values("event_date", ascending=False).drop_duplicates("label").head(180)
    event_options = option_rows["label"].tolist()
    if not event_options:
        event_options = ["Custom competition"]
    elif "Custom competition" not in event_options:
        event_options.append("Custom competition")
    event_choice = st.selectbox(
        "Competition", event_options, key="tag_event_choice",
        help="The full competition name stays visible in the open list.",
    )
    custom_event = ""
    if event_choice == "Custom competition":
        custom_event = st.text_input("Competition name", key="tag_custom_event").strip()
    chosen_event = custom_event or (
        event_choice.split(" - ", 1)[1] if " - " in event_choice else event_choice
    )
    chosen_date = event_choice.split(" - ", 1)[0] if " - " in event_choice else ""
    event_rows = event_catalog.loc[event_catalog["label"].eq(event_choice)] if not event_catalog.empty else pd.DataFrame()
    source_scope = (
        str(event_rows["source_scope"].mode().iloc[0])
        if not event_rows.empty and "source_scope" in event_rows and not event_rows["source_scope"].dropna().empty
        else ""
    )

    available_rounds = [
        value for value in ["Qualification", "Semi-final", "Final"]
        if event_rows.empty or value in set(event_rows["round_clean"].dropna())
    ] or ["Qualification", "Semi-final", "Final"]
    round_name = st.segmented_control(
        "Round", available_rounds,
        default=available_rounds[0] if "tag_round_choice" not in st.session_state else None,
        key="tag_round_choice",
    )
    if round_name is None:
        round_name = available_rounds[0]

    youth_event = any(token in plain_key(chosen_event) for token in ["youth", "junior", "jeunesse"])
    if source_scope == "CEC":
        terrain_options = [
            "Youth A + Junior (shared Canadian terrain)", "Youth B", "Youth C", "Other",
        ] if youth_event else [
            "Open / Senior", "Youth A + Junior (shared Canadian terrain)",
            "Youth B", "Youth C", "Other",
        ]
    else:
        terrain_options = [
            "Youth A", "Junior", "Youth B", "Youth C", "Other",
        ] if youth_event else [
            "Open / Senior", "Youth A", "Junior", "Youth B", "Youth C", "Other",
        ]
    terrain_group = st.selectbox(
        "Age-class terrain", terrain_options, key="tag_terrain_group",
        help=(
            "Canadian Youth A and Junior may share a terrain. Youth Worlds keeps every age "
            "category separate. Youth B and C are always separate from each other and A/Junior."
        ),
    )
    st.caption(
        "Terrain identity rule: Canadian A/Junior can share; B and C never merge. "
        "At Youth Worlds, A and Junior also remain separate."
    )

    round_rows = event_rows.loc[event_rows["round_clean"].eq(round_name)] if not event_rows.empty else pd.DataFrame()
    available_genders = []
    if not round_rows.empty and "pool" in round_rows:
        pools = set(round_rows["pool"].dropna().astype(str))
        if "Boulder_Men" in pools:
            available_genders.append("Men")
        if "Boulder_Women" in pools:
            available_genders.append("Women")
    available_genders = available_genders or ["Men", "Women"]
    gender = st.segmented_control(
        "Gender terrain", available_genders,
        default=available_genders[0] if "tag_gender_choice" not in st.session_state else None,
        key="tag_gender_choice",
    )
    if gender is None:
        gender = available_genders[0]

    governed_rows = round_rows.copy()
    if not governed_rows.empty and "gender" in governed_rows:
        governed_rows = governed_rows.loc[
            governed_rows["gender"].astype(str).eq(gender)
        ]
    if not governed_rows.empty and "terrain_group" in governed_rows:
        governed_rows = governed_rows.loc[
            governed_rows["terrain_group"].astype(str).eq(terrain_group)
        ]

    def context_counts(target_round: str, target_gender: str) -> str:
        subset = records.loc[
            _tag_context_mask(
                records, chosen_event, target_round, target_gender, terrain_group
            )
        ]
        styled = subset["boulder"].map(_tag_boulder_number).dropna().nunique()
        tagged = subset.loc[
            subset["optional_tags_completed"].astype(str).str.lower().isin(["true", "1"]),
            "boulder",
        ].map(_tag_boulder_number).dropna().nunique()
        return f"{styled}S/{tagged}T"

    round_progress = " · ".join(
        f"{item}: {context_counts(item, gender)}" for item in available_rounds
    )
    gender_progress = " · ".join(
        f"{item}: {context_counts(round_name, item)}" for item in available_genders
    )
    st.caption(f"Round responses — {round_progress}")
    st.caption(f"Gender responses — {gender_progress}")

    count_key = plain_key("|".join([chosen_event, round_name, gender, terrain_group]))
    matching = records.loc[_tag_context_mask(records, chosen_event, round_name, gender, terrain_group)]
    recorded_counts = pd.to_numeric(matching.get("expected_boulders"), errors="coerce").dropna()
    count_evidence = _governed_boulder_count(
        governed_rows, recorded_counts, round_name=round_name
    )
    default_count = int(count_evidence["count"])
    count_status = str(count_evidence["status"])
    count_source = str(count_evidence["source"])
    count_candidates = list(count_evidence["candidates"])
    if not bool(count_evidence["editable"]):
        expected_boulders = default_count
        st.success(
            f"{expected_boulders} boulders in this terrain · confirmed by the results source",
            icon="✅",
        )
    else:
        if count_status == "source-conflict":
            details = f" ({', '.join(map(str, count_candidates))})" if count_candidates else ""
            st.warning(
                f"The source contains conflicting boulder counts{details}. "
                "Choose the count visible in the official round."
            )
        elif count_status == "format-assumption":
            st.info(
                f"{default_count} boulders is a format-based proposal; this older round has no "
                "source-confirmed count."
            )
        elif count_status == "contributor-proposed":
            st.info(f"{default_count} boulders was proposed in an earlier response; verify it once.")
        else:
            st.warning("No boulder count is available in the source metadata. Confirm it before tagging.")
        expected_boulders = int(st.number_input(
            "Boulders in this terrain", min_value=1, max_value=12, value=default_count,
            step=1, key=f"tag_count_{count_key}",
            help="A correction is stored as contributor-proposed evidence, not source-confirmed metadata.",
        ))
        if expected_boulders != default_count:
            count_status = "contributor-proposed"
            count_source = "public tagger count correction"

    source_event_ids: list[str] = []
    source_round_ids: list[str] = []
    if not governed_rows.empty and "source_event_id" in governed_rows:
        source_event_ids = sorted(set(
            governed_rows["source_event_id"].dropna().astype(str)
        ) - {"", "nan"})
    if not governed_rows.empty and "category_round_id" in governed_rows:
        source_round_ids = sorted(set(
            governed_rows["category_round_id"].dropna().astype(str)
        ) - {"", "nan"})
    with st.expander("Boulder-count evidence", expanded=False):
        st.write(f"**Status:** {count_status.replace('-', ' ')}")
        st.write(f"**Source:** {count_source}")
        if source_round_ids:
            st.write("**Source round IDs:** " + ", ".join(source_round_ids))
    round_uid = "round-" + _tag_uid(
        source_scope, "|".join(source_event_ids), chosen_date, chosen_event,
        round_name, gender, terrain_group,
    )

    boulder_labels = []
    for number in range(1, expected_boulders + 1):
        boulder_group = matching.loc[
            matching["boulder"].map(_tag_boulder_number).eq(number)
        ]
        style_count = len(boulder_group)
        tag_count = int(boulder_group["optional_tags_completed"].astype(str).str.lower().isin(["true", "1"]).sum())
        boulder_labels.append(f"B{number} · {style_count}S/{tag_count}T")
    existing_boulder = st.session_state.get("tag_boulder_choice")
    if existing_boulder not in boulder_labels:
        existing_number = _tag_boulder_number(existing_boulder)
        if existing_number and 1 <= existing_number <= expected_boulders:
            st.session_state.tag_boulder_choice = boulder_labels[existing_number - 1]
        else:
            st.session_state.pop("tag_boulder_choice", None)
    boulder_choice = st.segmented_control(
        "Boulder", boulder_labels,
        default=boulder_labels[0] if "tag_boulder_choice" not in st.session_state else None,
        key="tag_boulder_choice",
    ) or boulder_labels[0]
    boulder_number = _tag_boulder_number(boulder_choice) or 1
    st.caption("S = complete style responses · T = responses with optional movement/hold tags")

    current_boulder = matching.loc[
        matching["boulder"].map(_tag_boulder_number).eq(boulder_number)
    ]
    image_items: list[tuple[str, object]] = []
    for _, item in current_boulder.iterrows():
        url = str(item.get("image_public_url") or item.get("image_url") or "").strip()
        if url:
            image_items.append((str(item.get("contributor") or "Existing image"), url))
        local_path = str(item.get("image_in_bundle") or "")
        if local_path in st.session_state.style_tag_images:
            image_items.append((str(item.get("contributor") or "Session image"), st.session_state.style_tag_images[local_path]))
    if image_items:
        st.markdown("#### Existing boulder images")
        columns = st.columns(min(3, len(image_items)))
        for index, (caption, source) in enumerate(image_items[:6]):
            columns[index % len(columns)].image(source, caption=caption, width="stretch")
    image_file = st.file_uploader(
        "Add another boulder image", type=["png", "jpg", "jpeg", "webp"],
        key=f"tag_image_{count_key}_{boulder_number}",
        help="Use the full boulder, no climber, and ideally no surrounding boulders.",
    )
    st.caption("Best image: full boulder visible · no climber · surrounding boulders cropped out.")
    if image_file is not None:
        st.image(image_file, caption="New image preview", width="stretch")

    add_optional_tags = st.checkbox(
        "I would like to help further and identify holds and movements",
        value=False, key=f"tag_optional_{count_key}_{boulder_number}",
    )

    def compact_pair(label: str, key: str, help_text: str = "") -> tuple[int, int]:
        columns = st.columns([1.7, 2.2, 2.2])
        columns[0].markdown(f"**{label}**")
        pre_zone = columns[1].slider(
            f"Before zone {label}", 0, 3, 0, key=f"z_{count_key}_{boulder_number}_{key}",
            label_visibility="collapsed", help=help_text or None,
        )
        post_zone = columns[2].slider(
            f"After zone {label}", 0, 3, 0, key=f"t_{count_key}_{boulder_number}_{key}",
            label_visibility="collapsed", help=help_text or None,
        )
        return pre_zone, post_zone

    with st.form(f"style_tag_fast_{count_key}_{boulder_number}", clear_on_submit=False):
        st.markdown('<div class="zt-key"><span></span><span class="zt-zone">Pre-zone · Start → Zone</span><span class="zt-top">Post-zone · Zone → Top</span></div>', unsafe_allow_html=True)
        core_scores = {
            "physical": compact_pair("Physical", "physical", STYLE_DEFINITIONS["physical"]),
            "technical": compact_pair("Technical", "technical", STYLE_DEFINITIONS["technical"]),
            "coordination": compact_pair("Coordination", "coordination", STYLE_DEFINITIONS["coordination"]),
            "verticality": compact_pair("Wall angle", "verticality", "0 slab · 1 vertical · 2 overhang · 3 roof"),
        }
        direction = st.columns([1.7, 2.2, 2.2])
        direction[0].markdown("**Direction**")
        pre_zone_direction = direction[1].selectbox(
            "Before zone direction", ["Up", "Diagonal", "Sideways", "Mixed / unclear"],
            label_visibility="collapsed",
        )
        post_zone_direction = direction[2].selectbox(
            "After zone direction", ["Up", "Diagonal", "Sideways", "Mixed / unclear"],
            label_visibility="collapsed",
        )
        optional_scores: dict[str, tuple[int, int]] = {}
        if add_optional_tags:
            st.markdown("#### Optional tags")
            st.caption("Leave 0 when absent or uncertain.")
            for theme, items in STYLE_TAG_GROUPS.items():
                with st.expander(theme, expanded=theme in {"Physical qualities", "Move types · Dynamic"}):
                    for tag_key, tag_label in items:
                        optional_scores[tag_key] = compact_pair(tag_label, tag_key)
        details = st.columns(2)
        notes = details[0].text_area(
            "Useful context (optional)", placeholder="Alternative beta, deceptive feature, uncertainty..."
        )
        confidence = details[1].select_slider(
            "Confidence", options=["Low", "Moderate", "High"], value="Moderate"
        )
        contributor = details[1].text_input("Contributor name or initials (optional)")
        submitted = st.form_submit_button("Save and continue", type="primary")

    if not submitted and st.session_state.get("tag_last_message"):
        st.success(str(st.session_state.tag_last_message))
    if submitted:
        image_bytes = image_file.getvalue() if image_file is not None else b""
        if len(image_bytes) > 10 * 1024 * 1024:
            st.error("Please use an image smaller than 10 MB.")
        elif not chosen_event:
            st.error("Choose or enter a competition first.")
        else:
            image_hash = hashlib.sha256(image_bytes).hexdigest() if image_bytes else ""
            suffix = Path(image_file.name).suffix.lower() if image_file is not None else ""
            stored_image = f"images/{image_hash}{suffix}" if image_hash else ""
            boulder_uid = f"{round_uid}-b{boulder_number}"
            record: dict[str, object] = {
                "schema_version": "4.0", "record_type": "style",
                "tag_taxonomy_version": STYLE_TAG_TAXONOMY_VERSION,
                "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
                "competition": chosen_event, "competition_date": chosen_date,
                "source_scope": source_scope,
                "source_event_ids": "|".join(source_event_ids),
                "source_round_ids": "|".join(source_round_ids),
                "round": round_name, "round_uid": round_uid,
                "gender_terrain": gender, "terrain_group": terrain_group,
                "boulder": f"B{boulder_number}", "expected_boulders": expected_boulders,
                "boulder_count_status": count_status,
                "boulder_count_source": count_source,
                "boulder_uid": boulder_uid,
                "pre_zone_segment_uid": f"{boulder_uid}-pre-zone",
                "post_zone_segment_uid": f"{boulder_uid}-post-zone",
                "image_name": image_file.name if image_file is not None else "",
                "image_sha256": image_hash, "image_in_bundle": stored_image,
                "pre_zone_direction": pre_zone_direction,
                "post_zone_direction": post_zone_direction,
                "optional_tags_completed": add_optional_tags,
                "notes": notes.strip(), "confidence": confidence,
                "contributor": contributor.strip(),
            }
            for score_key, (pre_zone_value, post_zone_value) in {**core_scores, **optional_scores}.items():
                record[f"pre_zone_{score_key}_0_3"] = pre_zone_value
                record[f"post_zone_{score_key}_0_3"] = post_zone_value
            st.session_state.style_tag_rows.append(record)
            if image_bytes and stored_image:
                st.session_state.style_tag_images[stored_image] = image_bytes
            message = "Saved in this session."
            if backend_url:
                saved, remote_message = save_style_tag_remotely(backend_url, record, image_bytes)
                message = remote_message if saved else f"Session saved; shared save failed: {remote_message}"
                fetch_style_tags_remotely.clear()
            st.session_state.tag_last_message = message
            st.session_state.tag_last_boulder = boulder_number
            st.rerun()

    if st.session_state.get("tag_last_boulder") == boulder_number:
        actions = st.columns(3)
        if boulder_number < expected_boulders:
            if actions[0].button("Next boulder ->", type="primary"):
                st.session_state.tag_pending_navigation = {
                    "tag_boulder_choice": boulder_labels[boulder_number],
                    "tag_last_message": None,
                }
                st.rerun()
        else:
            round_index = available_rounds.index(round_name)
            if round_index + 1 < len(available_rounds):
                if actions[0].button("Next round ->", type="primary"):
                    st.session_state.tag_pending_navigation = {
                        "tag_round_choice": available_rounds[round_index + 1],
                        "tag_boulder_choice": None, "tag_last_message": None,
                    }
                    st.rerun()
            elif len(available_genders) > 1:
                other_gender = next(item for item in available_genders if item != gender)
                if actions[0].button(f"Start {other_gender} ->", type="primary"):
                    st.session_state.tag_pending_navigation = {
                        "tag_gender_choice": other_gender,
                        "tag_round_choice": available_rounds[0],
                        "tag_boulder_choice": None, "tag_last_message": None,
                    }
                    st.rerun()

    with st.expander("Flag a mistake or suggest an improvement"):
        with st.form(f"tag_flag_{count_key}_{boulder_number}"):
            flag_type = st.selectbox(
                "Issue", ["Wrong boulder image", "Wrong competition / round / terrain",
                          "Duplicate", "Form improvement", "Other"]
            )
            flag_note = st.text_area("What should be corrected or improved?")
            flag_submit = st.form_submit_button("Send flag")
        if flag_submit:
            if not flag_note.strip():
                st.warning("Add a short explanation so the flag can be acted on.")
            else:
                flag_record = {
                    "schema_version": "4.0", "record_type": "flag",
                    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
                    "competition": chosen_event, "competition_date": chosen_date,
                    "source_scope": source_scope,
                    "source_event_ids": "|".join(source_event_ids),
                    "source_round_ids": "|".join(source_round_ids),
                    "round": round_name, "round_uid": round_uid,
                    "gender_terrain": gender,
                    "terrain_group": terrain_group, "boulder": f"B{boulder_number}",
                    "boulder_uid": f"{round_uid}-b{boulder_number}",
                    "flag_type": flag_type, "flag_note": flag_note.strip(),
                }
                st.session_state.style_tag_rows.append(flag_record)
                if backend_url:
                    save_style_tag_remotely(backend_url, flag_record, b"")
                    fetch_style_tags_remotely.clear()
                st.success("Flag saved. Thank you.")

    session_records = pd.DataFrame(st.session_state.style_tag_rows)
    if not session_records.empty:
        with st.expander("Session backup", expanded=False):
            st.dataframe(session_records, hide_index=True, width="stretch", height=240)
            csv_bytes = session_records.to_csv(index=False).encode("utf-8")
            st.download_button("Download session CSV", csv_bytes, "boulder_style_tags_v4.csv", "text/csv")
    st.caption(
        "Shared database connected." if backend_url else
        "Shared database is not connected in this deployment; session CSV remains available as a backup."
    )
    if standalone:
        st.info("This separate public tool keeps image and tagging traffic away from the analysis dashboard.")


def render_rating_contract() -> None:
    """Keep the scale contract visible before users interpret any graph."""
    with st.expander("How to read every Elo on this page"):
        st.write(rating_help())
        st.markdown(
            "- **Global-ELO:** all usable, de-duplicated competitions on one Open-readiness scale.\n"
            "- **WC+-ELO:** World Cups/Series, World Championships and Olympic qualification evidence only.\n"
            "- **IFSC-ELO:** every non-para competition published by IFSC.\n"
            "- **Round and format variants:** only the named qualifying, semifinal, final, flash, onsight or scramble evidence.\n"
            "- **Performance-ELO:** posterior mean WC performance implied by every beat/lost-to pairing; uncertainty and the raw estimate remain auditable."
        )
        st.caption(
            "A newcomer can move quickly while labelled provisional; uncertainty falls only after "
            "independent competitions agree. Pairwise probabilities are calibrated separately from "
            "field ordering on competitions that occurred later than model tuning."
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
    # Keep a previous compact artifact usable during an atomic model refresh.
    # Missing families remain visibly missing instead of crashing a view.
    for family in ALL_RATINGS:
        if family not in athletes:
            athletes[family] = np.nan
        evidence_column = f"{family} evidence"
        if evidence_column not in athletes:
            athletes[evidence_column] = np.nan
    athletes["display_name"] = athletes["athlete_name"].map(friendly_name)
    selected, _ = top_ribbon(athletes, data["history"], data["rosters"])
    if not selected:
        st.info("Select at least one athlete to begin.")
        st.stop()
    render_rating_contract()
    workspace = st.segmented_control(
        "Workspace",
        ["Overview", "Actionable Analysis", "Physical Strength", "Tag Boulder Styles", "Maths behind"],
        default="Overview", label_visibility="collapsed",
    )
    if workspace == "Overview":
        render_rating_detail(
            athletes, data["history"], selected, data["calibration"],
            data["context_benchmarks"],
        )
        st.header("Overview")
        section = st.segmented_control(
            "Overview section",
            ["Canadian Pool", "IFSC Pool", "WC+ / CUWR Pool", "Global progression", "Towards Olympics"],
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
            "WC+ / CUWR Pool": lambda: render_wr_pool(
                athletes, selected, data["correlations"], data["calibration"],
                data["country_entry"], data["cuwr_history"],
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
    elif workspace == "Actionable Analysis":
        render_actionable_analysis(
            athletes, data["history"], selected, data["calibration"]
        )
    elif workspace == "Physical Strength":
        render_physical_strength(athletes, selected, data)
    elif workspace == "Tag Boulder Styles":
        st.header("Boulder Style Tagging")
        st.write(
            "The public annotator now runs separately so images and community tagging "
            "do not consume the analysis dashboard's memory."
        )
        st.link_button(
            "Open the public Boulder Style Tagger",
            "https://comp-climbing-boulder-tags.streamlit.app/",
            type="primary",
        )
        st.caption(
            "It includes searchable competitions, paired Zone/Top scoring, direction and "
            "optional hold, foothold, physical, technical and movement tags."
        )
    else:
        render_maths_behind(
            athletes, data["correlations"], data["calibration"], data
        )

if __name__ == "__main__":
    main()
