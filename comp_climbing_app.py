"""Boulder-first interface for Comp Climbing Projections.

The legacy product stays on its own release branch and URL.  This module loads
only the compact artifacts needed by the Overview so Streamlit Community Cloud
does not retain the full research warehouse in memory.
"""

from __future__ import annotations

from datetime import date
import hashlib
import hmac
import json
import os
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RATING_ORDER = ["Global-ELO", "IFSC-ELO", "WC+-ELO"]
FORMAT_OPTIONS = ["All formats", "Onsight", "Flash", "Scramble"]
ALL_RATINGS = [
    "Global-ELO", "Global-ELO-Onsight", "Global-ELO-Scramble",
    "Global-ELO-Flash", "WC+-ELO", "WC+-ELO-Open", "WC+-ELO-Qualies",
    "WC+-ELO-Semies", "WC+-ELO-Finals", "IFSC-ELO", "IFSC-ELO-Qualies",
    "IFSC-ELO-Semies", "IFSC-ELO-Finals",
]
DEFAULT_ATHLETES = ["Oscar Baudrand", "Matthew Rodriguez", "Hugo Dorval"]
FROZEN_2026_HOLDOUT_ARTIFACT = (
    "a8225902e3181c58308586b1fe24338fd542e617ffe490202f5d55eaa28f94ce"
)
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
AGE_PROGRESSION_FIELDS = (
    "pool",
    "age_center_years",
    "age_bin_lower_years",
    "age_bin_upper_years",
    "athletes",
    "observations",
    "median_annual_performance_elo_change",
    "bootstrap_se_annual_performance_elo_change",
    "bootstrap_draws",
    "minimum_athletes",
    "age_assignment_status",
    "source_scope",
    "source_window_start",
    "source_window_end",
    "source_as_of_date",
    "method",
    "status",
    "research_only",
)
AGE_PROGRESSION_METHOD = (
    "median_within_interval_safe_centered_age_year_segment_"
    "min_0_20_years_bootstrap_se"
)
AGE_PROGRESSION_STATUS = "RESEARCH_AGGREGATE_MIN_20_NOT_CAUSAL"
OBVIOUS_FIXTURE_EVENT_PATTERN = r"(?i)\b(?:test|mock|demo|dummy|sandbox|hidden)\b"


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


def quarantine_obvious_fixture_exposure(
    athletes: pd.DataFrame,
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove fixture event keys but retain identities and legitimate rows.

    Directly exposed identities retain their roster/CNR fields so the clean
    current-WC replay can still be used. Their unreplayed legacy ELO diagnostics
    are blanked rather than applying a whole-identity penalty.
    """
    if athletes.empty or history.empty or "event_name" not in history:
        audit = pd.DataFrame([{
            "fixture_event_rows": 0,
            "fixture_source_events": 0,
            "fixture_pool_event_keys": 0,
            "fixture_exposed_athlete_ids": 0,
            "legacy_rating_rows_blanked": 0,
            "legacy_history_rating_rows_blanked": 0,
            "retained_canadian_identities": 0,
        }])
        return athletes.copy(), history.copy(), audit
    fixture = history["event_name"].astype("string").str.contains(
        OBVIOUS_FIXTURE_EVENT_PATTERN, regex=True, na=False
    )
    fixture_rows = history.loc[fixture]
    affected_ids = set(fixture_rows["global_id"].dropna().astype(str))
    key_columns = [
        column for column in ("source_scope", "source_event_id", "pool")
        if column in history.columns
    ]
    if len(key_columns) < 2:
        raise ValueError("fixture history lacks a stable source-event key")
    fixture_keys = fixture_rows[key_columns].drop_duplicates()
    keyed_history = history.merge(
        fixture_keys.assign(_fixture_event_key=True), on=key_columns, how="left"
    )
    safe_history = keyed_history.loc[
        keyed_history["_fixture_event_key"].ne(True)
    ].drop(columns="_fixture_event_key")
    if safe_history["event_name"].astype("string").str.contains(
        OBVIOUS_FIXTURE_EVENT_PATTERN, regex=True, na=False
    ).any():
        raise RuntimeError("an obvious fixture event survived event-key quarantine")

    history_exposed = safe_history["global_id"].astype(str).isin(affected_ids)
    history_rating_columns = [
        column
        for column in safe_history.columns
        if (
            column.startswith("rating_")
            or column.startswith("event_start_")
            or "performance_elo" in column
        )
    ]
    safe_history.loc[history_exposed, history_rating_columns] = np.nan
    safe_history["legacy_fixture_exposed"] = history_exposed

    safe_athletes = athletes.copy()
    exposed = safe_athletes["global_id"].astype(str).isin(affected_ids)
    legacy_rating_columns = [
        column
        for column in safe_athletes.columns
        if (
            "ELO" in column
            or column in {
                "momentum",
                "canada_projection_all_evidence",
                "cec_projected_rating",
                "cec_context_offset",
            }
        )
    ]
    safe_athletes.loc[exposed, legacy_rating_columns] = np.nan
    safe_athletes["legacy_fixture_exposed"] = exposed
    cnr_rank = pd.to_numeric(
        safe_athletes.get(
            "cnr_rank", pd.Series(np.nan, index=safe_athletes.index)
        ),
        errors="coerce",
    )
    audit = pd.DataFrame([{
        "fixture_event_rows": int(len(history) - len(safe_history)),
        "fixture_source_events": int(
            fixture_rows[["source_scope", "source_event_id"]]
            .drop_duplicates()
            .shape[0]
        ),
        "fixture_pool_event_keys": int(len(fixture_keys)),
        "fixture_exposed_athlete_ids": len(affected_ids),
        "legacy_rating_rows_blanked": int(exposed.sum()),
        "legacy_history_rating_rows_blanked": int(history_exposed.sum()),
        "retained_canadian_identities": int((exposed & cnr_rank.notna()).sum()),
    }])
    return safe_athletes, safe_history, audit


def load_current_wc_projection_artifact(
    data_dir: Path = DATA,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load the current pilot only when its clean replay contract is intact."""

    csv_path = data_dir / "canadian_current_wc_projection_v3_youth_world_complete.csv"
    metadata_path = (
        data_dir
        / "canadian_current_wc_projection_v3_youth_world_complete.metadata.json"
    )
    if not csv_path.is_file() or not metadata_path.is_file():
        return pd.DataFrame(), {
            "verified": False,
            "reason": "projection CSV or metadata is missing",
        }
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        csv_record = metadata["csv"]
        clean = metadata["clean_input"]
        model = metadata["model"]
        calibration = metadata["calibration"]
        low_wc = metadata["low_wc_evidence_calibration"]
        claims = metadata["claims"]
        digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        if metadata.get("schema") != "canadian-current-wc-projection-public-metadata-v1":
            raise ValueError("unexpected projection metadata schema")
        if csv_record.get("filename") != csv_path.name:
            raise ValueError("projection filename binding mismatch")
        if digest != csv_record.get("sha256"):
            raise ValueError("projection CSV hash mismatch")
        if csv_path.stat().st_size != int(csv_record.get("bytes", -1)):
            raise ValueError("projection CSV byte-count mismatch")
        if model.get("name") != "joint_rank_form_target":
            raise ValueError("unexpected governing projection model")
        if not bool(model.get("initializer_converged")):
            raise ValueError("current initializer did not converge")
        if model.get("initializer_warm_start_sha256") not in (None, ""):
            raise ValueError("current initializer was not refit from priors")
        routing = model.get("target_domain_routing", {})
        if routing.get("schema") != "ifsc-youth-world-separate-target-head-v1":
            raise ValueError("Youth-World target-domain routing is not sealed")
        if routing.get("replay_target_domain") != "ifsc_youth_world":
            raise ValueError("Youth Worlds do not have the expected target head")
        if routing.get("senior_open_world_major_target_domain") != "wc+":
            raise ValueError("Senior World-major target routing changed")
        if any(
            int(routing.get(field, -1)) != expected
            for field, expected in (
                ("rows_routed", 5752),
                ("source_events_routed", 11),
                ("pool_events_routed", 22),
                ("athletes_with_youth_world_evidence", 2515),
                ("mixed_target_batches", 0),
                ("youth_world_rows_in_senior_wc_target_state", 0),
                ("ranking_values_changed", 0),
                ("identity_values_changed", 0),
                ("graph_edges_removed", 0),
            )
        ):
            raise ValueError("Youth-World routing closure failed")
        if routing.get("senior_open_world_major_preserved_in_wc_plus") is not True:
            raise ValueError("Senior World-major WC+ routing was not preserved")
        post_cutoff = routing.get("post_cutoff_replay", {})
        if any(
            int(post_cutoff.get(field, -1)) != expected
            for field, expected in (
                ("pool_events", 6),
                ("athlete_events", 1111),
                ("source_pool_events", 6),
                ("youth_world_events_in_wc_plus", 0),
                ("all_history_source_event_inventory_count", 11),
            )
        ):
            raise ValueError("Youth-World post-cutoff replay closure failed")
        if post_cutoff.get("target_domain") != "ifsc_youth_world":
            raise ValueError("Youth-World replay target changed")
        if post_cutoff.get("all_history_source_event_inventory_sha256") != (
            "28fe20328b6eb6c6ed8893a045ff2eea66940f3a4cd47aa83d70ac6daef9005a"
        ):
            raise ValueError("Youth-World source-event inventory changed")
        if post_cutoff.get("asserted_exact") is not True:
            raise ValueError("Youth-World replay closure is not exact")
        if any(
            int(clean.get(field, -1)) != 0
            for field in (
                "surviving_obvious_fixture_rows",
                "surviving_flagged_event_keys",
                "ambiguous_mixed_name_event_keys",
            )
        ):
            raise ValueError("fixture-event closure failed")
        if not bool(calibration.get("event_clean_refit")):
            raise ValueError("base advancement calibration is not event-clean")
        clean_slope = float(calibration.get("slope_per_100", np.nan))
        if not np.isfinite(clean_slope) or clean_slope <= 0.0:
            raise ValueError("advancement calibration slope is not positive")
        if not bool(low_wc.get("event_clean_refit")):
            raise ValueError("low-WC calibration is not event-clean")
        if low_wc.get("class") != "zero_or_one_prior_senior_open_wc_plus_competition":
            raise ValueError("unexpected low-WC evidence class")
        if low_wc.get("central_route") != (
            "separate_k0_k1_intercept_adjusted_links"
        ):
            raise ValueError("unexpected low-WC calibration route")
        zero_prior = low_wc.get("zero_prior", {})
        one_prior = low_wc.get("one_prior", {})
        for expected_k, record in ((0, zero_prior), (1, one_prior)):
            if int(record.get("prior_competitions", -1)) != expected_k:
                raise ValueError("low-WC evidence class count mismatch")
            if record.get("tau_selection") != (
                "minimum_2025_leave_source_event_class_log_loss"
            ):
                raise ValueError("unexpected low-WC shrinkage selection rule")
            if record.get("slope_policy") != "fixed_to_clean_base_slope":
                raise ValueError("low-WC score discrimination policy changed")
            intercept = float(record.get("intercept", np.nan))
            slope = float(record.get("slope_per_100", np.nan))
            tau = float(record.get("shrinkage_tau", np.nan))
            if not all(np.isfinite(value) for value in (intercept, slope, tau)):
                raise ValueError("nonfinite low-WC calibration coefficient")
            if slope <= 0.0 or tau <= 0.0:
                raise ValueError("invalid low-WC transport calibration")
            if not np.isclose(slope, clean_slope, rtol=0.0, atol=1e-12):
                raise ValueError(
                    "intercept-adjusted low-WC link changed the clean score slope"
                )
        for record, prefix in (
            (calibration, "clean_2026"),
            (calibration, "clean_2026_canadian"),
            (low_wc, "clean_2026"),
            (zero_prior, "clean_2026"),
            (one_prior, "clean_2026"),
        ):
            rows = int(record.get(f"{prefix}_rows", 0))
            positives = int(record.get(f"{prefix}_positives", -1))
            predicted = float(record.get(f"{prefix}_predicted_rate", np.nan))
            observed = float(record.get(f"{prefix}_observed_rate", np.nan))
            log_loss = float(record.get(f"{prefix}_log_loss", np.nan))
            brier = float(record.get(f"{prefix}_brier", np.nan))
            if rows <= 0 or positives < 0 or positives > rows:
                raise ValueError(f"invalid {prefix} validation counts")
            if not all(
                np.isfinite(value)
                for value in (predicted, observed, log_loss, brier)
            ):
                raise ValueError(f"nonfinite {prefix} validation metric")
            if not (0.0 <= predicted <= 1.0 and 0.0 <= observed <= 1.0):
                raise ValueError(f"invalid {prefix} validation rate")
        if claims.get("bridge_governs_central_probability") is not False:
            raise ValueError("bridge is not restricted to sensitivity analysis")
        if claims.get("rating_state_sensitivity_uses_bridge_sd") is not False:
            raise ValueError("bridge uncertainty leaked into rating-state sensitivity")
        if claims.get("whole_identity_fixture_penalty") is not False:
            raise ValueError("whole-identity fixture penalty is active")
        if claims.get("youth_world_directly_updates_senior_wc_offset") is not False:
            raise ValueError("Youth Worlds directly update the Senior-WC target head")
        if claims.get("youth_world_shared_skill_graph_preserved") is not True:
            raise ValueError("Youth-World pairwise graph evidence was discarded")
        frame = pd.read_csv(csv_path, low_memory=False)
        if len(frame) != int(csv_record.get("rows", -1)):
            raise ValueError("projection row-count mismatch")
        required = {
            "projection_status",
            "direct_senior_open_wc_plus_competitions",
            "evidence_class",
            "score_route",
            "governing_calibration_intercept",
            "governing_calibration_slope_per_100",
            "zero_prior_calibration_intercept",
            "zero_prior_calibration_slope_per_100",
            "one_prior_calibration_intercept",
            "one_prior_calibration_slope_per_100",
            "base_calibration_intercept",
            "calibration_slope_per_100",
            "direct_youth_world_competitions",
            "youth_world_projection_score",
            "youth_world_projection_score_sd",
            "youth_world_target_adjustment",
            "youth_world_minus_wc_target_adjustment",
        }
        if not required.issubset(frame.columns):
            raise ValueError("projection evidence-calibration schema mismatch")
        available = frame.loc[
            frame["projection_status"].eq(
                "exploratory_current_reference_available"
            )
        ].copy()
        counts = pd.to_numeric(
            available["direct_senior_open_wc_plus_competitions"],
            errors="raise",
        )
        expected_class = np.select(
            [counts.eq(0), counts.eq(1)],
            [
                "zero_prior_senior_open_wc_plus",
                "one_prior_senior_open_wc_plus",
            ],
            default="two_or_more_prior_senior_open_wc_plus",
        )
        expected_route = np.select(
            [counts.eq(0), counts.eq(1)],
            [
                "wc_target_score_zero_prior_intercept_adjusted_link",
                "wc_target_score_one_prior_intercept_adjusted_link",
            ],
            default="wc_target_score_standard_link",
        )
        if not np.array_equal(available["evidence_class"].to_numpy(), expected_class):
            raise ValueError("projection evidence class does not match prior count")
        if not np.array_equal(available["score_route"].to_numpy(), expected_route):
            raise ValueError("projection score route does not match prior count")
        expected_intercept = np.select(
            [counts.eq(0), counts.eq(1)],
            [float(zero_prior["intercept"]), float(one_prior["intercept"])],
            default=float(calibration["intercept"]),
        )
        expected_slope = np.select(
            [counts.eq(0), counts.eq(1)],
            [float(zero_prior["slope_per_100"]), float(one_prior["slope_per_100"])],
            default=clean_slope,
        )
        if not np.allclose(
            pd.to_numeric(
                available["governing_calibration_intercept"], errors="raise"
            ),
            expected_intercept,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("projection governing intercept mismatch")
        if not np.allclose(
            pd.to_numeric(
                available["governing_calibration_slope_per_100"], errors="raise"
            ),
            expected_slope,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("projection governing slope mismatch")
        constant_bindings = {
            "zero_prior_calibration_intercept": float(zero_prior["intercept"]),
            "zero_prior_calibration_slope_per_100": float(
                zero_prior["slope_per_100"]
            ),
            "one_prior_calibration_intercept": float(one_prior["intercept"]),
            "one_prior_calibration_slope_per_100": float(
                one_prior["slope_per_100"]
            ),
            "base_calibration_intercept": float(calibration["intercept"]),
            "calibration_slope_per_100": clean_slope,
        }
        for column, expected in constant_bindings.items():
            if not np.allclose(
                pd.to_numeric(available[column], errors="raise"),
                expected,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(f"projection constant binding mismatch: {column}")
        metadata["verified"] = True
        return frame, metadata
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        return pd.DataFrame(), {"verified": False, "reason": str(error)}


@st.cache_data(show_spinner=False, ttl=900, max_entries=2)
def read_data() -> dict[str, object]:
    # Bump this value whenever a byte-bound data artifact is replaced. Streamlit
    # otherwise may retain the previous verification result until the TTL ends.
    projection_cache_release = "v3-youth-world-complete-sealed-bytes"
    files = {
        "athletes": ("boulder_overview_athletes.parquet", "parquet"),
        "history": ("boulder_overview_history.parquet", "parquet"),
        "age_progression": ("boulder_age_progression_reference.csv", "csv"),
        "correlations": ("boulder_rating_correlations.csv", "csv"),
        "rosters": ("program_rosters.csv", "csv"),
    }
    output: dict[str, object] = {}
    output["projection_cache_release"] = projection_cache_release
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
    projection, projection_metadata = load_current_wc_projection_artifact()
    output["current_wc_projection"] = projection
    output["current_wc_projection_metadata"] = projection_metadata
    safe_athletes, safe_history, fixture_audit = quarantine_obvious_fixture_exposure(
        output["athletes"], output["history"]
    )
    output["athletes"] = safe_athletes
    output["history"] = safe_history
    output["fixture_quarantine"] = fixture_audit
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
        "cec_projected_rating": "canada_projection_all_evidence",
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


def athlete_selection_id(pool: object, global_id: object) -> str:
    """Return the stable UI identity; display names are never selection keys."""
    return f"{pool}::{global_id}"


def athlete_selection_ids(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [
            athlete_selection_id(pool, global_id)
            for pool, global_id in zip(frame["pool"], frame["global_id"])
        ],
        index=frame.index,
        dtype="string",
    )


def athlete_selector_frame(athletes: pd.DataFrame) -> pd.DataFrame:
    """Build unique stable choices and visibly disambiguate duplicate names."""
    selectors = athletes.dropna(subset=["pool", "global_id", "athlete_name"]).copy()
    selectors = selectors.drop_duplicates(["pool", "global_id"], keep="first")
    selectors["_selection_id"] = athlete_selection_ids(selectors)
    selectors["_selection_label"] = selectors["athlete_name"].map(friendly_name)
    duplicate_name = selectors["name_key"].map(selectors["name_key"].value_counts()).gt(1)
    country = selectors.get(
        "country", pd.Series("", index=selectors.index, dtype="string")
    ).fillna("").astype(str).str.strip()
    nationality = selectors.get(
        "nationality", pd.Series("", index=selectors.index, dtype="string")
    ).fillna("").astype(str).str.strip()
    jurisdiction = country.where(country.ne(""), nationality).replace("", "source unverified")
    selectors.loc[duplicate_name, "_selection_label"] = [
        f"{friendly_name(name)} · {nation} · {global_id}"
        for name, nation, global_id in zip(
            selectors.loc[duplicate_name, "athlete_name"],
            jurisdiction.loc[duplicate_name],
            selectors.loc[duplicate_name, "global_id"],
        )
    ]
    return selectors


def preferred_selection_ids(selectors: pd.DataFrame, names: list[str]) -> list[str]:
    """Resolve roster display names once, without selecting every same-name record."""
    requested = {plain_key(name) for name in names}
    matches = selectors.loc[selectors["name_key"].isin(requested)].copy()
    if matches.empty:
        return []
    country = matches.get(
        "country", pd.Series("", index=matches.index, dtype="string")
    ).fillna("").astype(str).str.upper()
    nationality = matches.get(
        "nationality", pd.Series("", index=matches.index, dtype="string")
    ).fillna("").astype(str).str.upper()
    matches["_canadian"] = country.eq("CAN") | nationality.eq("CAN")
    matches["_established"] = matches.get(
        "Global-ELO status", pd.Series("", index=matches.index, dtype="string")
    ).eq("Established")
    matches["_rated"] = pd.to_numeric(matches.get("Global-ELO"), errors="coerce").notna()
    matches["_evidence"] = pd.to_numeric(
        matches.get("Global-ELO evidence"), errors="coerce"
    ).fillna(-1)
    matches = matches.sort_values(
        ["name_key", "_canadian", "_established", "_rated", "_evidence", "_selection_id"],
        ascending=[True, False, False, False, False, True],
        kind="stable",
    ).drop_duplicates("name_key", keep="first")
    by_key = dict(zip(matches["name_key"], matches["_selection_id"]))
    return [by_key[key] for key in map(plain_key, names) if key in by_key]


def selected_rows(athletes: pd.DataFrame, selection_ids: list[str]) -> pd.DataFrame:
    """Select exact athlete records by stable identity, never by a colliding name."""
    tokens = {str(value) for value in selection_ids}
    stable = (
        athlete_selection_ids(athletes)
        if {"pool", "global_id"}.issubset(athletes.columns)
        else pd.Series("", index=athletes.index, dtype="string")
    )
    legacy_global_ids = athletes["global_id"].astype(str)
    return athletes.loc[stable.isin(tokens) | legacy_global_ids.isin(tokens)].copy()


def rating_help() -> str:
    return (
        "These rating families are diagnostic ledgers, not interchangeable "
        "probability scales. Global-ELO uses every de-duplicated Boulder result. "
        "IFSC-ELO uses IFSC results only. WC+-ELO uses World Cups/"
        "World Series, World Championships and Olympic-pathway events. The actual "
        "IFSC World Ranking remains a separate rank/points field. Specialist ratings are shown only "
        "with at least two eligible rounds and enough athletes to calibrate the "
        "family; they shrink toward Global-ELO while evidence is limited. "
        "Performance-ELO is the posterior mean WC-level rating left plausible "
        "after combining frozen Cumulative-ELO with every beat/lost-to pairing. "
        "WC+ uses the full result likelihood; lower-level evidence is tempered "
        "when translated to WC terrain."
    )


def rating_transform_controls(key: str, default: str) -> str:
    columns = st.columns([1.55, 1])
    family = columns[0].segmented_control(
        "Rating evidence", RATING_ORDER, default=default,
        help=rating_help(), key=f"{key}_family",
    )
    format_name = columns[1].selectbox(
        "Round format", FORMAT_OPTIONS, index=0,
        disabled=family != "Global-ELO",
        help=(
            "Format-specific ratings are available for the Global evidence pool. "
            "IFSC and WC+ transformations already describe a narrower event pool."
        ),
        key=f"{key}_format",
    )
    if family == "Global-ELO" and format_name != "All formats":
        return f"Global-ELO-{format_name}"
    return family


def correlation_note(
    correlations: pd.DataFrame, family: str, pool: str | None = None
) -> str:
    if correlations.empty or family == "WC+-ELO":
        return "WC+-ELO is the highest-circuit rating reference in this view."
    rows = correlations.loc[correlations["rating_family"].eq(family)]
    if pool:
        rows = rows.loc[rows["pool"].eq(pool)]
    rows = rows.dropna(subset=["spearman_correlation"])
    if rows.empty:
        return "Not enough paired evidence to report a stable relationship with WC+-ELO."
    value = float(np.average(rows["spearman_correlation"], weights=rows["athletes"]))
    n = int(rows["athletes"].sum())
    return (
        f"Rank correlation with WC+-ELO: {value:.2f} across {n} paired athlete-pools. "
        "This shows association, not why the ratings differ; setting specificity, "
        "training environment, attendance, travel and selection remain mixed together."
    )


def add_selected_highlight(
    figure: go.Figure,
    frame: pd.DataFrame,
    selected: list[str],
    x: str,
    y: str,
) -> None:
    focus = selected_rows(frame, selected).dropna(subset=[x, y])
    order = {selection_id: index for index, selection_id in enumerate(selected)}
    focus["_selection_order"] = athlete_selection_ids(focus).map(
        lambda selection_id: order.get(selection_id, len(order))
    )
    focus = focus.sort_values("_selection_order")
    offsets = [(-58, -34), (-68, 18), (-54, 42)]
    for position, (_, row) in enumerate(focus.iterrows()):
        figure.add_trace(
            go.Scatter(
                x=[row[x]], y=[row[y]], mode="markers",
                marker={
                    "size": 14, "color": "rgba(0,0,0,0)",
                    "line": {"width": 2, "color": "#36524E"},
                    "symbol": "diamond" if row.get("gender") == "Women" else "circle",
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )
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


def pool_scatter(
    frame: pd.DataFrame,
    x: str,
    rating: str,
    selected: list[str],
    title: str,
    canadian_outline: bool = False,
) -> go.Figure:
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
    # Do not draw outcome thresholds here. The old pilot reused one 2025
    # Global-ELO calibration across incompatible families; 2026 testing showed
    # those lines were not calibrated current advancement probabilities.
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
            f"The same athlete is CNR #{int(rank)}. This chart is the all-source "
            "Global-ELO diagnostic; disagreement with CNR or the target-matched "
            "WC benchmark is the useful review signal. "
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
        rank = integer_observation(
            leader.get("world_event_rank", np.nan), minimum=1
        )
        starts = integer_observation(leader.get("starts_365", np.nan))
        detail = (
            f"Current World Ranking: {rank} from {starts} eligible starts in 365 days. "
        )
    elif context == "Progression":
        detail = f"Their Global-ELO changed {leader.get('momentum', 0):+.0f} over the latest 365-day window. "
    else:
        detail = ""
    if context == "Canadian":
        suffix = (
            "Read this as an all-source rating gap, not the target-matched World "
            "Cup benchmark or a selection verdict."
        )
    elif context == "WR":
        suffix = (
            "Read this as a WC+ rating gap relative to this World Ranking pool, "
            "not a projected placing or a selection verdict."
        )
    else:
        suffix = (
            "Read the gap with evidence count and last-result date; it is a projection "
            "difference, not a selection verdict."
        )
    return base + detail + suffix


def render_canadian_pool(
    athletes: pd.DataFrame,
    selected: list[str],
    correlations: pd.DataFrame,
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
    )
    if x == "cnr_rank":
        figure.update_xaxes(autorange="reversed")
    st.plotly_chart(figure, width="stretch", theme=None)
    st.info(compare_text(canadian, selected, stage, "Canadian"), icon="↗️")
    st.caption(correlation_note(correlations, stage))
    missing = int(canadian[stage].isna().sum()) if stage in canadian else len(canadian)
    if missing:
        st.caption(f"{missing} CNR athlete(s) are retained in the pool but cannot be plotted on {stage} yet because eligible evidence is missing.")


def render_canadian_projection_pilot(
    athletes: pd.DataFrame,
    selected: list[str],
    current_projection: pd.DataFrame,
    projection_metadata: dict[str, object] | None = None,
) -> None:
    """Show the fixture-clean, form-and-target current WC pilot."""
    st.header("Canadian performance benchmark pilot")
    st.caption(
        "Current form, target-specific quality and a conditional semifinal "
        "benchmark for Canadian climbers."
    )
    st.info(
        "Representative-2026-field estimate, conditional on starting. The score "
        "uses all within-contest pairwise orderings, a 100-day form component and "
        "WC-specific transfer. It is not an attendance, selection, injury or "
        "route-style-specific forecast; a known entry list and route set should "
        "replace the representative field for named-event guidance.",
        icon="ℹ️",
    )
    focus = selected_rows(athletes, selected).copy()
    if focus.empty:
        st.warning("No selected athlete has a matched Boulder rating snapshot.")
        return
    selection_order = {selection_id: index for index, selection_id in enumerate(selected)}
    focus["_selection_order"] = athlete_selection_ids(focus).map(selection_order)
    focus = focus.sort_values("_selection_order", kind="stable")
    if not projection_metadata or projection_metadata.get("verified") is not True:
        reason = (
            str(projection_metadata.get("reason", "unverified projection artifact"))
            if projection_metadata
            else "projection metadata is missing"
        )
        st.error(
            "The fixture-clean current projection failed its artifact check: "
            f"{reason}. The legacy probability curves are intentionally not used."
        )
        return
    if current_projection.empty:
        st.error(
            "The fixture-clean current projection extract is missing. The old "
            "2025 display curves are intentionally not used as a fallback."
        )
        return
    required = {
        "athlete_id", "pool", "projection_status", "wc_projection_score",
        "wc_projection_score_sd", "wc_projection_score_sd_source", "score_route",
        "evidence_class", "governing_calibration_slope_per_100",
        "model_provisional", "model_gate_anchored_events",
        "model_gate_anchored_comparisons",
        "model_gate_unique_anchored_opponents",
        "form_adjustment_100d",
        "wc_target_adjustment", "direct_wc_competitions",
        "direct_senior_open_wc_plus_competitions",
        "direct_youth_world_competitions",
        "youth_world_target_adjustment",
        "youth_world_minus_wc_target_adjustment",
        "semifinal_probability_central",
        "semifinal_probability_easiest_observed_field",
        "semifinal_probability_hardest_observed_field",
        "semifinal_probability_rating_state_low",
        "semifinal_probability_rating_state_high",
    }
    if not required.issubset(current_projection.columns):
        st.error("The current projection extract failed its schema check.")
        return
    available_source = current_projection.loc[
        current_projection["projection_status"].eq(
            "exploratory_current_reference_available"
        ),
        "wc_projection_score_sd_source",
    ]
    if not available_source.eq("wc_latent_readiness_sd").all():
        st.error("The current projection extract has mixed uncertainty provenance.")
        return

    selected_projection = focus[["global_id", "pool", "athlete_name"]].merge(
        current_projection,
        left_on=["global_id", "pool"],
        right_on=["athlete_id", "pool"],
        how="left",
        validate="one_to_one",
        suffixes=("_overview", ""),
    )
    missing = selected_projection["projection_status"].isna()
    if missing.any():
        names = selected_projection.loc[missing, "athlete_name_overview"].map(
            friendly_name
        )
        st.caption(
            "No fixture-clean current WC state is available for: "
            + ", ".join(names.astype(str))
            + ". No legacy probability is substituted."
        )
    available = selected_projection.loc[
        selected_projection["projection_status"].eq(
            "exploratory_current_reference_available"
        )
    ].copy()
    if available.empty:
        st.warning("None of the selected athletes has an available current WC projection.")
        return

    def probability_text(value: object) -> str:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return f"{numeric:.1%}" if np.isfinite(numeric) else "—"

    def senior_open_wc_count(row: pd.Series) -> int:
        value = pd.to_numeric(
            pd.Series([row.get("direct_senior_open_wc_plus_competitions", np.nan)]),
            errors="coerce",
        ).iloc[0]
        return int(value) if np.isfinite(value) else 0

    def route_text(row: pd.Series) -> str:
        senior_wc_text = senior_open_wc_count(row)
        youth_world = pd.to_numeric(
            pd.Series([row.get("direct_youth_world_competitions", np.nan)]),
            errors="coerce",
        ).iloc[0]
        youth_text = (
            f" · {int(youth_world)} Youth World competition(s)"
            if np.isfinite(youth_world) and youth_world > 0
            else ""
        )
        route = str(row.get("score_route", ""))
        if route == "wc_target_score_standard_link":
            return (
                f"Direct Senior/Open WC+ · {senior_wc_text} competition(s)"
                f"{youth_text}"
            )
        source = str(row.get("bridge_source_domain", "")).replace("_", " ")
        pairs = pd.to_numeric(
            pd.Series([row.get("bridge_paired_athletes")]), errors="coerce"
        ).iloc[0]
        pair_text = f" · {int(pairs)} connected athletes" if np.isfinite(pairs) else ""
        source_text = f" · {source} bridge sensitivity{pair_text}" if source else ""
        if route == "wc_target_score_zero_prior_intercept_adjusted_link":
            prefix = "No prior Senior/Open WC+ · lower-baseline transfer link"
        elif route == "wc_target_score_one_prior_intercept_adjusted_link":
            prefix = "One prior Senior/Open WC+ · one-start transfer link"
        else:
            prefix = "Unrecognized evidence calibration"
        return f"{prefix}{youth_text}{source_text}"

    def graph_evidence_text(row: pd.Series) -> str:
        events = pd.to_numeric(
            pd.Series([row.get("model_gate_anchored_events")]), errors="coerce"
        ).iloc[0]
        opponents = pd.to_numeric(
            pd.Series([row.get("model_gate_unique_anchored_opponents")]),
            errors="coerce",
        ).iloc[0]
        status = "provisional" if bool(row.get("model_provisional")) else "established"
        indirect = str(row.get("score_route", "")) == (
            "wc_target_score_zero_prior_intercept_adjusted_link"
        )
        prefix = (
            f"Indirect-to-WC · {status} graph"
            if indirect
            else f"{status.capitalize()} graph"
        )
        if not np.isfinite(events) or not np.isfinite(opponents):
            return prefix
        return (
            f"{prefix} · {int(events)} connected events · "
            f"{int(opponents)} opponents"
        )

    def projection_evidence_text(row: pd.Series) -> str:
        senior_wc = pd.to_numeric(
            pd.Series([row.get("direct_senior_open_wc_plus_competitions")]),
            errors="coerce",
        ).iloc[0]
        if not np.isfinite(senior_wc) or senior_wc <= 0:
            return "Very low absolute certainty · no Senior/Open WC+ history"
        if senior_wc == 1:
            return "Low absolute certainty · one Senior/Open WC+ competition"
        if bool(row.get("model_provisional")):
            return "Moderate target evidence · provisional connected state"
        return "Higher target evidence · 2+ Senior/Open WC+ competitions"

    cards = st.columns(min(3, len(available)))
    for card, (_, row) in zip(cards, available.iterrows()):
        central = probability_text(row["semifinal_probability_central"])
        field_low = probability_text(
            row["semifinal_probability_hardest_observed_field"]
        )
        field_high = probability_text(
            row["semifinal_probability_easiest_observed_field"]
        )
        rating_low = probability_text(
            row["semifinal_probability_rating_state_low"]
        )
        rating_high = probability_text(
            row["semifinal_probability_rating_state_high"]
        )
        prior_wc = senior_open_wc_count(row)
        prior_label = "competition" if prior_wc == 1 else "competitions"
        card.metric(friendly_name(row["athlete_name_overview"]), central)
        card.caption(
            f"Observed 2026 field-strength sensitivity: {field_low}–{field_high}"
        )
        card.caption(f"Rating-state sensitivity: {rating_low}–{rating_high}")
        card.caption(
            f"Prior Senior/Open WC+: {prior_wc} {prior_label}"
        )
        card.caption(
            f"Projection confidence: {projection_evidence_text(row)}"
        )
        card.caption(f"Evidence route: {route_text(row)}")
        card.caption(f"Graph connectivity: {graph_evidence_text(row)}")
    st.caption(
        "These estimates are conditional semifinal scenarios for each athlete "
        "against a reference field; they are not head-to-head win probabilities "
        "or a firm ordering between athletes."
    )

    table = pd.DataFrame(
        {
            "Athlete": available["athlete_name_overview"].map(friendly_name),
            "Representative semifinal": available[
                "semifinal_probability_central"
            ].map(probability_text),
            "Projection confidence": [
                projection_evidence_text(row) for _, row in available.iterrows()
            ],
            "Rating-state sensitivity": [
                f"{probability_text(low)}–{probability_text(high)}"
                for low, high in zip(
                    available["semifinal_probability_rating_state_low"],
                    available["semifinal_probability_rating_state_high"],
                )
            ],
            "Evidence route": [route_text(row) for _, row in available.iterrows()],
            "Graph connectivity": [
                graph_evidence_text(row) for _, row in available.iterrows()
            ],
            "Observed-field sensitivity": [
                f"{probability_text(low)}–{probability_text(high)}"
                for low, high in zip(
                    available["semifinal_probability_hardest_observed_field"],
                    available["semifinal_probability_easiest_observed_field"],
                )
            ],
            "Connected-source sensitivity": [
                (
                    probability_text(
                        row.get("bridge_probability_evidence_class_sensitivity")
                    )
                    if str(row.get("evidence_class", "")).startswith(
                        ("zero_prior_", "one_prior_")
                    )
                    else "—"
                )
                for _, row in available.iterrows()
            ],
            "Current WC score": available["wc_projection_score"],
            "100-day form": available["form_adjustment_100d"],
            "WC target adjustment": available["wc_target_adjustment"],
            "Youth-World adjustment": available["youth_world_target_adjustment"],
        }
    )
    st.markdown("#### Current target-specific semifinal benchmark")
    st.dataframe(
        table.style.format(
            {
                "Current WC score": "{:.0f}",
                "100-day form": "{:+.0f}",
                "WC target adjustment": "{:+.0f}",
                "Youth-World adjustment": "{:+.0f}",
            },
            na_rep="—",
        ),
        hide_index=True,
        width="stretch",
    )

    all_available = current_projection.loc[
        current_projection["projection_status"].eq(
            "exploratory_current_reference_available"
        )
    ].copy()
    all_available["_cnr_rank"] = pd.to_numeric(
        all_available.get("cnr_rank"), errors="coerce"
    )
    all_available = all_available.sort_values(
        ["pool", "_cnr_rank", "athlete_name"], kind="stable", na_position="last"
    )
    all_table = pd.DataFrame(
        {
            "Athlete": all_available["athlete_name"].map(friendly_name),
            "Pool": all_available["pool"].map(
                {"Boulder_Men": "Men", "Boulder_Women": "Women"}
            ).fillna(all_available["pool"]),
            "CNR rank": all_available["_cnr_rank"],
            "Representative semifinal": all_available[
                "semifinal_probability_central"
            ].map(probability_text),
            "Projection confidence": [
                projection_evidence_text(row) for _, row in all_available.iterrows()
            ],
            "Rating-state sensitivity": [
                f"{probability_text(low)}–{probability_text(high)}"
                for low, high in zip(
                    all_available["semifinal_probability_rating_state_low"],
                    all_available["semifinal_probability_rating_state_high"],
                )
            ],
            "Evidence route": [
                route_text(row) for _, row in all_available.iterrows()
            ],
            "Graph connectivity": [
                graph_evidence_text(row) for _, row in all_available.iterrows()
            ],
            "100-day form": all_available["form_adjustment_100d"],
            "Direct Senior/Open WC+ comps": all_available.get(
                "direct_senior_open_wc_plus_competitions"
            ),
            "Youth World comps": all_available.get(
                "direct_youth_world_competitions"
            ),
        }
    )
    with st.expander(
        f"All current Canadian projections ({len(all_table):,})", expanded=True
    ):
        st.dataframe(
            all_table.style.format(
                {
                    "CNR rank": "{:.0f}",
                    "100-day form": "{:+.0f}",
                    "Direct Senior/Open WC+ comps": "{:.0f}",
                    "Youth World comps": "{:.0f}",
                },
                na_rep="—",
            ),
            hide_index=True,
            width="stretch",
        )
        unavailable_count = int(len(current_projection) - len(all_available))
        if unavailable_count:
            st.caption(
                f"{unavailable_count:,} additional CNR identity cluster(s) remain "
                "listed in the source roster but have no connected event-clean "
                "current state; no legacy probability is substituted."
            )
    st.caption(
        "Observed-field sensitivity changes only the 2026 reference field. "
        "Rating-state sensitivity varies the athlete score by one model SD; it "
        "is not a full confidence interval and does not include illness or a "
        "style-specific route set. Athletes with zero or one prior Senior/Open "
        "WC+ competition use separate, cross-validated baselines while retaining "
        "the full field-strength-weighted score slope. Youth and domestic evidence "
        "retains its opponent graph. Youth Worlds has a separate context head, so "
        "it can change shared ability through actual opponents without directly "
        "granting a Senior-WC target adjustment. The newcomer baseline prevents "
        "lower-pathway evidence from inheriting veteran advancement odds unchanged. "
        "Retained senior local, provincial, national and NACS fields still "
        "connect newcomers to established WC opponents inside the pairwise state. "
        "A connected lower-tier→WC bridge remains visible as a sensitivity, not "
        "as an arbitrary probability cap."
    )
    with st.expander("Validation status and how to use this pilot"):
        calibration_meta = dict(projection_metadata.get("calibration", {}))
        low_wc_meta = dict(
            projection_metadata.get("low_wc_evidence_calibration", {})
        )
        zero_prior_meta = dict(low_wc_meta.get("zero_prior", {}))
        one_prior_meta = dict(low_wc_meta.get("one_prior", {}))
        validation_rows = int(calibration_meta.get("clean_2026_rows", 0))
        validation_predicted = float(
            calibration_meta.get("clean_2026_predicted_rate", np.nan)
        )
        validation_observed = float(
            calibration_meta.get("clean_2026_observed_rate", np.nan)
        )
        validation_brier = float(calibration_meta.get("clean_2026_brier", np.nan))
        validation_log_loss = float(
            calibration_meta.get("clean_2026_log_loss", np.nan)
        )
        canadian_rows = int(calibration_meta.get("clean_2026_canadian_rows", 0))
        canadian_predicted = float(
            calibration_meta.get("clean_2026_canadian_predicted_rate", np.nan)
        )
        canadian_observed = float(
            calibration_meta.get("clean_2026_canadian_observed_rate", np.nan)
        )
        low_rows = int(low_wc_meta.get("clean_2026_rows", 0))
        low_positives = int(low_wc_meta.get("clean_2026_positives", 0))
        low_predicted = float(
            low_wc_meta.get("clean_2026_predicted_rate", np.nan)
        )
        low_observed = float(
            low_wc_meta.get("clean_2026_observed_rate", np.nan)
        )
        zero_rows = int(zero_prior_meta.get("clean_2026_rows", 0))
        zero_positives = int(zero_prior_meta.get("clean_2026_positives", 0))
        zero_predicted = float(
            zero_prior_meta.get("clean_2026_predicted_rate", np.nan)
        )
        zero_observed = float(
            zero_prior_meta.get("clean_2026_observed_rate", np.nan)
        )
        one_rows = int(one_prior_meta.get("clean_2026_rows", 0))
        one_positives = int(one_prior_meta.get("clean_2026_positives", 0))
        one_predicted = float(
            one_prior_meta.get("clean_2026_predicted_rate", np.nan)
        )
        one_observed = float(
            one_prior_meta.get("clean_2026_observed_rate", np.nan)
        )
        st.markdown(
            "- The dependence-aware score uses all pairwise orderings but caps "
            "each event's effective pair weight; one large youth field is not "
            "hundreds of independent wins.\n"
            "- Youth Worlds is isolated from the Senior-WC target head: all 5,752 "
            "retained Youth-World rows still update shared ability through their "
            "opponents, while zero directly update the Senior-WC context.\n"
            "- The semifinal link was refit on the event-clean 2025 Open World "
            f"Cup replay. On {validation_rows:,} untouched 2026 starts it predicted "
            f"{validation_predicted:.1%} advancement versus {validation_observed:.1%} "
            f"observed (Brier {validation_brier:.3f}; log loss "
            f"{validation_log_loss:.3f}).\n"
            f"- In the smaller Canadian 2026 subset (`n={canadian_rows:,}`), the "
            f"same governing link predicted {canadian_predicted:.1%} versus "
            f"{canadian_observed:.1%} observed. Treat this subgroup check as a "
            "calibration warning, not a separate refit.\n"
            "- Zero and one prior Senior/Open WC+ starts are calibrated separately "
            "using only 2025 event-held-out selection. In untouched 2026, the "
            f"zero-prior class had {zero_positives:,}/{zero_rows:,} semifinalists "
            f"(predicted {zero_predicted:.1%}; observed {zero_observed:.1%}); the "
            f"one-prior class had {one_positives:,}/{one_rows:,} (predicted "
            f"{one_predicted:.1%}; observed {one_observed:.1%}). Combined, the "
            f"low-evidence link predicted {low_predicted:.1%} versus "
            f"{low_observed:.1%} over {low_rows:,} starts. The lower-baseline "
            "links lower newcomer odds without flattening athlete-to-"
            "athlete quality differences; neither is a hard probability cap.\n"
            "- Form affects the current score, but the old `0.75 × momentum` "
            "shortcut is not used; it worsened probability calibration.\n"
            "- A chronological Youth-World/Senior/NACS pathway-residual bridge "
            "was tested as an additional predictor. It slightly improved pairwise "
            "ordering but slightly worsened held-event semifinal log loss, so it "
            "remains a displayed sensitivity instead of being counted twice in the "
            "central probability.\n"
            f"- Independent 2026 temporal-holdout context (rank model, not the "
            f"source of these probability coefficients): `{FROZEN_2026_HOLDOUT_ARTIFACT}`. "
            "Current results run through 2026-07-25."
        )


def render_ifsc_pool(
    athletes: pd.DataFrame,
    history: pd.DataFrame,
    selected: list[str],
    correlations: pd.DataFrame,
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
    stage = rating_transform_controls("wr", "WC+-ELO")
    figure = pool_scatter(
        pool, "world_event_rank", stage, selected,
        f"World Ranking pool — {stage}", canadian_outline=True,
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
    country_pool = pool.loc[pool["country"].isin(["CAN", *countries])].dropna(subset=["WC+-ELO"])
    if not country_pool.empty:
        comparison = px.strip(
            country_pool, x="country", y="WC+-ELO", color="country",
            hover_name="display_name",
            hover_data={"world_event_rank": True, "starts_365": True},
            title="WC+ rating distribution of current World Ranking participants",
        )
        comparison.update_traces(marker={"size": 10, "opacity": 0.65})
        comparison.update_layout(height=430, showlegend=False)
        st.plotly_chart(comparison, width="stretch", theme=None)


def age_group(age: float) -> str:
    if pd.isna(age):
        return "Age unknown"
    if age < 15:
        return "Under 15 years"
    if age < 17:
        return "15 to under 17 years"
    if age < 19:
        return "17 to under 19 years"
    if age < 21:
        return "19 to under 21 years"
    return "21 years and older"


def render_progression(
    athletes: pd.DataFrame,
    history: pd.DataFrame,
    age_progression: pd.DataFrame,
    selected: list[str],
    correlations: pd.DataFrame,
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
    figure = px.scatter(
        plot, x="age", y="Global-ELO", color="Age group", symbol="gender",
        hover_name="display_name",
        hover_data={"cnr_rank": True, "momentum": ":.1f", "country": True},
        title="Current Canadian pathway — diagnostic all-result rating",
    )
    figure.update_traces(marker={"size": 10, "opacity": 0.65})
    add_selected_highlight(figure, plot, selected, "age", "Global-ELO")

    show_lines = st.toggle(
        "Show pathway reference lines",
        value=True,
        help=(
            "Shows the mean Global-ELO of current CNR top-five athletes by age "
            "band. It is a descriptive pathway reference, not a World Cup "
            "advancement threshold."
        ),
    )
    if show_lines:
        top5 = athletes.loc[athletes["cnr_rank"].le(5)].dropna(subset=["age", "Global-ELO"])
        if not top5.empty:
            grouped = (
                top5.assign(age_year=top5["age"].map(centered_age_year))
                .dropna(subset=["age_year"])
                .groupby(["gender", "age_year"], as_index=False)["Global-ELO"]
                .mean()
            )
            for gender, gender_rows in grouped.groupby("gender"):
                figure.add_trace(go.Scatter(
                    x=gender_rows["age_year"], y=gender_rows["Global-ELO"], mode="lines",
                    name=f"Current CNR top-five mean — {gender}",
                    line={
                        "color": PALETTE["teal"] if gender == "Men" else PALETTE["blue"],
                        "width": 3, "dash": "dot",
                    },
                ))
    figure.update_layout(height=570, margin={"l": 20, "r": 20, "t": 70, "b": 20})
    st.plotly_chart(figure, width="stretch", theme=None)
    st.info(compare_text(cohort, selected, "Global-ELO", "Progression"), icon="↗️")

    projection_figure = progression_projection(
        athletes, history, age_progression, selected
    )
    st.plotly_chart(projection_figure, width="stretch", theme=None)
    reviewed_rates = typical_age_progression(age_progression)
    if reviewed_rates:
        st.caption(
            "The minimum-20 IFSC age table passed its descriptive research "
            "contract, but it is not authorized for named-athlete projection."
        )
    else:
        st.warning(
            "The descriptive minimum-20 IFSC age table is missing or invalid."
        )
    st.warning(
        "Individual future projections are withheld. Observed rating histories "
        "remain visible while the professional forecast contract is validated."
    )
    st.caption(correlation_note(correlations, "Global-ELO"))


def progression_projection(
    athletes: pd.DataFrame,
    history: pd.DataFrame,
    age_progression: pd.DataFrame,
    selected: list[str],
) -> go.Figure:
    """Show observed histories while individual forecast use is unauthorized."""

    del age_progression
    figure = go.Figure()
    focus = selected_rows(athletes, selected)
    if history.empty or focus.empty:
        return figure.update_layout(
            title="Individual future projection withheld; observed history unavailable"
        )
    rows_source = history.copy()
    rows_source["event_date"] = pd.to_datetime(
        rows_source["event_date"], errors="coerce"
    )
    as_of = rows_source["event_date"].max()
    colors = [PALETTE["teal"], PALETTE["coral"], PALETTE["blue"]]
    for color, (_, athlete) in zip(colors, focus.iterrows()):
        rows = rows_source.loc[
            rows_source["global_id"].eq(athlete["global_id"])
            & rows_source["pool"].eq(athlete["pool"])
        ].dropna(subset=["rating_after"]).sort_values("event_date")
        if rows.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=rows["event_date"],
                y=rows["rating_after"],
                mode="lines",
                name=f"{athlete['athlete_name']} - observed",
                line={"color": color, "width": 3},
            )
        )
    if pd.notna(as_of):
        figure.add_vline(
            x=as_of.timestamp() * 1000,
            line_dash="dash",
            line_color="#555",
            annotation_text="Now",
        )
    figure.update_layout(
        title=(
            "Observed Global-ELO; individual future projection withheld "
            "pending a validated forecast contract"
        ),
        height=500,
        yaxis_title="Global-ELO",
        xaxis_title="Event date",
        hovermode="x unified",
    )
    return figure


def typical_age_progression(
    reference: pd.DataFrame,
) -> dict[tuple[str, int], float]:
    """Read the privacy-thresholded age-progression reference.

    Individual event ages are deliberately absent from the public history.
    The compact reference is built from the restricted internal history and
    publishes a cell only when at least 20 distinct athletes contribute.
    """

    if reference.empty or tuple(reference.columns) != AGE_PROGRESSION_FIELDS:
        return {}
    rows = reference.copy()
    age = pd.to_numeric(rows["age_center_years"], errors="coerce")
    lower = pd.to_numeric(rows["age_bin_lower_years"], errors="coerce")
    upper = pd.to_numeric(rows["age_bin_upper_years"], errors="coerce")
    athletes = pd.to_numeric(rows["athletes"], errors="coerce")
    observations = pd.to_numeric(rows["observations"], errors="coerce")
    minimum = pd.to_numeric(rows["minimum_athletes"], errors="coerce")
    draws = pd.to_numeric(rows["bootstrap_draws"], errors="coerce")
    value = pd.to_numeric(
        rows["median_annual_performance_elo_change"], errors="coerce"
    )
    standard_error = pd.to_numeric(
        rows["bootstrap_se_annual_performance_elo_change"], errors="coerce"
    )
    dates = {}
    for column in (
        "source_window_start",
        "source_window_end",
        "source_as_of_date",
    ):
        text = rows[column].astype(str)
        parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
        canonical = parsed.dt.strftime("%Y-%m-%d").eq(text)
        if parsed.isna().any() or not canonical.all():
            return {}
        dates[column] = parsed
    valid = (
        age.between(12, 45)
        & age.eq(np.floor(age))
        & lower.eq(age - 0.5)
        & upper.eq(age + 0.5)
        & athletes.ge(20)
        & athletes.eq(np.floor(athletes))
        & observations.ge(athletes)
        & observations.eq(np.floor(observations))
        & minimum.eq(20)
        & draws.eq(2000)
        & np.isfinite(value)
        & np.isfinite(standard_error)
        & standard_error.ge(0)
        & rows["age_assignment_status"].eq(
            "FULL_SOURCE_INTERVAL_WITHIN_CENTERED_BIN"
        )
        & rows["source_scope"].eq("IFSC")
        & rows["method"].eq(AGE_PROGRESSION_METHOD)
        & rows["status"].eq(AGE_PROGRESSION_STATUS)
        & rows["research_only"].eq(True)  # noqa: E712
        & dates["source_window_start"].le(dates["source_window_end"])
        & dates["source_window_end"].le(dates["source_as_of_date"])
    )
    if (
        not valid.all()
        or rows[["pool", "age_center_years"]].duplicated().any()
        or rows["pool"].astype(str).str.strip().eq("").any()
        or rows["source_as_of_date"].nunique(dropna=False) != 1
    ):
        return {}
    return {
        (str(pool), int(age_value)): float(rate)
        for pool, age_value, rate in zip(
            rows.loc[valid, "pool"], age.loc[valid], value.loc[valid], strict=False
        )
    }


def centered_age_year(value: object) -> int | None:
    """Return the same half-up centered age bin used by the producer."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    center = int(np.floor(numeric + 0.5))
    return center if 12 <= center <= 45 else None


def integer_observation(
    value: object,
    *,
    minimum: int = 0,
    missing: str = "not recorded",
) -> str:
    """Format a count/rank without treating missing evidence as zero."""

    if isinstance(value, (bool, np.bool_)):
        return missing
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return missing
    if (
        not np.isfinite(numeric)
        or numeric < minimum
        or not numeric.is_integer()
    ):
        return missing
    return str(int(numeric))


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
        rank = integer_observation(
            athlete.get("world_event_rank", np.nan), minimum=1
        )
        starts = integer_observation(athlete.get("starts_365", np.nan))
        column.caption(
            f"Current World Ranking: {rank} · starts/365d: {starts}"
        )
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
        selectors = athlete_selector_frame(athletes)
        selectors = selectors.sort_values(
            ["_selection_label", "_selection_id"], key=lambda values: values.astype(str).str.casefold()
        )
        selection_options = selectors["_selection_id"].tolist()
        label_by_id = dict(zip(selectors["_selection_id"], selectors["_selection_label"]))
        selected: list[str]
        if mode == "Compare 3":
            columns = st.columns(3)
            defaults = preferred_selection_ids(selectors, DEFAULT_ATHLETES)
            while len(defaults) < 3 and selection_options:
                candidate = selection_options[min(len(defaults), len(selection_options) - 1)]
                if candidate not in defaults:
                    defaults.append(candidate)
                else:
                    break
            selected = []
            for index, column in enumerate(columns):
                default = defaults[index] if index < len(defaults) else selection_options[0]
                selected.append(column.selectbox(
                    "Main athlete" if index == 0 else f"Comparison {index + 1}",
                    selection_options,
                    index=selection_options.index(default),
                    format_func=lambda value: label_by_id[value],
                    key=f"athlete_{index}",
                ))
        else:
            preset = roster_names(mode, athletes, history, rosters)
            matched = preferred_selection_ids(selectors, preset)
            selected = st.multiselect(
                f"{mode} athletes",
                selection_options,
                default=matched[:12],
                format_func=lambda value: label_by_id[value],
                help="Choose up to the athletes you need; the three first selections receive the strongest visual emphasis.",
            )
            if mode == "Canadian National Team proxy":
                st.caption("Proxy only: current CNR top 15 by gender. Replace with the official roster when supplied.")
        return selected[:3], discipline


def render_rating_detail(
    athletes: pd.DataFrame,
    history: pd.DataFrame,
    selected: list[str],
) -> None:
    with st.expander("Compared athletes · all rating evidence"):
        focus = selected_rows(athletes, selected)
        if focus.empty:
            st.caption("No matched rating evidence.")
            return
        tabs = st.tabs([friendly_name(row["athlete_name"]) for _, row in focus.iterrows()])
        for tab, (_, athlete) in zip(tabs, focus.iterrows()):
            with tab:
                metrics = st.columns(3)
                for column, family in zip(metrics, RATING_ORDER):
                    value = athlete.get(family, np.nan)
                    column.metric(family, f"{value:.0f}" if pd.notna(value) else "—")
                detail = pd.DataFrame([
                    {
                        "Rating family": family,
                        "Elo": athlete.get(family),
                        "Eligible contests": athlete.get(f"{family} evidence"),
                    }
                    for family in ALL_RATINGS
                ])
                st.dataframe(
                    detail.style.format(
                        {"Elo": "{:.0f}", "Eligible contests": "{:.0f}"},
                        na_rep="—",
                    ),
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "Missing Elo means the minimum evidence rule was not met. "
                    "Eligible contests describes evidence quantity, not athlete quality."
                )
                rounds = history.loc[
                    history["global_id"].eq(athlete["global_id"])
                    & history["pool"].eq(athlete["pool"])
                ].sort_values("event_date", ascending=False).head(5)
                if not rounds.empty:
                    latest = rounds[[
                        "event_date", "event_name", "round_group",
                        "confirmed_procedure", "performance_elo",
                    ]].rename(columns={
                        "event_date": "Event date", "event_name": "Competition",
                        "round_group": "Round",
                        "confirmed_procedure": "Procedure",
                        "performance_elo": "Performance-ELO",
                    })
                    st.markdown("**Latest isolated round performances**")
                    st.dataframe(
                        latest.style.format({"Performance-ELO": "{:.0f}"}, na_rep="—"),
                        hide_index=True,
                        width="stretch",
                    )


def startup_status(data: dict[str, pd.DataFrame]) -> None:
    missing = [
        key for key in ("athletes", "history")
        if data[key].empty
    ]
    age_reference_ready = bool(
        typical_age_progression(data.get("age_progression", pd.DataFrame()))
    )
    with st.expander(
        "Data health", expanded=bool(missing) or not age_reference_ready
    ):
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
            fixture_audit = data.get("fixture_quarantine", pd.DataFrame())
            if not fixture_audit.empty:
                row = fixture_audit.iloc[0]
                exposed_ids = int(row.get("fixture_exposed_athlete_ids", 0))
                if exposed_ids:
                    st.info(
                        "Source-semantic guard: "
                        f"{int(row.get('fixture_event_rows', 0)):,} rows across "
                        f"{int(row.get('fixture_pool_event_keys', 0)):,} clearly labelled "
                        "test/demo pool-event keys are removed from the legacy history view. "
                        f"All identities remain selectable; legacy ELO diagnostics are blanked "
                        f"for {exposed_ids:,} directly exposed identities. The current WC pilot "
                        "is independently replayed from event-clean inputs."
                    )
            current_projection = data.get("current_wc_projection", pd.DataFrame())
            if not current_projection.empty:
                available = int(
                    current_projection.get(
                        "projection_status",
                        pd.Series("", index=current_projection.index),
                    ).eq("exploratory_current_reference_available").sum()
                )
                st.caption(
                    f"Fixture-clean current WC pilot loaded for {available:,} "
                    "Canadian identity clusters; legacy universal outcome "
                    "thresholds are disabled."
                )
            if not age_reference_ready:
                st.warning(
                    "The descriptive minimum-20 aggregate age reference is "
                    "missing or invalid. Named-athlete projection remains "
                    "withheld in either case; no missing effect is treated as zero."
                )
        st.caption("Duplicate controls run before the rating build. Missing specialist evidence is withheld rather than silently replaced.")


def configured_access_password() -> str:
    """Read an optional deployment secret without placing it in source control."""
    configured = os.environ.get("ACCESS_PASSWORD", "")
    try:
        configured = str(st.secrets.get("ACCESS_PASSWORD", configured))
    except (FileNotFoundError, KeyError, RuntimeError, TypeError):
        pass
    return configured.strip()


def require_optional_password() -> None:
    """Protect a private deployment when ACCESS_PASSWORD is configured."""
    expected = configured_access_password()
    if not expected:
        return
    if st.session_state.get("pilot_access_granted", False):
        with st.sidebar:
            if st.button("Sign out", key="pilot_sign_out"):
                st.session_state.pop("pilot_access_granted", None)
                st.rerun()
        return
    supplied = st.text_input(
        "Access password", type="password", key="pilot_access_password"
    )
    if supplied:
        st.session_state["pilot_access_granted"] = hmac.compare_digest(
            supplied, expected
        )
        if st.session_state["pilot_access_granted"]:
            st.rerun()
        st.error("Incorrect password.")
    st.stop()


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
          .js-plotly-plot{min-height:420px}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    require_optional_password()
    st.title("Comp Climbing Projections")
    st.markdown("### Strength, depth and progression of canadian climbers: from local comps to the Olympics")
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
    render_rating_detail(athletes, data["history"], selected)
    render_canadian_projection_pilot(
        athletes,
        selected,
        data["current_wc_projection"],
        data.get("current_wc_projection_metadata"),
    )

    st.header("Overview")
    section = st.segmented_control(
        "Overview section",
        ["Canadian Pool", "IFSC Pool", "WR Pool", "Global progression", "Towards Olympics"],
        default="Canadian Pool",
        label_visibility="collapsed",
    )
    renderers = {
        "Canadian Pool": lambda: render_canadian_pool(
            athletes, selected, data["correlations"]
        ),
        "IFSC Pool": lambda: render_ifsc_pool(
            athletes, data["history"], selected, data["correlations"],
        ),
        "WR Pool": lambda: render_wr_pool(
            athletes, selected, data["correlations"]
        ),
        "Global progression": lambda: render_progression(
            athletes, data["history"], data["age_progression"],
            selected, data["correlations"],
        ),
        "Towards Olympics": lambda: render_olympics(athletes, selected, data["correlations"]),
    }
    renderers[section]()

    with st.expander("Rating glossary and model contract"):
        st.write(rating_help())
        st.markdown(
            "- **Global-ELO-Onsight / Scramble / Flash:** only rounds with a confirmed procedure.\n"
            "- **WC+-ELO-Qualies / Semies / Finals:** only the named round inside World Cup+, World Championship and Olympic-pathway events.\n"
            "- **IFSC-ELO-Qualies / Semies / Finals:** only the named round of non-para IFSC events.\n"
            "- **Performance-ELO:** the isolated level shown in one round, calculated from ratings frozen before the event."
        )


if __name__ == "__main__":
    main()
