"""Release-safe guards for a reviewed identity correction and WC eligibility.

This module intentionally does *not* rewrite published ratings.  Joining an
alias into a canonical athlete changes the chronological rating state and must
be rebuilt by the rating producer, not approximated in the Streamlit process.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


IDENTITY_OVERRIDE_COLUMNS = (
    "source_scope",
    "athlete_source_id",
    "ifsc_athlete_id",
    "identity_confidence",
    "reviewer_note",
    "evidence_url",
)
REVIEWED_ALIAS_GLOBAL_ID = "IFSC:18545"
REVIEWED_CANONICAL_GLOBAL_ID = "IFSC:14843"


def quarantine_reviewed_split_identity_outputs(
    athletes: pd.DataFrame,
    projection: pd.DataFrame,
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Withhold derived outputs whose chronology predates a reviewed merge.

    The canonical and alias histories cannot be joined after ratings have
    already been computed.  Until the producer replays the accepted mapping,
    keep the canonical profile selectable but remove its derived ratings and
    current projection.  Raw history remains available for audit.
    """

    athlete_ids = athletes.get(
        "global_id", pd.Series("", index=athletes.index, dtype="string")
    ).astype(str)
    history_ids = history.get(
        "global_id", pd.Series("", index=history.index, dtype="string")
    ).astype(str)
    alias_history_rows = int(history_ids.eq(REVIEWED_ALIAS_GLOBAL_ID).sum())
    canonical_mask = athlete_ids.eq(REVIEWED_CANONICAL_GLOBAL_ID)
    if alias_history_rows <= 0:
        safe_projection = projection.copy()
        projection_rows_withheld = 0
        if not safe_projection.empty and "athlete_id" in safe_projection:
            projection_mask = safe_projection["athlete_id"].astype(str).eq(
                REVIEWED_CANONICAL_GLOBAL_ID
            )
            projection_rows_withheld = int(projection_mask.sum())
            safe_projection = safe_projection.loc[~projection_mask].copy()
        return athletes, safe_projection, {
            "quarantine_active": False,
            "canonical_profile_count": int(canonical_mask.sum()),
            "alias_history_rows": 0,
            "athlete_rating_values_withheld": 0,
            "projection_rows_withheld": projection_rows_withheld,
            "projection_rebuild_pending": bool(projection_rows_withheld),
        }
    if int(canonical_mask.sum()) != 1:
        raise ValueError("reviewed split identity canonical profile cardinality changed")

    safe_athletes = athletes.copy()
    rating_columns = [
        column
        for column in safe_athletes.columns
        if "ELO" in column.upper()
    ]
    rating_columns.extend(
        column
        for column in (
            "momentum",
            "canada_projection_all_evidence",
            "Canada context adjustment",
            "cec_projected_rating",
            "cec_context_offset",
        )
        if column in safe_athletes.columns and column not in rating_columns
    )
    numeric_values = sum(
        int(pd.notna(safe_athletes.loc[canonical_mask, column]).sum())
        for column in rating_columns
    )
    if rating_columns:
        safe_athletes.loc[canonical_mask, rating_columns] = pd.NA
    if "Global-ELO status" in safe_athletes:
        safe_athletes.loc[
            canonical_mask, "Global-ELO status"
        ] = "Withheld pending identity rebuild"
    safe_athletes.loc[canonical_mask, "identity_rebuild_pending"] = True

    safe_projection = projection.copy()
    projection_rows_withheld = 0
    if not safe_projection.empty and "athlete_id" in safe_projection:
        projection_mask = safe_projection["athlete_id"].astype(str).eq(
            REVIEWED_CANONICAL_GLOBAL_ID
        )
        projection_rows_withheld = int(projection_mask.sum())
        safe_projection = safe_projection.loc[~projection_mask].copy()

    return safe_athletes, safe_projection, {
        "quarantine_active": True,
        "canonical_global_id": REVIEWED_CANONICAL_GLOBAL_ID,
        "alias_global_id": REVIEWED_ALIAS_GLOBAL_ID,
        "canonical_profile_count": 1,
        "alias_history_rows": alias_history_rows,
        "athlete_rating_values_withheld": numeric_values,
        "projection_rows_withheld": projection_rows_withheld,
        "history_rows_changed": 0,
        "runtime_merge_performed": False,
        "requires_producer_rebuild": True,
    }


def load_reviewed_identity_overrides(path: Path) -> pd.DataFrame:
    """Load the deliberately narrow, reviewed override ledger.

    The release lane permits exactly the reviewed IFSC 18545 -> 14843 mapping.
    Additional rows are a new identity decision and must not silently enter a
    public build.
    """

    overrides = pd.read_csv(path, dtype="string")
    if tuple(overrides.columns) != IDENTITY_OVERRIDE_COLUMNS:
        raise ValueError("unexpected reviewed identity override schema")
    if len(overrides) != 1:
        raise ValueError("release override ledger must contain exactly one mapping")
    row = overrides.iloc[0]
    if (
        str(row["source_scope"]).upper() != "IFSC"
        or str(row["athlete_source_id"]) != "18545"
        or str(row["ifsc_athlete_id"]) != "14843"
        or float(row["identity_confidence"]) != 1.0
    ):
        raise ValueError("release override ledger is not the reviewed IFSC mapping")
    return overrides


def senior_wc_direct_evidence_mask(history: pd.DataFrame) -> pd.Series:
    """Return direct Senior/Open World-level evidence, never youth or regional.

    This is a display/release guard.  The producer must use the same semantic
    rule when it rebuilds the WC+ rating family.
    """

    if history.empty:
        return pd.Series(False, index=history.index, dtype=bool)
    names = history.get(
        "event_name", pd.Series("", index=history.index, dtype="string")
    ).astype("string")
    senior_name = names.str.contains(
        r"(?i)\b(?:world\s+(?:climbing\s+)?(?:cup|series|championships?)|"
        r"olympic(?:\s+qualifier)?(?:\s+series)?|oqs)\b",
        regex=True,
        na=False,
    )
    youth = names.str.contains(r"(?i)\byouth\b", regex=True, na=False)
    source = history.get(
        "source_scope", pd.Series("", index=history.index, dtype="string")
    ).astype("string").str.upper()
    return source.eq("IFSC") & senior_name & ~youth


def identity_rebuild_status(
    athletes: pd.DataFrame,
    history: pd.DataFrame,
) -> dict[str, object]:
    """Describe whether release inputs still require the producer rebuild."""

    athlete_ids = athletes.get("global_id", pd.Series(dtype="string")).astype(str)
    history_ids = history.get("global_id", pd.Series(dtype="string")).astype(str)
    alias_rows = int(history_ids.eq(REVIEWED_ALIAS_GLOBAL_ID).sum())
    return {
        "reviewed_alias_global_id": REVIEWED_ALIAS_GLOBAL_ID,
        "reviewed_canonical_global_id": REVIEWED_CANONICAL_GLOBAL_ID,
        "canonical_profile_count": int(athlete_ids.eq(REVIEWED_CANONICAL_GLOBAL_ID).sum()),
        "alias_profile_count": int(athlete_ids.eq(REVIEWED_ALIAS_GLOBAL_ID).sum()),
        "alias_history_rows": alias_rows,
        "requires_producer_rebuild": bool(alias_rows),
        "runtime_merge_performed": False,
    }


def suppress_reviewed_alias_profile(
    athletes: pd.DataFrame,
    overrides: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Hide a reviewed alias, or accept its exact rebuilt terminal state.

    The deployed predecessor contains one canonical and one alias profile, so
    the alias is suppressed without changing either rating history.  A
    producer-rebuilt successor contains one canonical and zero alias profiles;
    that state must pass through unchanged.  Every other cardinality remains a
    hard error.
    """

    if len(overrides) != 1:
        raise ValueError("reviewed override ledger must be validated before suppression")
    ids = athletes.get("global_id", pd.Series("", index=athletes.index)).astype(str)
    alias_mask = ids.eq(REVIEWED_ALIAS_GLOBAL_ID)
    canonical_count = int(ids.eq(REVIEWED_CANONICAL_GLOBAL_ID).sum())
    alias_count = int(alias_mask.sum())
    if canonical_count != 1 or alias_count not in (0, 1):
        raise ValueError("reviewed canonical/alias profile cardinality changed")
    if alias_count == 0:
        return athletes.copy(), {
            "reviewed_alias_global_id": REVIEWED_ALIAS_GLOBAL_ID,
            "reviewed_canonical_global_id": REVIEWED_CANONICAL_GLOBAL_ID,
            "suppressed_profile_count": 0,
            "history_rows_changed": 0,
            "ratings_merged": False,
            "requires_producer_rebuild": False,
            "identity_state": "REVIEWED_REBUILD_RESOLVED",
        }
    return athletes.loc[~alias_mask].copy(), {
        "reviewed_alias_global_id": REVIEWED_ALIAS_GLOBAL_ID,
        "reviewed_canonical_global_id": REVIEWED_CANONICAL_GLOBAL_ID,
        "suppressed_profile_count": alias_count,
        "history_rows_changed": 0,
        "ratings_merged": False,
        "requires_producer_rebuild": True,
        "identity_state": "REVIEWED_SPLIT_PREDECESSOR",
    }
