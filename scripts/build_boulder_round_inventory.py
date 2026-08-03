"""Build the governed event/round/boulder inventory used by the public tagger.

The input is the normalized ``source_results.csv.gz`` produced by the research
repository.  ``n_routes`` is the source-reported number of boulders in a round.
Rows without that field remain explicitly assumed or unknown; the builder never
silently presents a format default as an observed count.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd


ROUND_ORDER = {"Qualification": 0, "Semi-final": 1, "Final": 2, "Other": 3}


def plain_key(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    return re.sub(r"[^a-z0-9]+", " ", text.encode("ascii", "ignore").decode().casefold()).strip()


def round_group(value: object) -> str:
    text = plain_key(value)
    if "qual" in text or "clasific" in text:
        return "Qualification"
    if "semi" in text:
        return "Semi-final"
    if "final" in text:
        return "Final"
    return "Other"


def category_round_id(source_url: object) -> str:
    match = re.search(r"category_rounds/(\d+)", str(source_url or ""))
    return match.group(1) if match else ""


def terrain_group(source_scope: object, age_class: object, category: object) -> str:
    source = plain_key(source_scope)
    age = plain_key(age_class)
    category_key = plain_key(category)
    combined = f"{age} {category_key}"
    youth = "youth" in age or any(
        token in combined
        for token in ("u15", "u 15", "u17", "u 17", "u19", "u 19", "u21", "u 21", "junior")
    )
    if not youth:
        return "Open / Senior"

    is_junior = any(token in combined for token in ("u21", "u 21", "junior", "20"))
    is_youth_a = any(token in combined for token in ("u19", "u 19", "youth a", "19"))
    is_youth_b = any(token in combined for token in ("u17", "u 17", "youth b", "17"))
    is_youth_c = any(token in combined for token in ("u15", "u 15", "youth c", "15"))
    if source == "cec" and (is_junior or is_youth_a):
        return "Youth A + Junior (shared Canadian terrain)"
    if is_junior:
        return "Junior"
    if is_youth_a:
        return "Youth A"
    if is_youth_b:
        return "Youth B"
    if is_youth_c:
        return "Youth C"
    return "Other youth"


def format_default(round_name: str) -> int | None:
    if round_name == "Qualification":
        return 5
    if round_name in {"Semi-final", "Final"}:
        return 4
    return None


def stable_uid(*values: object) -> str:
    def identity_key(value: object) -> str:
        text = unicodedata.normalize("NFKD", str(value or ""))
        plain = "".join(
            char.lower() if char.isalnum() else " "
            for char in text
            if not unicodedata.combining(char)
        )
        return "".join(sorted(plain.split()))

    canonical = "|".join(identity_key(value) for value in values)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def build_inventory(source_path: Path) -> pd.DataFrame:
    wanted = [
        "source_scope", "source_event_id", "event_date", "event_name", "source_url",
        "pool", "gender", "category", "round_name", "age_class", "n_routes",
        "format_identifier",
    ]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(source_path, chunksize=100_000, low_memory=False):
        chunk = chunk.loc[
            chunk["pool"].astype(str).str.startswith("Boulder_")
            & ~chunk["round_name"].astype(str).map(plain_key).str.contains("ranking", regex=False),
            [column for column in wanted if column in chunk],
        ].copy()
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        return pd.DataFrame()
    frame = pd.concat(parts, ignore_index=True)
    frame["round_group"] = frame["round_name"].map(round_group)
    frame["category_round_id"] = frame["source_url"].map(category_round_id)
    frame["terrain_group"] = frame.apply(
        lambda row: terrain_group(row["source_scope"], row.get("age_class"), row.get("category")),
        axis=1,
    )
    frame["boulder_count_observed"] = pd.to_numeric(frame["n_routes"], errors="coerce")

    keys = [
        "source_scope", "source_event_id", "event_date", "event_name", "source_url",
        "category_round_id", "pool", "gender", "category", "age_class", "round_group",
        "terrain_group", "format_identifier",
    ]
    rows: list[dict[str, object]] = []
    for values, group in frame.groupby(keys, dropna=False, sort=False):
        row = dict(zip(keys, values))
        observed = sorted(
            pd.to_numeric(group["boulder_count_observed"], errors="coerce")
            .dropna().astype(int).loc[lambda values: values.gt(0)].unique().tolist()
        )
        if len(observed) == 1:
            count = observed[0]
            status = "source-confirmed"
            count_source = "normalized results n_routes"
        elif len(observed) > 1:
            count = max(observed)
            status = "source-conflict"
            count_source = "conflicting normalized results n_routes: " + ", ".join(map(str, observed))
        else:
            default = format_default(str(row["round_group"]))
            count = default if default is not None else pd.NA
            status = "format-assumption" if default is not None else "unknown"
            count_source = "round-format assumption" if default is not None else "unavailable"
        row.update({
            "boulder_count": count,
            "boulder_count_status": status,
            "boulder_count_source": count_source,
            "observed_count_values": "|".join(map(str, observed)),
        })
        rows.append(row)
    output = pd.DataFrame(rows)
    output["round_order"] = output["round_group"].map(ROUND_ORDER).fillna(3)
    output = output.sort_values(
        ["event_date", "event_name", "pool", "terrain_group", "round_order", "category"],
        kind="stable",
    ).drop(columns="round_order")
    return output


def build_problem_inventory(rounds: pd.DataFrame) -> pd.DataFrame:
    """Collapse shared terrains and materialize one row per governed boulder."""
    keys = [
        "source_scope", "source_event_id", "event_date", "event_name", "pool", "gender",
        "round_group", "terrain_group",
    ]
    rows: list[dict[str, object]] = []
    for values, group in rounds.groupby(keys, dropna=False, sort=False):
        context = dict(zip(keys, values))
        counts = pd.to_numeric(group["boulder_count"], errors="coerce").dropna()
        counts = sorted(counts.loc[counts.gt(0)].astype(int).unique().tolist())
        statuses = set(group["boulder_count_status"].dropna().astype(str))
        if len(counts) == 1 and statuses == {"source-confirmed"}:
            status = "source-confirmed"
        elif len(counts) > 1 or "source-conflict" in statuses:
            status = "source-conflict"
        elif counts:
            status = "format-assumption"
        else:
            status = "unknown"
        count = max(counts) if counts else None
        if count is None:
            continue
        source_event_ids = sorted(set(group["source_event_id"].dropna().astype(str)))
        source_round_ids = sorted(set(group["category_round_id"].dropna().astype(str)) - {""})
        round_uid = "round-" + stable_uid(
            context["source_scope"], "|".join(source_event_ids), context["event_date"],
            context["event_name"], context["round_group"], context["gender"],
            context["terrain_group"],
        )
        for number in range(1, count + 1):
            boulder_uid = f"{round_uid}-b{number}"
            rows.append({
                **context,
                "source_event_ids": "|".join(source_event_ids),
                "source_round_ids": "|".join(source_round_ids),
                "round_uid": round_uid,
                "boulder_count": count,
                "boulder_count_status": status,
                "boulder_number": number,
                "boulder_uid": boulder_uid,
                "pre_zone_segment_uid": f"{boulder_uid}-pre-zone",
                "post_zone_segment_uid": f"{boulder_uid}-post-zone",
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--problems-output", type=Path)
    args = parser.parse_args()
    inventory = build_inventory(args.source_results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(args.output, index=False)
    status = inventory["boulder_count_status"].value_counts(dropna=False).to_dict()
    print(f"Wrote {len(inventory):,} category-round rows to {args.output}")
    print(status)
    if args.problems_output:
        problems = build_problem_inventory(inventory)
        args.problems_output.parent.mkdir(parents=True, exist_ok=True)
        problems.to_csv(args.problems_output, index=False, compression="infer")
        print(f"Wrote {len(problems):,} governed boulder rows to {args.problems_output}")


if __name__ == "__main__":
    main()
