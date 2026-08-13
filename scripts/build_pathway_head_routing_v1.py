"""Classify replay events into demonstrated and projected pathway heads.

This is a research routing contract, not a model change.  In particular, the
legacy ``ifsc_non_wc`` component is split using governed ``age_class`` metadata
so that youth regional events are never presented as senior/open REG-IFSC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


SCHEMA = "pathway-head-routing-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_event(target_domain: str, age_classes: tuple[str, ...]) -> str:
    if target_domain == "wc+":
        return "WC_PLUS"
    if target_domain == "ifsc_youth_world":
        return "YW_IFSC"
    if target_domain == "ifsc_non_wc":
        if age_classes == ("Senior / Open",):
            return "REG_IFSC"
        if age_classes == ("Youth",):
            return "YOUTH_REG_IFSC_GRAPH_ONLY"
        return "HOLD_MIXED_OR_UNKNOWN_AGE_CLASS"
    return "OTHER_GRAPH_EVIDENCE"


def build_routing(plan: dict, prepared: pd.DataFrame) -> pd.DataFrame:
    required = {"event_date", "event_name", "pool", "age_class"}
    missing = required - set(prepared.columns)
    if missing:
        raise ValueError(f"prepared rows missing columns: {sorted(missing)}")
    meta = (
        prepared.assign(event_date=pd.to_datetime(prepared["event_date"]))
        .groupby(["event_date", "event_name", "pool"], dropna=False)["age_class"]
        .agg(lambda values: tuple(sorted(set(values.dropna().astype(str)))))
        .rename("age_classes")
        .reset_index()
    )
    events = pd.DataFrame(
        [
            {
                "event_id": event["event_id"],
                "event_date": pd.Timestamp(event["event_date"]),
                "event_name": event["event_name"],
                "pool": event["pool"],
                "legacy_target_domain": event["target_domain"],
            }
            for event in plan["events"]
        ]
    )
    output = events.merge(
        meta, on=["event_date", "event_name", "pool"], how="left", validate="many_to_one"
    )
    governed_age_required = output["legacy_target_domain"].eq("ifsc_non_wc")
    missing_governed = governed_age_required & output["age_classes"].isna()
    if missing_governed.any():
        missing_ids = output.loc[missing_governed, "event_id"].tolist()
        raise ValueError(f"event metadata join failed: {missing_ids[:5]}")
    output["age_classes"] = output["age_classes"].map(
        lambda value: value if isinstance(value, tuple) else tuple()
    )
    output["pathway_head"] = [
        classify_event(domain, ages)
        for domain, ages in zip(output["legacy_target_domain"], output["age_classes"])
    ]
    output["direct_demonstrated_head"] = output["pathway_head"].map(
        {"YW_IFSC": "YW-IFSC", "REG_IFSC": "REG-IFSC", "WC_PLUS": "WC+"}
    )
    output["projection_use"] = output["pathway_head"].map(
        {
            "YW_IFSC": "direct YW-IFSC evidence and shared-graph evidence",
            "REG_IFSC": "direct REG-IFSC evidence and shared-graph evidence",
            "WC_PLUS": "direct WC+ evidence and shared-graph evidence",
            "YOUTH_REG_IFSC_GRAPH_ONLY": "shared-graph evidence; not direct REG-IFSC",
            "OTHER_GRAPH_EVIDENCE": "shared-graph evidence only",
            "HOLD_MIXED_OR_UNKNOWN_AGE_CLASS": "withheld pending event review",
        }
    )
    output["age_classes"] = output["age_classes"].map(lambda x: "|".join(x))
    return output.sort_values(["event_date", "event_id"], kind="stable").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    prepared = pd.read_parquet(args.prepared)
    output = build_routing(plan, prepared)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "pathway_head_routing.csv"
    output.to_csv(output_path, index=False, encoding="utf-8", lineterminator="\n")
    counts = output["pathway_head"].value_counts().sort_index().to_dict()
    receipt = {
        "schema": SCHEMA,
        "status": "RESEARCH_ROUTING_CONTRACT_NOT_A_MODEL_CHANGE",
        "inputs": {"plan": sha256(args.plan), "prepared": sha256(args.prepared)},
        "rows": len(output),
        "counts": {key: int(value) for key, value in counts.items()},
        "output_sha256": sha256(output_path),
        "next_model_domains": ["yw_ifsc", "youth_reg_ifsc", "reg_ifsc", "wc+"],
        "authority": {"production": False, "app": False, "deployment": False},
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
