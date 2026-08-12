"""Plan the next official 2026 Boulder semi/final anchor batches, offline."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

INITIAL_EVENT_IDS = {1479, 1480, 1482}

def build_queue(manifest: pd.DataFrame) -> pd.DataFrame:
    required = {"event_id", "category_round_id", "event", "gender", "round", "video_id", "official_channel", "metadata_status", "official_youtube_url", "duration_seconds"}
    missing = required - set(manifest)
    if missing: raise ValueError("manifest is missing: " + ", ".join(sorted(missing)))
    rows = manifest.loc[
        manifest["round"].isin(["Semi-final", "Final"])
        & manifest["official_channel"].eq("World Climbing")
        & manifest["metadata_status"].eq("Verified by public YouTube oEmbed")
        & ~manifest["event_id"].isin(INITIAL_EVENT_IDS)
    ].copy()
    if rows.empty: return rows
    if rows["video_id"].isna().any() or rows["duration_seconds"].isna().any() or rows["category_round_id"].duplicated().any():
        raise ValueError("expansion source identity or duration is incomplete")
    rows["batch_id"] = rows["event_id"].map(lambda value: f"boulder-anchor-expansion-v2-event-{int(value)}")
    rows["review_status"] = "PENDING_BOUNDED_ANCHOR_DISCOVERY"
    rows["execution_authorized"] = False
    rows["external_transmission_authorized"] = False
    return rows.sort_values(["event_id", "gender", "round", "category_round_id"], kind="stable").reset_index(drop=True)

def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--required-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if _sha(args.manifest) != args.required_manifest_sha256: raise SystemExit("manifest hash mismatch")
    queue = build_queue(pd.read_csv(args.manifest))
    payload = {"source_manifest_sha256": _sha(args.manifest), "batch_count": int(queue["batch_id"].nunique()) if not queue.empty else 0, "source_count": len(queue), "records": queue.to_dict("records")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Planned {len(queue)} expansion sources across {payload['batch_count']} event batches; no model was called.")
    return 0
if __name__ == "__main__": raise SystemExit(main())
