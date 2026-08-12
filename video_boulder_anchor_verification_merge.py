"""Deterministically merge event-scoped anchor-verification checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

from scripts.run_boulder_anchor_discovery import (
    SAFETY_GATES,
    _load_failed_windows,
    _load_records,
    _validate_cached,
    _window_from_row,
)
from video_boulder_anchor_discovery import PIPELINE_VERSION, AnchorWindow
from video_boulder_segmentation import sha256_file


MERGE_VERSION = "ifsc-boulder-anchor-verification-merge-v1"
EXPECTED_EVENT_WINDOWS = {1479: 73, 1480: 73, 1482: 67}
FROZEN_DISCOVERY_CHECKPOINT_SHA256 = (
    "49fdce904c9b5a720c14f38033253378def7b22d95b291ac0cb30b72daa0e3ab"
)
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "8f6ce23cfd3da6d01ef75dbf795e24cba3040aee8aa143add8b7e69feadc8772"
)
CONTRACT_NAMESPACE = "boulder-anchor-verification-v4-frozen77-anchors1fps"
MERGEABLE_RUN_STATUSES = {
    "BOUNDED PARTIAL",
    "BOUNDED PARTIAL WITH QUARANTINE",
    "COMPLETE",
}
COMMON_CONTRACT_FIELDS = (
    "pipeline_version",
    "pass_name",
    "model",
    "source_manifest_sha256",
    "source_discovery_checkpoint_sha256",
    "fps",
    "window_seconds",
    "verification_seconds",
    "verification_candidate_source",
    "contract_namespace",
    "max_validation_retries",
    "max_transient_retries",
    "request_timeout_ms",
    "structured_output_profile",
)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _read_json(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _source_hash(path: Path) -> str:
    return sha256_file(path) if path.is_file() else ""


def _enrich_record(
    record: dict[str, object],
    window: AnchorWindow,
    *,
    event_id: int,
    hashes: dict[str, str],
) -> dict[str, object]:
    return {
        **record,
        "event_id": window.event_id,
        "category_round_id": window.category_round_id,
        "event": window.event,
        "gender": window.gender,
        "round": window.round,
        "source_candidate_id": window.source_candidate_id,
        "source_discovery_window_id": window.source_discovery_window_id,
        "source_candidate_json": window.source_candidate_json,
        "source_checkpoint_id": f"event-{event_id}",
        "source_checkpoint_contract_sha256": hashes["checkpoint_contract_sha256"],
        "source_window_plan_sha256": hashes["window_plan_sha256"],
        "source_review_windows_sha256": hashes["review_windows_sha256"],
        "source_failed_windows_sha256": hashes["failed_windows_sha256"],
        "source_run_manifest_sha256": hashes["run_manifest_sha256"],
    }


def merge_verification_checkpoints(
    input_dirs: list[Path],
    output_dir: Path,
) -> dict[str, object]:
    """Validate and merge the three bounded event checkpoints; never call a model."""

    if len(input_dirs) != 3:
        raise ValueError("exactly three event verification checkpoints are required")
    common_contract: dict[str, object] | None = None
    sources: list[dict[str, object]] = []
    all_windows: list[AnchorWindow] = []
    merged_reviews: list[dict[str, object]] = []
    merged_failed: list[dict[str, object]] = []
    seen_events: set[int] = set()
    seen_windows: set[str] = set()

    for input_dir in input_dirs:
        contract_path = input_dir / "checkpoint_contract.json"
        plan_path = input_dir / "window_plan.csv"
        review_path = input_dir / "review_windows.jsonl"
        failed_path = input_dir / "failed_windows.jsonl"
        run_path = input_dir / "run_manifest.json"
        contract = _read_json(contract_path, "checkpoint contract")
        run = _read_json(run_path, "run manifest")
        for path, label in (
            (plan_path, "window plan"),
            (review_path, "review checkpoint"),
            (failed_path, "failed-window quarantine"),
        ):
            if not path.is_file():
                raise ValueError(f"{label} is missing: {path}")
        if contract.get("window_plan_sha256") != sha256_file(plan_path):
            raise ValueError(f"verification window-plan hash mismatch: {input_dir}")
        if contract.get("pipeline_version") != PIPELINE_VERSION or contract.get("pass_name") != "verification":
            raise ValueError(f"only governed verification checkpoints can be merged: {input_dir}")
        if contract.get("verification_candidate_source") != "anchors-only":
            raise ValueError(f"verification checkpoint is not discovery-anchor-only: {input_dir}")
        if contract.get("source_discovery_checkpoint_sha256") != FROZEN_DISCOVERY_CHECKPOINT_SHA256:
            raise ValueError(f"verification checkpoint is not derived from the frozen 77-window packet: {input_dir}")
        if contract.get("source_manifest_sha256") != FROZEN_SOURCE_MANIFEST_SHA256:
            raise ValueError(f"verification checkpoint source manifest changed: {input_dir}")
        if float(contract.get("fps", 0)) != 1.0 or int(contract.get("verification_seconds", 0)) != 60:
            raise ValueError(f"verification checkpoint must use 60-second windows at 1 FPS: {input_dir}")
        if contract.get("contract_namespace") != CONTRACT_NAMESPACE:
            raise ValueError(f"verification checkpoint requires an isolated contract namespace: {input_dir}")
        if run.get("status") not in MERGEABLE_RUN_STATUSES:
            raise ValueError(f"verification run status is not mergeable: {input_dir}")
        for field in (*COMMON_CONTRACT_FIELDS, "event_id_filter", "window_plan_sha256"):
            if run.get(field) != contract.get(field):
                raise ValueError(f"verification run contract mismatch for {field}: {input_dir}")
        for gate in SAFETY_GATES:
            if run.get(gate) is not False:
                raise ValueError(f"unsafe verification run-manifest gate {gate}: {input_dir}")

        event_id = int(contract.get("event_id_filter", 0))
        if event_id not in EXPECTED_EVENT_WINDOWS or event_id in seen_events:
            raise ValueError(f"verification event checkpoint must be unique and governed: {event_id}")
        seen_events.add(event_id)
        source_common = {field: contract.get(field) for field in COMMON_CONTRACT_FIELDS}
        if common_contract is None:
            common_contract = source_common
        elif source_common != common_contract:
            raise ValueError(f"verification event contract is incompatible: {event_id}")

        plan = pd.read_csv(plan_path, keep_default_na=False)
        windows = [_window_from_row(row) for _, row in plan.iterrows()]
        if len(windows) != EXPECTED_EVENT_WINDOWS[event_id]:
            raise ValueError(f"verification event {event_id} anchor count changed")
        if {window.event_id for window in windows} != {event_id}:
            raise ValueError(f"verification event {event_id} plan contains another event")
        if any(
            window.pass_name != "verification"
            or window.fps != 1.0
            or window.end_seconds - window.start_seconds != 60
            or not window.source_candidate_id
            or not window.source_discovery_window_id
            or not window.source_candidate_json
            for window in windows
        ):
            raise ValueError(f"verification event {event_id} plan violates the frozen window contract")
        for window in windows:
            try:
                source_candidate = json.loads(window.source_candidate_json)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"verification event {event_id} plan has invalid source candidate JSON: "
                    f"{window.window_id}"
                ) from exc
            if (
                not isinstance(source_candidate, dict)
                or str(source_candidate.get("candidate_id", ""))
                != window.source_candidate_id
                or not str(source_candidate.get("anchor_type", "")).strip()
            ):
                raise ValueError(
                    f"verification event {event_id} plan source candidate does not match "
                    f"the governed anchor: {window.window_id}"
                )
        duplicates = {window.window_id for window in windows}.intersection(seen_windows)
        if duplicates:
            raise ValueError(f"duplicate verification windows across events: {sorted(duplicates)}")
        seen_windows.update(window.window_id for window in windows)
        all_windows.extend(windows)
        by_id = {window.window_id: window for window in windows}

        reviews = _load_records(review_path)
        failed = _load_failed_windows(failed_path)
        completed_ids: set[str] = set()
        for record in reviews:
            window_id = str(record["window_id"])
            window = by_id.get(window_id)
            if window is None:
                raise ValueError(f"completed verification window is outside event plan: {window_id}")
            errors = _validate_cached(
                record, window, model=str(contract["model"]), pass_name="verification"
            )
            if errors:
                raise ValueError(f"invalid completed verification window {window_id}: {'; '.join(errors)}")
            completed_ids.add(window_id)
        failed_ids: set[str] = set()
        for record in failed:
            window_id = str(record["window_id"])
            window = by_id.get(window_id)
            if window is None:
                raise ValueError(f"quarantined verification window is outside event plan: {window_id}")
            if window_id in completed_ids:
                raise ValueError(f"verification window is both completed and quarantined: {window_id}")
            for key, expected in (
                ("pipeline_version", PIPELINE_VERSION),
                ("pass_name", "verification"),
                ("model", contract["model"]),
                ("structured_output_profile", contract["structured_output_profile"]),
                ("event_id", window.event_id),
                ("category_round_id", window.category_round_id),
                ("video_id", window.video_id),
                ("youtube_url", window.youtube_url),
                ("window_start_seconds", window.start_seconds),
                ("window_end_seconds", window.end_seconds),
                ("fps", window.fps),
                ("source_candidate_id", window.source_candidate_id),
                ("source_discovery_window_id", window.source_discovery_window_id),
            ):
                if record.get(key) != expected:
                    raise ValueError(f"verification quarantine metadata mismatch: {window_id} {key}")
            failed_ids.add(window_id)

        hashes = {
            "checkpoint_contract_sha256": sha256_file(contract_path),
            "window_plan_sha256": sha256_file(plan_path),
            "review_windows_sha256": _source_hash(review_path),
            "failed_windows_sha256": _source_hash(failed_path),
            "run_manifest_sha256": sha256_file(run_path),
        }
        review_by_id = {str(record["window_id"]): record for record in reviews}
        failed_by_id = {str(record["window_id"]): record for record in failed}
        for window in windows:
            if window.window_id in review_by_id:
                merged_reviews.append(_enrich_record(
                    review_by_id[window.window_id], window,
                    event_id=event_id, hashes=hashes,
                ))
            elif window.window_id in failed_by_id:
                merged_failed.append(_enrich_record(
                    failed_by_id[window.window_id], window,
                    event_id=event_id, hashes=hashes,
                ))
        sources.append({
            "source_checkpoint_id": f"event-{event_id}",
            "event_id": event_id,
            "planned_window_count": len(windows),
            "completed_window_count": len(completed_ids),
            "quarantined_window_count": len(failed_ids),
            "pending_window_count": len(windows) - len(completed_ids) - len(failed_ids),
            "run_status": str(run.get("status", "")),
            **hashes,
        })

    if seen_events != set(EXPECTED_EVENT_WINDOWS):
        raise ValueError("verification merge requires Madrid, Prague and Innsbruck event checkpoints")
    sources.sort(key=lambda value: int(value["event_id"]))
    all_windows.sort(key=lambda value: (value.event_id, value.video_id, value.start_seconds, value.window_id))
    order = {window.window_id: number for number, window in enumerate(all_windows)}
    merged_reviews.sort(key=lambda value: order[str(value["window_id"])])
    merged_failed.sort(key=lambda value: order[str(value["window_id"])])
    completed_ids = {str(record["window_id"]) for record in merged_reviews}
    failed_ids = {str(record["window_id"]) for record in merged_failed}
    source_by_event = {int(source["event_id"]): source for source in sources}
    coverage_rows = []
    for window in all_windows:
        status = (
            "completed" if window.window_id in completed_ids
            else "quarantined_retryable" if window.window_id in failed_ids
            else "pending_unreviewed"
        )
        source = source_by_event[window.event_id]
        coverage_rows.append({
            **window.__dict__,
            "status": status,
            "source_checkpoint_id": source["source_checkpoint_id"],
            "source_checkpoint_contract_sha256": source["checkpoint_contract_sha256"],
            "source_window_plan_sha256": source["window_plan_sha256"],
            "media_download_required": False,
            **{gate: False for gate in SAFETY_GATES},
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    review_output = output_dir / "merged_review_windows.jsonl"
    failed_output = output_dir / "merged_failed_windows.jsonl"
    coverage_output = output_dir / "coverage_windows.csv"
    _atomic_jsonl(review_output, merged_reviews)
    _atomic_jsonl(failed_output, merged_failed)
    _atomic_csv(coverage_output, pd.DataFrame(coverage_rows))
    by_event = [
        {
            "event_id": source["event_id"],
            "planned": source["planned_window_count"],
            "completed": source["completed_window_count"],
            "quarantined_retryable": source["quarantined_window_count"],
            "pending": source["pending_window_count"],
        }
        for source in sources
    ]
    summary = {
        "merge_version": MERGE_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "common_contract": common_contract or {},
        "source_event_ids": sorted(seen_events),
        "source_count": len(sources),
        "planned_window_count": len(all_windows),
        "completed_window_count": len(completed_ids),
        "quarantined_retryable_window_count": len(failed_ids),
        "pending_window_count": len(all_windows) - len(completed_ids) - len(failed_ids),
        "coverage_by_event": by_event,
        "source_verification_model_executed": True,
        "merge_executes_model": False,
        "media_download_required": False,
        **{gate: False for gate in SAFETY_GATES},
    }
    summary_path = output_dir / "coverage_summary.json"
    _atomic_json(summary_path, summary)
    manifest = {
        "merge_version": MERGE_VERSION,
        "sources": sources,
        "outputs": {
            "merged_review_windows_sha256": sha256_file(review_output),
            "merged_failed_windows_sha256": sha256_file(failed_output),
            "coverage_windows_sha256": sha256_file(coverage_output),
            "coverage_summary_sha256": sha256_file(summary_path),
        },
        "deterministic_order": "event_id, video_id, start_seconds, window_id",
        "source_verification_model_executed": True,
        "merge_executes_model": False,
        **{gate: False for gate in SAFETY_GATES},
    }
    _atomic_json(output_dir / "merge_manifest.json", manifest)
    return summary
