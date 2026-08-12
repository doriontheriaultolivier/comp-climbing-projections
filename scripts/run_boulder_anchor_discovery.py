"""Run the bounded two-pass Boulder broadcast-anchor discovery lane.

Dry run is the default. Execution reads official YouTube URLs directly through
Gemini and checkpoints every validated window atomically.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from video_boulder_anchor_discovery import (  # noqa: E402
    DISCOVERY_FPS,
    DISCOVERY_SECONDS,
    PIPELINE_VERSION,
    VERIFICATION_CONTEXT_SECONDS,
    VERIFICATION_FPS,
    AnchorWindow,
    build_discovery_plan,
    build_verification_plan,
    build_video_part,
    plan_frame,
    validate_response,
)
from video_boulder_anchor_gemini_compat import (  # noqa: E402
    SCHEMA_PROFILE,
    build_gemini_prompt,
    build_gemini_response_schema,
    validate_gemini_output_budget,
)
from video_boulder_segmentation import sha256_file  # noqa: E402


REVIEWER_ID = "ai-boulder-anchor-discovery"
SAFETY_GATES = (
    "production_use_allowed",
    "athlete_scoring_allowed",
    "athlete_comparison_allowed",
    "elo_update_allowed",
)
VERIFICATION_CANDIDATE_FIELDS = {
    "anchors-only": ("anchor_candidates",),
    "anchors-and-scenes": ("anchor_candidates", "scene_candidates"),
}


class WindowReviewFailure(RuntimeError):
    """A bounded window review exhausted its governed retry allowance."""

    def __init__(self, failure: dict[str, object]):
        super().__init__(str(failure.get("error_message", "window review failed")))
        self.failure = failure


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_records(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Checkpoint JSONL is corrupt at line {line_number}") from exc
        window_id = str(record.get("window_id", "")) if isinstance(record, dict) else ""
        if not window_id:
            raise ValueError(f"Checkpoint line {line_number} has no window_id")
        if window_id in seen:
            raise ValueError(f"Checkpoint contains duplicate window_id: {window_id}")
        if not isinstance(record.get("response"), dict):
            raise ValueError(f"Checkpoint line {line_number} has no response object")
        seen.add(window_id)
        records.append(record)
    return records


def _load_failed_windows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed-window JSONL is corrupt at line {line_number}") from exc
        window_id = str(record.get("window_id", "")) if isinstance(record, dict) else ""
        if not window_id or window_id in seen:
            raise ValueError(f"Failed-window line {line_number} has a blank or duplicate window_id")
        if record.get("quarantined") is not True or any(record.get(gate) is not False for gate in SAFETY_GATES):
            raise ValueError(f"Failed-window line {line_number} has unsafe quarantine gates")
        seen.add(window_id)
        records.append(record)
    return records


def _window_from_row(row: pd.Series) -> AnchorWindow:
    def clean(value: object) -> str:
        return "" if pd.isna(value) else str(value)

    return AnchorWindow(
        window_id=str(row["window_id"]), pass_name=str(row["pass_name"]),
        event_id=int(row["event_id"]), category_round_id=int(row["category_round_id"]),
        event=str(row["event"]), gender=str(row["gender"]), round=str(row["round"]),
        video_id=str(row["video_id"]), youtube_url=str(row["youtube_url"]),
        start_seconds=int(row["start_seconds"]), end_seconds=int(row["end_seconds"]),
        fps=float(row["fps"]),
        source_candidate_id=clean(row.get("source_candidate_id", "")),
        source_discovery_window_id=clean(row.get("source_discovery_window_id", "")),
        source_candidate_json=clean(row.get("source_candidate_json", "")),
    )


def _verification_inputs(records: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    for record in records:
        response = record["response"]
        assert isinstance(response, dict)
        merged.append({**record, **response})
    return merged


def _parse_response(response: object) -> dict[str, object]:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        return parsed
    text = str(getattr(response, "text", "") or "").strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Gemini response must be a JSON object")
    return value


def _is_transient_exception(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    markers = (
        "timeout", "temporar", "rate limit", "resource exhausted", "unavailable",
        "connection", "disconnected", "remoteprotocol", "protocol error",
        "429", "500", "502", "503", "504", "deadline exceeded",
    )
    return any(marker in name or marker in message for marker in markers)


def _retry_delay(attempt: int) -> float:
    return min(2.0 ** max(0, attempt - 1), 30.0)


def _failure_record(
    window: AnchorWindow,
    *,
    model: str,
    failure_kind: str,
    exc: BaseException,
    total_attempts: int,
    validation_failures: int,
    transient_failures: int,
    validation_errors: list[str],
    attempt_history: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "window_id": window.window_id,
        "pipeline_version": PIPELINE_VERSION,
        "pass_name": window.pass_name,
        "reviewer_id": REVIEWER_ID,
        "reviewer_type": "AI",
        "model": model,
        "structured_output_profile": SCHEMA_PROFILE,
        "event_id": window.event_id,
        "category_round_id": window.category_round_id,
        "video_id": window.video_id,
        "youtube_url": window.youtube_url,
        "window_start_seconds": window.start_seconds,
        "window_end_seconds": window.end_seconds,
        "fps": window.fps,
        "source_candidate_id": window.source_candidate_id,
        "source_discovery_window_id": window.source_discovery_window_id,
        "failure_kind": failure_kind,
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:2000],
        "validation_errors": validation_errors[:20],
        "attempt_history": attempt_history,
        "total_attempts": total_attempts,
        "validation_failures": validation_failures,
        "transient_failures": transient_failures,
        "quarantined": True,
        "retryable_next_run": True,
        "quarantined_at_utc": datetime.now(timezone.utc).isoformat(),
        "media_download_required": False,
        **{gate: False for gate in SAFETY_GATES},
    }


def _review_window(
    client: object,
    types_module: object,
    window: AnchorWindow,
    *,
    model: str,
    seed: int,
    max_validation_retries: int,
    max_transient_retries: int,
    sleep_fn=time.sleep,
) -> dict[str, object]:
    """Return one valid response or raise a structured bounded failure."""

    validation_failures = 0
    transient_failures = 0
    total_attempts = 0
    last_errors: list[str] = []
    attempt_history: list[dict[str, object]] = []
    while True:
        total_attempts += 1
        retry_note = ""
        if last_errors:
            retry_note = (
                "\n\nThe previous response was rejected. Return a complete replacement JSON "
                "object. Correct these validation errors: " + "; ".join(last_errors[:12])
            )
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    build_video_part(types_module, window),
                    build_gemini_prompt(window) + retry_note,
                ],
                config=types_module.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=build_gemini_response_schema(window),
                    max_output_tokens=16384, temperature=0.1, seed=seed + total_attempts,
                ),
            )
        except Exception as exc:
            attempt_history.append({
                "attempt": total_attempts,
                "kind": "transient_api" if _is_transient_exception(exc) else "permanent_api",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:1000],
            })
            if _is_transient_exception(exc) and transient_failures < max_transient_retries:
                transient_failures += 1
                sleep_fn(_retry_delay(transient_failures))
                continue
            raise WindowReviewFailure(_failure_record(
                window, model=model,
                failure_kind="transient_exhausted" if _is_transient_exception(exc) else "api_permanent",
                exc=exc, total_attempts=total_attempts,
                validation_failures=validation_failures,
                transient_failures=transient_failures,
                validation_errors=last_errors,
                attempt_history=attempt_history,
            )) from exc
        try:
            payload = _parse_response(response)
            last_errors = [
                *validate_response(payload, window),
                *validate_gemini_output_budget(payload),
            ]
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            payload = {}
            last_errors = [f"{type(exc).__name__}: {str(exc)[:1000]}"]
        if not last_errors:
            return payload
        attempt_history.append({
            "attempt": total_attempts,
            "kind": "validation",
            "errors": last_errors[:20],
        })
        validation_failures += 1
        if validation_failures <= max_validation_retries:
            continue
        exc = ValueError("; ".join(last_errors))
        raise WindowReviewFailure(_failure_record(
            window, model=model, failure_kind="validation_exhausted", exc=exc,
            total_attempts=total_attempts, validation_failures=validation_failures,
            transient_failures=transient_failures, validation_errors=last_errors,
            attempt_history=attempt_history,
        )) from exc


def _validate_cached(
    record: dict[str, object], window: AnchorWindow, *, model: str, pass_name: str
) -> list[str]:
    response = record.get("response")
    errors = (
        [*validate_response(response, window), *validate_gemini_output_budget(response)]
        if isinstance(response, dict)
        else ["missing response"]
    )
    expected = {
        "pipeline_version": PIPELINE_VERSION, "pass_name": pass_name,
        "reviewer_id": REVIEWER_ID, "reviewer_type": "AI", "model": model,
        "structured_output_profile": SCHEMA_PROFILE,
        "youtube_url": window.youtube_url, "video_id": window.video_id,
        "window_start_seconds": window.start_seconds,
        "window_end_seconds": window.end_seconds, "fps": window.fps,
        "media_download_required": False,
    }
    if pass_name == "verification":
        expected.update({
            "event_id": window.event_id,
            "category_round_id": window.category_round_id,
            "source_candidate_id": window.source_candidate_id,
            "source_discovery_window_id": window.source_discovery_window_id,
        })
    for key, value in expected.items():
        if record.get(key) != value:
            errors.append(f"cached metadata mismatch: {key}")
    for gate in SAFETY_GATES:
        if record.get(gate) is not False:
            errors.append(f"cached safety gate is not false: {gate}")
    return errors


def _prepare_verification_records(
    discovery_records: list[dict[str, object]],
    manifest: pd.DataFrame,
    *,
    event_id: int | None,
    window_seconds: int,
) -> list[dict[str, object]]:
    """Validate the complete discovery checkpoint, then filter by trusted plan event."""

    discovery_by_id = {
        window.window_id: window
        for window in build_discovery_plan(
            manifest, window_seconds=window_seconds, event_id=None
        )
    }
    duration_by_video = {
        str(row["video_id"]): int(row["duration_seconds"])
        for _, row in manifest.iterrows()
    }
    selected: list[dict[str, object]] = []
    for record in discovery_records:
        window_id = str(record.get("window_id", ""))
        source = discovery_by_id.get(window_id)
        if source is None:
            raise ValueError(f"Unknown discovery window: {window_id}")
        errors = _validate_cached(
            record, source, model=str(record.get("model", "")), pass_name="discovery"
        )
        if errors:
            raise ValueError(f"Invalid frozen discovery window {window_id}: {'; '.join(errors)}")
        recorded_event = record.get("source_event_id")
        if recorded_event is not None and int(recorded_event) != source.event_id:
            raise ValueError(f"Frozen discovery provenance event mismatch: {window_id}")
        recorded_checkpoint = str(record.get("source_checkpoint_id", ""))
        if recorded_checkpoint and recorded_checkpoint != f"event-{source.event_id}":
            raise ValueError(f"Frozen discovery checkpoint event mismatch: {window_id}")
        if event_id is not None and source.event_id != event_id:
            continue
        if source.video_id not in duration_by_video:
            raise ValueError(f"Missing official duration for discovery window: {window_id}")
        selected.append({
            **record,
            "event_id": source.event_id,
            "category_round_id": source.category_round_id,
            "event": source.event,
            "gender": source.gender,
            "round": source.round,
            "video_id": source.video_id,
            "youtube_url": source.youtube_url,
            "video_duration_seconds": duration_by_video[source.video_id],
        })
    if not selected:
        scope = f"event {event_id}" if event_id is not None else "the requested scope"
        raise ValueError(f"No validated frozen discovery records remain for {scope}")
    return selected


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass", dest="pass_name", choices=("discovery", "verification"), required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/video_2026_source_manifest.csv")
    parser.add_argument("--required-source-manifest-sha256", default="")
    parser.add_argument("--event-id", type=int)
    parser.add_argument("--target-event-ids", default="1479,1480,1482")
    parser.add_argument("--discovery-checkpoint", type=Path)
    parser.add_argument("--required-discovery-checkpoint-sha256", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--window-seconds", type=int, default=DISCOVERY_SECONDS)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--verification-seconds", type=int, default=VERIFICATION_CONTEXT_SECONDS)
    parser.add_argument(
        "--verification-candidate-source",
        choices=tuple(VERIFICATION_CANDIDATE_FIELDS),
        default="anchors-only",
    )
    parser.add_argument("--contract-namespace", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--max-validation-retries", type=int, default=1)
    parser.add_argument("--max-transient-retries", type=int, default=2)
    parser.add_argument("--request-timeout-ms", type=int, default=900000)
    parser.add_argument("--continue-on-window-failure", action="store_true")
    return parser.parse_args()


def _require_bounded_execution(
    pass_name: str,
    *,
    execute: bool,
    max_windows: int | None,
) -> None:
    if not execute:
        return
    maximum = 12 if pass_name == "verification" else 30
    if max_windows is None or not 1 <= max_windows <= maximum:
        raise SystemExit(
            f"--execute requires --max-windows between 1 and {maximum} for {pass_name}"
        )


def main() -> int:
    args = _arguments()
    try:
        target_events = {int(item) for item in args.target_event_ids.split(",") if item.strip()}
    except ValueError:
        raise SystemExit("target-event-ids must be comma-separated positive integers")
    if not target_events or any(value <= 0 for value in target_events):
        raise SystemExit("target-event-ids must be comma-separated positive integers")
    if args.event_id is not None and args.event_id not in target_events:
        raise SystemExit("event-id must be in target-event-ids")
    _require_bounded_execution(
        args.pass_name, execute=args.execute, max_windows=args.max_windows
    )
    if not 0 <= args.max_validation_retries <= 3:
        raise SystemExit("--max-validation-retries must be between 0 and 3")
    if not 0 <= args.max_transient_retries <= 5:
        raise SystemExit("--max-transient-retries must be between 0 and 5")
    if not 60000 <= args.request_timeout_ms <= 1800000:
        raise SystemExit("--request-timeout-ms must be between 60000 and 1800000")
    if args.contract_namespace and (
        len(args.contract_namespace) > 96
        or not args.contract_namespace[0].isalnum()
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789.-" for character in args.contract_namespace)
    ):
        raise SystemExit("--contract-namespace must be a lowercase alphanumeric, dot or dash token of at most 96 characters")
    if not args.manifest.is_file():
        raise SystemExit(f"Missing manifest: {args.manifest}")
    source_manifest_hash = sha256_file(args.manifest)
    if (
        args.required_source_manifest_sha256
        and source_manifest_hash != args.required_source_manifest_sha256.casefold()
    ):
        raise SystemExit("Source manifest does not match the required frozen SHA-256")
    manifest = pd.read_csv(args.manifest)
    source_checkpoint_hash = ""
    if args.pass_name == "discovery":
        fps = args.fps if args.fps is not None else DISCOVERY_FPS
        windows = build_discovery_plan(
            manifest, window_seconds=args.window_seconds, fps=fps, event_id=args.event_id,
            target_events=target_events,
        )
    else:
        if args.discovery_checkpoint is None or not args.discovery_checkpoint.is_file():
            raise SystemExit("Verification requires --discovery-checkpoint review_windows.jsonl")
        source_checkpoint_hash = sha256_file(args.discovery_checkpoint)
        if (
            args.required_discovery_checkpoint_sha256
            and source_checkpoint_hash != args.required_discovery_checkpoint_sha256.casefold()
        ):
            raise SystemExit("Verification discovery checkpoint does not match the required frozen SHA-256")
        discovery_records = _prepare_verification_records(
            _load_records(args.discovery_checkpoint), manifest,
            event_id=args.event_id, window_seconds=args.window_seconds,
        )
        fps = args.fps if args.fps is not None else VERIFICATION_FPS
        windows = build_verification_plan(
            _verification_inputs(discovery_records),
            context_seconds=args.verification_seconds, fps=fps,
            candidate_fields=VERIFICATION_CANDIDATE_FIELDS[args.verification_candidate_source],
        )
    if not windows:
        raise SystemExit("No windows were planned")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "window_plan.csv"
    plan_frame(windows).to_csv(plan_path, index=False, lineterminator="\n")
    contract = {
        "pipeline_version": PIPELINE_VERSION, "reviewer_id": REVIEWER_ID,
        "pass_name": args.pass_name, "model": args.model,
        "source_manifest_sha256": source_manifest_hash,
        "event_id_filter": args.event_id or 0,
        "source_discovery_checkpoint_sha256": source_checkpoint_hash,
        "window_plan_sha256": sha256_file(plan_path), "fps": fps,
        "window_seconds": args.window_seconds,
        "verification_seconds": args.verification_seconds,
        "max_validation_retries": args.max_validation_retries,
        "max_transient_retries": args.max_transient_retries,
        "request_timeout_ms": args.request_timeout_ms,
        "structured_output_profile": SCHEMA_PROFILE,
    }
    if args.pass_name == "verification":
        contract.update({
            "verification_candidate_source": args.verification_candidate_source,
            "contract_namespace": args.contract_namespace,
        })
    contract_path = args.output_dir / "checkpoint_contract.json"
    if contract_path.is_file() and json.loads(contract_path.read_text(encoding="utf-8")) != contract:
        raise SystemExit("Checkpoint contract changed; use a clean output directory")
    _atomic_json(contract_path, contract)

    checkpoint_path = args.output_dir / "review_windows.jsonl"
    cached = _load_records(checkpoint_path)
    by_id = {window.window_id: window for window in windows}
    invalid: list[dict[str, object]] = []
    valid: list[dict[str, object]] = []
    for record in cached:
        window = by_id.get(str(record["window_id"]))
        errors = ["window is not in the current plan"] if window is None else _validate_cached(
            record, window, model=args.model, pass_name=args.pass_name
        )
        if errors:
            invalid.append({"window_id": record["window_id"], "errors": errors})
        else:
            valid.append(record)
    if invalid:
        _atomic_jsonl(checkpoint_path, valid)
    cached = valid
    completed = {str(record["window_id"]) for record in cached}
    failed_path = args.output_dir / "failed_windows.jsonl"
    failed_by_id: dict[str, dict[str, object]] = {}
    for record in _load_failed_windows(failed_path):
        window_id = str(record["window_id"])
        if window_id not in by_id:
            raise SystemExit(f"Quarantined checkpoint contains unknown window: {window_id}")
        for key, expected in (
            ("pipeline_version", PIPELINE_VERSION),
            ("pass_name", args.pass_name),
            ("model", args.model),
        ):
            if record.get(key) != expected:
                raise SystemExit(f"Quarantined checkpoint metadata mismatch: {window_id} {key}")
        if window_id not in completed:
            failed_by_id[window_id] = record
    if failed_path.is_file():
        _atomic_jsonl(failed_path, failed_by_id.values())
    fresh_pending = [
        window for window in windows
        if window.window_id not in completed and window.window_id not in failed_by_id
    ]
    retry_pending = [
        window for window in windows
        if window.window_id not in completed and window.window_id in failed_by_id
    ]
    pending = [*fresh_pending, *retry_pending]
    selected = pending[: args.max_windows] if args.max_windows is not None else pending
    run_path = args.output_dir / "run_manifest.json"
    run = {
        **contract, "reviewer_type": "AI", "structured_output": "response_json_schema",
        "status": "DRY RUN" if not args.execute else "RUNNING",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "window_count": len(windows), "video_count": len({window.video_id for window in windows}),
        "resumed_window_count": len(completed), "selected_pending_window_count": len(selected),
        "repaired_invalid_cached_windows": invalid, "media_download_required": False,
        "continue_on_window_failure": args.continue_on_window_failure,
        "max_validation_retries": args.max_validation_retries,
        "max_transient_retries": args.max_transient_retries,
        "quarantined_failed_windows": sorted(failed_by_id),
        "quarantined_failed_window_count": len(failed_by_id),
        "fresh_pending_window_count": len(fresh_pending),
        "retry_pending_window_count": len(retry_pending),
        "selected_retry_window_count": sum(window.window_id in failed_by_id for window in selected),
        "one_video_per_request": True, **{gate: False for gate in SAFETY_GATES},
    }
    _atomic_json(run_path, run)
    if not args.execute:
        print(f"Dry run planned {len(windows)} {args.pass_name} windows across {run['video_count']} official videos; no download.")
        return 0

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is required only with --execute")
    from google import genai
    from google.genai import types

    if not checkpoint_path.is_file():
        _atomic_jsonl(checkpoint_path, cached)
    if not failed_path.is_file():
        _atomic_jsonl(failed_path, failed_by_id.values())
    run["google_genai_version"] = importlib.metadata.version("google-genai")
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=args.request_timeout_ms),
    )
    try:
        for number, window in enumerate(selected, 1):
            try:
                payload = _review_window(
                    client, types, window, model=args.model, seed=20260801 + number * 10,
                    max_validation_retries=args.max_validation_retries,
                    max_transient_retries=args.max_transient_retries,
                )
            except WindowReviewFailure as exc:
                failed_by_id[window.window_id] = exc.failure
                _atomic_jsonl(failed_path, failed_by_id.values())
                run["quarantined_failed_windows"] = sorted(failed_by_id)
                run["quarantined_failed_window_count"] = len(failed_by_id)
                run["last_failed_window"] = exc.failure
                _atomic_json(run_path, run)
                if args.continue_on_window_failure:
                    continue
                run["status"] = "FAILED WITH QUARANTINE"
                run["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
                _atomic_json(run_path, run)
                raise
            record = {
                "window_id": window.window_id, "response": payload,
                "pipeline_version": PIPELINE_VERSION, "pass_name": args.pass_name,
                "reviewer_id": REVIEWER_ID, "reviewer_type": "AI", "model": args.model,
                "structured_output_profile": SCHEMA_PROFILE,
                "youtube_url": window.youtube_url, "video_id": window.video_id,
                "window_start_seconds": window.start_seconds,
                "window_end_seconds": window.end_seconds, "fps": window.fps,
                "event_id": window.event_id,
                "category_round_id": window.category_round_id,
                "source_candidate_id": window.source_candidate_id,
                "source_discovery_window_id": window.source_discovery_window_id,
                "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
                "media_download_required": False, **{gate: False for gate in SAFETY_GATES},
            }
            cached.append(record)
            _atomic_jsonl(checkpoint_path, cached)
            completed.add(window.window_id)
            failed_by_id.pop(window.window_id, None)
            _atomic_jsonl(failed_path, failed_by_id.values())
            run["completed_window_count"] = len(completed)
            run["quarantined_failed_windows"] = sorted(failed_by_id)
            run["quarantined_failed_window_count"] = len(failed_by_id)
            _atomic_json(run_path, run)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    if len(completed) == len(windows):
        run["status"] = "COMPLETE"
    elif failed_by_id:
        run["status"] = "BOUNDED PARTIAL WITH QUARANTINE"
    else:
        run["status"] = "BOUNDED PARTIAL"
    run["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    run["completed_window_count"] = len(completed)
    _atomic_json(run_path, run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
